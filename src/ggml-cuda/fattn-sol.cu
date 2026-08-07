#include "fattn-sol.cuh"

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <vector>

namespace {

constexpr int kSolHeadDim = 128;
// Reference Sol-Attn geometry.  These are deliberately distinct from the
// legacy Q32/KV32 route-list experiment below: BLOCK is tokens, GROUP is
// blocks, not tokens.
constexpr int kSolFusedTokenBlock = 64;
// The reference pointer kernel autotunes 32 and 64 block epochs.  At H3's
// 128-wide value head, G64 halves the expensive proxy P@V epochs while using
// the same 64x64 score/probability and K/V staging allocation already needed
// by one exact block.  It is therefore a real work-decomposition improvement,
// not a routing-threshold change.
constexpr int kSolFusedGroupBlocks = 64;
// The paired-selected path is a separate opt-in dispatch.  It is deliberately
// kept distinct from the verified single-K64 recurrence and dense-exact path.
constexpr int kSolFusedPairedSelectedBlocks = 2;
// Q32 is the current production candidate: it keeps the whole-head tile under
// the shared-memory occupancy cliff without the end-to-end regression observed
// with the smaller Q16 probe.  K/V routing remains in 32-token chunks; exact
// mode still reduces to dense attention.
constexpr int kSolQueryBlock = 32;
constexpr int kSolKVGroup    = 32;

using namespace nvcuda;

int sol_env_i(const char * name, int fallback, int min_value, int max_value) {
    const char * e = getenv(name);
    if (e == nullptr || *e == '\0') return fallback;
    char * end = nullptr;
    const long value = strtol(e, &end, 10);
    if (end == e || *end != '\0' || value < min_value || value > max_value) {
        static int warned = 0;
        if (warned++ < 4) fprintf(stderr, "[sol] %s=%s ignored; using %d\n", name, e, fallback);
        return fallback;
    }
    return static_cast<int>(value);
}

float sol_env_f(const char * name, float fallback, float min_value, float max_value) {
    const char * e = getenv(name);
    if (e == nullptr || *e == '\0') return fallback;
    char * end = nullptr;
    const float value = strtof(e, &end);
    if (end == e || *end != '\0' || !std::isfinite(value) || value < min_value || value > max_value) {
        static int warned = 0;
        if (warned++ < 4) fprintf(stderr, "[sol] %s=%s ignored; using %.3f\n", name, e, fallback);
        return fallback;
    }
    return value;
}

// GGML_FLASH_ATTN_EXT presents all H3 operands in [D,L,H,1] here.  Keeping
// those native strides avoids a BTHD conversion buffer for every H3 block.
__global__ void sol_summarize(const half * k, const half * v, float * kc, float * vc,
                              int l, int heads, int kv_blocks, int64_t k_s1, int64_t k_s2,
                              int64_t v_s1, int64_t v_s2) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int kb = blockIdx.y;
    const int h = blockIdx.z;
    if (d >= kSolHeadDim || kb >= kv_blocks || h >= heads) return;
    const int begin = kb * kSolKVGroup;
    const int end = min(l, begin + kSolKVGroup);
    float ksum = 0.0f, vsum = 0.0f;
    for (int t = begin; t < end; ++t) {
        ksum += __half2float(k[(size_t) h * k_s2 + (size_t) t * k_s1 + d]);
        vsum += __half2float(v[(size_t) h * v_s2 + (size_t) t * v_s1 + d]);
    }
    const size_t out = ((size_t) h * kv_blocks + kb) * kSolHeadDim + d;
    kc[out] = ksum / (end - begin);
    vc[out] = vsum;
}

// The fused path uses the same 64-token block summaries as the Triton
// reference.  K is a mean, V is a sum: the latter lets proxy probabilities
// retain the correct token multiplicity in the online softmax recurrence.
__global__ void sol_fused_summarize64(const half * k, const half * v, float * kc, float * vc,
                                      int l, int heads, int blocks, int64_t k_s1, int64_t k_s2,
                                      int64_t v_s1, int64_t v_s2) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    const int h = blockIdx.z;
    if (d >= kSolHeadDim || b >= blocks || h >= heads) return;
    const int begin = b * kSolFusedTokenBlock;
    const int end = min(l, begin + kSolFusedTokenBlock);
    float ksum = 0.0f, vsum = 0.0f;
    for (int t = begin; t < end; ++t) {
        ksum += __half2float(k[(size_t) h * k_s2 + (size_t) t * k_s1 + d]);
        vsum += __half2float(v[(size_t) h * v_s2 + (size_t) t * v_s1 + d]);
    }
    const size_t out = ((size_t) h * blocks + b) * kSolHeadDim + d;
    kc[out] = ksum / (end - begin);
    vc[out] = vsum;
}

// Compute the reference's diagonal-independent global centroid moments.
// One block owns a (head, channel), avoiding a document-sized route matrix.
__global__ void sol_fused_moments(const float * kc, float * mean, float * variance,
                                  int heads, int blocks) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (d >= kSolHeadDim || h >= heads) return;
    float sum = 0.0f, sum2 = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const float x = kc[((size_t) h * blocks + b) * kSolHeadDim + d];
        sum += x; sum2 += x * x;
    }
    const float m = sum / blocks;
    mean[(size_t) h * kSolHeadDim + d] = m;
    variance[(size_t) h * kSolHeadDim + d] = fmaxf(0.0f, sum2 / blocks - m * m);
}

// Match _diag_threshold_kernel: use the Q64 centroid and global per-channel
// K-centroid moments.  This is intentionally not the legacy per-query-block
// scan / route-list threshold.
__global__ void sol_fused_threshold64(const float * q, const float * mean, const float * variance,
                                      float * threshold, int l, int heads, int q_blocks, float scale, float tau,
                                      int64_t q_s1, int64_t q_s2) {
    const int qb = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int begin = qb * kSolFusedTokenBlock;
    const int end = min(l, begin + kSolFusedTokenBlock);
    float local_mean = 0.0f, local_var = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i;
        float qsum = 0.0f;
        for (int t = begin; t < end; ++t) qsum += q[(size_t) h * q_s2 + (size_t) t * q_s1 + d];
        const float qc = qsum / (end - begin);
        local_mean += qc * mean[(size_t) h * kSolHeadDim + d];
        local_var += qc * qc * variance[(size_t) h * kSolHeadDim + d];
    }
    for (int off = 16; off; off >>= 1) {
        local_mean += __shfl_down_sync(0xffffffff, local_mean, off);
        local_var += __shfl_down_sync(0xffffffff, local_var, off);
    }
    if (lane == 0) threshold[(size_t) h * q_blocks + qb] =
        local_mean * scale + tau * sqrtf(fmaxf(0.0f, local_var * scale * scale) + 1.0e-6f);
}

// One warp derives the routing threshold for one query block/head.  Each lane
// owns four of the D=128 channels, so the centroid and every centroid score
// are reduced with warp shuffles rather than a 128-thread shared reduction.
__global__ void sol_thresholds(const float * q, const float * kc, float * threshold,
                               int l, int heads, int q_blocks, int kv_blocks, float scale, float tau,
                               int64_t q_s1, int64_t q_s2) {
    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int begin = qb * kSolQueryBlock;
    const int end = min(l, begin + kSolQueryBlock);
    float qmean[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i;
        float sum = 0.0f;
        for (int t = begin; t < end; ++t) sum += q[(size_t) h * q_s2 + (size_t) t * q_s1 + d];
        qmean[i] = sum / (end - begin);
    }

    float score_sum = 0.0f;
    float score_sq_sum = 0.0f;
    for (int kb = 0; kb < kv_blocks; ++kb) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dot += qmean[i] * kc[((size_t) h * kv_blocks + kb) * kSolHeadDim + lane * 4 + i];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane == 0) {
            const float score = dot * scale;
            score_sum += score;
            score_sq_sum += score * score;
        }
    }
    if (lane == 0) {
        const float mean = score_sum / kv_blocks;
        const float variance = fmaxf(0.0f, score_sq_sum / kv_blocks - mean * mean);
        threshold[(size_t) h * q_blocks + qb] = mean + tau * sqrtf(variance + 1.0e-6f);
    }
}

// Build a compact selected-KV list from the same 64-query centroid that
// produced the threshold.  Every query block gets a fixed-capacity slice of
// `kv_blocks` uint16 entries and its actual count.  This avoids a dense route
// matrix and is the handoff format for tiled exact QK/PV consumers.
__global__ void sol_route_masks(const float * q, const float * kc, const float * threshold,
                                uint16_t * route_lists, uint16_t * route_counts,
                                uint32_t * route_bits, int route_words,
                                int l, int heads, int q_blocks, int kv_blocks, float scale,
                                int sink_blocks, int sink_query_blocks, bool force_exact,
                                int64_t q_s1, int64_t q_s2) {
    const int qb = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int begin = qb * kSolQueryBlock;
    const int end = min(l, begin + kSolQueryBlock);
    float qmean[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i;
        float sum = 0.0f;
        for (int t = begin; t < end; ++t) sum += q[(size_t) h * q_s2 + (size_t) t * q_s1 + d];
        qmean[i] = sum / (end - begin);
    }
    const float route_threshold = threshold[(size_t) h * q_blocks + qb];
    int count = 0;
    for (int kb = 0; kb < kv_blocks; ++kb) {
        float score = 0.0f;
        const size_t summary = ((size_t) h * kv_blocks + kb) * kSolHeadDim + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) score += qmean[i] * kc[summary + i];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) score += __shfl_down_sync(0xffffffff, score, offset);
        if (lane == 0) {
            const bool exact = force_exact || qb < sink_query_blocks || kb < sink_blocks ||
                               abs(qb - kb) <= 1 || score * scale > route_threshold;
            if (exact) {
                route_lists[((size_t) h * q_blocks + qb) * kv_blocks + count++] = kb;
                // One CTA owns a (head, query-block), and lane zero is the
                // only writer, so this is a plain non-atomic bit set.  The
                // packed bitmap is the O(1) membership companion to the
                // compact list: the list is for exact-token traversal, the
                // bitmap is for the centroid pass.
                route_bits[((size_t) h * q_blocks + qb) * route_words + kb / 32] |= 1u << (kb & 31);
            }
        }
    }
    if (lane == 0) route_counts[(size_t) h * q_blocks + qb] = count;
}

__device__ __forceinline__ bool sol_list_contains(const uint16_t * route_lists, int route_count,
                                                   size_t route_base, int kb) {
    for (int i = 0; i < route_count; ++i) {
        if (route_lists[route_base + i] == kb) return true;
    }
    return false;
}

__device__ __forceinline__ bool sol_route_contains(const uint32_t * route_bits, int route_words,
                                                    int h, int qb, int q_blocks, int kb) {
    const uint32_t word = route_bits[((size_t) h * q_blocks + qb) * route_words + kb / 32];
    return (word & (1u << (kb & 31))) != 0;
}

// Eight query rows share one CTA (eight independent warps).  A lane owns four
// output channels, turning D=128 dot products into a shuffle reduction and
// eliminating the old CTA-per-row shared-memory/barrier inner loop.  This is
// the pointer/strided counterpart of the reference Triton kernel's query
// tiles; it intentionally keeps routing decisions per row for fidelity.
template<typename T>
__global__ void sol_attention_warp8(const float * q, const half * k, const half * v,
                              const float * kc, const float * vc, const float * threshold,
                              const uint16_t * route_lists, const uint16_t * route_counts, const uint32_t * route_bits, int route_words, bool block_routes,
                              T * out, int l, int heads, int q_blocks, int kv_blocks, float scale,
                              int sink_blocks, int sink_query_blocks, bool force_exact,
                              int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2,
                              int64_t v_s1, int64_t v_s2, int64_t out_h_stride, int64_t out_t_stride) {
    constexpr int kWarpsPerBlock = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int t = ((int) blockIdx.x * kWarpsPerBlock) + warp;
    const int h = blockIdx.y;
    if (t >= l || h >= heads) return;
    const int qb = t / kSolQueryBlock;
    const bool dense_query = qb < sink_query_blocks;
    const float route_threshold = threshold[(size_t) h * q_blocks + qb];
    const size_t route_base = ((size_t) h * q_blocks + qb) * kv_blocks;
    const int route_count = block_routes ? route_counts[(size_t) h * q_blocks + qb] : 0;
    const float q0 = q[(size_t) h * q_s2 + (size_t) t * q_s1 + lane * 4 + 0];
    const float q1 = q[(size_t) h * q_s2 + (size_t) t * q_s1 + lane * 4 + 1];
    const float q2 = q[(size_t) h * q_s2 + (size_t) t * q_s1 + lane * 4 + 2];
    const float q3 = q[(size_t) h * q_s2 + (size_t) t * q_s1 + lane * 4 + 3];
    float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (int kb = 0; kb < kv_blocks; ++kb) {
        const size_t summary = ((size_t) h * kv_blocks + kb) * kSolHeadDim + lane * 4;
        // Block routing changes only the selected/exact decision.  The
        // unselected centroid still needs its real Q·Kc score for the sparse
        // softmax.  Leaving this at zero made the scalar block-route consumer
        // an invalid same-mask reference (and silently changed its output).
        float score = q0 * kc[summary + 0] + q1 * kc[summary + 1] +
                      q2 * kc[summary + 2] + q3 * kc[summary + 3];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) score += __shfl_down_sync(0xffffffff, score, offset);
        score = __shfl_sync(0xffffffff, score * scale, 0);
        // `force_exact` is intentionally a diagnostic gate, not a quality
        // mode.  It turns this same strided kernel into ordinary dense
        // softmax(QK^T)V so that a real H3 call can be compared with cuDNN
        // before any sparse result is trusted.
        const bool exact = force_exact || (block_routes
            ? sol_route_contains(route_bits, route_words, h, qb, q_blocks, kb)
            : dense_query || kb < sink_blocks || abs(qb - kb) <= 1 || score > route_threshold);

        if (!exact) {
            const float next_max = fmaxf(row_max, score);
            const float alpha = expf(row_max - next_max);
            const float p = expf(score - next_max);
            row_sum = row_sum * alpha + p * min(kSolKVGroup, l - kb * kSolKVGroup);
            row_max = next_max;
            value0 = value0 * alpha + p * vc[summary + 0];
            value1 = value1 * alpha + p * vc[summary + 1];
            value2 = value2 * alpha + p * vc[summary + 2];
            value3 = value3 * alpha + p * vc[summary + 3];
            continue;
        }

        const int begin = kb * kSolKVGroup;
        const int end = min(l, begin + kSolKVGroup);
        for (int kt = begin; kt < end; ++kt) {
            const size_t kv = (size_t) h * k_s2 + (size_t) kt * k_s1 + lane * 4;
            float next_score = q0 * __half2float(k[kv + 0]) + q1 * __half2float(k[kv + 1]) +
                               q2 * __half2float(k[kv + 2]) + q3 * __half2float(k[kv + 3]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) next_score += __shfl_down_sync(0xffffffff, next_score, offset);
            next_score = __shfl_sync(0xffffffff, next_score * scale, 0);
            const float next_max = fmaxf(row_max, next_score);
            const float alpha = expf(row_max - next_max);
            const float p = expf(next_score - next_max);
            row_sum = row_sum * alpha + p;
            row_max = next_max;
            const size_t vv = (size_t) h * v_s2 + (size_t) kt * v_s1 + lane * 4;
            value0 = value0 * alpha + p * __half2float(v[vv + 0]);
            value1 = value1 * alpha + p * __half2float(v[vv + 1]);
            value2 = value2 * alpha + p * __half2float(v[vv + 2]);
            value3 = value3 * alpha + p * __half2float(v[vv + 3]);
        }
    }
    if constexpr (std::is_same<T, float>::value) {
        // FLASH_ATTN_EXT's destination is BSHD: [D,H,L,N], unlike H3's
        // Q/K/V [D,L,H,N] inputs.  Keep the two layouts explicit.
        const size_t o = (size_t) h * out_h_stride + (size_t) t * out_t_stride + lane * 4;
        out[o + 0] = value0 / row_sum; out[o + 1] = value1 / row_sum;
        out[o + 2] = value2 / row_sum; out[o + 3] = value3 / row_sum;
    } else {
        const size_t o = (size_t) h * out_h_stride + (size_t) t * out_t_stride + lane * 4;
        out[o + 0] = __float2half_rn(value0 / row_sum); out[o + 1] = __float2half_rn(value1 / row_sum);
        out[o + 2] = __float2half_rn(value2 / row_sum); out[o + 3] = __float2half_rn(value3 / row_sum);
    }
}

// Tensor-core exact-score path.  One CTA owns a (64 query rows, 1 head,
// 32-value-channel) tile.  It consumes the compact selected-group list: the
// approximate groups are folded first with their pooled values, then selected
// 32-token groups use WMMA 64x128 x 128x32 score tiles and an F32 online
// softmax/PV update.  Q is converted from native F32 directly into shared
// half storage; K/V keep their native strided F16 source layout.  The output
// write remains explicitly BSHD.
__global__ void sol_attention_wmma_q64_v32(
        const float * q, const half * k, const half * v, const float * kc, const float * vc,
        const uint16_t * route_lists, const uint16_t * route_counts,
        float * out, int l, int heads, int q_blocks, int kv_blocks, float scale,
        int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2, int64_t v_s1, int64_t v_s2,
        int64_t out_h_stride, int64_t out_t_stride) {
    const int qb = blockIdx.x;
    const int h = blockIdx.y;
    const int vtile = blockIdx.z;
    const int tid = threadIdx.x;
    const int q_begin = qb * kSolQueryBlock;
    constexpr int kValueTile = 32;
    const int d_begin = vtile * kValueTile;
    const int route_count = route_counts[(size_t) h * q_blocks + qb];
    const size_t route_base = ((size_t) h * q_blocks + qb) * kv_blocks;

    extern __shared__ uint8_t storage[];
    half * sq = reinterpret_cast<half *>(storage);                         // 64 x 128
    half * skt = sq + kSolQueryBlock * kSolHeadDim;                        // 128 x 32, column-major B
    float * scores = reinterpret_cast<float *>(skt + kSolHeadDim * kSolKVGroup); // 64 x 32
    float * accum = scores + kSolQueryBlock * kSolKVGroup;                 // 64 x 32
    float * row_max = accum + kSolQueryBlock * kValueTile;
    float * row_sum = row_max + kSolQueryBlock;

    for (int i = tid; i < kSolQueryBlock * kSolHeadDim; i += blockDim.x) {
        const int row = i / kSolHeadDim;
        const int d = i % kSolHeadDim;
        const int t = q_begin + row;
        sq[i] = t < l ? __float2half_rn(q[(size_t) h * q_s2 + (size_t) t * q_s1 + d]) : __float2half(0.0f);
    }
    for (int i = tid; i < kSolQueryBlock * kValueTile; i += blockDim.x) accum[i] = 0.0f;
    for (int row = tid; row < kSolQueryBlock; row += blockDim.x) {
        row_max[row] = -INFINITY;
        row_sum[row] = 0.0f;
    }
    __syncthreads();

    // Summary-only groups: a single row thread evaluates the centroid proxy
    // and updates its 32-value slice.  It deliberately skips groups named in
    // the compact exact list rather than scanning a dense route matrix.
    for (int row = tid; row < kSolQueryBlock; row += blockDim.x) {
        const int t = q_begin + row;
        if (t >= l) continue;
        for (int kb = 0; kb < kv_blocks; ++kb) {
            if (sol_list_contains(route_lists, route_count, route_base, kb)) continue;
            const size_t summary = ((size_t) h * kv_blocks + kb) * kSolHeadDim;
            float score = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < kSolHeadDim; ++d) {
                score += q[(size_t) h * q_s2 + (size_t) t * q_s1 + d] * kc[summary + d];
            }
            score *= scale;
            const float next_max = fmaxf(row_max[row], score);
            const float alpha = expf(row_max[row] - next_max);
            const float p = expf(score - next_max);
            row_sum[row] = row_sum[row] * alpha + p * min(kSolKVGroup, l - kb * kSolKVGroup);
            #pragma unroll
            for (int vi = 0; vi < kValueTile; ++vi) {
                accum[row * kValueTile + vi] = accum[row * kValueTile + vi] * alpha + p * vc[summary + d_begin + vi];
            }
            row_max[row] = next_max;
        }
    }
    __syncthreads();

    for (int ri = 0; ri < route_count; ++ri) {
        const int kb = route_lists[route_base + ri];
        const int k_begin = kb * kSolKVGroup;
        // Stage B as col-major [D,N] for WMMA.  The native K tensor itself is
        // [D,L,H], so this is a local transpose only for the selected group.
        for (int i = tid; i < kSolHeadDim * kSolKVGroup; i += blockDim.x) {
            const int n = i / kSolHeadDim;
            const int d = i % kSolHeadDim;
            const int kt = k_begin + n;
            // column-major B: element (K=d, N=n) lives at d + n*D.
            skt[i] = kt < l ? k[(size_t) h * k_s2 + (size_t) kt * k_s1 + d] : __float2half(0.0f);
        }
        __syncthreads();
        const int warp = tid >> 5;
        if (warp < (kSolQueryBlock / 16) * 2) {
            const int mi = warp >> 1;
            const int ni = warp & 1;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::load_matrix_sync(b, skt + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores + mi * 16 * kSolKVGroup + ni * 16, c, kSolKVGroup, wmma::mem_row_major);
        }
        __syncthreads();
        for (int row = tid; row < kSolQueryBlock; row += blockDim.x) {
            const int t = q_begin + row;
            if (t >= l) continue;
            const int valid = min(kSolKVGroup, l - k_begin);
            float local_max = -INFINITY;
            #pragma unroll
            for (int n = 0; n < kSolKVGroup; ++n) {
                if (n < valid) local_max = fmaxf(local_max, scores[row * kSolKVGroup + n] * scale);
            }
            const float next_max = fmaxf(row_max[row], local_max);
            const float alpha = expf(row_max[row] - next_max);
            float add_sum = 0.0f;
            float weighted[kValueTile] = {};
            #pragma unroll
            for (int n = 0; n < kSolKVGroup; ++n) {
                if (n >= valid) continue;
                const float p = expf(scores[row * kSolKVGroup + n] * scale - next_max);
                add_sum += p;
                const size_t vv = (size_t) h * v_s2 + (size_t) (k_begin + n) * v_s1 + d_begin;
                #pragma unroll
                for (int vi = 0; vi < kValueTile; ++vi) weighted[vi] += p * __half2float(v[vv + vi]);
            }
            row_sum[row] = row_sum[row] * alpha + add_sum;
            #pragma unroll
            for (int vi = 0; vi < kValueTile; ++vi) accum[row * kValueTile + vi] = accum[row * kValueTile + vi] * alpha + weighted[vi];
            row_max[row] = next_max;
        }
        __syncthreads();
    }
    for (int i = tid; i < kSolQueryBlock * kValueTile; i += blockDim.x) {
        const int row = i / kValueTile;
        const int vi = i % kValueTile;
        const int t = q_begin + row;
        if (t < l) out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d_begin + vi] = accum[i] / row_sum[row];
    }
}

// Collective whole-head consumer.  Unlike the V=32 experiment above, one CTA
// owns all 128 output channels for a (Q64, head) tile.  This is important for
// the real sparse path: score generation, K staging, and route traversal are
// performed exactly once rather than once per value slice.  The online state
// lives in shared memory, so each worker has one scalar accumulator while
// applying a selected 32-token group (instead of the V32 kernel's 32-element
// register array, which pushed it to 128 registers on Ada/Blackwell).
__global__ void sol_attention_wmma_q64_full_d(
        const float * q, const half * k, const half * v, const float * kc, const float * vc,
        const uint16_t * route_lists, const uint16_t * route_counts, const uint32_t * route_bits, int route_words,
        float * out, int l, int heads, int q_blocks, int kv_blocks, float scale,
        int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2, int64_t v_s1, int64_t v_s2,
        int64_t out_h_stride, int64_t out_t_stride) {
    const int qb = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int q_begin = qb * kSolQueryBlock;
    const int route_count = route_counts[(size_t) h * q_blocks + qb];
    const size_t route_base = ((size_t) h * q_blocks + qb) * kv_blocks;

    extern __shared__ uint8_t storage[];
    half * sq = reinterpret_cast<half *>(storage); // [Q=64,D=128]
    half * skt = sq + kSolQueryBlock * kSolHeadDim; // col-major [D=128,N=32]
    float * scores = reinterpret_cast<float *>(skt + kSolHeadDim * kSolKVGroup); // [Q,N]
    float * accum = scores + kSolQueryBlock * kSolKVGroup; // [Q,D]
    float * row_max = accum + kSolQueryBlock * kSolHeadDim;
    float * row_sum = row_max + kSolQueryBlock;
    float * row_alpha = row_sum + kSolQueryBlock;
    // The pooled PV tensor-core result.  It is deliberately separate from
    // `accum`: online softmax must retain old_accum until alpha is known.
    // A Q32,D128 tile costs 16 KiB and keeps the whole pooled path on tensor
    // cores rather than replacing only its QK half.
    float * pv = row_alpha + kSolQueryBlock;
    // P is half precision solely for the tensor-core input.  It must not
    // alias Q: Q is reused for every centroid tile, not only the exact pass.
    half * sp = reinterpret_cast<half *>(pv + kSolQueryBlock * kSolHeadDim);

    for (int i = tid; i < kSolQueryBlock * kSolHeadDim; i += blockDim.x) {
        const int row = i / kSolHeadDim;
        const int d = i % kSolHeadDim;
        const int t = q_begin + row;
        sq[i] = t < l ? __float2half_rn(q[(size_t) h * q_s2 + (size_t) t * q_s1 + d]) : __float2half(0.0f);
        accum[i] = 0.0f;
    }
    for (int row = tid; row < kSolQueryBlock; row += blockDim.x) {
        row_max[row] = -INFINITY;
        row_sum[row] = 0.0f;
        row_alpha[row] = 0.0f;
    }
    __syncthreads();

    // Pool unselected groups in 32-centroid tiles.  The old path did a scalar
    // 128-channel Q.dot(K-centroid) for every query/group pair, which becomes
    // the dominant O(L^2) cost at production resolution.  These centroids are
    // the approximate part of Sol, so converting their F32 summaries to the
    // same F16 WMMA staging used by the selected exact tiles is intentional.
    // ALL_EXACT selects every group; it consequently executes no pooled
    // updates and falls through to the dense-equivalent exact loop below.
    uint32_t * summary_selected = reinterpret_cast<uint32_t *>(row_alpha);
    for (int kb0 = 0; kb0 < kv_blocks; kb0 += kSolKVGroup) {
        if (tid == 0) summary_selected[0] = 0;
        __syncthreads();
        if (tid < kSolKVGroup) {
            const int kb = kb0 + tid;
            if (kb < kv_blocks && sol_route_contains(route_bits, route_words, h, qb, q_blocks, kb)) {
                atomicOr(summary_selected, 1u << tid);
            }
        }
        for (int i = tid; i < kSolHeadDim * kSolKVGroup; i += blockDim.x) {
            const int n = i / kSolHeadDim;
            const int d = i % kSolHeadDim;
            const int kb = kb0 + n;
            skt[i] = kb < kv_blocks ? __float2half_rn(kc[((size_t) h * kv_blocks + kb) * kSolHeadDim + d]) : __float2half(0.0f);
        }
        __syncthreads();
        const int warp = tid >> 5;
        if (warp < (kSolQueryBlock / 16) * 2) {
            const int mi = warp >> 1;
            const int ni = warp & 1;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::load_matrix_sync(b, skt + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores + mi * 16 * kSolKVGroup + ni * 16, c, kSolKVGroup, wmma::mem_row_major);
        }
        __syncthreads();
        const uint32_t selected = summary_selected[0];
        for (int row = tid; row < kSolQueryBlock; row += blockDim.x) {
            const int t = q_begin + row;
            if (t >= l) continue;
            float local_max = -INFINITY;
            #pragma unroll
            for (int n = 0; n < kSolKVGroup; ++n) {
                const int kb = kb0 + n;
                if (kb < kv_blocks && !(selected & (1u << n))) {
                    local_max = fmaxf(local_max, scores[row * kSolKVGroup + n] * scale);
                }
            }
            if (local_max == -INFINITY) {
                row_alpha[row] = 1.0f;
                continue;
            }
            const float next_max = fmaxf(row_max[row], local_max);
            const float alpha = expf(row_max[row] - next_max);
            float add_sum = 0.0f;
            #pragma unroll
            for (int n = 0; n < kSolKVGroup; ++n) {
                const int kb = kb0 + n;
                const float p = kb < kv_blocks && !(selected & (1u << n)) ? expf(scores[row * kSolKVGroup + n] * scale - next_max) : 0.0f;
                scores[row * kSolKVGroup + n] = p;
                add_sum += p * (kb < kv_blocks ? min(kSolKVGroup, l - kb * kSolKVGroup) : 0);
            }
            row_alpha[row] = alpha;
            row_sum[row] = row_sum[row] * alpha + add_sum;
            row_max[row] = next_max;
        }
        __syncthreads();
        // Reuse the K-centroid staging buffer as a column-major V-summary
        // matrix [K=32,N=128].  `scores` now contains probabilities, so eight
        // warps can form P[32,32] @ Vsummary[32,128] as sixteen WMMA tiles.
        // This removes the old scalar 32-term loop for every (Q,D).
        for (int i = tid; i < kSolKVGroup * kSolHeadDim; i += blockDim.x) {
            const int d = i / kSolKVGroup;
            const int n = i % kSolKVGroup;
            const int kb = kb0 + n;
            skt[i] = (kb < kv_blocks && !(selected & (1u << n)))
                ? __float2half_rn(vc[((size_t) h * kv_blocks + kb) * kSolHeadDim + d])
                : __float2half(0.0f);
        }
        // P has only K=32 columns.  Convert it once for the tensor-core input
        // without touching Q, which is reused by the next centroid tile.
        for (int i = tid; i < kSolQueryBlock * kSolKVGroup; i += blockDim.x) {
            const int row = i / kSolKVGroup;
            const int n = i % kSolKVGroup;
            sp[i] = __float2half_rn(scores[row * kSolKVGroup + n]);
        }
        __syncthreads();
        if (warp < 8) {
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int tile = warp + part * 8;
                const int mi = tile / 8;
                const int ni = tile % 8;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < kSolKVGroup; kk += 16) {
                    wmma::load_matrix_sync(a, sp + mi * 16 * kSolKVGroup + kk, kSolKVGroup);
                    wmma::load_matrix_sync(b, skt + ni * 16 * kSolKVGroup + kk, kSolKVGroup);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(pv + mi * 16 * kSolHeadDim + ni * 16,
                                        c, kSolHeadDim, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int i = tid; i < kSolQueryBlock * kSolHeadDim; i += blockDim.x) {
            const int row = i / kSolHeadDim;
            const int d = i % kSolHeadDim;
            const int t = q_begin + row;
            if (t < l) accum[i] = accum[i] * row_alpha[row] + pv[i];
        }
        __syncthreads();
    }

    for (int ri = 0; ri < route_count; ++ri) {
        const int kb = route_lists[route_base + ri];
        const int k_begin = kb * kSolKVGroup;
        for (int i = tid; i < kSolHeadDim * kSolKVGroup; i += blockDim.x) {
            const int n = i / kSolHeadDim;
            const int d = i % kSolHeadDim;
            const int kt = k_begin + n;
            skt[i] = kt < l ? k[(size_t) h * k_s2 + (size_t) kt * k_s1 + d] : __float2half(0.0f);
        }
        __syncthreads();
        const int warp = tid >> 5;
        if (warp < (kSolQueryBlock / 16) * 2) {
            const int mi = warp >> 1;
            const int ni = warp & 1;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::load_matrix_sync(b, skt + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores + mi * 16 * kSolKVGroup + ni * 16, c, kSolKVGroup, wmma::mem_row_major);
        }
        __syncthreads();

        // Convert the score tile to probabilities once.  `scores` can be
        // overwritten after the max reduction because later work needs only
        // p, not QK itself.
        for (int row = tid; row < kSolQueryBlock; row += blockDim.x) {
            const int t = q_begin + row;
            if (t >= l) continue;
            const int valid = min(kSolKVGroup, l - k_begin);
            float local_max = -INFINITY;
            #pragma unroll
            for (int n = 0; n < kSolKVGroup; ++n) {
                if (n < valid) local_max = fmaxf(local_max, scores[row * kSolKVGroup + n] * scale);
            }
            const float next_max = fmaxf(row_max[row], local_max);
            const float alpha = expf(row_max[row] - next_max);
            float add_sum = 0.0f;
            #pragma unroll
            for (int n = 0; n < kSolKVGroup; ++n) {
                const float p = n < valid ? expf(scores[row * kSolKVGroup + n] * scale - next_max) : 0.0f;
                scores[row * kSolKVGroup + n] = p;
                add_sum += p;
            }
            row_alpha[row] = alpha;
            row_sum[row] = row_sum[row] * alpha + add_sum;
            row_max[row] = next_max;
        }
        __syncthreads();

        // Exact selected groups used to be the one scalar residue in this
        // otherwise tensor-core consumer: every (Q,D) worker walked the 32
        // selected V tokens itself.  Route stats from real H3 show that this
        // still covers ~13% of groups at 864p (and grows in absolute work at
        // production L), so reuse the pooled-path P x V WMMA form here too.
        // `skt` is no longer needed as K after the score/probability phase;
        // repack native V [D,L,H] as column-major B [K=32,N=128].  `sp` is
        // the half P input and `pv` receives F32 P@V.  This preserves the
        // existing online-softmax recurrence exactly apart from the same
        // intentional F16 WMMA input precision already used by the pooled
        // path and accepted by ALL_EXACT dense checks.
        for (int i = tid; i < kSolKVGroup * kSolHeadDim; i += blockDim.x) {
            const int d = i / kSolKVGroup;
            const int n = i % kSolKVGroup;
            const int vt = k_begin + n;
            skt[i] = vt < l
                ? v[(size_t) h * v_s2 + (size_t) vt * v_s1 + d]
                : __float2half(0.0f);
        }
        for (int i = tid; i < kSolQueryBlock * kSolKVGroup; i += blockDim.x) {
            sp[i] = __float2half_rn(scores[i]);
        }
        __syncthreads();
        if (warp < 8) {
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int tile = warp + part * 8;
                const int mi = tile / 8;
                const int ni = tile % 8;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < kSolKVGroup; kk += 16) {
                    wmma::load_matrix_sync(a, sp + mi * 16 * kSolKVGroup + kk, kSolKVGroup);
                    wmma::load_matrix_sync(b, skt + ni * 16 * kSolKVGroup + kk, kSolKVGroup);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(pv + mi * 16 * kSolHeadDim + ni * 16,
                                        c, kSolHeadDim, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int i = tid; i < kSolQueryBlock * kSolHeadDim; i += blockDim.x) {
            const int row = i / kSolHeadDim;
            const int t = q_begin + row;
            if (t < l) accum[i] = accum[i] * row_alpha[row] + pv[i];
        }
        __syncthreads();
    }
    for (int i = tid; i < kSolQueryBlock * kSolHeadDim; i += blockDim.x) {
        const int row = i / kSolHeadDim;
        const int d = i % kSolHeadDim;
        const int t = q_begin + row;
        if (t < l) out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d] = accum[i] / row_sum[row];
    }
}

// Reference-shaped Sol consumer.  A CTA owns one (head,Q64) tile and keeps
// route decisions only for the current 64 *blocks* epoch in shared memory.
// In particular this does not allocate or revisit a document-sized route map.
// The score tile is produced once, reused for route/proxy softmax, and selected
// 64-token blocks are consumed immediately in the same online recurrence.
// Persistent-fragment follow-up uses the empirically verified SM120 WMMA C
// layout (see /tmp/wmma_mapper.cu): lane l owns rows l/4 and l/4+8, with
// x[0,1,4,5] and x[2,3,6,7] respectively.  Keep this adjacent to the fused
// kernel so any 32-warp dense implementation scales its F32 fragments by the
// correct online-softmax row alpha rather than relying on undocumented layout.
__global__ void sol_attention_fused_q64_g32(
        const float * q, const half * k, const half * v, const float * kc, const float * vc,
        const float * threshold, float * out, int l, int heads, int q_blocks, int kv_blocks,
        float scale, int sink_blocks, int sink_query_blocks, bool force_exact,
        int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2, int64_t v_s1, int64_t v_s2,
        int64_t out_h_stride, int64_t out_t_stride, bool resident_values, bool fp16_state) {
    const int qb = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5;
    const int q_begin = qb * kSolFusedTokenBlock;
    extern __shared__ uint8_t storage[];
    half * sq = reinterpret_cast<half *>(storage);                         // [64,128]
    half * sb = sq + kSolFusedTokenBlock * kSolHeadDim;                    // max [128,64]
    float * scores = reinterpret_cast<float *>(sb + kSolHeadDim * kSolFusedTokenBlock); // [64,64]
    // The F32 online output lives directly in `out`, which is already an F32
    // temporary for F16 destinations.  This removes the 32 KiB shared
    // accumulator; keep P separate from scores because an in-place F32->F16
    // conversion races across the overlapping representations.
    half * probs = reinterpret_cast<half *>(scores + kSolFusedTokenBlock * kSolFusedTokenBlock);
    float * row_max = reinterpret_cast<float *>(probs + kSolFusedTokenBlock * kSolFusedTokenBlock);
    float * row_sum = row_max + kSolFusedTokenBlock;
    float * row_alpha = row_sum + kSolFusedTokenBlock;
    int * selected = reinterpret_cast<int *>(row_alpha + kSolFusedTokenBlock); // 64 flags
    // The default bin65 path keeps its online state in the F32 destination to
    // stay below the shared-memory limit.  The opt-in value-tile path instead
    // owns D=64 and retains that half of the state on chip across every G64
    // epoch.  The two CTAs recompute identical score/routing state, which is
    // valid because the online softmax weights do not depend on value channel.
    float * accum = reinterpret_cast<float *>(selected + kSolFusedGroupBlocks);
    half * fp16_accum = reinterpret_cast<half *>(selected + kSolFusedGroupBlocks);
    const int value_width = resident_values ? 64 : kSolHeadDim;
    const int value_tiles = value_width / 16;
    const int value_offset = resident_values ? (int) blockIdx.z * value_width : 0;

    for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
        const int row = i / kSolHeadDim, d = i % kSolHeadDim, t = q_begin + row;
        sq[i] = t < l ? __float2half_rn(q[(size_t) h * q_s2 + (size_t) t * q_s1 + d]) : __float2half(0.0f);
        if (!resident_values && !fp16_state && t < l) out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d] = 0.0f;
    }
    if (resident_values) for (int i = tid; i < kSolFusedTokenBlock * value_width; i += blockDim.x) accum[i] = 0.0f;
    if (fp16_state) for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) fp16_accum[i] = __float2half(0.0f);
    for (int row = tid; row < kSolFusedTokenBlock; row += blockDim.x) {
        row_max[row] = -INFINITY; row_sum[row] = 0.0f; row_alpha[row] = 1.0f;
    }
    __syncthreads();

    const bool q_sink = qb < sink_query_blocks;
    const int q_len = min(kSolFusedTokenBlock, l - q_begin);
    const float route_threshold = threshold[(size_t) h * q_blocks + qb];
    for (int group_start = 0; group_start < kv_blocks; group_start += kSolFusedGroupBlocks) {
        const int group_n = min(kSolFusedGroupBlocks, kv_blocks - group_start);
        // K centroid matrix [D,K] in column-major form for WMMA.
        for (int i = tid; i < kSolHeadDim * kSolFusedGroupBlocks; i += blockDim.x) {
            const int n = i / kSolHeadDim, d = i % kSolHeadDim, kb = group_start + n;
            sb[i] = n < group_n ? __float2half_rn(kc[((size_t) h * kv_blocks + kb) * kSolHeadDim + d]) : __float2half(0.0f);
        }
        __syncthreads();
        // Q64 x Kcentroid64 (4x4 WMMA tiles).
        if (warp < 8) for (int pass = 0; pass < 2; ++pass) {
            const int tile = warp + pass * 8, mi = tile / 4, ni = tile % 4;
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::load_matrix_sync(b, sb + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores + mi * 16 * kSolFusedTokenBlock + ni * 16, c,
                                    kSolFusedTokenBlock, wmma::mem_row_major);
        }
        __syncthreads();
        // Routing is derived from this existing proxy score tile, exactly once.
        if (tid < kSolFusedGroupBlocks) {
            const int n = tid, kb = group_start + n;
            float sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < kSolFusedTokenBlock; ++r) sum += scores[r * kSolFusedTokenBlock + n] * scale;
            selected[n] = n < group_n && (force_exact || q_sink || kb < sink_blocks || abs(qb - kb) <= 1 || sum / q_len > route_threshold);
        }
        __syncthreads();
        // Proxy blocks: softmax QxKc and P x Vc.  First form P and scale the
        // retained online state, then WMMA accumulates directly into it.
        for (int r = tid; r < kSolFusedTokenBlock; r += blockDim.x) {
            const int t = q_begin + r;
            float local_max = -INFINITY;
            if (t < l) for (int n = 0; n < group_n; ++n) if (!selected[n])
                local_max = fmaxf(local_max, scores[r * kSolFusedTokenBlock + n] * scale);
            if (local_max == -INFINITY) { row_alpha[r] = 1.0f; }
            else {
                const float nm = fmaxf(row_max[r], local_max), alpha = expf(row_max[r] - nm);
                float add = 0.0f;
                for (int n = 0; n < kSolFusedGroupBlocks; ++n) {
                    const float p = n < group_n && !selected[n] ? expf(scores[r * kSolFusedTokenBlock + n] * scale - nm) : 0.0f;
                    scores[r * kSolFusedTokenBlock + n] = p;
                    if (n < group_n) add += p * min(kSolFusedTokenBlock, l - (group_start + n) * kSolFusedTokenBlock);
                }
                row_alpha[r] = alpha; row_sum[r] = row_sum[r] * alpha + add; row_max[r] = nm;
            }
            if (local_max == -INFINITY) for (int n = 0; n < kSolFusedGroupBlocks; ++n) scores[r * kSolFusedTokenBlock + n] = 0.0f;
        }
        __syncthreads();
        for (int i = tid; i < kSolFusedTokenBlock * kSolFusedGroupBlocks; i += blockDim.x) probs[i] = __float2half_rn(scores[i]);
        __syncthreads();
        for (int i = tid; i < kSolFusedTokenBlock * value_width; i += blockDim.x) {
            const int row = i / value_width, d = i % value_width, t = q_begin + row;
            if (t < l) {
                if (resident_values) accum[i] *= row_alpha[row];
                else if (fp16_state) fp16_accum[i] = __float2half_rn(__half2float(fp16_accum[i]) * row_alpha[row]);
                else out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d] *= row_alpha[row];
            }
        }
        // Reuse B staging as [K=64,D=128] column-major V centroid matrix.
        for (int i = tid; i < kSolFusedGroupBlocks * value_width; i += blockDim.x) {
            const int d = i / kSolFusedGroupBlocks, n = i % kSolFusedGroupBlocks, kb = group_start + n;
            sb[i] = n < group_n && !selected[n] ? __float2half_rn(vc[((size_t) h * kv_blocks + kb) * kSolHeadDim + d + value_offset]) : __float2half(0.0f);
        }
        __syncthreads();
        if (warp < 8) for (int pass = 0; pass < (4 * value_tiles + 7) / 8; ++pass) {
            const int tile = warp + pass * 8, mi = tile / value_tiles, ni = tile % value_tiles;
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            if (resident_values) wmma::load_matrix_sync(c, accum + (size_t) (mi * 16) * value_width + ni * 16, value_width, wmma::mem_row_major);
            else if (fp16_state) wmma::fill_fragment(c, 0.0f);
            else wmma::load_matrix_sync(c, out + (size_t) h * out_h_stride + (size_t) (q_begin + mi * 16) * out_t_stride + ni * 16, out_t_stride, wmma::mem_row_major);
            #pragma unroll
            for (int kk = 0; kk < kSolFusedGroupBlocks; kk += 16) {
                wmma::load_matrix_sync(a, probs + mi * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                wmma::load_matrix_sync(b, sb + ni * 16 * kSolFusedGroupBlocks + kk, kSolFusedGroupBlocks);
                wmma::mma_sync(c, a, b, c);
            }
            if (resident_values) wmma::store_matrix_sync(accum + (size_t) (mi * 16) * value_width + ni * 16, c, value_width, wmma::mem_row_major);
            else if (fp16_state) {
                float * tile = scores + warp * 16 * 16;
                wmma::store_matrix_sync(tile, c, 16, wmma::mem_row_major);
                __syncwarp();
                const int lane = tid & 31;
                for (int j = lane; j < 16 * 16; j += 32) {
                    const int row = mi * 16 + j / 16, col = ni * 16 + j % 16;
                    fp16_accum[row * kSolHeadDim + col] = __float2half_rn(
                        __half2float(fp16_accum[row * kSolHeadDim + col]) + tile[j]);
                }
            }
            else wmma::store_matrix_sync(out + (size_t) h * out_h_stride + (size_t) (q_begin + mi * 16) * out_t_stride + ni * 16, c, out_t_stride, wmma::mem_row_major);
        }
        __syncthreads();

        // Exact selected 64-token blocks are merged immediately, never put in
        // a global route list.  QK and PV are both tensor-core tiles.
        for (int n = 0; n < group_n; ++n) if (selected[n]) {
            const int kb = group_start + n, k_begin = kb * kSolFusedTokenBlock;
            for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
                const int col = i / kSolHeadDim, d = i % kSolHeadDim, kt = k_begin + col;
                sb[i] = kt < l ? k[(size_t) h * k_s2 + (size_t) kt * k_s1 + d] : __float2half(0.0f);
            }
            __syncthreads();
            if (warp < 8) for (int pass = 0; pass < 2; ++pass) {
                const int tile = warp + pass * 8, mi = tile / 4, ni = tile % 4;
                wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
                wmma::fragment<wmma::accumulator,16,16,16,float> c;
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                    wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                    wmma::load_matrix_sync(b, sb + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(scores + mi * 16 * kSolFusedTokenBlock + ni * 16, c, kSolFusedTokenBlock, wmma::mem_row_major);
            }
            __syncthreads();
            for (int r = tid; r < kSolFusedTokenBlock; r += blockDim.x) {
                const int t = q_begin + r, valid = min(kSolFusedTokenBlock, l - k_begin);
                float lm = -INFINITY;
                if (t < l) for (int col = 0; col < valid; ++col) lm = fmaxf(lm, scores[r * kSolFusedTokenBlock + col] * scale);
                const float nm = fmaxf(row_max[r], lm), alpha = expf(row_max[r] - nm);
                float add = 0.0f;
                for (int col = 0; col < kSolFusedTokenBlock; ++col) {
                    const float p = col < valid ? expf(scores[r * kSolFusedTokenBlock + col] * scale - nm) : 0.0f;
                    scores[r * kSolFusedTokenBlock + col] = p; add += p;
                }
                row_alpha[r] = alpha; row_sum[r] = row_sum[r] * alpha + add; row_max[r] = nm;
            }
            __syncthreads();
            for (int i = tid; i < kSolFusedTokenBlock * kSolFusedTokenBlock; i += blockDim.x) probs[i] = __float2half_rn(scores[i]);
            __syncthreads();
            for (int i = tid; i < kSolFusedTokenBlock * value_width; i += blockDim.x) {
                const int row = i / value_width, d = i % value_width, t = q_begin + row;
                if (t < l) {
                    if (resident_values) accum[i] *= row_alpha[row];
                    else if (fp16_state) fp16_accum[i] = __float2half_rn(__half2float(fp16_accum[i]) * row_alpha[row]);
                    else out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d] *= row_alpha[row];
                }
            }
            for (int i = tid; i < kSolFusedTokenBlock * value_width; i += blockDim.x) {
                const int d = i / kSolFusedTokenBlock, col = i % kSolFusedTokenBlock, vt = k_begin + col;
                sb[i] = vt < l ? v[(size_t) h * v_s2 + (size_t) vt * v_s1 + d + value_offset] : __float2half(0.0f);
            }
            __syncthreads();
            if (warp < 8) for (int pass = 0; pass < (4 * value_tiles + 7) / 8; ++pass) {
                const int tile = warp + pass * 8, mi = tile / value_tiles, ni = tile % value_tiles;
                wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
                wmma::fragment<wmma::accumulator,16,16,16,float> c;
                if (resident_values) wmma::load_matrix_sync(c, accum + (size_t) (mi * 16) * value_width + ni * 16, value_width, wmma::mem_row_major);
                else if (fp16_state) wmma::fill_fragment(c, 0.0f);
                else wmma::load_matrix_sync(c, out + (size_t) h * out_h_stride + (size_t) (q_begin + mi * 16) * out_t_stride + ni * 16, out_t_stride, wmma::mem_row_major);
                #pragma unroll
                for (int kk = 0; kk < kSolFusedTokenBlock; kk += 16) {
                    wmma::load_matrix_sync(a, probs + mi * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                    wmma::load_matrix_sync(b, sb + ni * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                    wmma::mma_sync(c, a, b, c);
                }
                if (resident_values) wmma::store_matrix_sync(accum + (size_t) (mi * 16) * value_width + ni * 16, c, value_width, wmma::mem_row_major);
                else if (fp16_state) {
                    float * tile = scores + warp * 16 * 16;
                    wmma::store_matrix_sync(tile, c, 16, wmma::mem_row_major);
                    __syncwarp();
                    const int lane = tid & 31;
                    for (int j = lane; j < 16 * 16; j += 32) {
                        const int row = mi * 16 + j / 16, col = ni * 16 + j % 16;
                        fp16_accum[row * kSolHeadDim + col] = __float2half_rn(
                            __half2float(fp16_accum[row * kSolHeadDim + col]) + tile[j]);
                    }
                }
                else wmma::store_matrix_sync(out + (size_t) h * out_h_stride + (size_t) (q_begin + mi * 16) * out_t_stride + ni * 16, c, out_t_stride, wmma::mem_row_major);
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < kSolFusedTokenBlock * value_width; i += blockDim.x) {
        const int row = i / value_width, d = i % value_width, t = q_begin + row;
        if (t < l) {
            if (resident_values) out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d + value_offset] = accum[i] / row_sum[row];
            else if (fp16_state) out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d] = __half2float(fp16_accum[i]) / row_sum[row];
            else out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d] /= row_sum[row];
        }
    }
}

// Dense, all-exact persistent variant of the fused Q64 path.  One 1024-thread
// CTA owns a (Q=64, H, D=128) tile.  Its 32 warps retain a 16x16 F32 P@V
// fragment each across every K64 epoch.  This is intentionally separate from
// sol_attention_fused_q64_g32: the latter is the verified sparse fallback and
// updates the destination between epochs.  The C-fragment row ownership below
// is measured on SM120 (not assumed from an older WMMA ABI): lane l owns rows
// l/4 and l/4+8; x[0,1,4,5] belong to the first, x[2,3,6,7] to the second.
__device__ __forceinline__ void sol_scale_f32_c_rows(
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> & c,
        const float * factors, int row_base) {
    const int lane = threadIdx.x & 31;
    const float a0 = factors[row_base + lane / 4];
    const float a1 = factors[row_base + lane / 4 + 8];
    c.x[0] *= a0; c.x[1] *= a0; c.x[4] *= a0; c.x[5] *= a0;
    c.x[2] *= a1; c.x[3] *= a1; c.x[6] *= a1; c.x[7] *= a1;
}

__global__ void sol_attention_fused_dense_persistent_q64(
        const float * q, const half * k, const half * v, float * out,
        int l, int heads, float scale,
        int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2,
        int64_t v_s1, int64_t v_s2, int64_t out_h_stride, int64_t out_t_stride) {
    const int qb = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int q_begin = qb * kSolFusedTokenBlock;
    const int kv_blocks = (l + kSolFusedTokenBlock - 1) / kSolFusedTokenBlock;

    extern __shared__ uint8_t storage[];
    half * sq = reinterpret_cast<half *>(storage);                         // [64,128]
    half * sb = sq + kSolFusedTokenBlock * kSolHeadDim;                    // [K=128,N=64] or [K=64,N=128]
    float * scores = reinterpret_cast<float *>(sb + kSolHeadDim * kSolFusedTokenBlock); // [64,64]
    half * probs = reinterpret_cast<half *>(scores + kSolFusedTokenBlock * kSolFusedTokenBlock); // [64,64]
    float * row_max = reinterpret_cast<float *>(probs + kSolFusedTokenBlock * kSolFusedTokenBlock);
    float * row_sum = row_max + kSolFusedTokenBlock;
    float * row_scale = row_sum + kSolFusedTokenBlock;

    // Every warp owns one 16x16 output tile: 4 query tiles x 8 value tiles.
    const int mi_out = warp / 8;
    const int ni_out = warp & 7;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
    wmma::fill_fragment(accum, 0.0f);

    for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
        const int row = i / kSolHeadDim, d = i % kSolHeadDim, t = q_begin + row;
        sq[i] = t < l ? __float2half_rn(q[(size_t) h * q_s2 + (size_t) t * q_s1 + d]) : __float2half(0.0f);
    }
    for (int row = tid; row < kSolFusedTokenBlock; row += blockDim.x) {
        row_max[row] = -INFINITY; row_sum[row] = 0.0f; row_scale[row] = 1.0f;
    }
    __syncthreads();

    for (int kb = 0; kb < kv_blocks; ++kb) {
        const int k_begin = kb * kSolFusedTokenBlock;
        // K is staged column-major [D=128,N=64] for Q@K^T.
        for (int i = tid; i < kSolHeadDim * kSolFusedTokenBlock; i += blockDim.x) {
            const int n = i / kSolHeadDim, d = i % kSolHeadDim, kt = k_begin + n;
            sb[i] = kt < l ? k[(size_t) h * k_s2 + (size_t) kt * k_s1 + d] : __float2half(0.0f);
        }
        __syncthreads();
        // The first sixteen warps form the complete Q64xK64 score matrix.
        if (warp < 16) {
            const int mi = warp / 4, ni = warp & 3;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::load_matrix_sync(b, sb + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores + mi * 16 * kSolFusedTokenBlock + ni * 16, c,
                                    kSolFusedTokenBlock, wmma::mem_row_major);
        }
        __syncthreads();
        // Online softmax for this K64 epoch.  It is all-exact: no centroid or
        // routing decisions enter this path, making it a direct dense gate.
        for (int row = tid; row < kSolFusedTokenBlock; row += blockDim.x) {
            const int t = q_begin + row, valid = min(kSolFusedTokenBlock, l - k_begin);
            float local_max = -INFINITY;
            if (t < l) for (int n = 0; n < valid; ++n)
                local_max = fmaxf(local_max, scores[row * kSolFusedTokenBlock + n] * scale);
            const float next_max = fmaxf(row_max[row], local_max);
            const float alpha = t < l ? expf(row_max[row] - next_max) : 1.0f;
            float add = 0.0f;
            for (int n = 0; n < kSolFusedTokenBlock; ++n) {
                const float p = (t < l && n < valid) ? expf(scores[row * kSolFusedTokenBlock + n] * scale - next_max) : 0.0f;
                probs[row * kSolFusedTokenBlock + n] = __float2half_rn(p);
                add += p;
            }
            row_scale[row] = alpha;
            if (t < l) { row_sum[row] = row_sum[row] * alpha + add; row_max[row] = next_max; }
        }
        __syncthreads();

        // Row-scale the retained F32 fragments before adding P@V for this epoch.
        sol_scale_f32_c_rows(accum, row_scale, mi_out * 16);
        // Reuse B staging as column-major V [K=64,N=128].
        for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
            const int d = i / kSolFusedTokenBlock, n = i % kSolFusedTokenBlock, vt = k_begin + n;
            sb[i] = vt < l ? v[(size_t) h * v_s2 + (size_t) vt * v_s1 + d] : __float2half(0.0f);
        }
        __syncthreads();
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            #pragma unroll
            for (int kk = 0; kk < kSolFusedTokenBlock; kk += 16) {
                wmma::load_matrix_sync(a, probs + mi_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                wmma::load_matrix_sync(b, sb + ni_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                wmma::mma_sync(accum, a, b, accum);
            }
        }
        __syncthreads();
    }
    // Normalize the persistent F32 accumulator in-register, then make the
    // single final BSHD write for the tile.
    for (int lane_scale = 0; lane_scale < 1; ++lane_scale) {
        (void) lane_scale;
        const int lane = tid & 31;
        const float s0 = 1.0f / row_sum[mi_out * 16 + lane / 4];
        const float s1 = 1.0f / row_sum[mi_out * 16 + lane / 4 + 8];
        accum.x[0] *= s0; accum.x[1] *= s0; accum.x[4] *= s0; accum.x[5] *= s0;
        accum.x[2] *= s1; accum.x[3] *= s1; accum.x[6] *= s1; accum.x[7] *= s1;
    }
    // Q and B staging are dead after the final epoch.  Overlay their combined
    // 32 KiB with a complete F32 output tile so a partial last Q64 block can
    // be bounds-checked before its only global write.
    float * final_tile = reinterpret_cast<float *>(sq);
    wmma::store_matrix_sync(final_tile + mi_out * 16 * kSolHeadDim + ni_out * 16,
                            accum, kSolHeadDim, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
        const int row = i / kSolHeadDim, d = i % kSolHeadDim, t = q_begin + row;
        if (t < l) out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d] = final_tile[i];
    }
}

// Sparse counterpart to the persistent dense kernel above.  It deliberately
// retains the same Q64 / K64 / F32-fragment ownership: only the contribution
// schedule differs.  Each G64 epoch first scores K centroids, contributes the
// unselected proxy blocks (Vc is a token sum), then immediately merges each
// selected K64 block exactly into the very same online-softmax recurrence.
// There is no document-sized route materialization: `selected` is one shared
// 64-entry epoch-local mask.  This path is opt-in until its sparse gate clears.
// The paired instantiation consumes two already-selected K64 blocks together.
// It needs a second K/V, score and probability tile (about 97 KiB total
// dynamic shared memory), but importantly does *one* online-softmax update and
// one retained-fragment scale for both blocks.  The ordinary instantiation is
// compile-time separate so its proven 1x layout/code path remains unchanged.
template<bool collect_route_stats, bool paired_selected_blocks = false>
// SM120 compiles the paired specialization to 80/95 registers otherwise;
// with 1024 threads that exceeds this device's 65,536-register CTA limit
// even though its 99,328-byte dynamic shared allocation fits the 101,376-byte
// opt-in limit.  One resident CTA is all this persistent design can use.
__global__ __launch_bounds__(1024, 1) void sol_attention_fused_sparse_persistent_q64(
        const float * q, const half * k, const half * v, const float * kc, const float * vc,
        const float * threshold, float * out, int l, int heads, int q_blocks, int kv_blocks,
        float scale, int sink_blocks, int sink_query_blocks,
        int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2,
        int64_t v_s1, int64_t v_s2, int64_t out_h_stride, int64_t out_t_stride,
        uint8_t * route_stats, int route_epochs) {
    const int qb = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int q_begin = qb * kSolFusedTokenBlock;

    extern __shared__ uint8_t storage[];
    half * sq = reinterpret_cast<half *>(storage);
    half * sb = sq + kSolFusedTokenBlock * kSolHeadDim;
    float * scores = reinterpret_cast<float *>(sb + kSolHeadDim * kSolFusedTokenBlock);
    half * probs = reinterpret_cast<half *>(scores + kSolFusedTokenBlock * kSolFusedTokenBlock);
    float * row_max = reinterpret_cast<float *>(probs + kSolFusedTokenBlock * kSolFusedTokenBlock);
    float * row_sum = row_max + kSolFusedTokenBlock;
    float * row_scale = row_sum + kSolFusedTokenBlock;
    int * selected = reinterpret_cast<int *>(row_scale + kSolFusedTokenBlock);
    half * sb_pair = paired_selected_blocks ? reinterpret_cast<half *>(selected + kSolFusedGroupBlocks) : nullptr;
    float * scores_pair = paired_selected_blocks ? reinterpret_cast<float *>(sb_pair + kSolFusedTokenBlock * kSolHeadDim) : nullptr;
    half * probs_pair = paired_selected_blocks ? reinterpret_cast<half *>(scores_pair + kSolFusedTokenBlock * kSolFusedTokenBlock) : nullptr;

    const int mi_out = warp / 8, ni_out = warp & 7;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
    wmma::fill_fragment(accum, 0.0f);

    for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
        const int row = i / kSolHeadDim, d = i % kSolHeadDim, t = q_begin + row;
        sq[i] = t < l ? __float2half_rn(q[(size_t) h * q_s2 + (size_t) t * q_s1 + d]) : __float2half(0.0f);
    }
    for (int row = tid; row < kSolFusedTokenBlock; row += blockDim.x) {
        row_max[row] = -INFINITY; row_sum[row] = 0.0f; row_scale[row] = 1.0f;
    }
    __syncthreads();

    const bool q_sink = qb < sink_query_blocks;
    const int q_len = min(kSolFusedTokenBlock, l - q_begin);
    const float route_threshold = threshold[(size_t) h * q_blocks + qb];
    for (int group_start = 0; group_start < kv_blocks; group_start += kSolFusedGroupBlocks) {
        const int group_n = min(kSolFusedGroupBlocks, kv_blocks - group_start);
        // Score the 64 K centroids once.  `sb` is [D,K] col-major.
        for (int i = tid; i < kSolHeadDim * kSolFusedGroupBlocks; i += blockDim.x) {
            const int n = i / kSolHeadDim, d = i % kSolHeadDim, kb = group_start + n;
            sb[i] = n < group_n ? __float2half_rn(kc[((size_t) h * kv_blocks + kb) * kSolHeadDim + d]) : __float2half(0.0f);
        }
        __syncthreads();
        if (warp < 16) {
            const int mi = warp / 4, ni = warp & 3;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::load_matrix_sync(b, sb + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores + mi * 16 * kSolFusedTokenBlock + ni * 16, c,
                                    kSolFusedTokenBlock, wmma::mem_row_major);
        }
        __syncthreads();
        // Reference-shaped block decision from the same score tile.  Sink and
        // neighbour blocks are exact so tails never lose their local context.
        if (tid < kSolFusedGroupBlocks) {
            const int n = tid, kb = group_start + n;
            float sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < kSolFusedTokenBlock; ++r) sum += scores[r * kSolFusedTokenBlock + n] * scale;
            selected[n] = n < group_n && (q_sink || kb < sink_blocks || abs(qb - kb) <= 1 || sum / q_len > route_threshold);
        }
        __syncthreads();
        if constexpr (collect_route_stats) {
            if (tid == 0) {
                int selected_count = 0;
                #pragma unroll
                for (int n = 0; n < kSolFusedGroupBlocks; ++n) selected_count += n < group_n && selected[n];
                route_stats[((size_t) h * q_blocks + qb) * route_epochs + group_start / kSolFusedGroupBlocks] =
                    static_cast<uint8_t>(selected_count);
            }
        }

        // Unselected centroids form one proxy epoch.  Vc is a sum rather than
        // mean, while row_sum includes the represented token multiplicity.
        for (int row = tid; row < kSolFusedTokenBlock; row += blockDim.x) {
            const int t = q_begin + row;
            float local_max = -INFINITY;
            if (t < l) for (int n = 0; n < group_n; ++n) if (!selected[n])
                local_max = fmaxf(local_max, scores[row * kSolFusedTokenBlock + n] * scale);
            if (local_max == -INFINITY) {
                row_scale[row] = 1.0f;
                for (int n = 0; n < kSolFusedGroupBlocks; ++n) scores[row * kSolFusedTokenBlock + n] = 0.0f;
            } else {
                const float nm = fmaxf(row_max[row], local_max), alpha = expf(row_max[row] - nm);
                float add = 0.0f;
                for (int n = 0; n < kSolFusedGroupBlocks; ++n) {
                    const float p = n < group_n && !selected[n] ? expf(scores[row * kSolFusedTokenBlock + n] * scale - nm) : 0.0f;
                    scores[row * kSolFusedTokenBlock + n] = p;
                    if (n < group_n) add += p * min(kSolFusedTokenBlock, l - (group_start + n) * kSolFusedTokenBlock);
                }
                row_scale[row] = alpha; row_sum[row] = row_sum[row] * alpha + add; row_max[row] = nm;
            }
        }
        __syncthreads();
        for (int i = tid; i < kSolFusedTokenBlock * kSolFusedTokenBlock; i += blockDim.x) probs[i] = __float2half_rn(scores[i]);
        __syncthreads();
        sol_scale_f32_c_rows(accum, row_scale, mi_out * 16);
        // V centroid matrix [K,D] is represented as a column-major [K,N].
        for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
            const int d = i / kSolFusedTokenBlock, n = i % kSolFusedTokenBlock, kb = group_start + n;
            sb[i] = n < group_n && !selected[n] ? __float2half_rn(vc[((size_t) h * kv_blocks + kb) * kSolHeadDim + d]) : __float2half(0.0f);
        }
        __syncthreads();
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            #pragma unroll
            for (int kk = 0; kk < kSolFusedTokenBlock; kk += 16) {
                wmma::load_matrix_sync(a, probs + mi_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                wmma::load_matrix_sync(b, sb + ni_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                wmma::mma_sync(accum, a, b, accum);
            }
        }
        __syncthreads();

        // The normal producer consumes exactly one selected K64 at a time.
        // Keep it as a compile-time path: the paired experiment below must not
        // perturb the validated sparse recurrence or its shared-memory shape.
        if constexpr (!paired_selected_blocks) for (int n = 0; n < group_n; ++n) if (selected[n]) {
            const int kb = group_start + n, k_begin = kb * kSolFusedTokenBlock;
            for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
                const int col = i / kSolHeadDim, d = i % kSolHeadDim, kt = k_begin + col;
                sb[i] = kt < l ? k[(size_t) h * k_s2 + (size_t) kt * k_s1 + d] : __float2half(0.0f);
            }
            __syncthreads();
            if (warp < 16) {
                const int mi = warp / 4, ni = warp & 3;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                    wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                    wmma::load_matrix_sync(b, sb + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(scores + mi * 16 * kSolFusedTokenBlock + ni * 16, c,
                                        kSolFusedTokenBlock, wmma::mem_row_major);
            }
            __syncthreads();
            for (int row = tid; row < kSolFusedTokenBlock; row += blockDim.x) {
                const int t = q_begin + row, valid = min(kSolFusedTokenBlock, l - k_begin);
                float local_max = -INFINITY;
                if (t < l) for (int col = 0; col < valid; ++col)
                    local_max = fmaxf(local_max, scores[row * kSolFusedTokenBlock + col] * scale);
                const float nm = fmaxf(row_max[row], local_max), alpha = t < l ? expf(row_max[row] - nm) : 1.0f;
                float add = 0.0f;
                for (int col = 0; col < kSolFusedTokenBlock; ++col) {
                    const float p = t < l && col < valid ? expf(scores[row * kSolFusedTokenBlock + col] * scale - nm) : 0.0f;
                    scores[row * kSolFusedTokenBlock + col] = p; add += p;
                }
                row_scale[row] = alpha;
                if (t < l) { row_sum[row] = row_sum[row] * alpha + add; row_max[row] = nm; }
            }
            __syncthreads();
            for (int i = tid; i < kSolFusedTokenBlock * kSolFusedTokenBlock; i += blockDim.x) probs[i] = __float2half_rn(scores[i]);
            __syncthreads();
            sol_scale_f32_c_rows(accum, row_scale, mi_out * 16);
            for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
                const int d = i / kSolFusedTokenBlock, col = i % kSolFusedTokenBlock, vt = k_begin + col;
                sb[i] = vt < l ? v[(size_t) h * v_s2 + (size_t) vt * v_s1 + d] : __float2half(0.0f);
            }
            __syncthreads();
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                #pragma unroll
                for (int kk = 0; kk < kSolFusedTokenBlock; kk += 16) {
                    wmma::load_matrix_sync(a, probs + mi_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                    wmma::load_matrix_sync(b, sb + ni_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                    wmma::mma_sync(accum, a, b, accum);
                }
            }
            __syncthreads();
        }
        // Pair consecutive selected K64 blocks.  The two score tiles are
        // retained simultaneously so their maxima/probabilities form a single
        // 128-key online-softmax epoch; applying row_scale once is essential.
        // The two P@V products then accumulate into the same F32 fragments.
        if constexpr (paired_selected_blocks) for (int n = 0; n < group_n; ++n) if (selected[n]) {
            int n1 = n + 1;
            while (n1 < group_n && !selected[n1]) ++n1;
            const bool have_pair = n1 < group_n;
            const int kb0 = group_start + n, kb1 = have_pair ? group_start + n1 : group_start + n;
            const int k_begin0 = kb0 * kSolFusedTokenBlock, k_begin1 = kb1 * kSolFusedTokenBlock;
            for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
                const int col = i / kSolHeadDim, d = i % kSolHeadDim;
                const int kt0 = k_begin0 + col, kt1 = k_begin1 + col;
                sb[i] = kt0 < l ? k[(size_t) h * k_s2 + (size_t) kt0 * k_s1 + d] : __float2half(0.0f);
                sb_pair[i] = have_pair && kt1 < l ? k[(size_t) h * k_s2 + (size_t) kt1 * k_s1 + d] : __float2half(0.0f);
            }
            __syncthreads();
            if (warp < 16) {
                const int mi = warp / 4, ni = warp & 3;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                // Do not retain two score accumulator fragments at once:
                // 1024 resident threads already retain the F32 P@V fragment,
                // and c0+c1 crosses this GPU's launch resource limit when
                // combined with the 97 KiB paired shared allocation.
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                    wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                    wmma::load_matrix_sync(b, sb + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(scores + mi * 16 * kSolFusedTokenBlock + ni * 16, c,
                                        kSolFusedTokenBlock, wmma::mem_row_major);
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < kSolHeadDim; kk += 16) {
                    wmma::load_matrix_sync(a, sq + mi * 16 * kSolHeadDim + kk, kSolHeadDim);
                    wmma::load_matrix_sync(b, sb_pair + ni * 16 * kSolHeadDim + kk, kSolHeadDim);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(scores_pair + mi * 16 * kSolFusedTokenBlock + ni * 16, c,
                                        kSolFusedTokenBlock, wmma::mem_row_major);
            }
            __syncthreads();
            for (int row = tid; row < kSolFusedTokenBlock; row += blockDim.x) {
                const int t = q_begin + row;
                const int valid0 = min(kSolFusedTokenBlock, l - k_begin0);
                const int valid1 = have_pair ? min(kSolFusedTokenBlock, l - k_begin1) : 0;
                float local_max = -INFINITY;
                if (t < l) {
                    for (int col = 0; col < valid0; ++col) local_max = fmaxf(local_max, scores[row * kSolFusedTokenBlock + col] * scale);
                    for (int col = 0; col < valid1; ++col) local_max = fmaxf(local_max, scores_pair[row * kSolFusedTokenBlock + col] * scale);
                }
                const float nm = fmaxf(row_max[row], local_max);
                const float alpha = t < l ? expf(row_max[row] - nm) : 1.0f;
                float add = 0.0f;
                for (int col = 0; col < kSolFusedTokenBlock; ++col) {
                    const float p0 = t < l && col < valid0 ? expf(scores[row * kSolFusedTokenBlock + col] * scale - nm) : 0.0f;
                    const float p1 = t < l && col < valid1 ? expf(scores_pair[row * kSolFusedTokenBlock + col] * scale - nm) : 0.0f;
                    probs[row * kSolFusedTokenBlock + col] = __float2half_rn(p0);
                    probs_pair[row * kSolFusedTokenBlock + col] = __float2half_rn(p1);
                    add += p0 + p1;
                }
                row_scale[row] = alpha;
                if (t < l) { row_sum[row] = row_sum[row] * alpha + add; row_max[row] = nm; }
            }
            __syncthreads();
            sol_scale_f32_c_rows(accum, row_scale, mi_out * 16);
            for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
                const int d = i / kSolFusedTokenBlock, col = i % kSolFusedTokenBlock;
                const int vt0 = k_begin0 + col, vt1 = k_begin1 + col;
                sb[i] = vt0 < l ? v[(size_t) h * v_s2 + (size_t) vt0 * v_s1 + d] : __float2half(0.0f);
                sb_pair[i] = have_pair && vt1 < l ? v[(size_t) h * v_s2 + (size_t) vt1 * v_s1 + d] : __float2half(0.0f);
            }
            __syncthreads();
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                #pragma unroll
                for (int kk = 0; kk < kSolFusedTokenBlock; kk += 16) {
                    wmma::load_matrix_sync(a, probs + mi_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                    // V is staged as column-major [K=64,N=128]: K, not
                    // head width, is its leading dimension.
                    wmma::load_matrix_sync(b, sb + ni_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                    wmma::mma_sync(accum, a, b, accum);
                    wmma::load_matrix_sync(a, probs_pair + mi_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                    wmma::load_matrix_sync(b, sb_pair + ni_out * 16 * kSolFusedTokenBlock + kk, kSolFusedTokenBlock);
                    wmma::mma_sync(accum, a, b, accum);
                }
            }
            __syncthreads();
            if (have_pair) n = n1;
        }
    }
    const int lane = tid & 31;
    const float s0 = 1.0f / row_sum[mi_out * 16 + lane / 4];
    const float s1 = 1.0f / row_sum[mi_out * 16 + lane / 4 + 8];
    accum.x[0] *= s0; accum.x[1] *= s0; accum.x[4] *= s0; accum.x[5] *= s0;
    accum.x[2] *= s1; accum.x[3] *= s1; accum.x[6] *= s1; accum.x[7] *= s1;
    float * final_tile = reinterpret_cast<float *>(sq);
    wmma::store_matrix_sync(final_tile + mi_out * 16 * kSolHeadDim + ni_out * 16,
                            accum, kSolHeadDim, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < kSolFusedTokenBlock * kSolHeadDim; i += blockDim.x) {
        const int row = i / kSolHeadDim, d = i % kSolHeadDim, t = q_begin + row;
        if (t < l) out[(size_t) h * out_h_stride + (size_t) t * out_t_stride + d] = final_tile[i];
    }
}

__global__ void sol_fused_cast_bshd_f16(const float * src, half * dst, size_t count,
                                        int l, int heads, int64_t dst_h_stride, int64_t dst_t_stride) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int d = i % kSolHeadDim;
    const size_t row = i / kSolHeadDim;
    const int t = row % l;
    const int h = row / l;
    if (h < heads) dst[(size_t) h * dst_h_stride + (size_t) t * dst_t_stride + d] = __float2half_rn(src[i]);
}

// Debug-only postcondition check.  A NaN/Inf or absurd magnitude in one
// attention result poisons every later H3 block and manifests as decoder
// garbage, so detect it at the producer rather than calling it "quality".
template<typename T>
__global__ void sol_check_output(const T * out, size_t count, int l, int heads,
                                 int64_t out_h_stride, int64_t out_t_stride, int * bad) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || *bad) return;
    const int d = i % kSolHeadDim;
    const size_t row = i / kSolHeadDim;
    const int t = row % l;
    const int h = row / l;
    if (h >= heads) return;
    const size_t offset = (size_t) h * out_h_stride + (size_t) t * out_t_stride + d;
    const float x = std::is_same<T, float>::value ? ((const float *) out)[offset] : __half2float(((const half *) out)[offset]);
    if (!isfinite(x) || fabsf(x) > 1.0e4f) atomicExch(bad, 1);
}

template<typename T>
__global__ void sol_compare_output(const T * native_out, const T * dense_out,
                                   size_t count, unsigned int * max_abs_bits,
                                   unsigned int * max_ref_bits, int l, int heads,
                                   int64_t out_h_stride, int64_t out_t_stride) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int d = i % kSolHeadDim;
    const size_t row = i / kSolHeadDim;
    const int t = row % l;
    const int h = row / l;
    if (h >= heads) return;
    const size_t offset = (size_t) h * out_h_stride + (size_t) t * out_t_stride + d;
    const float native_x = std::is_same<T, float>::value ? ((const float *) native_out)[offset] : __half2float(((const half *) native_out)[offset]);
    const float dense_x = std::is_same<T, float>::value ? ((const float *) dense_out)[offset] : __half2float(((const half *) dense_out)[offset]);
    const float delta = fabsf(native_x - dense_x);
    atomicMax(max_abs_bits, __float_as_uint(delta));
    atomicMax(max_ref_bits, __float_as_uint(fabsf(dense_x)));
}

// A sparse result has no dense oracle by design.  This optional paired check
// runs the scalar route consumer against the *identical compact route list and
// bitmap* used by the WMMA consumer, so any reported delta is staging/MMA
// error rather than a different routing decision.
void sol_compare_same_route(ggml_backend_cuda_context & ctx, const float * wmma_out,
                            const float * scalar_out, int l, int heads,
                            int64_t out_h_stride, int64_t out_t_stride) {
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<unsigned int> stats(ctx.pool());
    stats.alloc(2);
    CUDA_CHECK(cudaMemsetAsync(stats.get(), 0, 2 * sizeof(unsigned int), stream));
    const size_t count = (size_t) kSolHeadDim * l * heads;
    sol_compare_output<<<(count + 255)/256, 256, 0, stream>>>(wmma_out, scalar_out, count,
        stats.get(), stats.get() + 1, l, heads, out_h_stride, out_t_stride);
    unsigned int host_bits[2] = {};
    CUDA_CHECK(cudaMemcpyAsync(host_bits, stats.get(), sizeof(host_bits), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float max_abs = 0.0f, max_ref = 0.0f;
    memcpy(&max_abs, &host_bits[0], sizeof(max_abs));
    memcpy(&max_ref, &host_bits[1], sizeof(max_ref));
    fprintf(stderr, "[sol] SPARSE_SAME_ROUTE_CHECK L=%d H=%d max_abs=%.8g max_ref=%.8g rel_to_global_max=%.8g\n",
            l, heads, max_abs, max_ref, max_ref > 0.0f ? max_abs / max_ref : max_abs);
}

// Reads logical [D,L,H] coordinates through the tensor's actual strides.  It
// is intentionally tiny diagnostic plumbing: a native all-exact result cannot
// be judged until Q/K/V themselves are proven finite on the very invocation.
template<typename T>
__global__ void sol_tensor_stats(const T * data, size_t count, int l, int heads,
                                 int64_t s1, int64_t s2, unsigned int * nonfinite,
                                 unsigned int * max_abs_bits) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int d = i % kSolHeadDim;
    const size_t row = i / kSolHeadDim;
    const int t = row % l;
    const int h = row / l;
    if (h >= heads) return;
    const float x = std::is_same<T, float>::value
        ? ((const float *) data)[(size_t) h * s2 + (size_t) t * s1 + d]
        : __half2float(((const half *) data)[(size_t) h * s2 + (size_t) t * s1 + d]);
    if (!isfinite(x)) {
        atomicAdd(nonfinite, 1);
    } else {
        atomicMax(max_abs_bits, __float_as_uint(fabsf(x)));
    }
}

template<typename T>
void sol_log_tensor_stats(const char * name, const T * data, int l, int heads,
                          int64_t s1, int64_t s2, cudaStream_t stream,
                          ggml_backend_cuda_context & ctx) {
    ggml_cuda_pool_alloc<unsigned int> stats(ctx.pool());
    stats.alloc(2);
    CUDA_CHECK(cudaMemsetAsync(stats.get(), 0, 2 * sizeof(unsigned int), stream));
    const size_t count = (size_t) kSolHeadDim * l * heads;
    sol_tensor_stats<<<(count + 255)/256, 256, 0, stream>>>(data, count, l, heads, s1, s2,
                                                               stats.get(), stats.get() + 1);
    unsigned int host[2] = {};
    CUDA_CHECK(cudaMemcpyAsync(host, stats.get(), sizeof(host), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float max_abs = 0.0f;
    memcpy(&max_abs, &host[1], sizeof(max_abs));
    fprintf(stderr, "[sol] input %s finite_bad=%u max_abs=%.8g stride=[1,%lld,%lld]\n",
            name, host[0], max_abs, (long long) s1, (long long) s2);
}

bool sol_contract_ok(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                     const ggml_tensor * mask, const ggml_tensor * dst) {
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) return false;
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) return false;
    if (mask != nullptr || q->ne[0] != kSolHeadDim || k->ne[0] != kSolHeadDim || v->ne[0] != kSolHeadDim) return false;
    if (q->ne[1] != k->ne[1] || q->ne[1] != v->ne[1] || q->ne[2] != k->ne[2] || q->ne[2] != v->ne[2]) return false;
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || q->ne[2] != 56) return false;
    // FLASH_ATTN_EXT's output is BSHD [D,H,L,N], whereas its Q/K/V inputs
    // are [D,L,H,N].  Do not silently apply the Q layout to dst.
    if (dst->ne[0] != kSolHeadDim || dst->ne[1] != q->ne[2] || dst->ne[2] != q->ne[1] || dst->ne[3] != 1) return false;
    return q->nb[0] == sizeof(float) && k->nb[0] == sizeof(half) && v->nb[0] == sizeof(half) &&
           dst->nb[0] == ggml_type_size(dst->type);
}

} // namespace

bool ggml_cuda_flash_attn_ext_sol(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale, dst->op_params, sizeof(scale));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(softcap));

    const int min_tokens = sol_env_i("GGML_H3_SOL_ATTN_MIN_TOKENS", 4096, 64, 1 << 20);
    if (max_bias != 0.0f || softcap != 0.0f || q->ne[1] < min_tokens ||
        !sol_contract_ok(q, k, v, dst->src[3], dst)) {
        static int rejected = 0;
        if (rejected++ < 3) {
            fprintf(stderr, "[sol] fallback: q=[%lld,%lld,%lld,%lld] k=[%lld,%lld,%lld,%lld] v=[%lld,%lld,%lld,%lld] q/k/v=%d/%d/%d dst=%d mask=%d contig=%d/%d/%d/%d min=%d\n",
                    (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2], (long long) q->ne[3],
                    (long long) k->ne[0], (long long) k->ne[1], (long long) k->ne[2], (long long) k->ne[3],
                    (long long) v->ne[0], (long long) v->ne[1], (long long) v->ne[2], (long long) v->ne[3],
                    q->type, k->type, v->type, dst->type, dst->src[3] != nullptr,
                    ggml_is_contiguous(q), ggml_is_contiguous(k), ggml_is_contiguous(v), ggml_is_contiguous(dst), min_tokens);
        }
        return false;
    }

    // Benchmark safety valve: permit a bounded number of real Sol calls in an
    // otherwise complete H3 sample, then decline into the established dense
    // path.  Zero is the normal unlimited behaviour.  This is intentionally a
    // dispatcher fallback rather than a partial kernel result.
    static int dispatch_count = 0;
    const int max_calls = sol_env_i("GGML_H3_SOL_ATTN_MAX_CALLS", 0, 0, 1000000);
    if (max_calls != 0 && dispatch_count >= max_calls) return false;

    const int l = static_cast<int>(q->ne[1]);
    const int heads = static_cast<int>(q->ne[2]);
    // The reference-shaped implementation is opt-in while it clears the
    // all-exact gate.  Unlike the legacy WMMA experiment it has no global
    // route-list allocation and uses Q64 / KV64 / GROUP=64 blocks.
    const bool fused_reference = sol_env_i("GGML_H3_SOL_ATTN_FUSED", 0, 0, 1) != 0;
    // Experimental long-context variant.  Two D64 CTAs retain their online
    // value state in shared memory, avoiding a destination read/scale/write at
    // every G64 epoch.  Leave it opt-in until it clears the exact and sparse
    // gates; the single-CTA global-state route is the known-correct fallback.
    const bool fused_value_tiles = fused_reference &&
        sol_env_i("GGML_H3_SOL_ATTN_VALUE_TILES", 0, 0, 1) != 0;
    const bool fused_fp16_state = fused_reference && !fused_value_tiles &&
        sol_env_i("GGML_H3_SOL_ATTN_FP16_STATE", 0, 0, 1) != 0;
    // Persistent F32 fragment implementation.  Sparse is a separate opt-in
    // from the dense exact gate: enabling PERSISTENT alone never changes the
    // verified all-exact behavior or silently approximates an attention node.
    const bool fused_persistent = fused_reference &&
        sol_env_i("GGML_H3_SOL_ATTN_PERSISTENT", 0, 0, 1) != 0;
    const bool fused_persistent_sparse = fused_persistent &&
        sol_env_i("GGML_H3_SOL_ATTN_PERSISTENT_SPARSE", 0, 0, 1) != 0;
    // Conservative opt-in for the 2x selected-K64 consumer.  It only changes
    // the sparse persistent producer; exact/dense and the 1x sparse fallback
    // retain their established kernels and shared-memory footprint.
    const bool fused_persistent_sparse_paired = fused_persistent_sparse &&
        sol_env_i("GGML_H3_SOL_ATTN_PERSISTENT_SPARSE_PAIRED", 0, 0, 1) != 0;
    const int q_blocks = (l + kSolQueryBlock - 1) / kSolQueryBlock;
    const int kv_blocks = (l + kSolKVGroup - 1) / kSolKVGroup;
    const float tau = sol_env_f("GGML_H3_SOL_ATTN_TAU", 1.0f, 0.0f, 4.0f);
    const int sink_tokens = sol_env_i("GGML_H3_SOL_ATTN_SINK_TOKENS", 0, 0, l);
    const int sink_query_tokens = sol_env_i("GGML_H3_SOL_ATTN_SINK_QUERY_TOKENS", 0, 0, l);
    const int sink_blocks = (sink_tokens + kSolKVGroup - 1) / kSolKVGroup;
    const int sink_query_blocks = (sink_query_tokens + kSolQueryBlock - 1) / kSolQueryBlock;
    // DENSE_CHECK is a paired native-vs-cuDNN diagnostic and therefore always
    // means all-exact, even if the caller forgot ALL_EXACT explicitly.
    const bool dense_check = sol_env_i("GGML_H3_SOL_ATTN_DENSE_CHECK", 0, 0, 1) != 0;
    const bool force_exact = dense_check || sol_env_i("GGML_H3_SOL_ATTN_ALL_EXACT", 0, 0, 1) != 0;
    // DENSE_CHECK/ALL_EXACT always select the dense persistent recurrence.
    // Sparse must be explicitly requested and is separately guarded below.
    const bool fused_persistent_exact = fused_persistent && force_exact;
    const bool fused_persistent_sparse_active = fused_persistent_sparse && !force_exact;
    const bool validate_output = sol_env_i("GGML_H3_SOL_ATTN_VALIDATE", 0, 0, 1) != 0;
    const bool sparse_same_route_check = sol_env_i("GGML_H3_SOL_ATTN_SPARSE_SAME_ROUTE_CHECK", 0, 0, 1) != 0;
    const bool route_stats = sol_env_i("GGML_H3_SOL_ATTN_ROUTE_STATS", 0, 0, 1) != 0;
    // Experimental performance path: route one K-block decision per 64-query
    // block.  It is opt-in until a conservative multi-step visual validates
    // this approximation.  The exact diagnostic ignores its route choices.
    const bool block_routes = sol_env_i("GGML_H3_SOL_ATTN_BLOCK_ROUTES", 0, 0, 1) != 0;
    // F32-output-only exploratory tensor-core path.  It requires block routes
    // because its exact phase consumes their compact selected-group list.
    const bool wmma_tiles = block_routes && !force_exact ?
        sol_env_i("GGML_H3_SOL_ATTN_WMMA", 0, 0, 1) != 0 :
        (block_routes && force_exact && sol_env_i("GGML_H3_SOL_ATTN_WMMA", 0, 0, 1) != 0);
    const int64_t q_s1 = q->nb[1] / sizeof(float), q_s2 = q->nb[2] / sizeof(float);
    const int64_t k_s1 = k->nb[1] / sizeof(half),  k_s2 = k->nb[2] / sizeof(half);
    const int64_t v_s1 = v->nb[1] / sizeof(half),  v_s2 = v->nb[2] / sizeof(half);
    // Q/K/V are [D,L,H,1], but FLASH_ATTN_EXT output is [D,H,L,1].
    if (dst->ne[0] != kSolHeadDim || dst->ne[1] != heads || dst->ne[2] != l || dst->ne[3] != 1) {
        fprintf(stderr, "[sol] fallback: unexpected output ne=[%lld,%lld,%lld,%lld], expected [128,H,L,1]\n",
                (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3]);
        return false;
    }
    const int64_t o_h_stride = dst->nb[1] / ggml_type_size(dst->type);
    const int64_t o_t_stride = dst->nb[2] / ggml_type_size(dst->type);
    cudaStream_t stream = ctx.stream();

    if (dense_check || validate_output) {
        sol_log_tensor_stats("Q", (const float *) q->data, l, heads, q_s1, q_s2, stream, ctx);
        sol_log_tensor_stats("K", (const half *) k->data, l, heads, k_s1, k_s2, stream, ctx);
        sol_log_tensor_stats("V", (const half *) v->data, l, heads, v_s1, v_s2, stream, ctx);
        fprintf(stderr, "[sol] output BSHD stride=[1,h=%lld,t=%lld] type=%d contiguous=%d\n",
                (long long) o_h_stride, (long long) o_t_stride, dst->type, ggml_is_contiguous(dst));
    }

    if (fused_reference) {
        // Route telemetry is deliberately one bounded real attention call per
        // process.  The normal fused sparse instantiation never allocates,
        // writes, or synchronizes for it.
        static int fused_route_stats_logged = 0;
        const bool collect_fused_route_stats = fused_persistent_sparse_active && route_stats && fused_route_stats_logged == 0;
        const int fused_q_blocks = (l + kSolFusedTokenBlock - 1) / kSolFusedTokenBlock;
        const int fused_kv_blocks = fused_q_blocks;
        const int fused_sink_blocks = (sink_tokens + kSolFusedTokenBlock - 1) / kSolFusedTokenBlock;
        const int fused_sink_query_blocks = (sink_query_tokens + kSolFusedTokenBlock - 1) / kSolFusedTokenBlock;
        ggml_cuda_pool_alloc<float> fused_kc(ctx.pool()), fused_vc(ctx.pool()), fused_mean(ctx.pool()), fused_var(ctx.pool()), fused_threshold(ctx.pool());
        ggml_cuda_pool_alloc<uint8_t> fused_route_stats(ctx.pool());
        const int fused_route_epochs = (fused_kv_blocks + kSolFusedGroupBlocks - 1) / kSolFusedGroupBlocks;
        fused_kc.alloc((size_t) heads * fused_kv_blocks * kSolHeadDim);
        fused_vc.alloc((size_t) heads * fused_kv_blocks * kSolHeadDim);
        fused_mean.alloc((size_t) heads * kSolHeadDim);
        fused_var.alloc((size_t) heads * kSolHeadDim);
        fused_threshold.alloc((size_t) heads * fused_q_blocks);
        if (fused_persistent_sparse_active && collect_fused_route_stats) {
            fused_route_stats.alloc((size_t) heads * fused_q_blocks * fused_route_epochs);
        }
        ggml_cuda_pool_alloc<float> fused_out(ctx.pool());
        const bool fused_f16_dst = dst->type == GGML_TYPE_F16;
        if (fused_f16_dst) fused_out.alloc((size_t) kSolHeadDim * l * heads);
        sol_fused_summarize64<<<dim3(1, (unsigned) fused_kv_blocks, (unsigned) heads), kSolHeadDim, 0, stream>>>(
            (const half *) k->data, (const half *) v->data, fused_kc.get(), fused_vc.get(), l, heads, fused_kv_blocks, k_s1, k_s2, v_s1, v_s2);
        sol_fused_moments<<<dim3(1, (unsigned) heads), kSolHeadDim, 0, stream>>>(
            fused_kc.get(), fused_mean.get(), fused_var.get(), heads, fused_kv_blocks);
        sol_fused_threshold64<<<dim3((unsigned) fused_q_blocks, (unsigned) heads), 32, 0, stream>>>(
            (const float *) q->data, fused_mean.get(), fused_var.get(), fused_threshold.get(), l, heads, fused_q_blocks,
            scale, tau, q_s1, q_s2);
        constexpr size_t fused_smem_base =
            (kSolFusedTokenBlock * kSolHeadDim + kSolHeadDim * kSolFusedTokenBlock) * sizeof(half) +
            kSolFusedTokenBlock * kSolFusedTokenBlock * sizeof(float) +
            kSolFusedTokenBlock * kSolFusedTokenBlock * sizeof(half) +
            (3 * kSolFusedTokenBlock) * sizeof(float) + kSolFusedGroupBlocks * sizeof(int);
        constexpr size_t fused_value_tile_smem = fused_smem_base +
            kSolFusedTokenBlock * 64 * sizeof(float);
        const size_t fused_smem = (fused_value_tiles || fused_fp16_state) ? fused_value_tile_smem : fused_smem_base;
        constexpr size_t fused_persistent_smem =
            (kSolFusedTokenBlock * kSolHeadDim + kSolHeadDim * kSolFusedTokenBlock) * sizeof(half) +
            kSolFusedTokenBlock * kSolFusedTokenBlock * sizeof(float) +
            kSolFusedTokenBlock * kSolFusedTokenBlock * sizeof(half) +
            3 * kSolFusedTokenBlock * sizeof(float);
        constexpr size_t fused_persistent_sparse_smem =
            fused_persistent_smem + kSolFusedGroupBlocks * sizeof(int);
        constexpr size_t fused_persistent_sparse_paired_smem =
            fused_persistent_sparse_smem +
            kSolFusedTokenBlock * kSolHeadDim * sizeof(half) +
            kSolFusedTokenBlock * kSolFusedTokenBlock * sizeof(float) +
            kSolFusedTokenBlock * kSolFusedTokenBlock * sizeof(half);
        CUDA_CHECK(cudaFuncSetAttribute(sol_attention_fused_q64_g32,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, fused_smem));
        float * fused_dst = fused_f16_dst ? fused_out.get() : (float *) dst->data;
        const int64_t fused_h_stride = fused_f16_dst ? kSolHeadDim : o_h_stride;
        const int64_t fused_t_stride = fused_f16_dst ? (int64_t) kSolHeadDim * heads : o_t_stride;
        if (fused_persistent_exact) {
            CUDA_CHECK(cudaFuncSetAttribute(sol_attention_fused_dense_persistent_q64,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            fused_persistent_smem));
            sol_attention_fused_dense_persistent_q64<<<dim3((unsigned) fused_q_blocks, (unsigned) heads),
                                                             1024, fused_persistent_smem, stream>>>(
                (const float *) q->data, (const half *) k->data, (const half *) v->data, fused_dst,
                l, heads, scale, q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, fused_h_stride, fused_t_stride);
        } else if (fused_persistent_sparse_active) {
            if (collect_fused_route_stats) {
                if (fused_persistent_sparse_paired) {
                    CUDA_CHECK(cudaFuncSetAttribute(sol_attention_fused_sparse_persistent_q64<true, true>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    fused_persistent_sparse_paired_smem));
                    sol_attention_fused_sparse_persistent_q64<true, true><<<dim3((unsigned) fused_q_blocks, (unsigned) heads),
                                                                              1024, fused_persistent_sparse_paired_smem, stream>>>(
                        (const float *) q->data, (const half *) k->data, (const half *) v->data,
                        fused_kc.get(), fused_vc.get(), fused_threshold.get(), fused_dst,
                        l, heads, fused_q_blocks, fused_kv_blocks, scale,
                        fused_sink_blocks, fused_sink_query_blocks,
                        q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, fused_h_stride, fused_t_stride,
                        fused_route_stats.get(), fused_route_epochs);
                } else {
                    CUDA_CHECK(cudaFuncSetAttribute(sol_attention_fused_sparse_persistent_q64<true>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    fused_persistent_sparse_smem));
                    sol_attention_fused_sparse_persistent_q64<true><<<dim3((unsigned) fused_q_blocks, (unsigned) heads),
                                                                           1024, fused_persistent_sparse_smem, stream>>>(
                        (const float *) q->data, (const half *) k->data, (const half *) v->data,
                        fused_kc.get(), fused_vc.get(), fused_threshold.get(), fused_dst,
                        l, heads, fused_q_blocks, fused_kv_blocks, scale,
                        fused_sink_blocks, fused_sink_query_blocks,
                        q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, fused_h_stride, fused_t_stride,
                        fused_route_stats.get(), fused_route_epochs);
                }
            } else {
                if (fused_persistent_sparse_paired) {
                    CUDA_CHECK(cudaFuncSetAttribute(sol_attention_fused_sparse_persistent_q64<false, true>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    fused_persistent_sparse_paired_smem));
                    sol_attention_fused_sparse_persistent_q64<false, true><<<dim3((unsigned) fused_q_blocks, (unsigned) heads),
                                                                               1024, fused_persistent_sparse_paired_smem, stream>>>(
                        (const float *) q->data, (const half *) k->data, (const half *) v->data,
                        fused_kc.get(), fused_vc.get(), fused_threshold.get(), fused_dst,
                        l, heads, fused_q_blocks, fused_kv_blocks, scale,
                        fused_sink_blocks, fused_sink_query_blocks,
                        q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, fused_h_stride, fused_t_stride,
                        nullptr, 0);
                } else {
                    CUDA_CHECK(cudaFuncSetAttribute(sol_attention_fused_sparse_persistent_q64<false>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    fused_persistent_sparse_smem));
                    sol_attention_fused_sparse_persistent_q64<false><<<dim3((unsigned) fused_q_blocks, (unsigned) heads),
                                                                            1024, fused_persistent_sparse_smem, stream>>>(
                        (const float *) q->data, (const half *) k->data, (const half *) v->data,
                        fused_kc.get(), fused_vc.get(), fused_threshold.get(), fused_dst,
                        l, heads, fused_q_blocks, fused_kv_blocks, scale,
                        fused_sink_blocks, fused_sink_query_blocks,
                        q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, fused_h_stride, fused_t_stride,
                        nullptr, 0);
                }
            }
        } else {
            if (fused_persistent && dispatch_count < 4) {
                fprintf(stderr, "[sol] persistent requires ALL_EXACT/DENSE_CHECK or PERSISTENT_SPARSE=1; using fused fallback\n");
            }
            sol_attention_fused_q64_g32<<<dim3((unsigned) fused_q_blocks, (unsigned) heads,
                                                fused_value_tiles ? 2U : 1U), 256, fused_smem, stream>>>(
                (const float *) q->data, (const half *) k->data, (const half *) v->data, fused_kc.get(), fused_vc.get(),
                fused_threshold.get(), fused_dst, l, heads, fused_q_blocks, fused_kv_blocks, scale,
                fused_sink_blocks, fused_sink_query_blocks, force_exact, q_s1, q_s2, k_s1, k_s2, v_s1, v_s2,
                fused_h_stride, fused_t_stride, fused_value_tiles, fused_fp16_state);
        }
        if (fused_f16_dst) {
            const size_t fused_count = (size_t) kSolHeadDim * l * heads;
            sol_fused_cast_bshd_f16<<<(fused_count + 255) / 256, 256, 0, stream>>>(
                fused_out.get(), (half *) dst->data, fused_count, l, heads, o_h_stride, o_t_stride);
        }
        if (collect_fused_route_stats) {
            const size_t route_count = (size_t) heads * fused_q_blocks * fused_route_epochs;
            std::vector<uint8_t> host_route_stats(route_count);
            CUDA_CHECK(cudaMemcpyAsync(host_route_stats.data(), fused_route_stats.get(), route_count,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::sort(host_route_stats.begin(), host_route_stats.end());
            size_t ge_two = 0;
            for (const uint8_t x : host_route_stats) ge_two += x >= 2;
            const size_t p50 = route_count / 2;
            const size_t p95 = (route_count * 95) / 100;
            fprintf(stderr, "[sol] ROUTE_STATS Q64/K64/G64 samples=%zu mean=%.3f p50=%u p95=%u ge2=%zu (%.2f%%)\n",
                    route_count,
                    std::accumulate(host_route_stats.begin(), host_route_stats.end(), 0.0) / route_count,
                    (unsigned) host_route_stats[p50], (unsigned) host_route_stats[p95], ge_two,
                    100.0 * ge_two / route_count);
            ++fused_route_stats_logged;
        }
        CUDA_CHECK(cudaGetLastError());
        // Keep the normal log compact, but allow a bounded all-Sol gate to
        // account for every real dispatcher acceptance without altering the
        // routing/result.  This is diagnostic only.
        const int dispatch_log_calls = sol_env_i("GGML_H3_SOL_ATTN_LOG_CALLS", 4, 0, 1000000);
        if (dispatch_count++ < dispatch_log_calls) {
            if (fused_persistent_exact) {
                fprintf(stderr, "[sol] dispatch fused-persistent-dense H=%d L=%d Q64 KV64 all_exact=1\n", heads, l);
            } else if (fused_persistent_sparse_active) {
                fprintf(stderr, "[sol] dispatch fused-persistent-sparse%s H=%d L=%d Q64 KV64 G64 all_exact=0\n",
                        fused_persistent_sparse_paired ? "-paired" : "", heads, l);
            } else {
                fprintf(stderr, "[sol] dispatch fused-reference H=%d L=%d Q64 KV64 G64 all_exact=%d\n", heads, l, force_exact);
            }
        }
        return true;
    }

    ggml_cuda_pool_alloc<float> kc(ctx.pool()), vc(ctx.pool()), thresholds(ctx.pool());
    ggml_cuda_pool_alloc<uint16_t> route_lists(ctx.pool()), route_counts(ctx.pool());
    ggml_cuda_pool_alloc<uint32_t> route_bits(ctx.pool());
    kc.alloc((size_t) heads * kv_blocks * kSolHeadDim);
    vc.alloc((size_t) heads * kv_blocks * kSolHeadDim);
    thresholds.alloc((size_t) heads * q_blocks);
    sol_summarize<<<dim3(1, (unsigned) kv_blocks, (unsigned) heads), kSolHeadDim, 0, stream>>>(
        (const half *) k->data, (const half *) v->data, kc.get(), vc.get(), l, heads, kv_blocks, k_s1, k_s2, v_s1, v_s2);
    sol_thresholds<<<dim3((unsigned) q_blocks, (unsigned) heads), 32, 0, stream>>>(
        (const float *) q->data, kc.get(), thresholds.get(), l, heads, q_blocks, kv_blocks, scale, tau, q_s1, q_s2);
    if (block_routes) {
        route_lists.alloc((size_t) heads * q_blocks * kv_blocks);
        route_counts.alloc((size_t) heads * q_blocks);
        const int route_words = (kv_blocks + 31) / 32;
        route_bits.alloc((size_t) heads * q_blocks * route_words);
        CUDA_CHECK(cudaMemsetAsync(route_bits.get(), 0,
                                   (size_t) heads * q_blocks * route_words * sizeof(uint32_t), stream));
        sol_route_masks<<<dim3((unsigned) q_blocks, (unsigned) heads), 32, 0, stream>>>(
            (const float *) q->data, kc.get(), thresholds.get(), route_lists.get(), route_counts.get(), route_bits.get(), route_words,
            l, heads, q_blocks, kv_blocks, scale,
            sink_blocks, sink_query_blocks, force_exact, q_s1, q_s2);
        // Route density determines whether the remaining selected-token pass
        // can materially affect end-to-end time.  This is diagnostic-only:
        // synchronize once on bounded real H3 calls before deciding whether
        // to grow the exact K/PV staging machinery.
        if (route_stats) {
            const size_t route_tiles = (size_t) heads * q_blocks;
            std::vector<uint16_t> host_counts(route_tiles);
            CUDA_CHECK(cudaMemcpyAsync(host_counts.data(), route_counts.get(),
                                       route_tiles * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint64_t total = 0;
            for (uint16_t count : host_counts) total += count;
            std::sort(host_counts.begin(), host_counts.end());
            const auto percentile = [&](int p) -> uint16_t {
                return host_counts[(host_counts.size() - 1) * p / 100];
            };
            const double density = kv_blocks ? (double) total / ((double) route_tiles * kv_blocks) : 0.0;
            fprintf(stderr, "[sol] ROUTE_STATS L=%d H=%d tiles=%zu kv_groups=%d selected_total=%llu density=%.6f min=%u p50=%u p95=%u max=%u\n",
                    l, heads, route_tiles, kv_blocks, (unsigned long long) total, density,
                    host_counts.front(), percentile(50), percentile(95), host_counts.back());
        }
    }
    const unsigned query_tiles = (unsigned) ((l + 7) / 8);
    const int route_words = (kv_blocks + 31) / 32;
    if (wmma_tiles && dst->type == GGML_TYPE_F32) {
        constexpr size_t smem_bytes =
            (kSolQueryBlock * kSolHeadDim + kSolHeadDim * kSolKVGroup) * sizeof(half) +
            (kSolQueryBlock * kSolKVGroup + 2 * kSolQueryBlock * kSolHeadDim + 3 * kSolQueryBlock) * sizeof(float) +
            kSolQueryBlock * kSolKVGroup * sizeof(half);
        // One CTA owns all D=128 channels.  It stages and scores a selected
        // KV group once, then collectively applies it to the whole head.
        // This is deliberately above CUDA's portable 48 KiB default (about
        // 65 KiB), so opt in explicitly rather than silently launching an
        // invalid configuration and leaving dst uninitialised.
        CUDA_CHECK(cudaFuncSetAttribute(sol_attention_wmma_q64_full_d,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_bytes));
        sol_attention_wmma_q64_full_d<<<dim3((unsigned) q_blocks, (unsigned) heads), 256, smem_bytes, stream>>>(
            (const float *) q->data, (const half *) k->data, (const half *) v->data, kc.get(), vc.get(),
            route_lists.get(), route_counts.get(), route_bits.get(), route_words, (float *) dst->data, l, heads, q_blocks, kv_blocks, scale,
            q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, o_h_stride, o_t_stride);
        if (sparse_same_route_check && !force_exact) {
            ggml_cuda_pool_alloc<float> scalar_reference(ctx.pool());
            scalar_reference.alloc((size_t) kSolHeadDim * l * heads);
            sol_attention_warp8<<<dim3(query_tiles, (unsigned) heads), 256, 0, stream>>>(
                (const float *) q->data, (const half *) k->data, (const half *) v->data,
                kc.get(), vc.get(), thresholds.get(), route_lists.get(), route_counts.get(), route_bits.get(), route_words,
                true, scalar_reference.get(), l, heads, q_blocks, kv_blocks, scale,
                sink_blocks, sink_query_blocks, false, q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, o_h_stride, o_t_stride);
            sol_compare_same_route(ctx, (const float *) dst->data, scalar_reference.get(), l, heads, o_h_stride, o_t_stride);
        }
    } else if (dst->type == GGML_TYPE_F32) {
        sol_attention_warp8<<<dim3(query_tiles, (unsigned) heads), 256, 0, stream>>>(
            (const float *) q->data, (const half *) k->data, (const half *) v->data,
            kc.get(), vc.get(), thresholds.get(), route_lists.get(), route_counts.get(), route_bits.get(), route_words, block_routes, (float *) dst->data, l, heads, q_blocks, kv_blocks, scale,
            sink_blocks, sink_query_blocks, force_exact, q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, o_h_stride, o_t_stride);
    } else {
        sol_attention_warp8<<<dim3(query_tiles, (unsigned) heads), 256, 0, stream>>>(
            (const float *) q->data, (const half *) k->data, (const half *) v->data,
            kc.get(), vc.get(), thresholds.get(), route_lists.get(), route_counts.get(), route_bits.get(), route_words, block_routes, (half *) dst->data, l, heads, q_blocks, kv_blocks, scale,
            sink_blocks, sink_query_blocks, force_exact, q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, o_h_stride, o_t_stride);
    }
    if (validate_output) {
        ggml_cuda_pool_alloc<int> bad(ctx.pool());
        bad.alloc(1);
        CUDA_CHECK(cudaMemsetAsync(bad.get(), 0, sizeof(int), stream));
        const size_t out_count = (size_t) q->ne[0] * q->ne[1] * q->ne[2] * q->ne[3];
        if (dst->type == GGML_TYPE_F32) {
            sol_check_output<<<(out_count + 255)/256, 256, 0, stream>>>((const float *) dst->data, out_count, l, heads, o_h_stride, o_t_stride, bad.get());
        } else {
            sol_check_output<<<(out_count + 255)/256, 256, 0, stream>>>((const half *) dst->data, out_count, l, heads, o_h_stride, o_t_stride, bad.get());
        }
        int host_bad = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_bad, bad.get(), sizeof(host_bad), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (host_bad) GGML_ABORT("[sol] invalid output (non-finite or |x| > 1e4)");
    }
    CUDA_CHECK(cudaGetLastError());
    if (dispatch_count++ < 4) {
        fprintf(stderr, "[sol] dispatch H3 B=1 H=%d L=%d D=128 tau=%.3f sink_kv=%d sink_q=%d all_exact=%d block_routes=%d wmma=%d validate=%d\n",
                heads, l, tau, sink_blocks, sink_query_blocks, force_exact, block_routes, wmma_tiles, validate_output);
    }
    return true;
}

void ggml_cuda_flash_attn_ext_sol_compare(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor * dense_out,
                                          const void * sol_out) {
    if (dense_out->type != GGML_TYPE_F32 && dense_out->type != GGML_TYPE_F16) {
        GGML_ABORT("[sol] DENSE_CHECK requires an F32/F16 attention output");
    }
    const size_t count = ggml_nelements(dense_out);
    // dense output is BSHD [D,H,L,N], while the logical comparison loops Q's
    // sequence/head coordinates supplied by the native H3 contract.
    const int l = dense_out->ne[2], heads = dense_out->ne[1];
    const int64_t out_h_stride = dense_out->nb[1] / ggml_type_size(dense_out->type);
    const int64_t out_t_stride = dense_out->nb[2] / ggml_type_size(dense_out->type);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<unsigned int> stats(ctx.pool());
    stats.alloc(2);
    CUDA_CHECK(cudaMemsetAsync(stats.get(), 0, 2 * sizeof(unsigned int), stream));
    if (dense_out->type == GGML_TYPE_F32) {
        sol_compare_output<<<(count + 255)/256, 256, 0, stream>>>((const float *) sol_out,
            (const float *) dense_out->data, count, stats.get(), stats.get() + 1, l, heads, out_h_stride, out_t_stride);
    } else {
        sol_compare_output<<<(count + 255)/256, 256, 0, stream>>>((const half *) sol_out,
            (const half *) dense_out->data, count, stats.get(), stats.get() + 1, l, heads, out_h_stride, out_t_stride);
    }
    unsigned int host_bits[2] = {};
    CUDA_CHECK(cudaMemcpyAsync(host_bits, stats.get(), sizeof(host_bits), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float max_abs = 0.0f, max_ref = 0.0f;
    // CUDA's __uint_as_float is device-only in this toolkit; preserve the
    // positive IEEE-754 bit pattern on the host without invoking a device
    // intrinsic.
    memcpy(&max_abs, &host_bits[0], sizeof(max_abs));
    memcpy(&max_ref, &host_bits[1], sizeof(max_ref));
    fprintf(stderr, "[sol] DENSE_CHECK L=%lld H=%lld D=%lld max_abs=%.8g max_ref=%.8g rel_to_global_max=%.8g\n",
            (long long) dense_out->ne[2], (long long) dense_out->ne[1], (long long) dense_out->ne[0],
            max_abs, max_ref, max_ref > 0.0f ? max_abs / max_ref : max_abs);
}
