#include "concat.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// contiguous kernels — value-copy only, dtype is just a width.
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_T_cont(const T * x,
                                                                                const T * y,
                                                                                T *       dst,
                                                                                int64_t   ne00,
                                                                                int64_t   ne01,
                                                                                int64_t   ne02,
                                                                                int64_t   ne0,
                                                                                int64_t   ne1,
                                                                                int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

// Single-launch contiguous concat over the FULL 4D extent. Folds the outer ne3
// loop into the kernel so one ggml CONCAT op == one kernel launch (was ne3 launches
// — pathological for high-channel-count VAE/conv graphs: a 1024-channel concat used
// to fire 1024 tiny kernels). Bit-identical to the per-slice path; only the launch
// count changes. The non-concat dims of src0/src1 match dst, so each i3 slice is a
// fixed src0_slice / src1_slice element block.
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_T_cont_4d(const T * x,
                                                                                   const T * y,
                                                                                   T *       dst,
                                                                                   int64_t   ne00,
                                                                                   int64_t   ne01,
                                                                                   int64_t   ne02,
                                                                                   int64_t   ne0,
                                                                                   int64_t   ne1,
                                                                                   int64_t   ne2,
                                                                                   int64_t   ne3) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t src0_slice = ne00 * ne01 * ne02;  // src0 elems per i3 (non-dim dims == dst)
    const int64_t dst_slice  = ne0 * ne1 * ne2;
    const int64_t src1_slice = dst_slice - src0_slice;
    const int64_t n          = dst_slice * ne3;

    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t i3 = i / dst_slice;
        const int64_t j  = i - i3 * dst_slice;
        const T *     xb = x + i3 * src0_slice;
        const T *     yb = y + i3 * src1_slice;

        if constexpr (dim == 0) {
            const int64_t row = j / ne0;
            const int64_t i0  = j - row * ne0;
            dst[i] = (i0 < ne00) ? xb[row * ne00 + i0] : yb[row * (ne0 - ne00) + (i0 - ne00)];
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = j / dst_plane;
            const int64_t i01        = j - i2 * dst_plane;
            dst[i] = (i01 < src0_plane) ? xb[i2 * src0_plane + i01] : yb[i2 * src1_plane + (i01 - src0_plane)];
        } else {
            dst[i] = (j < src0_slice) ? xb[j] : yb[j - src0_slice];
        }
    }
}

struct concat_profile_key {
    int     type;
    int32_t dim;
    bool    contiguous;
    int64_t src0_ne[4];
    int64_t src1_ne[4];
    int64_t dst_ne[4];

    bool operator==(const concat_profile_key & other) const {
        if (type != other.type || dim != other.dim || contiguous != other.contiguous) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (src0_ne[i] != other.src0_ne[i] || src1_ne[i] != other.src1_ne[i] || dst_ne[i] != other.dst_ne[i]) {
                return false;
            }
        }
        return true;
    }
};

struct concat_profile_key_hash {
    size_t operator()(const concat_profile_key & k) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&](int64_t v) {
            h ^= (uint64_t) v;
            h *= 1099511628211ull;
        };
        mix(k.type);
        mix(k.dim);
        mix(k.contiguous ? 1 : 0);
        for (int i = 0; i < 4; ++i) {
            mix(k.src0_ne[i]);
            mix(k.src1_ne[i]);
            mix(k.dst_ne[i]);
        }
        return h;
    }
};

struct concat_profile_stats {
    uint64_t calls     = 0;
    uint64_t dst_bytes = 0;
};

static std::unordered_map<concat_profile_key, concat_profile_stats, concat_profile_key_hash> & concat_profile_map() {
    static auto * map = new std::unordered_map<concat_profile_key, concat_profile_stats, concat_profile_key_hash>();
    return *map;
}

static std::mutex & concat_profile_mutex() {
    static auto * mutex = new std::mutex();
    return *mutex;
}

static bool concat_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void concat_profile_dump() {
    std::vector<std::pair<concat_profile_key, concat_profile_stats>> rows;
    uint64_t total_calls = 0;
    uint64_t total_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(concat_profile_mutex());
        rows.reserve(concat_profile_map().size());
        for (const auto & item : concat_profile_map()) {
            rows.push_back(item);
            total_calls += item.second.calls;
            total_bytes += item.second.dst_bytes;
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        if (a.second.dst_bytes != b.second.dst_bytes) {
            return a.second.dst_bytes > b.second.dst_bytes;
        }
        return a.second.calls > b.second.calls;
    });

    int top = 48;
    if (const char * env = getenv("LONGCAT_CONCAT_PROFILE_TOP")) {
        const int parsed = atoi(env);
        if (parsed > 0) {
            top = parsed;
        }
    }

    fprintf(stderr,
            "[concat-profile] keys=%zu calls=%llu dst_mb=%.2f\n",
            rows.size(),
            (unsigned long long) total_calls,
            (double) total_bytes / (1024.0 * 1024.0));
    for (int i = 0; i < (int) rows.size() && i < top; ++i) {
        const auto & k = rows[i].first;
        const auto & s = rows[i].second;
        fprintf(stderr,
                "[concat-profile] #%02d calls=%llu dst_mb=%.2f avg_kb=%.1f type=%s dim=%d contig=%d "
                "src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                i + 1,
                (unsigned long long) s.calls,
                (double) s.dst_bytes / (1024.0 * 1024.0),
                s.calls == 0 ? 0.0 : (double) s.dst_bytes / (double) s.calls / 1024.0,
                ggml_type_name((ggml_type) k.type),
                k.dim,
                k.contiguous ? 1 : 0,
                (long long) k.src0_ne[0],
                (long long) k.src0_ne[1],
                (long long) k.src0_ne[2],
                (long long) k.src0_ne[3],
                (long long) k.src1_ne[0],
                (long long) k.src1_ne[1],
                (long long) k.src1_ne[2],
                (long long) k.src1_ne[3],
                (long long) k.dst_ne[0],
                (long long) k.dst_ne[1],
                (long long) k.dst_ne[2],
                (long long) k.dst_ne[3]);
    }
}

static bool concat_profile_enabled() {
    static bool enabled = [] {
        const bool on = concat_env_enabled("LONGCAT_CONCAT_PROFILE");
        if (on) {
            atexit(concat_profile_dump);
        }
        return on;
    }();
    return enabled;
}

static void concat_profile_note(const ggml_tensor * dst, int32_t dim, bool contiguous) {
    if (!concat_profile_enabled()) {
        return;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    concat_profile_key  key  = {};
    key.type                 = src0->type;
    key.dim                  = dim;
    key.contiguous           = contiguous;
    for (int i = 0; i < 4; ++i) {
        key.src0_ne[i] = src0->ne[i];
        key.src1_ne[i] = src1->ne[i];
        key.dst_ne[i]  = dst->ne[i];
    }

    std::lock_guard<std::mutex> lock(concat_profile_mutex());
    concat_profile_stats & stats = concat_profile_map()[key];
    stats.calls += 1;
    stats.dst_bytes += ggml_nbytes(dst);
}

template <typename T>
static void concat_T_cuda(const T *    x,
                          const T *    y,
                          T *          dst,
                          int64_t      ne00,
                          int64_t      ne01,
                          int64_t      ne02,
                          int64_t      ne0,
                          int64_t      ne1,
                          int64_t      ne2,
                          int64_t      ne3,
                          int          dim,
                          cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2 * ne3;
    const int64_t blk        = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    const int     num_blocks = (int) std::min<int64_t>(blk, 65535);

    if (dim == 0) {
        concat_T_cont_4d<T, 0>
            <<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
        return;
    }
    if (dim == 1) {
        concat_T_cont_4d<T, 1>
            <<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
        return;
    }
    concat_T_cont_4d<T, 2>
        <<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
}

// non-contiguous kernel (slow) — value-copy only, dtype is just a width.
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_T_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const T *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const T *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const T *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


template <typename T>
static void concat_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();
    const int32_t dim = ((int32_t *) dst->op_params)[0];
    const bool contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);

    concat_profile_note(dst, dim, contiguous);

    if (contiguous) {
        const T * src0_d = (const T *)src0->data;
        const T * src1_d = (const T *)src1->data;
        T * dst_d = (T *)dst->data;

        if (dim != 3) {
            // single launch over the full 4D extent (folds the ne3 loop into the kernel)
            concat_T_cuda<T>(
                    src0_d, src1_d, dst_d,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    dst->ne[0],  dst->ne[1],  dst->ne[2], dst->ne[3], dim, stream);
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d,                       src0_d, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst_d + size0 / sizeof(T),   src1_d, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_T_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);

    switch (src0->type) {
        case GGML_TYPE_F32:
            concat_dispatch<float>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            concat_dispatch<half>(ctx, dst);
            break;
        case GGML_TYPE_I32:
            concat_dispatch<int32_t>(ctx, dst);
            break;
        default:
            GGML_ABORT("ggml_cuda_op_concat: unsupported dtype %s", ggml_type_name(src0->type));
    }
}
