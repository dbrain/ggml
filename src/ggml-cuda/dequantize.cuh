#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// Per-element dequant for Q4_K, matching the dequantize_kernel_t signature so
// it can plug into k_get_rows. Decodes 2 elements (positions iqs and
// iqs + qk/2 = iqs + 128) of super-block `ib`.
//
// Q4_K layout: super-block of QK_K=256 elements, organized as 8 sub-blocks of
// 32 elements each. Each sub-block has a 6-bit quantized scale (d) and a 6-bit
// quantized min (m), packed into the 12-byte `scales` array via the
// get_scale_min_k4 packing scheme. Per element, value = (dall*d)*nibble - dmin*m.
//
// For element position p ∈ [0, 256):
//   sub-block index s = p / 32       ∈ [0, 8)
//   nibble parity     = s & 1        (0 = low nibble, 1 = high nibble)
//   qs byte index     = (p/64)*32 + (p%32)
//
// p0 = iqs and p1 = iqs + 128 share nibble parity by construction:
//   s0 = iqs/32 ∈ [0, 4) ; s1 = s0 + 4 ∈ [4, 8) ; (s0 & 1) == (s1 & 1)
//   q_idx0 ∈ [0, 64) ; q_idx1 = q_idx0 + 64 ∈ [64, 128)
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_K * x = (const block_q4_K *) vx;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const int s0     = iqs >> 5;                                // iqs / 32
    const int s1     = s0 + 4;
    const int q_idx0 = ((iqs >> 6) << 5) | (iqs & 31);          // (iqs/64)*32 + (iqs%32)
    const int q_idx1 = q_idx0 + 64;
    const int hi     = s0 & 1;

    const uint8_t qb0 = x[ib].qs[q_idx0];
    const uint8_t qb1 = x[ib].qs[q_idx1];
    const int n0 = hi ? (qb0 >> 4) : (qb0 & 0xF);
    const int n1 = hi ? (qb1 >> 4) : (qb1 & 0xF);

    // get_scale_min_k4 inlined for the {s0 < 4, s1 ≥ 4} access pattern.
    const uint8_t * sc = x[ib].scales;
    const uint8_t sd0 = sc[s0]     & 63;
    const uint8_t sm0 = sc[s0 + 4] & 63;
    const uint8_t sd1 = (uint8_t)((sc[s1 + 4] & 0xF) | ((sc[s1 - 4] >> 6) << 4));
    const uint8_t sm1 = (uint8_t)((sc[s1 + 4] >>  4) | ((sc[s1    ] >> 6) << 4));

    v.x = (dall * (float)sd0) * (float)n0 - (dmin * (float)sm0);
    v.y = (dall * (float)sd1) * (float)n1 - (dmin * (float)sm1);
}
