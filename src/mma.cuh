#pragma once
#include "ptx.cuh"
#include "utils.cuh"

namespace mma {
/**
 * \brief C = A * B^T using wmma API
 * \note Launch 1 block with 32 threads only, and 16x16 each matrix, other
 * parameters settings will cause UB
 */
__global__ void mma16x16(half *c, half *a, half *b) {
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];

    int tx = threadIdx.x;
    ld_st_128bit(smem_a + 8 * tx, a + 8 * tx);
    ld_st_128bit(smem_b + 8 * tx, b + 8 * tx);

    __syncthreads();
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    load_matrix_sync(a_frag, smem_a, 16);
    load_matrix_sync(b_frag, smem_b, 16);

    fill_fragment(c_frag, 0.0f);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(smem_c, c_frag, 16, mem_row_major);

    // sync threads not necessary when only 1 warp, but we will generalize it in
    // the future, so just keep it here
    __syncthreads();
    ld_st_128bit(c + 8 * tx, smem_c + 8 * tx);
}

/**
 * \brief C = A * B^T using wmma API with PTX ISA mma instructions, this kernel
 * illustrates how to use PTX ISA mma wrappers.
 */
__global__ void mma16x16_ptx(half *c, half *a, half *b) {
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];

    int tx = threadIdx.x;
    ld_st_128bit(smem_a + 8 * tx, a + 8 * tx);
    ld_st_128bit(smem_b + 8 * tx, b + 8 * tx);
    __syncthreads();

    uint32_t row = tx % 16;
    uint32_t col = tx / 16;

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    fill_fragment(c_frag, 0.0f);
    // you can also manually set the register to 0 like:
    // for (int i = 0; i < 8; i++) {
    //     c_frag.x[i] = 0.0f;
    // }

    ptx::ldmatrix_sync(a_frag.x, smem_a + row * 16 + col * 8);
    ptx::ldmatrix_sync(b_frag.x, smem_b + row * 16 + col * 8);

    // swap R1 and R2 of B, this is required by B's layout, more info see PTX
    // ISA mma instruction
    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;
    // 2 m16n8k16 HMMA to achieve m16n16k16 matrix multiplication
    ptx::mma_sync_m16n8k16(c_frag.x, a_frag.x, b_frag.x);
    ptx::mma_sync_m16n8k16(c_frag.x + 4, a_frag.x, b_frag.x + 4);
    // store the result back to shared memory, this can be hand coded, but we
    // are interested in LDSM now
    store_matrix_sync(smem_c, c_frag, 16, mem_row_major);

    __syncthreads();
    ld_st_128bit(c + 8 * tx, smem_c + 8 * tx);
}

/**
 * \brief C = A * B^T with applying swizzle load
 */
__global__ void mma16x16_swizzle(half *c, half *a, half *b) {
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];
    // swizzle load A and B
    int tx = threadIdx.x;
    // each thread load 8 bytes, so tx * 8 is the offset
    int gRow = tx * 8 / 16;
    int gCol = tx * 8 % 16;
    // map gAddr -> sAddr
    int g2sRow = gRow; // row unchanged
    // [xxxx] [x] [xxx]
    // [row] [col] [8 fp16]
    // permute col using row's least 3rd bit
    int g2sCol = gCol ^ (((gRow >> 2) & 1) << 3);
    ld_st_128bit(smem_a + g2sRow * 16 + g2sCol, a + tx * 8);
    ld_st_128bit(smem_b + g2sRow * 16 + g2sCol, b + tx * 8);
    __syncthreads();

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    fill_fragment(c_frag, 0.0f);

    // swizzle load frag a and b
    int rRow = tx % 16;
    int rCol = (tx / 16) * 8;
    int r2sRow = rRow;
    int r2sCol = rCol ^ (((rRow >> 2) & 1) << 3);
    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sRow * 16 + r2sCol);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sRow * 16 + r2sCol);
    // swap R1 and R2 of B, this is required by B's layout, more info see PTX
    // ISA mma instruction
    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;
    // calc and store
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // store can also be swizzle, but we are interested in LDSM only
    store_matrix_sync(smem_c, c_frag, 16, mem_row_major);

    __syncthreads();
    ld_st_128bit(c + 8 * threadIdx.x, smem_c + 8 * threadIdx.x);
}

/**
 * \brief C = A * B^T with applying swizzle load
 * \note this kernel serves as a practice of understanding swizzle load, which 4
 * warps cooperate to load and compute separately
 */
__global__ void mma16x16_x4_swizzle(half *c, half *a, half *b) {
    __shared__ half smem_a[16 * 64];
    __shared__ half smem_b[16 * 64];
    __shared__ half smem_c[16 * 64];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * blockDim.x;
    // swizzle load A and B
    constexpr int stride = 64;
    int gRow = tid * 8 / stride;
    int gCol = tid * 8 % stride;

    int g2sRow = gRow;
    // [xxxx] [xxx] [xxx]
    // [16row] [8col] [8fp16]
    int g2sCol = gCol ^ ((gRow & 0x7) << 3);

    ld_st_128bit(smem_a + g2sRow * stride + g2sCol, a + tid * 8);
    ld_st_128bit(smem_b + g2sRow * stride + g2sCol, b + tid * 8);
    __syncthreads();

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    fill_fragment(c_frag, 0.0f);
    // swizzle load frag a and b
    int rRow = tx % 16;
    int rCol = (ty * 2 + tx / 16) * 8;
    int r2sRow = rRow;
    int r2sCol = rCol ^ ((rRow & 0x7) << 3);
    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sRow * stride + r2sCol);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sRow * stride + r2sCol);
    // swap R1 and R2 of B, this is required by B's layout, more info see PTX
    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;
    // calc and store
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(smem_c + 16 * ty, c_frag, 16 * 4, mem_row_major);
    __syncthreads();
    ld_st_128bit(c + 8 * tid, smem_c + 8 * tid);
}

/**
 * \brief 2 patterns are resolved in this kernel, one is a row 1x256 regarded as
 * 16x16, the other is block 16x16
 *
 * \note this kernel has serious LDSM bank
 * conflicts calculated as follows
 *
 * Pattern 1: 1x256 regarded as 16x16, each 1x256 has 4 bank conflicts, result
 * in 4x16(rows)x2(matrix A/B) = 128 bank conflicts
 *
 * Pattern 2: 16x16 block, each 16x16 has 7x4 bank conflicts, result in
 * 7x4x16(blocks)x2(matrix A/B) = 896 bank conflicts
 *
 * Total bank conflicts = 128 + 896 = 1024
 */
__global__ void mma_multi_pattern_simple(half *c, half *a, half *b) {
    __shared__ half smem_a[16 * 256];
    __shared__ half smem_b[16 * 256];
    __shared__ half smem_c[16 * 256];

    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y; // 0-15

    int tid = tx + ty * blockDim.x;
    // TODO: swizzle load A and B
    ld_st_128bit(smem_a + tid * 8, a + tid * 8);
    ld_st_128bit(smem_b + tid * 8, b + tid * 8);

    __syncthreads();

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    fill_fragment(c_frag, 0.0f);

    // 1x256 regarded as 16x16, compute C = A * B^T
    load_matrix_sync(a_frag, smem_a + 256 * ty, 16);
    load_matrix_sync(b_frag, smem_b + 256 * ty, 16);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(smem_c + 256 * ty, c_frag, 16, mem_row_major);

    __syncthreads();

    // 16x16 block, compute C = A * B^T
    load_matrix_sync(a_frag, smem_a + 16 * ty, 256);
    load_matrix_sync(b_frag, smem_b + 16 * ty, 256);

    fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(smem_c + 16 * ty, c_frag, 256, mem_row_major);

    __syncthreads();

    ld_st_128bit(c + tid * 8, smem_c + tid * 8);
}

__global__ void mma_multi_pattern_swizzle(half *c, half *a, half *b) {
    __shared__ half smem_a[16 * 256];
    __shared__ half smem_b[16 * 256];
    __shared__ half smem_c[16 * 256];

    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y; // 0-15
    int tid = tx + ty * blockDim.x;

    // TODO: swizzle load A and B
    // [xxxx]    [xxxxx]    [xxx]
    // [16rows]  [32cols]    [8fp16]
    // split cols into 4 groups:
    // [xxxx]    [xx] [xxx]     [xxx]

    int gAddr = tid * 8;
    // swizzle for 16x16 blocks
    int g2sAddr = (((gAddr >> 8) & 0x7) << 3) ^ gAddr;
    // swizzle for 1x256 blocks
    g2sAddr ^= ((g2sAddr >> 6) & 0x3) << 3;
    ld_st_128bit(smem_a + g2sAddr, a + gAddr);
    ld_st_128bit(smem_b + g2sAddr, b + gAddr);

    __syncthreads();

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;
    // TODO: 1x256 regarded as 16x16
    int rAddr = ty * 256 + (tx % 16) * 16 + (tx / 16 * 8);
    int r2sAddr = (((rAddr >> 8) & 0x7) << 3) ^ rAddr;
    r2sAddr ^= ((r2sAddr >> 6) & 0x3) << 3;

    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sAddr);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sAddr);

    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;

    fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(smem_c + 256 * ty, c_frag, 16, mem_row_major);
    __syncthreads();
    // TODO: 16x16 block
    rAddr = ty * 16 + (tx % 16) * 256 + (tx / 16 * 8);
    r2sAddr = (((rAddr >> 8) & 0x7) << 3) ^ rAddr;
    r2sAddr ^= ((r2sAddr >> 6) & 0x3) << 3;
    // r2sAddr ^= ((rAddr >> 6) & 0x3) << 3;
    // r2sAddr = (((r2sAddr >> 9) & 0x7) << 3) ^ r2sAddr;

    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sAddr);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sAddr);

    tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;

    fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(smem_c + 16 * ty, c_frag, 256, mem_row_major);
    __syncthreads();
    ld_st_128bit(c + tid * 8, smem_c + tid * 8);
}

} // namespace mma
