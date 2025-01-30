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
} // namespace mma
