#include "utils.cuh"

__global__ void mma_simple(half *a, half *b, half *c) {
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

    ld_st_128bit(c + 8 * tx, smem_c + 8 * tx);
}

__global__ void mma_swizzle_16x64(half *a, half *b, half *c) {
    // we will launch (1,1,1) x (32,4,1) threads
    // We will calc 4 16x16 matrix multiplication on 16x64 A,B and C
    __shared__ half smem_a[16 * 64];
    __shared__ half smem_b[16 * 64];
    __shared__ half smem_c[16 * 64];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tIdx = tx + ty * blockDim.x;
    int gAddr = tIdx * 8;
    int gRow = gAddr / 64;
    int gCol = gAddr % 64;
    int sCol = (gCol / 8) ^ (gRow & 0x7);
    int sAddr = gRow * 64 + sCol * 8;
    // here we only do swizzle on A
    ld_st_128bit(smem_a + sAddr, a + gAddr);

    // ld_st_128bit(smem_a + gAddr, a + gAddr);
    ld_st_128bit(smem_b + gAddr, b + gAddr);
    __syncthreads();
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    int r_ = threadIdx.x % 16;
    int c_ = (r_ & 0x7) ^ (2 * threadIdx.y + threadIdx.x / 16);

    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(REG(a_frag.x[0])),
          "=r"(REG(a_frag.x[2])),
          "=r"(REG(a_frag.x[4])),
          "=r"(REG(a_frag.x[6]))
        : "l"(__cvta_generic_to_shared(smem_a + r_ * 64 + c_ * 8)));

    // load_matrix_sync(a_frag, smem_a + 16 * ty, 64);
    load_matrix_sync(b_frag, smem_b + 16 * ty, 64);

    fill_fragment(c_frag, 0.0f);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(smem_c + 16 * ty, c_frag, 64, mem_row_major);

    __syncthreads();
    ld_st_128bit(c + gAddr, smem_c + gAddr);
}

void mmaDriver(half *a, half *b, half *c, int n) {
    half *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, n * sizeof(half));
    cudaMalloc(&b_d, n * sizeof(half));
    cudaMalloc(&c_d, n * sizeof(half));

    cudaMemcpy(a_d, a, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n * sizeof(half), cudaMemcpyHostToDevice);

    // mma_simple<<<1, 32>>>(a_d, b_d, c_d);
    mma_swizzle_16x64<<<1, dim3(32, 4)>>>(a_d, b_d, c_d);
    cudaDeviceSynchronize();

    cudaMemcpy(c, c_d, n * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void matmulx4(half *c, half *a, half *b) {
    for (int i = 0; i < 4; i++) {
        matmul_m16n16k16(c + 16 * i, a + 16 * i, b + 16 * i, 16 * 4);
    }
}

void test_mma() {
    // int n = 16 * 16;
    int n = 16 * 64;

    half *a_h = new half[n];
    half *b_h = new half[n];

    fill_input(a_h, b_h, n);

    half *c_ref = new half[n];
    // matmul_m16n16k16(c_ref, a_h, b_h, n / 16);
    matmulx4(c_ref, a_h, b_h);

    half *c_h = new half[n];
    mmaDriver(a_h, b_h, c_h, n);

    if (!diff(c_ref, c_h, n)) {
        printf("\033[0;32m All Points Correct \033[0m\n");
    }

    delete[] a_h;
    delete[] b_h;
    delete[] c_ref;
    delete[] c_h;
}

int main() {
    test_mma();
    return 0;
}