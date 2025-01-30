#pragma once
#include <mma.h>
#include "utils.cuh"

#define CHECK_TEST(test)                                                       \
    if (!test) {                                                               \
        std::cout << "\033[1;31m" << #test << " failed\033[0m\n";              \
        return 1;                                                              \
    } else {                                                                   \
        std::cout << "\033[1;32m" << #test << " passed\033[0m\n";              \
    }

bool test_mma_16x16(void (*kernel)(half *, half *, half *)) {
    const int n = 16 * 16;
    half *a_h = (half *)malloc(n * sizeof(half));
    half *b_h = (half *)malloc(n * sizeof(half));
    half *c_h = (half *)malloc(n * sizeof(half));

    fill_data(a_h, n);
    fill_data(b_h, n);

    half *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, n * sizeof(half));
    cudaMalloc(&b_d, n * sizeof(half));
    cudaMalloc(&c_d, n * sizeof(half));

    cudaMemcpy(a_d, a_h, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n * sizeof(half), cudaMemcpyHostToDevice);

    kernel<<<1, 32>>>(c_d, a_d, b_d);

    cudaMemcpy(c_h, c_d, n * sizeof(half), cudaMemcpyDeviceToHost);

    half *c_ref = (half *)malloc(n * sizeof(half));
    // C = A * B^T
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            half sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                sum += a_h[i * 16 + k] * b_h[j * 16 + k];
            }
            c_ref[i * 16 + j] = sum;
        }
    }

    bool ret = !diff(c_h, c_ref, n);
    free(a_h);
    free(b_h);
    free(c_h);
    free(c_ref);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return ret;
}