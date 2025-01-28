#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mma.h>

// #define DEBUG

void matmul_m16n16k16(half *c, half *a, half *b, int stride) {
    // C = A * B^T
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            half sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                sum += a[i * stride + k] * b[j * stride + k];
            }
            c[i * stride + j] = sum;
        }
    }
}

bool diff(half *a, half *b, int n) {
    for (int i = 0; i < n; i++) {
        half error = a[i] - b[i];
        if (__half2float(error) > 1e-2) {
            printf("\033[0;31mDifference found at index %d: a[%d] = %f, b[%d] "
                   "= %f\033[0m\n",
                   i,
                   i,
                   __half2float(a[i]),
                   i,
                   __half2float(b[i]));
            return true;
        }
    }
    return false;
}

void fill_input(half *a, half *b, int n) {
#ifdef DEBUG
    for (int i = 0; i < n; i++) {
        a[i] = float(i);
        b[i] = float(n - i - 1);
    }
#else
    srand(time(0));
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
#endif
}

void print_data(half *data, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%.2f ", (float)(data[i * c + j]));
        }
        printf("\n");
    }
    printf("\n");
}

__device__ __forceinline__ void ld_st_128bit(void *dst, void *src) {
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
}

#define HALF2(val) (*reinterpret_cast<half2 *>(&(val)))
#define REG(val) (*reinterpret_cast<uint32_t *>(&(val)))
