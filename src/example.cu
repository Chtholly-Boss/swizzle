#include "swizzle.cuh"

__global__ void example1(float *gmem) {
    float smem[32 * 32];
    // 32 threads load colomn 0-3
    uint32_t tx = threadIdx.x;
    uint32_t gAddr = tx * 32;
    uint32_t sAddr = swizzle<5, 5, 0>(gAddr);
    *reinterpret_cast<float4 *>(smem + sAddr) =
        *reinterpret_cast<float4 *>(gmem + sAddr);
    // ... load other columns
    // each thread read colomn 0
    auto sum = smem[swizzle<5, 5, 0>(tx * 32)];
    // each thread read column 1
    sum += smem[swizzle<5, 5, 0>(tx * 32 + 1)];
    // store sum to shared memory
    smem[swizzle<5, 5, 0>(tx * 32)] = sum;
    // ...
}