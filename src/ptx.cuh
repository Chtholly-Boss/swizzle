#pragma once
#include <cstdint>
#include <mma.h>

#define REG(val) (*reinterpret_cast<uint32_t *>(&(val)))

namespace ptx {
using fp16 = half;

__device__ __forceinline__ void ldmatrix_sync(fp16 *dst, void *addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(REG(dst[0])),
          "=r"(REG(dst[2])),
          "=r"(REG(dst[4])),
          "=r"(REG(dst[6]))
        : "l"(__cvta_generic_to_shared(addr)));
}

__device__ __forceinline__ void ldmatrix_trans_sync(fp16 *dst, void *addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(REG(dst[0])),
          "=r"(REG(dst[2])),
          "=r"(REG(dst[4])),
          "=r"(REG(dst[6]))
        : "l"(__cvta_generic_to_shared(addr)));
}

// C = A * B^T
__device__ __forceinline__ void mma_sync_m16n8k16(fp16 *c, fp16 *a, fp16 *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3, %4, %5}, "
                 "{%6, %7}, "
                 "{%8, %9};"
                 : "=r"(REG(c[0])), "=r"(REG(c[2]))
                 : "r"(REG(a[0])),
                   "r"(REG(a[2])),
                   "r"(REG(a[4])),
                   "r"(REG(a[6])),
                   "r"(REG(b[0])),
                   "r"(REG(b[2])),
                   "r"(0),
                   "r"(0));
}
} // namespace ptx