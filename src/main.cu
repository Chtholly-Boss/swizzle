#include "mma.cuh"
#include "test.cuh"

int main(int argc, char **argv) {
    CHECK_TEST(test_mma_16x16(mma::mma16x16));
    CHECK_TEST(test_mma_16x16(mma::mma16x16_ptx));
    CHECK_TEST(test_mma_16x16(mma::mma16x16_swizzle));
}
