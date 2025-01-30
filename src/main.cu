#include "mma.cuh"
#include "test.cuh"

int main(int argc, char **argv) {
    CHECK_TEST(test_mma_16x16(mma::simple));
    CHECK_TEST(test_mma_16x16(mma::simple_ptx));
}
