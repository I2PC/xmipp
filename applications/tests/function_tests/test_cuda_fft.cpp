#include "reconstruction_cuda/gpu.h"
#include "reconstruction_cuda/cuda_fft.h"

template<typename T>
class AFT_Test;

#define SETUP \
    void SetUp() { \
        ft = new CudaFFT<T>(); \
    }

#define SETUPTESTCASE \
    static void SetUpTestCase() { \
        hw = new GPU(); \
        hw->set(); \
    }

#define TEST_VALUES \
    auto batch = std::vector<size_t>{1, 2, 3, 5, 6, 7, 8, 10, 23}; \
    auto nSet = std::vector<size_t>{1, 2, 4, 5, 6, 8, 10, 12, 14, 23, 24}; \
    auto zSet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 2048, 2049}; \
    auto ySet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 2048, 2049}; \
    auto xSet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 2048, 2049};

#define EXECUTIONS 20

#include "aft_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cuda, AFT_Test, TestTypes);
