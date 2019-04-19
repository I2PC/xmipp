#include "reconstruction_cuda/gpu.h"
#include "reconstruction_cuda/cuda_fft.h"

template<typename T>
class AFFT_Transformer;

#define SETUP \
    void SetUp() { \
        printf("SetUp\n"); \
        ft = new CudaFFT<T>(); \
    }

#define SETUPTESTCASE \
    static void SetUpTestCase() { \
        printf("SetUpTestCase\n"); \
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

#include "afft_transformer_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cuda, AFFT_Transformer, TestTypes);
