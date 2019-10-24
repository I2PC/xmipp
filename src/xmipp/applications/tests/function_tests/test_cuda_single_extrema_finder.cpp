#include "reconstruction_cuda/cuda_single_extrema_finder.h"

template<typename T>
class SingleExtremaFinder_Test;

#define SETUPTESTCASE_SPECIFIC \
    finder = new ExtremaFinder::CudaExtremaFinder<T>(); \
    for (int i = 0; i < 2; ++i) { \
        auto g = new GPU(); \
        g->set(); \
        hw.push_back(g); \
    }

#include "asingle_extrema_finder_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cuda, SingleExtremaFinder_Test, TestTypes);
