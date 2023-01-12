#include "reconstruction_cuda/gpu.h"
#include "reconstruction_cuda/cuda_fft.h"

template<typename T>
class AFT_Test;

int CUDA_VERSION = 0;

#define SETUP \
    void SetUp() { \
        ft = new CudaFFT<T>(); \
    }

#define SETUPTESTCASE \
    static void SetUpTestCase() { \
        hw = new GPU(); \
        hw->set(); \
        CUDA_VERSION = dynamic_cast<GPU*>(hw)->getCudaVersion(); \
    }

#define MUSTBESKIPPED \
    bool mustBeSkipped(const FFTSettings<T> &s, bool isBothDirection) { \
        const auto &d = s.sDim(); \
        if (10020 == CUDA_VERSION) { \
            if (std::is_same<T, float>::value) { \
                if ((2049 == d.x() && 1 == d.y() && 1 == d.z() && 6 == d.n() && 5 == s.batch() && s.isInPlace()) \
                    || (!isBothDirection && 2049 == d.x() && 1 == d.y() && 1 == d.z() && 6 == d.n() && 5 == s.batch() && s.isInPlace() && s.isForward()) \
                    || (!isBothDirection && 2049 == d.x() && 106 == d.y() && 2 == d.z() && 24 == d.n() && 23 == s.batch() && s.isInPlace() && s.isForward()) \
                    || (!isBothDirection && 15 == d.x() && 15 == d.y() && 2048 == d.z() && 12 == d.n() && 7 == s.batch() && s.isInPlace() && s.isForward()) \
                    || (isBothDirection && 2 == d.x() && 1 == d.y() && 1 == d.z() && 23 == d.n() && 2 == s.batch() && s.isInPlace()) \
                    || (isBothDirection && 2 == d.x() && 1 == d.y() && 1 == d.z() && 5 == d.n() && 2 == s.batch() && s.isInPlace()) \
                    || (isBothDirection && 3 == d.x() && 2048 == d.y() && 1 == d.z() && 24 == d.n() && 23 == s.batch() && s.isInPlace()) \
                    || (isBothDirection && 2049 == d.x() && 2048 == d.y() && 1 == d.z() && 24 == d.n() && 23 == s.batch() && s.isInPlace()) \
                    || (isBothDirection && 3 == d.x() && 15 == d.y() && 2049 == d.z() && 10 == d.n() && 7 == s.batch() && s.isInPlace()) \
                    ) { \
                    return true; \
                } \
            } else if (std::is_same<T, double>::value) { \
                if ((isBothDirection && 2 == d.x() && 1 == d.y() && 1 == d.z() && 23 == d.n() && 2 == s.batch() && s.isInPlace()) \
                    || (isBothDirection && 2 == d.x() && 1 == d.y() && 1 == d.z() && 5 == d.n() && 2 == s.batch() && s.isInPlace())) { \
                    return true; \
                } \
            } \
        } \
        return false; \
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
INSTANTIATE_TYPED_TEST_SUITE_P(Cuda, AFT_Test, TestTypes);
