#include "data/cpu.h"
#include "reconstruction/fftwT.h"

template<typename T>
class AFT_Test;

#define SETUP \
    void SetUp() { \
        ft = new FFTwT<T>(); \
    }

#define SETUPTESTCASE \
    static void SetUpTestCase() { \
        hw = new CPU(CPU::findCores()); \
        hw->set(); \
    }

#define TEST_VALUES \
    auto batch = std::vector<size_t>{1, 2, 3, 5, 6, 7, 8, 10, 14}; \
    auto nSet = std::vector<size_t>{1, 2, 4, 5, 6, 8, 10, 12, 14}; \
    auto zSet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 512, 513}; \
    auto ySet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 512, 513}; \
    auto xSet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 512, 513};

#define EXECUTIONS 10

#include "aft_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cpu, AFT_Test, TestTypes);
