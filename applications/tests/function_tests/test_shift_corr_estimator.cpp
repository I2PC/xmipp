#include "data/cpu.h"
#include "reconstruction/shift_corr_estimator.h"

template<typename T>
class AShiftCorrEstimator_Test;

#define SETUP \
    void SetUp() { \
        estimator = new Alignment::ShiftCorrEstimator<T>(); \
    }

#define SETUPTESTCASE \
    static void SetUpTestCase() { \
        hw = new CPU(); \
        hw->set(); \
    }

#include "ashift_corr_estimator_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cpu, AShiftCorrEstimator_Test, TestTypes);
