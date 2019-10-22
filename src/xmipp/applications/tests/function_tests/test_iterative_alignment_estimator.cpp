#include "reconstruction/polar_rotation_estimator.h"
#include "reconstruction/shift_corr_estimator.h"


template<typename T>
class IterativeAlignmentEstimator_Test;

#define SETUPTESTCASE_SPECIFIC \
    shiftAligner = new Alignment::ShiftCorrEstimator<T>(); \
    rotationAligner = new Alignment::PolarRotationEstimator<T>(); \
    hw.emplace_back(new CPU()); \

#include "aiterative_alignment_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(CPU, IterativeAlignmentEstimator_Test, TestTypes);
