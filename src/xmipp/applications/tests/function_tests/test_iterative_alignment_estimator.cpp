#include "reconstruction/polar_rotation_estimator.h"
#include "reconstruction/shift_corr_estimator.h"
#include "reconstruction/bspline_geo_transformer.h"
#include "reconstruction/correlation_computer.h"

template<typename T>
class IterativeAlignmentEstimator_Test;

#define SETUPTESTCASE_SPECIFIC \
    shiftAligner = new Alignment::ShiftCorrEstimator<T>(); \
    rotationAligner = new Alignment::PolarRotationEstimator<T>(); \
    transformer = new BSplineGeoTransformer<T>(); \
    meritComputer = new CorrelationComputer<T>(); \
    hw.emplace_back(new CPU()); \

#include "aiterative_alignment_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(CPU, IterativeAlignmentEstimator_Test, TestTypes);
