#include "reconstruction_cuda/cuda_rot_polar_estimator.h"
#include "reconstruction_cuda/cuda_shift_corr_estimator.h"
#include "reconstruction_cuda/cuda_bspline_geo_transformer.h"
#include "reconstruction_cuda/cuda_correlation_computer.h"

template<typename T>
class IterativeAlignmentEstimator_Test;

#define SETUPTESTCASE_SPECIFIC \
    shiftAligner = new Alignment::CudaShiftCorrEstimator<T>(); \
    rotationAligner = new Alignment::CudaRotPolarEstimator<T>(); \
    transformer = new CudaBSplineGeoTransformer<T>(); \
    meritComputer = new CudaCorrelationComputer<T>(); \
    for (int i = 0; i < 2; ++i) { \
        auto g = new GPU(); \
        g->set(); \
        hw.push_back(g); \
    }

#include "aiterative_alignment_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(GPU, IterativeAlignmentEstimator_Test, TestTypes);
