#include "reconstruction_cuda/gpu.h"
#include "reconstruction_cuda/cuda_rot_polar_estimator.h"

template<typename T>
class ARotationEstimator_Test;

#define SETUP \
    void SetUp() { \
        estimator = new Alignment::CudaRotPolarEstimator<T>(); \
    }

#define SETUPTESTCASE \
    static void SetUpTestCase() { \
        hw = new GPU(); \
        hw->set(); \
    }

#define INIT \
    estimator->init(*hw, AlignType::OneToN, dims, batch, maxRotation);

#include "arotation_estimator_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Gpu, ARotationEstimator_Test, TestTypes);
