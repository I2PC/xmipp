#include "reconstruction_cuda/cuda_rot_polar_estimator.h"
#include <gtest/gtest.h>
#include <numeric>
#include <random>

#pragma GCC push_options
#pragma GCC optimize ("O0")

template<typename T>
class ARotationEstimator_Test : public ::testing::Test {
public:
    void rotate2D()
       {
           using Alignment::AlignType;
           auto dims = Dimensions(7, 7, 1, 2);
           size_t batch = dims.n();

           auto estimator = Alignment::CudaRotPolarEstimator<T>();
           auto gpu = GPU();
           gpu.set();
           estimator.init(gpu, AlignType::OneToN, dims, batch, 360);

           auto others = new T[dims.size()]();
           auto mt = std::mt19937(42);
           auto dist = std::uniform_real_distribution<>(-1, 1.f);
           for (size_t i = 0; i < dims.size(); ++i) {
               others[i] = dist(mt);
           }
           estimator.compute(others);
           auto result = estimator.getRotations2D();

           delete[] others;
       }
};
TYPED_TEST_CASE_P(ARotationEstimator_Test);


TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToOne)
{
    XMIPP_TRY
    // test one reference vs one image
    ARotationEstimator_Test<TypeParam>::rotate2D();
    XMIPP_CATCH
}


REGISTER_TYPED_TEST_CASE_P(ARotationEstimator_Test,
    rotate2DOneToOne
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cpu, ARotationEstimator_Test, TestTypes);

#pragma GCC pop_options
