#include <gtest/gtest.h>
#include "reconstruction_cuda/cuda_shift_aligner.h"

template<typename T>
class CudaShiftAlignerTest : public ::testing::Test { };
TYPED_TEST_CASE_P(CudaShiftAlignerTest);

template<typename T>
void correlate2D(FFTSettingsNew<T> &dims) {
    using std::complex;
    using Alignment::CudaShiftAligner;

    // allocate and prepare data
    auto inOut = new complex<T>[dims.fDim().size()];
    auto ref = new complex<T>[dims.fDim().xy()];
    for (int n = 0; n < dims.fDim().n(); ++n) {
        for (int y = 0; y < dims.fDim().y(); ++y) {
            for (int x = 0; x < dims.fDim().x(); ++x) {
                int index = (n * dims.fDim().xy()) + (y * dims.fDim().x()) + x;
                inOut[index] = complex<T>(x + n, y + n);
                if (0 == n) { // ref is a single image
                    ref[index] = inOut[index];
                }
            }
        }
    }

    CudaShiftAligner<T>::computeCorrelations2DOneToN(inOut, ref, dims, true);

    T delta = 0.0001;
    for (int n = 0; n < dims.fDim().n(); ++n) {
        for (int y = 0; y < dims.fDim().y(); ++y) {
            for (int x = 0; x < dims.fDim().x(); ++x) {
                int index = (n * dims.fDim().xy()) + (y * dims.fDim().x()) + x;
                auto expected = complex<T>(x, y)
                        * conj(complex<T>(x + n, y + n));
                auto actual = inOut[index];
                EXPECT_NEAR(expected.real(), actual.real(), delta);
                EXPECT_NEAR(expected.imag(), actual.imag(), delta);
            }
        }
    }

    delete[] inOut;
    delete[] ref;
}


TYPED_TEST_P( CudaShiftAlignerTest, correlate2DOneToOne)
{
    // test one reference vs one image
    FFTSettingsNew<TypeParam> dims(29, 13);
}


TYPED_TEST_P( CudaShiftAlignerTest, correlate2DOneToMany)
{
    // check that n == batch works properly
    FFTSettingsNew<TypeParam> dims(29, 13, 1, 5, 5);
}

TYPED_TEST_P( CudaShiftAlignerTest, correlate2DOneToManyBatched1)
{
    // test that n mod batch != 0 works
    FFTSettingsNew<TypeParam> dims(29, 13, 1, 5, 3);
    correlate2D(dims);
}

TYPED_TEST_P( CudaShiftAlignerTest, correlate2DOneToManyBatched2)
{
    // test that n mod batch = 0 works
    FFTSettingsNew<TypeParam> dims(29, 13, 1, 6, 3);
    correlate2D(dims);
}

REGISTER_TYPED_TEST_CASE_P(CudaShiftAlignerTest,
    correlate2DOneToOne,
    correlate2DOneToMany,
    correlate2DOneToManyBatched1,
    correlate2DOneToManyBatched2
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(SomeRandomText, CudaShiftAlignerTest, TestTypes);
