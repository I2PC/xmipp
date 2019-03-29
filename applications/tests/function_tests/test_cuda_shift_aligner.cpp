#include <gtest/gtest.h>
#include "reconstruction_cuda/cuda_shift_aligner.h"

template<typename T>
class CudaShiftAlignerTest : public ::testing::Test { };
TYPED_TEST_CASE_P(CudaShiftAlignerTest);

template<typename T>
void correlate2D(const FFTSettingsNew<T> &dims) {
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

    CudaShiftAligner<T>::computeCorrelations2DOneToN(inOut, ref, dims);

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

template<typename T>
void correlate2D(size_t n, size_t batch) {
    correlate2D<T>(FFTSettingsNew<T>(29, 13, 1, n, batch)); // odd, odd
    correlate2D<T>(FFTSettingsNew<T>(29, 14, 1, n, batch)); // odd, even
    correlate2D<T>(FFTSettingsNew<T>(30, 13, 1, n, batch)); // even, odd
    correlate2D<T>(FFTSettingsNew<T>(30, 14, 1, n, batch)); // even, even
}


TYPED_TEST_P( CudaShiftAlignerTest, correlate2DOneToOne)
{
    // test one reference vs one image
    correlate2D<TypeParam>(1, 1);
}


TYPED_TEST_P( CudaShiftAlignerTest, correlate2DOneToMany)
{
    // check that n == batch works properly
    correlate2D<TypeParam>(5, 5);
}

TYPED_TEST_P( CudaShiftAlignerTest, correlate2DOneToManyBatched1)
{
    // test that n mod batch != 0 works
    correlate2D<TypeParam>(5, 3);
}

TYPED_TEST_P( CudaShiftAlignerTest, correlate2DOneToManyBatched2)
{
    // test that n mod batch = 0 works
    correlate2D<TypeParam>(6, 3);
}

REGISTER_TYPED_TEST_CASE_P(CudaShiftAlignerTest,
    correlate2DOneToOne,
    correlate2DOneToMany,
    correlate2DOneToManyBatched1,
    correlate2DOneToManyBatched2
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(SomeRandomText, CudaShiftAlignerTest, TestTypes);
