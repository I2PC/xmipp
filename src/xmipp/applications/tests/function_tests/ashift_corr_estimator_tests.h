#include <gtest/gtest.h>
#include "reconstruction/ashift_corr_estimator.h"

template<typename T>
class AShiftCorrEstimator_Test : public ::testing::Test {
public:
    SETUP

    void TearDown() {
        delete estimator;
    }

    SETUPTESTCASE

    static void TearDownTestCase() {
        delete hw;
    }

    void correlate2DNoCenter(size_t n, size_t batch) {
        using std::complex;
        using Alignment::AlignType;

        FFTSettingsNew<T> dims(30, 14, 1, n, batch, false, false); // only even sizes are supported

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

        estimator->init2D(*hw, AlignType::OneToN, dims, 1, false, false);
        estimator->load2DReferenceOneToN(ref);
        estimator->computeCorrelations2DOneToN(inOut, false);
        hw->synch();

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

private:
    Alignment::AShiftCorrEstimator<T> *estimator;
    static HW *hw;

};
TYPED_TEST_CASE_P(AShiftCorrEstimator_Test);

template<typename T>
HW* AShiftCorrEstimator_Test<T>::hw;


//***********************************************
//              Correlation tests
//***********************************************

TYPED_TEST_P( AShiftCorrEstimator_Test, correlate2DOneToOne)
 {
     // test one reference vs one image
    AShiftCorrEstimator_Test<TypeParam>::correlate2DNoCenter(1, 1);
}

TYPED_TEST_P( AShiftCorrEstimator_Test, correlate2DOneToMany)
{
    // check that n == batch works properly
    AShiftCorrEstimator_Test<TypeParam>::correlate2DNoCenter(5, 5);
}

TYPED_TEST_P( AShiftCorrEstimator_Test, correlate2DOneToManyBatched1)
{
    // test that n mod batch != 0 works
    AShiftCorrEstimator_Test<TypeParam>::correlate2DNoCenter(5, 3);
}

TYPED_TEST_P( AShiftCorrEstimator_Test, correlate2DOneToManyBatched2)
{
    // test that n mod batch = 0 works
    AShiftCorrEstimator_Test<TypeParam>::correlate2DNoCenter(6, 3);
}

REGISTER_TYPED_TEST_CASE_P(AShiftCorrEstimator_Test,
        correlate2DOneToOne,
        correlate2DOneToMany,
        correlate2DOneToManyBatched1,
        correlate2DOneToManyBatched2
);
