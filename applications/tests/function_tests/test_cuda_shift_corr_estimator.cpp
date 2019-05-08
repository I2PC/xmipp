#include <gtest/gtest.h>
#include <random>

#include "reconstruction_cuda/cuda_shift_corr_estimator.h"

template<typename T>
class CudaShiftCorrEstimatorTest : public ::testing::Test { };
TYPED_TEST_CASE_P(CudaShiftCorrEstimatorTest);

template<typename T>
void correlate2DNoCenter(size_t n, size_t batch) {
    using std::complex;
    using Alignment::CudaShiftCorrEstimator;
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

    CudaShiftCorrEstimator<T> aligner;
    auto gpu = GPU();
    gpu.set();
    aligner.init2D(gpu, AlignType::OneToN, dims);
    aligner.load2DReferenceOneToN(ref);
    aligner.computeCorrelations2DOneToN(inOut, false);
    gpu.synch();

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

TYPED_TEST_P( CudaShiftCorrEstimatorTest, correlate2DOneToOne)
 {
     // test one reference vs one image
    correlate2DNoCenter<TypeParam>(1, 1);
}

TYPED_TEST_P( CudaShiftCorrEstimatorTest, correlate2DOneToMany)
{
    // check that n == batch works properly
    correlate2DNoCenter<TypeParam>(5, 5);
}

TYPED_TEST_P( CudaShiftCorrEstimatorTest, correlate2DOneToManyBatched1)
{
    // test that n mod batch != 0 works
    correlate2DNoCenter<TypeParam>(5, 3);
}

TYPED_TEST_P( CudaShiftCorrEstimatorTest, correlate2DOneToManyBatched2)
{
    // test that n mod batch = 0 works
    correlate2DNoCenter<TypeParam>(6, 3);
}

template<typename T>
void drawCross(T *data, size_t xDim, size_t yDim, T xPos, T yPos) {
    for (size_t y = 0; y < yDim; ++y) {
        for (size_t x = 0; x < xDim; ++x) {
            size_t index = y * xDim + x;
            data[index] = (xPos == x) || (yPos == y);
             // draw wave with crossing line
//            data[index] = std::cos((M_PI * 2.f * (x - xPos)) / (xDim / 16.f) ) * std::exp(-std::pow((x-xPos)/2,2.0));// || (yPos == y);
//            if (yPos == y) {
//                data[index] = 1;
//            }
        }
    }
}


template<typename T>
void shift2D(FFTSettingsNew<T> &dims)
{
    using Alignment::CudaShiftCorrEstimator;
    using Alignment::AlignType;
    // max shift must be sharply less than half of the size
    auto maxShift = std::min(dims.sDim().x() / 2, dims.sDim().y() / 2) - 1;
    auto maxShiftSq = maxShift * maxShift;
    // generate random shifts
    int seed = 42;
    std::mt19937 mt(seed);
//    mt.seed(seed);
    std::uniform_int_distribution<> dist(0, maxShift);
    auto shifts = std::vector<Point2D<T>>();
    shifts.reserve(dims.fDim().n());
    for(size_t n = 0; n < dims.fDim().n(); ++n) {
        // generate shifts so that the Euclidean distance is smaller than max shift
        int shiftX = dist(mt);
        int shiftXSq = shiftX * shiftX;
        int maxShiftY = std::floor(sqrt(maxShiftSq - shiftXSq));
        int shiftY = (0 == maxShiftY) ? 0 : dist(mt) % maxShiftY;
        shifts.emplace_back(shiftX, shiftY);
    }

    auto others = new T[dims.sDim().size()];
    auto ref = new T[dims.sDim().xy()];
    T centerX = dims.sDim().x() / 2;
    T centerY = dims.sDim().y() / 2;
    drawCross(ref, dims.sDim().x(), dims.sDim().y(), centerX, centerY);
    for (size_t n = 0; n < dims.fDim().n(); ++n) {
        drawCross(others + n * dims.sDim().xy(),
                dims.sDim().x(), dims.sDim().y(),
                centerX + shifts.at(n).x, centerY + shifts.at(n).y);
    }

    auto aligner = CudaShiftCorrEstimator<T>();
    auto gpu = GPU();
    gpu.set();
    aligner.init2D(gpu, AlignType::OneToN, dims, maxShift, true, true);
    aligner.load2DReferenceOneToN(ref);
    auto result = aligner.computeShift2DOneToN(others);

    EXPECT_EQ(shifts.size(), result.size());
    for (size_t n = 0; n < shifts.size(); ++n) {
        EXPECT_EQ(shifts.at(n).x, - result.at(n).x); // notice the -1 multiplication
        EXPECT_EQ(shifts.at(n).y, - result.at(n).y); // notice the -1 multiplication
//        std::cout << "expected " << "(" << shifts.at(n).x << "," << shifts.at(n).y << "), got "
//            << "(" << result.at(n).x << "," << result.at(n).y << ")\n";
    }

    delete[] others;
    delete[] ref;
}

TYPED_TEST_P( CudaShiftCorrEstimatorTest, shift2DOneToOne)
{
    // test one reference vs one image
    FFTSettingsNew<TypeParam> dims(100, 50, 1, 1, 1, false, false);
    shift2D(dims);
}

TYPED_TEST_P( CudaShiftCorrEstimatorTest, shift2DOneToMany)
{
    // check that n == batch works properly
    FFTSettingsNew<TypeParam> dims(100, 50, 1, 5, 5, false, false);
    shift2D(dims);
}


TYPED_TEST_P( CudaShiftCorrEstimatorTest, shift2DOneToManyBatched1)
{
    // test that n mod batch != 0 works
    FFTSettingsNew<TypeParam> dims(100, 50, 1, 5, 3, false, false);
    shift2D(dims);
}

TYPED_TEST_P( CudaShiftCorrEstimatorTest, shift2DOneToManyBatched2)
{
    // test that n mod batch = 0 works
    FFTSettingsNew<TypeParam> dims(100, 50, 1, 6, 3, false, false);
    shift2D(dims);
}


REGISTER_TYPED_TEST_CASE_P(CudaShiftCorrEstimatorTest,
    correlate2DOneToOne,
    correlate2DOneToMany,
    correlate2DOneToManyBatched1,
    correlate2DOneToManyBatched2,
    shift2DOneToOne,
    shift2DOneToMany,
    shift2DOneToManyBatched1,
    shift2DOneToManyBatched2
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Alignment, CudaShiftCorrEstimatorTest, TestTypes);
