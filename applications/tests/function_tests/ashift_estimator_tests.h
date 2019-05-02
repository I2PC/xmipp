#include <gtest/gtest.h>
#include <random>
#include "reconstruction/ashift_corr_estimator.h"

template<typename T>
class AShiftEstimator_Test : public ::testing::Test {
public:
    SETUP

    void TearDown() {
        delete estimator;
    }

    SETUPTESTCASE

    static void TearDownTestCase() {
        delete hw;
    }

    void shift2D(FFTSettingsNew<T> &dims)
    {
        using Alignment::AlignType;
        // max shift must be sharply less than half of the size
        auto maxShift = std::min(dims.sDim().x() / 2, dims.sDim().y() / 2) - 1;
        auto maxShiftSq = maxShift * maxShift;
        // generate random shifts
        int seed = 42;
        std::mt19937 mt(seed);
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

        INIT
        estimator->load2DReferenceOneToN(ref);
        estimator->computeShift2DOneToN(others);
        auto result = estimator->getShifts2D();

        EXPECT_EQ(shifts.size(), result.size());
        for (size_t n = 0; n < shifts.size(); ++n) {
            ASSERT_EQ(shifts.at(n).x, - result.at(n).x); // notice the -1 multiplication
            ASSERT_EQ(shifts.at(n).y, - result.at(n).y); // notice the -1 multiplication
    //        std::cout << "expected " << "(" << shifts.at(n).x << "," << shifts.at(n).y << "), got "
    //            << "(" << result.at(n).x << "," << result.at(n).y << ")\n";
        }

        delete[] others;
        delete[] ref;
    }

private:
    Alignment::AShiftEstimator<T> *estimator;
    static HW *hw;

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

};
TYPED_TEST_CASE_P(AShiftEstimator_Test);

template<typename T>
HW* AShiftEstimator_Test<T>::hw;


//***********************************************
//              Shift tests
//***********************************************


TYPED_TEST_P( AShiftEstimator_Test, shift2DOneToOne)
{
    // test one reference vs one image
    FFTSettingsNew<TypeParam> dims(100, 50, 1, 1, 1, false, false);
    AShiftEstimator_Test<TypeParam>::shift2D(dims);
}

TYPED_TEST_P( AShiftEstimator_Test, shift2DOneToMany)
{
    // check that n == batch works properly
    FFTSettingsNew<TypeParam> dims(100, 50, 1, 5, 5, false, false);
    AShiftEstimator_Test<TypeParam>::shift2D(dims);
}


TYPED_TEST_P( AShiftEstimator_Test, shift2DOneToManyBatched1)
{
    // test that n mod batch != 0 works
    FFTSettingsNew<TypeParam> dims(100, 50, 1, 5, 3, false, false);
    AShiftEstimator_Test<TypeParam>::shift2D(dims);
}

TYPED_TEST_P( AShiftEstimator_Test, shift2DOneToManyBatched2)
{
    // test that n mod batch = 0 works
    FFTSettingsNew<TypeParam> dims(100, 50, 1, 6, 3, false, false);
    AShiftEstimator_Test<TypeParam>::shift2D(dims);
}


REGISTER_TYPED_TEST_CASE_P(AShiftEstimator_Test,
    shift2DOneToOne,
    shift2DOneToMany,
    shift2DOneToManyBatched1,
    shift2DOneToManyBatched2
);
