#include <gtest/gtest.h>
#include <random>
#include "reconstruction/ashift_corr_estimator.h"
#include "alignment_test_utils.h"
#include "core/utils/memory_utils.h"

template<typename T>
class AShiftEstimator_Test : public ::testing::Test {
public:
    SETUP

    void TearDown() {
        delete estimator;
    }

    SETUPTESTCASE

    static void TearDownTestCase() {
        for (auto h : hw) {
            delete h;
        }
        hw.clear();
    }

    void generateAndTest2D(size_t n, size_t batch) {

        std::uniform_int_distribution<> dist1(0, 368);
        std::uniform_int_distribution<> dist2(369, 768);

        // only even inputs are valid
        size_t first;
        size_t second;

        // test x == y
        first = ((int)dist1(mt) / 2) * 2;
        shift2D(FFTSettingsNew<T>(first, first, 1, n, batch, false, false));

        // test x > y
        first = ((int)dist1(mt) / 2) * 2;
        second = ((int)dist2(mt) / 2) * 2;
        shift2D(FFTSettingsNew<T>(second, first, 1, n, batch, false, false));

        // test x < y
        first = ((int)dist1(mt) / 2) * 2;
        second = ((int)dist2(mt) / 2) * 2;
        shift2D(FFTSettingsNew<T>(first, second, 1, n, batch, false, false));
    }

    void shift2D(const FFTSettingsNew<T> &dims)
    {
        using Alignment::AlignType;
        // max shift must be sharply less than half of the size
        auto maxShift = getMaxShift(dims.sDim());
        auto shifts = generateShifts(dims.sDim(), maxShift, mt);

        auto others = memoryUtils::page_aligned_alloc<T>(dims.sDim().size(), true);
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

        TEARDOWN

        free(others);
        delete[] ref;
    }

private:
    Alignment::AShiftEstimator<T> *estimator;
    static std::vector<HW*> hw;
    static std::mt19937 mt;

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
std::vector<HW*> AShiftEstimator_Test<T>::hw;
template<typename T>
std::mt19937 AShiftEstimator_Test<T>::mt(42); // fixed seed to ensure reproducibility


//***********************************************
//              Shift tests
//***********************************************


TYPED_TEST_P( AShiftEstimator_Test, shift2DOneToOne)
{
    XMIPP_TRY
    // test one reference vs one image
    AShiftEstimator_Test<TypeParam>::generateAndTest2D(1, 1);
    XMIPP_CATCH
}

TYPED_TEST_P( AShiftEstimator_Test, shift2DOneToMany)
{
    // check that n == batch works properly
    AShiftEstimator_Test<TypeParam>::generateAndTest2D(5, 5);
}


TYPED_TEST_P( AShiftEstimator_Test, shift2DOneToManyBatched1)
{
    // test that n mod batch != 0 works
    AShiftEstimator_Test<TypeParam>::generateAndTest2D(5, 3);
}

TYPED_TEST_P( AShiftEstimator_Test, shift2DOneToManyBatched2)
{
    // test that n mod batch = 0 works
    AShiftEstimator_Test<TypeParam>::generateAndTest2D(6, 3);
}


REGISTER_TYPED_TEST_CASE_P(AShiftEstimator_Test,
    shift2DOneToOne,
    shift2DOneToMany,
    shift2DOneToManyBatched1,
    shift2DOneToManyBatched2
);
