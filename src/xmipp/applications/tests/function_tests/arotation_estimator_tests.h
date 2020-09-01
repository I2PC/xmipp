#include <gtest/gtest.h>
#include <random>
#include "reconstruction/arotation_estimator.h"
#include "core/transformations.h"
#include "core/xmipp_image.h"
#include "alignment_test_utils.h"
#include "core/utils/memory_utils.h"
#include "data/cpu.h"

template<typename T>
class ARotationEstimator_Test : public ::testing::Test {
public:
    SETUP

    void TearDown() {
        delete estimator;
    }

    SETUPTESTCASE

    static void TearDownTestCase() {
        for (auto device : hw) {
            delete device;
        }
        hw.clear();
    }

    template<bool ADD_NOISE>
    void generateAndTest2D(size_t n, size_t batch) {
        // images bellow 13 x 13 pixels are just too small for processing
        std::uniform_int_distribution<> distSizeSmall(13, 368);
        std::uniform_int_distribution<> distSizeBig(369, 768);

        // only square inputs are valid:
        size_t size = distSizeSmall(mt);
        rotate2D<ADD_NOISE>(Dimensions(size, size, 1, n), batch);
        size = distSizeBig(mt);
        rotate2D<ADD_NOISE>(Dimensions(size, size, 1, n), batch);
    }

    template<bool ADD_NOISE>
    void rotate2D(const Dimensions &dims, size_t batch)
    {
        using namespace Alignment;
        float maxRotation = RotationEstimationSetting::getMaxRotation();

//        printf("testing: %lu x %lu x %lu (batch %lu)\n",
//                dims.x(), dims.y(), dims.n(), batch);

        auto rotations = generateRotations(dims, maxRotation, mt);
        auto others = memoryUtils::page_aligned_alloc<T>(dims.size(), true);
        auto ref = memoryUtils::page_aligned_alloc<T>(dims.xy(), true);
        T centerX = dims.x() / 2;
        T centerY = dims.y() / 2;
        drawClockArms(ref, dims, centerX, centerY, 0.f);

        for (size_t n = 0; n < dims.n(); ++n) {
            T *d = others + (n * dims.xyzPadded());
            drawClockArms(d, dims, centerX, centerY, rotations.at(n));
        }
        if (ADD_NOISE) {
            addNoise(others, dims, mt);
        }
//        outputData("data.stk", others, dims);

        auto settings = Alignment::RotationEstimationSetting();
        settings.hw = hw;
        settings.type = AlignType::OneToN;
        settings.refDims = dims.createSingle();
        settings.otherDims = dims;
        settings.batch = batch;
        settings.maxRotDeg = maxRotation;
        settings.firstRing = RotationEstimationSetting::getDefaultFirstRing(dims);
        settings.lastRing = RotationEstimationSetting::getDefaultLastRing(dims);
        settings.fullCircle = true;
        settings.allowTuningOfNumberOfSamples = false; // to make sure that we always use the same settings
        settings.allowDataOverwrite = false; // we cannot reuse it, as this test needs to work both for CPU and GPU version

        estimator->init(settings, true);
        hw.at(0)->lockMemory(others, dims.size() * sizeof(T));

        estimator->loadReference(ref);
        estimator->compute(others);

        const auto *cEst = estimator;
        auto result = cEst->getRotations2D();
        EXPECT_EQ(rotations.size(), result.size());
        check<ADD_NOISE>(dims, rotations, result);

        hw.at(0)->unlockMemory(others);

        free(others);
        free(ref);
    }

private:
    template<bool IS_WITH_NOISE>
    void check(const Dimensions &dims, const std::vector<float> &rotations,
            const std::vector<float> &result) {
        float maxError = getTheoreticalRotationError(dims);
        if (IS_WITH_NOISE) {
            maxError *= 0.81f; // while testing, it seems that we can be more strict.
        } else {
            maxError *= 0.62f; // while testing, it seems that we can be more strict. The max error was usually half of the theoretical
        }
        for (size_t n = 0; n < result.size(); ++n) {
            // we rotated by angle, so we should detect rotation in '360 - angle' degrees
            auto actual = 360 - result.at(n);
            auto diff = 180 - abs(abs(actual - rotations.at(n)) - 180);
            EXPECT_NEAR(diff, 0, maxError) << "expected: "
                    << rotations.at(n) << " actual: " << actual << " signal " << n;
        }
    }

    Alignment::ARotationEstimator<T> *estimator;
    static std::vector<HW*> hw;
    static std::mt19937 mt;

    void outputData(const std::string &fn, T *data, const Dimensions &dims) {
        MultidimArray<T>wrapper(dims.n(), dims.z(), dims.y(), dims.x(), data);
        Image<T> img(wrapper);
        img.write(fn);
    }
};
TYPED_TEST_CASE_P(ARotationEstimator_Test);

template<typename T>
std::vector<HW*> ARotationEstimator_Test<T>::hw;
template<typename T>
std::mt19937 ARotationEstimator_Test<T>::mt(42); // fixed seed to ensure reproducibility


//***********************************************
//              Rotation tests
//***********************************************


TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToOne)
{
    XMIPP_TRY
    // test one reference vs one image
    ARotationEstimator_Test<TypeParam>::template generateAndTest2D<false>(1, 1);
    XMIPP_CATCH
}

TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToMany)
{
    // check that n == batch works properly
    ARotationEstimator_Test<TypeParam>::template generateAndTest2D<false>(5, 1);
}

TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToManyBatched1)
{
    // test that n mod batch != 0 works
    ARotationEstimator_Test<TypeParam>::template generateAndTest2D<false>(5, 3);
}

TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToManyBatched2)
{
    // test that n mod batch == 0 works
    ARotationEstimator_Test<TypeParam>::template generateAndTest2D<false>(12, 4);
}

TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToOneNoise)
{
    // test one reference vs one image
    ARotationEstimator_Test<TypeParam>::template generateAndTest2D<true>(1, 1);
}

TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToManyNoise)
{
    // check that n == batch works properly
    ARotationEstimator_Test<TypeParam>::template generateAndTest2D<true>(5, 1);
}

TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToManyBatched1Noise)
{
    // test that n mod batch != 0 works
    ARotationEstimator_Test<TypeParam>::template generateAndTest2D<true>(5, 3);
}

TYPED_TEST_P( ARotationEstimator_Test, rotate2DOneToManyBatched2Noise)
{
    // test that n mod batch == 0 works
    ARotationEstimator_Test<TypeParam>::template generateAndTest2D<true>(12, 4);
}

//TYPED_TEST_P( ARotationEstimator_Test, DEBUG)
//{
//    ARotationEstimator_Test<TypeParam>::rotate2D(Dimensions(13, 13, 1, 2), 1);
//}

REGISTER_TYPED_TEST_CASE_P(ARotationEstimator_Test,
//    DEBUG
    rotate2DOneToOne,
    rotate2DOneToMany,
    rotate2DOneToManyBatched1,
    rotate2DOneToManyBatched2,
    rotate2DOneToOneNoise,
    rotate2DOneToManyNoise,
    rotate2DOneToManyBatched1Noise,
    rotate2DOneToManyBatched2Noise
);
