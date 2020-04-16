#include <gtest/gtest.h>
#include "alignment_test_utils.h"
#include "reconstruction/iterative_alignment_estimator.h"
#include "core/utils/memory_utils.h"
#include "reconstruction/bspline_geo_transformer.h"
#include "reconstruction/amerit_computer.h"
#include "data/cpu.h"

template<typename T>
class IterativeAlignmentEstimatorHelper : public Alignment::IterativeAlignmentEstimator<T> {
public:
    static void applyTransform(ctpl::thread_pool &pool, const Dimensions &dims, const std::vector<Point2D<float>> &shifts, const std::vector<float> &rotations,
            const T * __restrict__ orig, T * __restrict__ copy) {
        Alignment::AlignmentEstimation e(dims.n());
        for (size_t i = 0; i < dims.n(); ++i) {
            auto &m = e.poses.at(i);
            auto &s = shifts.at(i);
            MAT_ELEM(m,0,2) += s.x;
            MAT_ELEM(m,1,2) += s.y;
            auto r = Matrix2D<float>();
            rotation2DMatrix(rotations.at(i), r);
            m = r * m;
        }
        Alignment::IterativeAlignmentEstimator<T>::sApplyTransform(pool, dims, e, orig, copy, true);
    }
    static void compensateTransform(ctpl::thread_pool &pool, const Alignment::AlignmentEstimation &est, const Dimensions &dims, T *in, T *out) {
        Alignment::IterativeAlignmentEstimator<T>::sApplyTransform(pool, dims, est, in, out, false);
    }
};

template<typename T>
class IterativeAlignmentEstimator_Test : public ::testing::Test {
public:

    static void SetUpTestCase() {
        others = new T[maxDims.size()]();
        ref = memoryUtils::page_aligned_alloc<T>(maxDims.sizeSingle(), true);
        SETUPTESTCASE_SPECIFIC
    }

    static void TearDownTestCase() {
        for (auto device : hw) {
            delete device;
        }
        hw.clear();
        delete shiftAligner;
        delete rotationAligner;
        delete transformer;
        delete meritComputer;
        delete[] others;
        others = nullptr;
        free(ref);
        ref = nullptr;
    }

    template<bool WITH_NOISE>
    void checkStatistics() {
        std::sort(diffsR.begin(), diffsR.end());
        std::sort(diffsX.begin(), diffsX.end());
        std::sort(diffsY.begin(), diffsY.end());
        std::sort(expR.begin(), expR.end());
        bool isCPU = dynamic_cast<CPU*>(hw.at(0));
        if (WITH_NOISE) {
            if (isCPU) {
                float refR = expR.at(std::floor((expR.size() - 1) * 0.67f));
                // FIXME this propably means that we have a bug in the CPU code, maybe also in the double version
                EXPECT_GE(10 * refR, diffsR.at(std::floor((expR.size() - 1) * 0.67f))) << "percentile 67";
                EXPECT_GE(1, diffsX.at(std::floor((diffsX.size() - 1) * 0.41f))) << "percentile 41";
                EXPECT_GE(2, diffsX.at(std::floor((diffsX.size() - 1) * 0.51f))) << "percentile 51";
                EXPECT_GE(1, diffsY.at(std::floor((diffsY.size() - 1) * 0.41f))) << "percentile 41";
                EXPECT_GE(2, diffsY.at(std::floor((diffsY.size() - 1) * 0.53f))) << "percentile 53";
            } else {
                float refR = expR.at(std::floor((expR.size() - 1) * 0.72f));
                EXPECT_GE(10 * refR, diffsR.at(std::floor((expR.size() - 1) * 0.77f))) << "percentile 77";
                EXPECT_GE(1, diffsX.at(std::floor((diffsX.size() - 1) * 0.51f))) << "percentile 51";//-
                EXPECT_GE(2, diffsX.at(std::floor((diffsX.size() - 1) * 0.63f))) << "percentile 63";
                EXPECT_GE(1, diffsY.at(std::floor((diffsY.size() - 1) * 0.48f))) << "percentile 48";//-
                EXPECT_GE(2, diffsY.at(std::floor((diffsY.size() - 1) * 0.63f))) << "percentile 63";
            }
        } else {
            float refR = expR.at(std::floor((expR.size() - 1) * 0.9f));
            if (isCPU) {
                EXPECT_GE(2 * refR, diffsR.at(std::floor((expR.size() - 1) * 0.9f))) << "percentile 90";
                EXPECT_GE(1, diffsX.at(std::floor((diffsX.size() - 1) * 0.8f))) << "percentile 80";
                EXPECT_GE(1.8, diffsX.at(std::floor((diffsX.size() - 1) * 0.9f))) << "percentile 90";
                EXPECT_GE(1, diffsY.at(std::floor((diffsY.size() - 1) * 0.8f))) << "percentile 80";
                EXPECT_GE(1.86f, diffsY.at(std::floor((diffsY.size() - 1) * 0.9f))) << "percentile 90";
            } else {
                EXPECT_GE(2 * refR, diffsR.at(std::floor((expR.size() - 1) * 0.99f))) << "percentile 90";
                EXPECT_GE(1, diffsX.at(std::floor((diffsX.size() - 1) * 0.87f))) << "percentile 87";
                EXPECT_GE(1.7, diffsX.at(std::floor((diffsX.size() - 1) * 0.96f))) << "percentile 96";
                EXPECT_GE(1, diffsY.at(std::floor((diffsY.size() - 1) * 0.89f))) << "percentile 80";
                EXPECT_GE(1.86f, diffsY.at(std::floor((diffsY.size() - 1) * 0.97f))) << "percentile 97";
            }
        }
//        printf("refR|diffR|diffX|diffY\n");
//        for (size_t n = 0; n < diffsR.size(); ++n) {
//            printf("%s|%f|%f|%f\n",
//                    (n < expR.size()? std::to_string(expR.at(n)).c_str() : ""), diffsR.at(n), diffsX.at(n), diffsY.at(n));
//        }
    }

    void clearStatistics() {
        expR.clear();
        diffsR.clear();
        diffsX.clear();
        diffsY.clear();
    }

    template<bool USE_NOISE>
    void generateAndTestStatistics2D(size_t n, size_t batch) {
        std::uniform_int_distribution<> dist1(0, 368);
        std::uniform_int_distribution<> dist2(369, 768);

        // only even inputs are valid
        // smaller
        size_t size = ((int)dist1(mt) / 2) * 2;
        testStatistics<USE_NOISE>(Dimensions(size, size, 1, n), batch);
        // bigger
        size = ((int)dist2(mt) / 2) * 2;
        testStatistics<USE_NOISE>(Dimensions(size, size, 1, n), batch);
    }

    template<bool ADD_NOISE>
    void testStatistics(const Dimensions &dims, size_t batch) {
        using namespace Alignment;

        if (dims.x() == 680 && dims.n() == 100 && batch == 50
                && std::is_same<double, T>::value) {
            std::cout << "Skipping " << dims << " (batch " << batch
                    << ", double): Insufficient GPU memory\n";
            return;
        }

        bool saveOutput = false && (dims.x() == 260 && dims.y() == 260 && batch == 1);
//        printf("sizes: %lu %lu %lu %lu, batch %lu\n", dims.x(), dims.y(), dims.z(), dims.n(), batch);

        auto maxShift = std::min((size_t)20, getMaxShift(dims));
        auto shifts = generateShifts(dims, maxShift, mt);
        auto maxRotation = RotationEstimationSetting::getMaxRotation();
        auto rotations = generateRotations(dims, maxRotation, mt);

        assert(dims.size() < maxDims.size());
        assert(dims.sizeSingle() < maxDims.sizeSingle());

        T *othersCorrected = nullptr;
        if (saveOutput) {
            othersCorrected = new T[maxDims.size()]();
        }

        T centerX = dims.x() / 2;
        T centerY = dims.y() / 2;
        // generate data
        drawClockArms(ref, dims, centerX, centerY, 0.f);
        // prepare aligner
        auto rotSettings = RotationEstimationSetting();
        rotSettings.hw = hw;
        rotSettings.type = AlignType::OneToN;
        rotSettings.refDims = dims.createSingle();
        rotSettings.otherDims = dims;
        rotSettings.batch = batch;
        rotSettings.maxRotDeg = maxRotation;
        rotSettings.firstRing = RotationEstimationSetting::getDefaultFirstRing(dims);
        rotSettings.lastRing = RotationEstimationSetting::getDefaultLastRing(dims);
        rotSettings.fullCircle = true;
        rotSettings.allowTuningOfNumberOfSamples = false;
        rotSettings.allowDataOverwrite = true;

        auto tSettings = BSplineTransformSettings<T>();
        tSettings.keepSrcCopy = true;
        tSettings.degree = InterpolationDegree::Linear;
        tSettings.dims = dims;
        tSettings.hw.push_back(hw.at(0));
        tSettings.type = InterpolationType::NToN;
        tSettings.doWrap = false;
        tSettings.defaultVal = (T)0;

        auto mSettings = MeritSettings();
        mSettings.hw.push_back(hw.at(0));
        mSettings.normalizeResult = true;
        mSettings.otherDims = dims;
        mSettings.refDims = dims.createSingle();
        mSettings.type = MeritType::OneToN;

        shiftAligner->init2D(hw, AlignType::OneToN, FFTSettingsNew<T>(dims, batch), maxShift, true, true, true);
        rotationAligner->init(rotSettings, true);
        transformer->init(tSettings, false);
        meritComputer->init(mSettings, true);
        ctpl::thread_pool threadPool(CPU::findCores());
        IterativeAlignmentEstimator<T> aligner(*rotationAligner, *shiftAligner, *transformer, *meritComputer, threadPool);
        IterativeAlignmentEstimatorHelper<T>::applyTransform(
                threadPool, dims, shifts, rotations, ref, others);

        if (ADD_NOISE) {
            // add noise to data
            // the reference should be without noise
            addNoise(others, dims, mt_noise);
        }
        // show data
        if (saveOutput) {
            outputData(others, dims, "dataBeforeAlignment.stk");
        }

        aligner.loadReference(ref);
        auto result = aligner.compute(others, 3); // use at least three iterations

        // show result
        if (saveOutput) {
            IterativeAlignmentEstimatorHelper<T>::compensateTransform(
                    threadPool, result, dims, others, othersCorrected);
            outputData(othersCorrected, dims, "dataAfterAlignment.stk");
            // show statistics
            outputStatistics(dims, shifts, rotations, result, ref, others);
        }

        saveResults(result, shifts, rotations, dims);

        if (saveOutput) {
            delete[] othersCorrected;
        }
    }

private:
    static std::mt19937 mt;
    static std::mt19937 mt_noise;
    static Alignment::ARotationEstimator<T> *rotationAligner;
    static Alignment::AShiftCorrEstimator<T> *shiftAligner;
    static BSplineGeoTransformer<T> *transformer;
    static AMeritComputer<T> *meritComputer;
    static std::vector<float> diffsX;
    static std::vector<float> diffsY;
    static std::vector<float> diffsR;
    static std::vector<float> expR;
    static std::vector<HW*> hw;
    static T *ref;
    static T *others;
    static Dimensions maxDims;

    void saveResults(const Alignment::AlignmentEstimation &est,
            const std::vector<Point2D<float>> &shifts,
            const std::vector<float> &rotations,
            const Dimensions &dims) {
        ASSERT_EQ(dims.n(), est.poses.size());

        // extract diffs
        for (size_t i = 0;i < dims.n();++i) {
            auto sE = shifts.at(i);
            auto rE = rotations.at(i);
            auto m = est.poses.at(i);
            auto sA = Point2D<float>(-MAT_ELEM(m, 0, 2), -MAT_ELEM(m, 1, 2));
            auto rA = fmod(360 + RAD2DEG(atan2(MAT_ELEM(m, 1, 0), MAT_ELEM(m, 0, 0))), 360);
            diffsR.push_back(180 - abs(abs(rA - rE) - 180));
            diffsX.push_back(std::abs(sA.x - sE.x));
            diffsY.push_back(std::abs(sA.y - sE.y));
        }
        expR.push_back(getTheoreticalRotationError(dims));
        // don't check for anything else, we do global check at the end of the test suite
    }

    void outputData(T *data, const Dimensions &dims, const std::string &name) {
        MultidimArray<T>wrapper(dims.n(), dims.z(), dims.y(), dims.x(), data);
        Image<T> img(wrapper);
        img.write(name);
    }

    Matrix2D<double> getReferenceTransform(const T *ref, const T *other, const Dimensions &dims, double &corr) {
        Matrix2D<double> M;
        auto I = convert(other, dims);
        I.setXmippOrigin();
        auto refWrapper = convert(ref, dims);
        refWrapper.setXmippOrigin();

        corr = alignImages(refWrapper, I, M, DONT_WRAP);
        return M;
    }

    MultidimArray<double> convert(const float *data, const Dimensions &dims) {
        auto wrapper = MultidimArray<double>(1, 1, dims.y(), dims.x());
        for (size_t i = 0; i < dims.xyz(); ++i) {
            wrapper.data[i] = data[i];
        }
        return wrapper;
    }

    MultidimArray<double> convert(const double *data, const Dimensions &dims) {
        return MultidimArray<double>(1, 1, dims.y(), dims.x(), (T*)data);
    }

    void outputStatistics(const Dimensions &dims,
        const std::vector<Point2D<float>> &shifts,
        const std::vector<float> &rotations,
        const Alignment::AlignmentEstimation &result,
        const T *ref,
        const T *others)
    {
        printf("Ground truth|shiftX|shiftY|rot|New version (CPU)|shiftX|shiftY|rot|Correlation|Orig version(up to 3 iter, CPU)|shiftX|shiftY|rot|Correlation||shiftX (GT-new)|shiftY(GT-new)|rot(GT-new)|Correlation||shiftX (GT-old)|shiftY(GT-old)|rot(GT-old)|Correlation\n");
        for (size_t i = 0;i < dims.n();++i) {
            auto sE = shifts.at(i);
            auto rE = rotations.at(i);
            auto m = result.poses.at(i);
            auto sA = Point2D<float>(-MAT_ELEM(m, 0, 2), -MAT_ELEM(m, 1, 2));
            auto rA = fmod(360 + RAD2DEG(atan2(MAT_ELEM(m, 1, 0), MAT_ELEM(m, 0, 0))), 360);
            // ground truth, new version
            printf("| %f | %f | %f ||%f | %f | %f | %f ||",// GT
                    // new
                    sE.x, sE.y, rE, sA.x, sA.y, rA, result.figuresOfMerit.at(i));
                    // original version
            size_t offset = i * dims.xyzPadded();
            double corr = std::numeric_limits<double>::lowest();
            auto M = getReferenceTransform(ref, others + offset, dims, corr);
            auto sR = Point2D<float>(-MAT_ELEM(M, 0, 2), -MAT_ELEM(M, 1, 2));
            auto rR = fmod(360 + RAD2DEG(atan2(MAT_ELEM(M, 1, 0), MAT_ELEM(M, 0, 0))), 360);
            printf("%f | %f | %f | %f ||",// orig
                    sR.x, sR.y, rR, corr);
            // comparison GT <-> new
            printf("%f | %f | %f | %f||", std::abs(sE.x - sA.x), std::abs(sE.y - sA.y), 180 - std::abs((std::abs(rA - rE) - 180)), result.figuresOfMerit.at(i));
            // comparison GT <-> orig
            printf("%f | %f | %f | %f\n", std::abs(sE.x - sR.x), std::abs(sE.y - sR.y), 180 - std::abs((std::abs(rR - rE) - 180)), corr);
        }
    }
};
TYPED_TEST_CASE_P(IterativeAlignmentEstimator_Test);

template<typename T>
Alignment::ARotationEstimator<T> *IterativeAlignmentEstimator_Test<T>::rotationAligner = nullptr;
template<typename T>
Alignment::AShiftCorrEstimator<T> *IterativeAlignmentEstimator_Test<T>::shiftAligner = nullptr;
template<typename T>
BSplineGeoTransformer<T> *IterativeAlignmentEstimator_Test<T>::transformer = nullptr;
template<typename T>
AMeritComputer<T> *IterativeAlignmentEstimator_Test<T>::meritComputer = nullptr;
template<typename T>
Dimensions IterativeAlignmentEstimator_Test<T>::maxDims(768, 768, 1, 1000);
template<typename T>
T *IterativeAlignmentEstimator_Test<T>::ref = nullptr;
template<typename T>
T *IterativeAlignmentEstimator_Test<T>::others = nullptr;
template<typename T>
std::vector<HW*> IterativeAlignmentEstimator_Test<T>::hw;
template<typename T>
std::vector<float> IterativeAlignmentEstimator_Test<T>::diffsX;
template<typename T>
std::vector<float> IterativeAlignmentEstimator_Test<T>::diffsY;
template<typename T>
std::vector<float> IterativeAlignmentEstimator_Test<T>::diffsR;
template<typename T>
std::vector<float> IterativeAlignmentEstimator_Test<T>::expR;
template<typename T>
std::mt19937 IterativeAlignmentEstimator_Test<T>::mt(42); // fixed seed to ensure reproducibility
template<typename T>
std::mt19937 IterativeAlignmentEstimator_Test<T>::mt_noise(23); // fixed seed to ensure reproducibility

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToOneNoNoise)
{
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTestStatistics2D<false>(1, 1);
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoNoiseNoBatch)
{
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTestStatistics2D<false>(100, 1);
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoNoiseBatch)
{
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTestStatistics2D<false>(100, 50);
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoNoiseBatchNotMultiple)
{
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTestStatistics2D<false>(100, 60);
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, checkStatisticsNoNoise)
{
    // this must be the last test in the 'test block'
    IterativeAlignmentEstimator_Test<TypeParam>::template checkStatistics<false>();
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, clearStatisticsNoNoise)
{
    // this must be the last test
    IterativeAlignmentEstimator_Test<TypeParam>::clearStatistics();
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToOneNoise)
{
    XMIPP_TRY
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTestStatistics2D<true>(1, 1);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoiseNoBatch)
{
    XMIPP_TRY
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTestStatistics2D<true>(100, 1);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoiseBatch)
{
    XMIPP_TRY
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTestStatistics2D<true>(100, 50);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoiseBatchNotMultiple)
{
    XMIPP_TRY
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTestStatistics2D<true>(100, 60);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, checkStatisticsNoise)
{
    // this must be the last test
    IterativeAlignmentEstimator_Test<TypeParam>::template checkStatistics<true>();
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, clearStatisticsNoise)
{
    // this must be the last test
    IterativeAlignmentEstimator_Test<TypeParam>::clearStatistics();
}

//TYPED_TEST_P( IterativeAlignmentEstimator_Test, debug)
//{
//    XMIPP_TRY
//    auto dims = Dimensions(680, 680, 1, 100);
////    auto dims = Dimensions(64, 64, 1, 50);
//    size_t batch = 50;
//    IterativeAlignmentEstimator_Test<TypeParam>::template testStatistics<false>(dims, batch);
////    IterativeAlignmentEstimator_Test<TypeParam>::template test<false>(dims, batch);
//    XMIPP_CATCH
//}

REGISTER_TYPED_TEST_CASE_P(IterativeAlignmentEstimator_Test,
//    debug
    // tests run different scenarios...
    align2DOneToOneNoNoise,
    align2DOneToManyNoNoiseNoBatch,
    align2DOneToManyNoNoiseBatch,
    align2DOneToManyNoNoiseBatchNotMultiple,
    // ...and at the end we check that it more or less works
    checkStatisticsNoNoise,
    clearStatisticsNoNoise,
    // tests run different scenarios...
    align2DOneToOneNoise,
    align2DOneToManyNoiseNoBatch,
    align2DOneToManyNoiseBatch,
    align2DOneToManyNoiseBatchNotMultiple,
    // ...and at the end we check that it more or less works
    checkStatisticsNoise,
    clearStatisticsNoise
);

