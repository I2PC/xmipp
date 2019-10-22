#include <gtest/gtest.h>
#include "alignment_test_utils.h"
#include "reconstruction/iterative_alignment_estimator.h"
#include "core/utils/memory_utils.h"


template<typename T>
class IterativeAlignmentEstimatorHelper : public Alignment::IterativeAlignmentEstimator<T> {
public:
    static void applyTransform(const Dimensions &dims, const std::vector<Point2D<float>> &shifts, const std::vector<float> &rotations,
            const T * __restrict__ orig, T * __restrict__ copy) {
        Alignment::AlignmentEstimation e(dims.n());
        for (size_t i = 0; i < dims.n(); ++i) {
            auto &m = e.poses.at(i);
            auto &s = shifts.at(i);
            MAT_ELEM(m,0,2) += s.x;
            MAT_ELEM(m,1,2) += s.y;
            auto r = Matrix2D<double>();
            rotation2DMatrix(rotations.at(i), r);
            m = r * m;
        }
        Alignment::IterativeAlignmentEstimator<T>::sApplyTransform(dims, e, orig, copy, true);
    }
    static void compensateTransform(const Alignment::AlignmentEstimation &est, const Dimensions &dims, T *in, T *out) {
        Alignment::IterativeAlignmentEstimator<T>::sApplyTransform(dims, est, in, out, false);
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
        delete[] others;
        others = nullptr;
        free(ref);
        ref = nullptr;
    }


    template<bool USE_NOISE>
    void generateAndTest2D(size_t n, size_t batch) {
        std::uniform_int_distribution<> dist1(0, 368);
        std::uniform_int_distribution<> dist2(369, 768);

        // only even inputs are valid
        // smaller
        size_t size = ((int)dist1(mt) / 2) * 2;
        test<USE_NOISE>(Dimensions(size, size, 1, n), batch);
        // bigger
        size = ((int)dist2(mt) / 2) * 2;
        test<USE_NOISE>(Dimensions(size, size, 1, n), batch);
    }

    template<bool ADD_NOISE>
    void test(const Dimensions &dims, size_t batch) {
        using namespace Alignment;

        bool saveOutput = false && (dims.x() == 260 && dims.y() == 260 && batch == 1);
//        printf("sizes: %lu %lu %lu %lu, batch %lu\n", dims.x(), dims.y(), dims.z(), dims.n(), batch);

        auto maxShift = std::min((size_t)20, getMaxShift(dims));
        auto shifts = generateShifts(dims, maxShift, mt);
        auto maxRotation = getMaxRotation();
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
        IterativeAlignmentEstimatorHelper<T>::applyTransform(
                dims, shifts, rotations, ref, others);

        if (ADD_NOISE) {
            // add noise to data
            // the reference should be without noise
            addNoise(others, dims, mt_noise);
        }
        // show data
        if (saveOutput) {
            outputData(others, dims, "dataBeforeAlignment.stk");
        }
        // prepare aligner
        auto rotSettings = Alignment::RotationEstimationSetting();
        rotSettings.hw = hw;
        rotSettings.type = AlignType::OneToN;
        rotSettings.refDims = dims.createSingle();
        rotSettings.otherDims = dims;
        rotSettings.batch = batch;
        rotSettings.maxRotDeg = maxRotation;
        rotSettings.fullCircle = true;

        shiftAligner->init2D(hw, AlignType::OneToN, FFTSettingsNew<T>(dims, batch), maxShift, true, true);
        rotationAligner->init(rotSettings, true);
        auto aligner = IterativeAlignmentEstimator<T>(*rotationAligner, *shiftAligner);

        auto result = aligner.compute(ref, others, 3); // use at least three iterations

        // show result
        if (saveOutput) {
            IterativeAlignmentEstimatorHelper<T>::compensateTransform(
                    result, dims, others, othersCorrected);
            outputData(othersCorrected, dims, "dataAfterAlignment.stk");
            // show statistics
            outputStatistics(dims, shifts, rotations, result, ref, others);
        }


        checkResults<ADD_NOISE>(result, shifts, rotations, dims);

        if (saveOutput) {
            delete[] othersCorrected;
        }
    }


private:
    static std::mt19937 mt;
    static std::mt19937 mt_noise;
    static Alignment::ARotationEstimator<T> *rotationAligner;
    static Alignment::AShiftCorrEstimator<T> *shiftAligner;
    static std::vector<HW*> hw;
    static T *ref;
    static T *others;
    static Dimensions maxDims;

    template<bool isWithNoise>
    void checkResults(const Alignment::AlignmentEstimation &est,
            const std::vector<Point2D<float>> &shifts,
            const std::vector<float> &rotations,
            const Dimensions &dims) {
        ASSERT_EQ(dims.n(), est.poses.size());

        if (isWithNoise
            && (260 == dims.x())
            && (260 == dims.y())
            && (1 == dims.n())) {
            // 'special case', the alignment is simply wrong, so skip the evaluation
            return;
        }

        auto rotErrors = std::vector<float>(dims.n());
        auto shiftXErrors = std::vector<float>(dims.n());
        auto shiftYErrors = std::vector<float>(dims.n());

        for (size_t i = 0;i < dims.n();++i) {
            auto sE = shifts.at(i);
            auto rE = rotations.at(i);
            auto m = est.poses.at(i);
            auto sA = Point2D<float>(-MAT_ELEM(m, 0, 2), -MAT_ELEM(m, 1, 2));
            auto rA = fmod(360 + RAD2DEG(atan2(MAT_ELEM(m, 1, 0), MAT_ELEM(m, 0, 0))), 360);
            rotErrors.at(i) = 180 - abs(abs(rA - rE) - 180);
            shiftXErrors.at(i) = std::abs(sA.x - sE.x);
            shiftYErrors.at(i) = std::abs(sA.y - sE.y);
        }
        std::sort(rotErrors.begin(), rotErrors.end());
        std::sort(shiftXErrors.begin(), shiftXErrors.end());
        std::sort(shiftYErrors.begin(), shiftYErrors.end());

        float x0 = 32; // small size
        float x1 = 256; // big size
        float y0 = 3; // coeff for small sizes (i.e. we are more relaxed)
        float y1 = 0.85f; // coeff for big sizes (i.e. we are more strict)
        float x2 = dims.x(); // assuming square inputs
        float sizeCoeff = (std::abs(x1 - x2) * y0 + (x2 - x0) * y1) / (x1 - x0);

        float medianMaxRotError = RAD2DEG(atan(2.0 / dims.x())) * 2.f * sizeCoeff; // degrees per one pixel. We allow for two pixels error
        float medianMaxShiftError = 0.9f * sizeCoeff;
        float percMaxRotError = 5.f * medianMaxRotError * sizeCoeff;
        float percMaxShiftError = 3.5f * medianMaxShiftError * sizeCoeff;

        if (isWithNoise) {
            medianMaxRotError = RAD2DEG(atan(2.0 / dims.x())) * 3 * sizeCoeff; // degrees per one pixel. We allow for three pixels error
            medianMaxShiftError = 1.2 * sizeCoeff;
            percMaxRotError = 90 * sizeCoeff;
            percMaxShiftError = 20 * medianMaxShiftError * sizeCoeff;
        }
        size_t medPosition = (1 == dims.n()) ? 0 : std::floor((dims.n() - 1) * 0.5f);
        size_t percPosition = (1 == dims.n()) ? 0 : std::floor((dims.n() - 1) * 0.95f);
        // test median
        ASSERT_LE(shiftXErrors.at(medPosition), medianMaxShiftError) << "median is over limit";
        ASSERT_LE(shiftYErrors.at(medPosition), medianMaxShiftError) << "median is over limit";
        ASSERT_LE(rotErrors.at(medPosition), medianMaxRotError) << "median is over limit";
        // test 95 percentil
        ASSERT_LE(shiftXErrors.at(percPosition), percMaxShiftError) << "95 percentil is over limit";
        ASSERT_LE(shiftYErrors.at(percPosition), percMaxShiftError) << "95 percentil is over limit";
        ASSERT_LE(rotErrors.at(percPosition), percMaxRotError) << "95 percentil is over limit";

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
                    sE.x, sE.y, rE, sA.x, sA.y, rA, result.correlations.at(i));
                    // original version
            size_t offset = i * dims.xyzPadded();
            double corr = std::numeric_limits<double>::lowest();
            auto M = getReferenceTransform(ref, others + offset, dims, corr);
            auto sR = Point2D<float>(-MAT_ELEM(M, 0, 2), -MAT_ELEM(M, 1, 2));
            auto rR = fmod(360 + RAD2DEG(atan2(MAT_ELEM(M, 1, 0), MAT_ELEM(M, 0, 0))), 360);
            printf("%f | %f | %f | %f ||",// orig
                    sR.x, sR.y, rR, corr);
            // comparison GT <-> new
            printf("%f | %f | %f | %f||", std::abs(sE.x - sA.x), std::abs(sE.y - sA.y), 180 - std::abs((std::abs(rA - rE) - 180)), result.correlations.at(i));
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
Dimensions IterativeAlignmentEstimator_Test<T>::maxDims(768, 768, 1, 1000);
template<typename T>
T *IterativeAlignmentEstimator_Test<T>::ref = nullptr;
template<typename T>
T *IterativeAlignmentEstimator_Test<T>::others = nullptr;
template<typename T>
std::vector<HW*> IterativeAlignmentEstimator_Test<T>::hw;
template<typename T>
std::mt19937 IterativeAlignmentEstimator_Test<T>::mt(42); // fixed seed to ensure reproducibility
template<typename T>
std::mt19937 IterativeAlignmentEstimator_Test<T>::mt_noise(23); // fixed seed to ensure reproducibility

//TYPED_TEST_P( IterativeAlignmentEstimator_Test, debug)
//{
//    XMIPP_TRY
//    auto dims = Dimensions(256, 256, 1, 50);
////    auto dims = Dimensions(64, 64, 1, 50);
//    size_t batch = 1;
//    IterativeAlignmentEstimator_Test<TypeParam>::template test<true>(dims, batch);
//    IterativeAlignmentEstimator_Test<TypeParam>::template test<false>(dims, batch);
//    XMIPP_CATCH
//}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToOneNoNoise)
{
    XMIPP_TRY
    // test one reference vs one image
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTest2D<false>(1, 1);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoNoiseNoBatch)
{
    XMIPP_TRY
    // test one reference vs one image
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTest2D<false>(100, 1);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoNoiseBatch)
{
    XMIPP_TRY
    // test one reference vs one image
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTest2D<false>(100, 50);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyNoNoiseBatchNotMultiple)
{
    XMIPP_TRY
    // test one reference vs one image
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTest2D<false>(100, 60);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToOneWithNoise)
{
    XMIPP_TRY
    // test one reference vs one image
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTest2D<true>(1, 1);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToManyWithNoiseBatch)
{
    XMIPP_TRY
    // test one reference vs one image
    IterativeAlignmentEstimator_Test<TypeParam>::template generateAndTest2D<true>(100, 50);
    XMIPP_CATCH
}

REGISTER_TYPED_TEST_CASE_P(IterativeAlignmentEstimator_Test,
//    debug
    align2DOneToOneNoNoise,
    align2DOneToManyNoNoiseNoBatch,
    align2DOneToManyNoNoiseBatch,
    align2DOneToManyNoNoiseBatchNotMultiple,
    align2DOneToOneWithNoise,
    align2DOneToManyWithNoiseBatch
);

