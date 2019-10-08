#include <gtest/gtest.h>
#include "alignment_test_utils.h"
#include "reconstruction/iterative_alignment_estimator.h"
#include "reconstruction/polar_rotation_estimator.h"
#include "reconstruction/shift_corr_estimator.h"


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
    void generateAndTest2D(size_t n, size_t batch) {
        std::uniform_int_distribution<> dist1(0, 368);
        std::uniform_int_distribution<> dist2(369, 768);

        // only even inputs are valid
        // smaller
        size_t size = ((int)dist1(mt) / 2) * 2;
        test(Dimensions(size, size, 1, n), batch);
        // bigger
        size = ((int)dist2(mt) / 2) * 2;
        test(Dimensions(size, size, 1, n), batch);
    }

    void test(const Dimensions &dims, size_t batch) {
        using namespace Alignment;

        auto maxShift = std::min((size_t)20, getMaxShift(dims));
        auto shifts = generateShifts(dims, maxShift, mt);
        auto maxRotation = getMaxRotation();
        auto rotations = generateRotations(dims, maxRotation, mt);

        auto others = new T[dims.size()]();
        auto othersCorrected = new T[dims.size()]();
        auto ref = new T[dims.xy()]();
        T centerX = dims.x() / 2;
        T centerY = dims.y() / 2;
        // generate data
        drawClockArms(ref, dims, centerX, centerY, 0.f);
        IterativeAlignmentEstimatorHelper<T>::applyTransform(
                dims, shifts, rotations, ref, others);

        // add noise to data
        // the reference should be without noise
        addNoise(others, dims, mt_noise);
        outputData(others, dims, "dataBeforeAlignment.stk");

        // prepare aligner
        auto cpu = CPU();
        auto hw = std::vector<HW*>{&cpu};
        auto rotSettings = Alignment::RotationEstimationSetting();
        rotSettings.hw = hw;
        rotSettings.type = AlignType::OneToN;
        rotSettings.refDims = dims.createSingle();
        rotSettings.otherDims = dims;
        rotSettings.batch = batch;
        rotSettings.maxRotDeg = maxRotation;
        rotSettings.fullCircle = true;

        auto shiftAligner = ShiftCorrEstimator<T>();
        auto rotationAligner = PolarRotationEstimator<T>();
        shiftAligner.init2D(hw, AlignType::OneToN, FFTSettingsNew<T>(dims, batch), maxShift, true, true);
        rotationAligner.init(rotSettings, false); // FIXME DS set reuse to true
        auto aligner = IterativeAlignmentEstimator<T>(rotationAligner, shiftAligner);

        auto result = aligner.compute(ref, others, 5);

        IterativeAlignmentEstimatorHelper<T>::compensateTransform(
                result, dims, others, othersCorrected);
        outputData(othersCorrected, dims, "dataAfterAlignment.stk");

        printf("Ground truth|shiftX|shiftY|rot|"
                "New version (CPU)|shiftX|shiftY|rot|Correlation|"
                "Orig version(up to 3 iter, CPU)|shiftX|shiftY|rot|Correlation||"
                "shiftX (GT-new)|shiftY(GT-new)|rot(GT-new)|Correlation||"
                "shiftX (GT-old)|shiftY(GT-old)|rot(GT-old)|Correlation\n");

        for (size_t i = 0; i < dims.n(); ++i) {
            auto sE = shifts.at(i);
            auto rE = rotations.at(i);
            auto m = result.poses.at(i);
            auto sA = Point2D<float>(-MAT_ELEM(m, 0, 2), -MAT_ELEM(m, 1, 2));
            auto rA = fmod(360 + RAD2DEG(atan2(MAT_ELEM(m, 1, 0), MAT_ELEM(m, 0, 0))), 360);
            // ground truth, new version
            printf("| %f | %f | %f ||" // GT
                   "%f | %f | %f | %f ||", // new
                    sE.x, sE.y, rE,
                    sA.x, sA.y, rA, result.correlations.at(i)
            );
            // original version
            size_t offset = i * dims.xyzPadded();
            double corr = std::numeric_limits<double>::lowest();
            auto M = getReferenceTransform(ref, others + offset, dims, corr);
            auto sR = Point2D<float>(-MAT_ELEM(M, 0, 2), -MAT_ELEM(M, 1, 2));
            auto rR = fmod(360 + RAD2DEG(atan2(MAT_ELEM(M, 1, 0), MAT_ELEM(M, 0, 0))), 360);
            printf("%f | %f | %f | %f ||", // orig
                    sR.x, sR.y, rR, corr);
            // comparison GT <-> new
            printf("%f | %f | %f | %f||",
                std::abs(sE.x - sA.x),
                std::abs(sE.y - sA.y),
                180 - std::abs((std::abs(rA - rE) - 180)),
                result.correlations.at(i));
            // comparison GT <-> orig
            printf("%f | %f | %f | %f\n",
                std::abs(sE.x - sR.x),
                std::abs(sE.y - sR.y),
                180 - std::abs((std::abs(rR - rE) - 180)),
                corr);

        }

        delete[] ref;
        delete[] others;
        delete[] othersCorrected;
    }


private:
    static std::mt19937 mt;
    static std::mt19937 mt_noise;

    void outputData(T *data, const Dimensions &dims, const std::string &name) {
        MultidimArray<T>wrapper(dims.n(), dims.z(), dims.y(), dims.x(), data);
        Image<T> img(wrapper);
        img.write(name);
    }

    Matrix2D<double> getReferenceTransform(T *ref, T *other, const Dimensions &dims, double &corr) {
        Matrix2D<double> M;
        auto I = convert(other, dims);
        I.setXmippOrigin();
        auto refWrapper = convert(ref, dims);
        refWrapper.setXmippOrigin();

        corr = alignImages(refWrapper, I, M, DONT_WRAP);
        return M;
    }

    MultidimArray<double> convert(float *data, const Dimensions &dims) {
        auto wrapper = MultidimArray<double>(1, 1, dims.y(), dims.x());
        for (size_t i = 0; i < dims.xyz(); ++i) {
            wrapper.data[i] = data[i];
        }
        return wrapper;
    }

    MultidimArray<double> convert(double *data, const Dimensions &dims) {
        return MultidimArray<double>(1, 1, dims.y(), dims.x(), data);
    }

};
TYPED_TEST_CASE_P(IterativeAlignmentEstimator_Test);

template<typename T>
std::mt19937 IterativeAlignmentEstimator_Test<T>::mt(42); // fixed seed to ensure reproducibility
template<typename T>
std::mt19937 IterativeAlignmentEstimator_Test<T>::mt_noise(23); // fixed seed to ensure reproducibility

TYPED_TEST_P( IterativeAlignmentEstimator_Test, debug)
{
    XMIPP_TRY
    //auto dims = Dimensions(256, 256, 1, 50);
    auto dims = Dimensions(64, 64, 1, 50);
    size_t batch = 1;
    IterativeAlignmentEstimator_Test<TypeParam>::test(dims, batch);
    XMIPP_CATCH
}

//TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToOne)
//{
//    XMIPP_TRY
//    // test one reference vs one image
//    IterativeAlignmentEstimator_Test<TypeParam>::generateAndTest2D(1, 1);
//    XMIPP_CATCH
//}
//
//TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToMany)
//{
//    XMIPP_TRY
//    // test one reference vs one image
//    IterativeAlignmentEstimator_Test<TypeParam>::generateAndTest2D(100, 50);
//    XMIPP_CATCH
//}

REGISTER_TYPED_TEST_CASE_P(IterativeAlignmentEstimator_Test,
    debug
//    align2DOneToOne,
//    align2DOneToMany
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(, IterativeAlignmentEstimator_Test, TestTypes);
