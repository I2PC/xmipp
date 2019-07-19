#include <gtest/gtest.h>
#include "alignment_test_utils.h"
#include "reconstruction/iterative_alignment_estimator.h"
#include "reconstruction/polar_rotation_estimator.h"
#include "reconstruction/shift_corr_estimator.h"


template<typename T>
class IterativeAlignmentEstimatorHelper : public Alignment::IterativeAlignmentEstimator<T> {
public:
    static void applyTransformSingle(const Dimensions &dims, const Point2D<float> &shift, float rotation,
                __restrict const T *orig, __restrict T *copy) {
            Alignment::AlignmentEstimation e(1);
            auto &m = e.poses.at(0);
            MAT_ELEM(m,0,2) += shift.x;
            MAT_ELEM(m,1,2) += shift.y;
            auto r = Matrix2D<double>();
            rotation2DMatrix(rotation, r);
            m = r * m;
            Alignment::IterativeAlignmentEstimator<T>::sApplyTransform(dims, e, orig, copy);
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
        auto ref = new T[dims.xy()]();
        T centerX = dims.x() / 2;
        T centerY = dims.y() / 2;
        drawClockArms(ref, dims, centerX, centerY, 0.f);

        for (size_t n = 0; n < dims.n(); ++n) {
            T *d = others + (n * dims.xyzPadded());
            IterativeAlignmentEstimatorHelper<T>::applyTransformSingle(
                    dims.createSingle(), shifts.at(n), rotations.at(n), ref, d);
        }
//        outputData(others, dims);

        // prepare aligner
        auto cpu = CPU();
        auto shiftAligner = ShiftCorrEstimator<T>();
        auto rotationAligner = PolarRotationEstimator<T>();
        shiftAligner.init2D(cpu, AlignType::OneToN, FFTSettingsNew<T>(dims, batch), maxShift, true, true);
        rotationAligner.init(cpu, AlignType::OneToN, dims, 1, maxRotation); // FIXME DS add test that batch is 1
        auto aligner = IterativeAlignmentEstimator<T>(rotationAligner, shiftAligner);

        auto result  = aligner.compute(ref, others);

        for (size_t i = 0; i < dims.n(); ++i) {
            auto sE = shifts.at(i);
            auto rE = rotations.at(i);
            auto m = result.poses.at(i);
            auto sA = Point2D<float>(-MAT_ELEM(m, 0, 2), -MAT_ELEM(m, 1, 2));
            auto rA = fmod(360 + RAD2DEG(atan2(MAT_ELEM(m, 1, 0), MAT_ELEM(m, 0, 0))), 360);

//            printf("exp: | %f | %f | %f | act: | %f | %f | %f ",
//                    sE.x, sE.y, rE,
//                    sA.x, sA.y, rA
//            );

//            size_t offset = i * dims.xyzPadded();
//            auto M = getReferenceTransform(ref, others + offset, dims);
//            auto sR = Point2D<float>(-MAT_ELEM(M, 0, 2), -MAT_ELEM(M, 1, 2));
//            auto rR = fmod(360 + RAD2DEG(atan2(MAT_ELEM(M, 1, 0), MAT_ELEM(M, 0, 0))), 360);
//            printf("| ref: | %f | %f | %f\n",
//                    sR.x, sR.y, rR);

//            printf("\n");
        }

        delete[] ref;
        delete[] others;
    }


private:
    static std::mt19937 mt;

    void outputData(T *data, const Dimensions &dims) {
        MultidimArray<T>wrapper(dims.n(), dims.z(), dims.y(), dims.x(), data);
        Image<T> img(wrapper);
        img.write("data.stk");
    }

    Matrix2D<double> getReferenceTransform(T *ref, T *other, const Dimensions &dims) {
        Matrix2D<double> M;
        auto I = convert(other, dims);
        I.setXmippOrigin();
        auto refWrapper = convert(ref, dims);
        refWrapper.setXmippOrigin();

        alignImages(refWrapper, I, M, DONT_WRAP);
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


TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToOne)
{
    XMIPP_TRY
    // test one reference vs one image
    IterativeAlignmentEstimator_Test<TypeParam>::generateAndTest2D(1, 1);
    XMIPP_CATCH
}

TYPED_TEST_P( IterativeAlignmentEstimator_Test, align2DOneToMany)
{
    XMIPP_TRY
    // test one reference vs one image
    IterativeAlignmentEstimator_Test<TypeParam>::generateAndTest2D(100, 50);
    XMIPP_CATCH
}

REGISTER_TYPED_TEST_CASE_P(IterativeAlignmentEstimator_Test,
    align2DOneToOne,
    align2DOneToMany
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(, IterativeAlignmentEstimator_Test, TestTypes);
