#include <gtest/gtest.h>
#include <random>
#include "reconstruction_cuda/cuda_fft.h"

template<typename T>
class CudaFFTTest : public ::testing::Test { };
TYPED_TEST_CASE_P(CudaFFTTest);

template<typename T>
void testFTInpulseShifted(const FFTSettingsNew<T> &s) {
    using std::complex;
    auto in = new T[s.sDim().size()]();
    auto out = new complex<T>[s.fDim().size()]();

    for (size_t n = 0; n < s.sDim().n(); ++n) {
        // shifted impulse ...
        in[n * s.sDim().xyz() + 1] = T(1);
    }

    CudaFFT<T> ft;
    ft.init(s);
    ft.fft(in, out);

    T delta = (T)0.00001;
    for (size_t i = 0; i < s.fDim().size(); ++i) {
        // ... will result in constant magnitude
        T re = out[i].real();
        T im = out[i].imag();
        T mag = (re * re) + (im * im);
        EXPECT_NEAR((T)1, std::sqrt(mag), delta);
    }

    delete[] in;
    delete[] out;
}

template<typename T>
void testFTInpulseOrigin(const FFTSettingsNew<T> &s) {
    using std::complex;
    auto in = new T[s.sDim().size()]();
    auto out = new complex<T>[s.fDim().size()]();

    for (size_t n = 0; n < s.sDim().n(); ++n) {
        // impulse at the origin ...
        in[n * s.sDim().xyz()] = T(1);
    }

    CudaFFT<T> ft;
    ft.init(s);
    ft.fft(in, out);

    T delta = (T)0.00001;
    for (size_t i = 0; i < s.fDim().size(); ++i) {
        // ... will result in constant real value, and no imag value
        EXPECT_NEAR((T)1, out[i].real(), delta);
        EXPECT_NEAR((T)0, out[i].imag(), delta);
    }

    delete[] in;
    delete[] out;
}



TYPED_TEST_P( CudaFFTTest, fft1DSingleOOP)
{
    // test a forward transform of a single 1D signal, out of place
    auto len = std::vector<size_t>{8, 15, 32, 42, 106, 2048, 2049};
    for (auto l : len) {
        testFTInpulseOrigin(FFTSettingsNew<TypeParam>(l));
        testFTInpulseShifted(FFTSettingsNew<TypeParam>(l));
    }
}

TYPED_TEST_P( CudaFFTTest, fft1DManyOOP)
{
    // test a forward transform of many 1D signals, out of place
    // check that n == batch works properly
    auto batches = std::vector<size_t>{1, 3, 4, 11, 12};
    auto len = std::vector<size_t>{8, 15, 32, 42, 106, 2048, 2049};
    for (auto b : batches) {
        for (auto l : len) {
            testFTInpulseOrigin(FFTSettingsNew<TypeParam>(l, 1, 1, b, b));
            testFTInpulseShifted(FFTSettingsNew<TypeParam>(l, 1, 1, b, b));
        }
    }
}

TYPED_TEST_P( CudaFFTTest, fft1DManyBatched1OOP)
{
    // test a forward transform of many 1D signals, out of place
    // test that n mod batch != 0 works
    auto sizes = std::vector<size_t>{3, 4, 5, 7};
    auto batches = std::vector<size_t>{2, 3, 4, 11, 12};
    auto len = std::vector<size_t>{8, 15, 32, 42, 106, 2048, 2049};
    for (auto s : sizes ) {
        for (auto b : batches) {
            if (b > s) continue;
            for (auto l : len) {
                testFTInpulseOrigin(FFTSettingsNew<TypeParam>(l, 1, 1, b, b));
                testFTInpulseShifted(FFTSettingsNew<TypeParam>(l, 1, 1, b, b));
            }
        }
    }
}

TYPED_TEST_P( CudaFFTTest, fft1DManyBatched2OOP)
{
    // test a forward transform of many 1D signals, out of place
    // test that n mod batch = 0 works
    auto sizes = std::vector<size_t>{3, 4, 5, 6, 7, 9, 16, 22};
    auto batches = std::vector<size_t>{2, 3, 4, 11};
    auto len = std::vector<size_t>{8, 15, 32, 42, 106, 2048, 2049};
    for (auto s : sizes ) {
        for (auto b : batches) {
            if (0 != (s % b)) continue;
            for (auto l : len) {
                testFTInpulseOrigin(FFTSettingsNew<TypeParam>(l, 1, 1, b, b));
                testFTInpulseShifted(FFTSettingsNew<TypeParam>(l, 1, 1, b, b));
            }
        }
    }
}

REGISTER_TYPED_TEST_CASE_P(CudaFFTTest,
    fft1DSingleOOP,
    fft1DManyOOP,
    fft1DManyBatched1OOP,
    fft1DManyBatched2OOP
);

typedef ::testing::Types<float> TestTypes; // FIXME add double
INSTANTIATE_TYPED_TEST_CASE_P(Cuda, CudaFFTTest, TestTypes);
