#include <gtest/gtest.h>
#include <random>
#include <set>
#include "core/utils/memory_utils.h"
#include "reconstruction_adapt_cuda/gpu.h"
#include "reconstruction_cuda/cuda_fft.h"

template<typename T>
class CudaFFTTest : public ::testing::Test { };
TYPED_TEST_CASE_P(CudaFFTTest);

template<typename T>
void testFTInpulseShifted(const FFTSettingsNew<T> &s) {
    using std::complex;

    // this test needs at least two elements in X dim
    if (s.sDim().x() == 1) return;

    auto in = new T[s.sDim().sizePadded()]();
    complex<T> *out;
    if (s.isInPlace()) {
        out = (std::complex<T>*)in;
    } else {
        out = new complex<T>[s.fDim().sizePadded()]();
    }

    for (size_t n = 0; n < s.sDim().n(); ++n) {
        // shifted impulse ...
        in[n * s.sDim().xyzPadded() + 1] = T(1);
    }

    CudaFFT<T> ft;
    ft.init(s);
    ft.fft(in, out);

//    printf("\nShifted:\n");
    T delta = (T)0.00001;
    for (size_t i = 0; i < s.fDim().size(); ++i) {
        // ... will result in constant magnitude
        T re = out[i].real();
        T im = out[i].imag();
        T mag = (re * re) + (im * im);
        EXPECT_NEAR((T)1, std::sqrt(mag), delta);
//        printf("%f %f\n", out[i].real(), out[i].imag());
    }

    delete[] in;
    if ((void*)in != (void*)out) {
        delete[] out;
    }
}

template<typename T>
void testFTInpulseOrigin(const FFTSettingsNew<T> &s) {
    using std::complex;
    auto in = new T[s.sDim().sizePadded()]();
    complex<T> *out;
    if (s.isInPlace()) {
        out = (std::complex<T>*)in;
    } else {
        out = new complex<T>[s.fDim().sizePadded()]();
    }

    for (size_t n = 0; n < s.sDim().n(); ++n) {
        // impulse at the origin ...
        in[n * s.sDim().xyzPadded()] = T(1);
    }

    CudaFFT<T> ft;
    ft.init(s);
    ft.fft(in, out);

//    printf("\nOrigin:\n");
    T delta = (T)0.00001;
    for (size_t i = 0; i < s.fDim().size(); ++i) {
        // ... will result in constant real value, and no imag value
        EXPECT_NEAR((T)1, out[i].real(), delta);
        EXPECT_NEAR((T)0, out[i].imag(), delta);
//        printf("%f %f\n", out[i].real(), out[i].imag());
    }

    delete[] in;
    if ((void*)in != (void*)out) {
        delete[] out;
    }
}

template<typename T, typename F>
void generateAndTest(F condition) {
    using namespace memoryUtils;
    size_t executed = 0;
    size_t skippedSize = 0;
    size_t skippedCondition = 0;
    auto batch = std::vector<size_t>{1, 2, 3, 5, 6, 7, 8, 10, 23};
    auto nSet = std::vector<size_t>{1, 2, 4, 5, 6, 8, 10, 12, 14, 23, 24};
    auto zSet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 2048, 2049};
    auto ySet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 2048, 2049};
    auto xSet = std::vector<size_t>{1, 2, 3, 8, 15, 32, 42, 106, 2048, 2049};
    size_t combinations = batch.size() * nSet.size() * zSet.size() * ySet.size() * xSet.size() * 4;

    auto settingsComparator = [] (const FFTSettingsNew<T> &l, const FFTSettingsNew<T> &r) {
      return ((l.sDim().x() >= r.sDim().x())
              && (l.sDim().y() >= r.sDim().y())
              && (l.sDim().z() >= r.sDim().z())
              && (l.sDim().n() >= r.sDim().n())
              && (l.batch() >= r.batch()));
    };
    auto tested = std::set<FFTSettingsNew<T>,decltype(settingsComparator)>(settingsComparator);

    int seed = 42;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<> dist(0, 4097);
    GPU gpu(0);
    T availableMem = gpu.lastFreeMem();
    while ((executed < 20)
            && ((skippedCondition + skippedSize) < combinations)) { // avoid endless loop
        size_t x = xSet.at(dist(mt) % xSet.size());
        size_t y = ySet.at(dist(mt) % ySet.size());
        size_t z = zSet.at(dist(mt) % zSet.size());
        size_t n = nSet.at(dist(mt) % nSet.size());
        size_t b = batch.at(dist(mt) % batch.size());
        if (b > n) continue; // batch must be smaller than n
        bool inPlace = dist(mt) % 2;
        bool isForward = dist(mt) % 2;
        auto settings = FFTSettingsNew<T>(x, y, z, n, b, inPlace, isForward);
        if (condition(x, y, z, n, b, inPlace, isForward)) {
            // make sure we have enough memory
            T size = MB(CudaFFT<T>::estimatePlanSize(settings)) + MB(settings.maxBytesBatch());
            if (availableMem < size) {
                skippedSize++;
                continue;
            }
            // make sure we did not test this before
            auto result = tested.insert(settings);
            if ( ! result.second) continue;

            printf("Testing %lu %lu %lu %lu %lu %s %s\n",
                    x, y, z, n, b, inPlace ? "inPlace" : "outOfPlace", isForward ? "fft" : "ifft");
            testFTInpulseOrigin(settings);
            testFTInpulseShifted(settings);
            executed++;
        } else {
            skippedCondition++;
        }
    }
//    std::cout << "Executed: " << executed
//            << "\nSkipped (condition): " << skippedCondition
//            << "\nSkipped (size):" << skippedSize << std::endl;
}

auto is1D = [] (size_t x, size_t y, size_t z) {
    return (z == 1) && (y == 1);
};

auto is2D = [] (size_t x, size_t y, size_t z) {
    return z == 1;
};

auto isBatchMultiple = [] (size_t n, size_t batch) {
    return 0 == (n % batch);
};

//*******************************************
//              1D out of place FFT tests
//*******************************************

TYPED_TEST_P( CudaFFTTest, fft_OOP_1D_Single)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward , out of place transform of a single 1D signal,
        return isForward && (!inPlace) && is1D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_1D_Batch1)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward , out of place transform of many 1D signals,
        // check that n == batch works properly
        return isForward && (!inPlace) && is1D(x, y, z) && (n == batch);
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_1D_Batch2)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward transform of many 1D signals, out of place
        // test that n mod batch != 0 works
        return isForward && (!inPlace) && is1D(x, y, z) && (!isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_1D_Batch3)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward transform of many 1D signals, out of place
        // test that n mod batch = 0 works
        return isForward && (!inPlace) && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
}

//*******************************************
//              1D in place FFT tests
//*******************************************

TYPED_TEST_P( CudaFFTTest, fft_IP_1D_Single)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward , out of place transform of a single 1D signal,
        return isForward && (inPlace) && is1D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_1D_Batch1)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward , out of place transform of many 1D signals,
        // check that n == batch works properly
        return isForward && (inPlace) && is1D(x, y, z) && (n == batch);
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_1D_Batch2)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward transform of many 1D signals, out of place
        // test that n mod batch != 0 works
        return isForward && (inPlace) && is1D(x, y, z) && (!isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_1D_Batch3)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward transform of many 1D signals, out of place
        // test that n mod batch = 0 works
        return isForward && (inPlace) && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
}

//*******************************************
//              2D out of place FFT tests
//*******************************************

TYPED_TEST_P( CudaFFTTest, fft_OOP_2D_Single)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward , out of place transform of a single 1D signal,
        return isForward && (!inPlace) && is2D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_2D_Batch1)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward , out of place transform of many 1D signals,
        // check that n == batch works properly
        return isForward && (!inPlace) && is2D(x, y, z) && (n == batch);
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_2D_Batch2)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward transform of many 1D signals, out of place
        // test that n mod batch != 0 works
        return isForward && (!inPlace) && is2D(x, y, z) && (!isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_2D_Batch3)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward transform of many 1D signals, out of place
        // test that n mod batch = 0 works
        return isForward && (!inPlace) && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
}

//*******************************************
//              2D in place FFT tests
//*******************************************

TYPED_TEST_P( CudaFFTTest, fft_IP_2D_Single)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward , out of place transform of a single 1D signal,
        return isForward && (inPlace) && is2D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_2D_Batch1)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward , out of place transform of many 1D signals,
        // check that n == batch works properly
        return isForward && (inPlace) && is2D(x, y, z) && (n == batch);
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_2D_Batch2)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward transform of many 1D signals, out of place
        // test that n mod batch != 0 works
        return isForward && (inPlace) && is2D(x, y, z) && (!isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_2D_Batch3)
{
    auto condition = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        // test a forward transform of many 1D signals, out of place
        // test that n mod batch = 0 works
        return isForward && (inPlace) && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
}

REGISTER_TYPED_TEST_CASE_P(CudaFFTTest,
    fft_OOP_1D_Single,
    fft_OOP_1D_Batch1,
    fft_OOP_1D_Batch2,
    fft_OOP_1D_Batch3,

    fft_IP_1D_Single,
    fft_IP_1D_Batch1,
    fft_IP_1D_Batch2,
    fft_IP_1D_Batch3,

    fft_OOP_2D_Single,
    fft_OOP_2D_Batch1,
    fft_OOP_2D_Batch2,
    fft_OOP_2D_Batch3,

    fft_IP_2D_Single,
    fft_IP_2D_Batch1,
    fft_IP_2D_Batch2,
    fft_IP_2D_Batch3
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cuda, CudaFFTTest, TestTypes);
