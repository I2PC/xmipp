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
void testFFTInpulseShifted(const FFTSettingsNew<T> &s) {
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

    auto gpu = GPUNew();
    gpu.set();
    auto ft = CudaFFT<T>();
    ft.init(gpu, s);
    ft.fft(in, out);
    gpu.synch();

    T delta = (T)0.00001;
    for (size_t i = 0; i < s.fDim().size(); ++i) {
        // ... will result in constant magnitude
        T re = out[i].real();
        T im = out[i].imag();
        T mag = (re * re) + (im * im);
        ASSERT_NEAR((T)1, std::sqrt(mag), delta);
    }

    delete[] in;
    if ((void*)in != (void*)out) {
        delete[] out;
    }
}

template<typename T>
void testFFTInpulseOrigin(const FFTSettingsNew<T> &s) {
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

    auto gpu = GPUNew();
    gpu.set();
    auto ft = CudaFFT<T>();
    ft.init(gpu, s);
    ft.fft(in, out);
    gpu.synch();

    T delta = (T)0.00001;
    for (size_t i = 0; i < s.fDim().size(); ++i) {
        // ... will result in constant real value, and no imag value
        ASSERT_NEAR((T)1, out[i].real(), delta);
        ASSERT_NEAR((T)0, out[i].imag(), delta);
    }

    delete[] in;
    if ((void*)in != (void*)out) {
        delete[] out;
    }
}

template<typename T>
void testIFFTInpulseOrigin(const FFTSettingsNew<T> &s) {
    using std::complex;

    auto in = new complex<T>[s.fDim().sizePadded()]();
    T *out;
    if (s.isInPlace()) {
        out = (T*)in;
    } else {
        out = new T[s.sDim().sizePadded()]();
    }

    for (size_t n = 0; n < s.fDim().sizePadded(); ++n) {
        // constant real value, and no imag value ...
        in[n] = {T(1), 0};
    }

    auto gpu = GPUNew();
    gpu.set();
    auto ft = CudaFFT<T>();
    ft.init(gpu, s);
    ft.ifft(in, out);
    gpu.synch();

    T delta = (T)0.0001;
    for (size_t n = 0; n < s.sDim().n(); ++n) {
        size_t offset = n * s.sDim().xyzPadded();
        // skip the padded area, it can contain garbage data
        for (size_t z = 0; z < s.sDim().z(); ++z) {
            for (size_t y = 0; y < s.sDim().y(); ++y) {
                for (size_t x = 0; x < s.sDim().x(); ++x) {
                    size_t index = offset + z * s.sDim().xyPadded() + y * s.sDim().xPadded() + x;
                    // output is not normalized, so normalize it to make the the test more stable
                    if (index == offset) {
                        // ... will result in impulse at the origin ...
                        ASSERT_NEAR((T)1, out[index] / s.sDim().xyz(), delta);
                    } else {
                        // ... and zeros elsewhere
                        ASSERT_NEAR((T)0, out[index] / s.sDim().xyz(), delta);
                    }
                }
            }
        }
    }

    delete[] in;
    if ((void*)in != (void*)out) {
        delete[] out;
    }
}

template<typename T>
void testFFTIFFT(const FFTSettingsNew<T> &s) {
    using std::complex;

    // allocate data
    auto inOut = new T[s.sDim().sizePadded()]();
    auto ref = new T[s.sDim().sizePadded()]();
    complex<T> *fd;
    if (s.isInPlace()) {
        fd = (complex<T>*)inOut;
    } else {
        fd = new complex<T>[s.fDim().sizePadded()]();
    }

    // generate random content
    int seed = 42;
    std::mt19937 mt(seed);
    std::uniform_real_distribution<> dist(-1, 1.1);
    for (size_t n = 0; n < s.sDim().n(); ++n) {
        size_t offset = n * s.sDim().xyzPadded();
        // skip the padded area, it can contain garbage data
        for (size_t z = 0; z < s.sDim().z(); ++z) {
            for (size_t y = 0; y < s.sDim().y(); ++y) {
                for (size_t x = 1; x < s.sDim().x(); ++x) {
                    size_t index = offset + z * s.sDim().xyPadded() + y * s.sDim().xPadded() + x;
                    inOut[index] = ref[index] = dist(mt);
                }
            }
        }
    }

    auto forward = s.isForward() ? s : s.createInverse();
    auto inverse = s.isForward() ? s.createInverse() : s;
    auto gpu = GPUNew();
    gpu.set();
    auto ft = CudaFFT<T>();
    ft.init(gpu, forward);
    ft.fft(inOut, fd);
    ft.init(gpu, inverse);
    ft.ifft(fd, inOut);
    gpu.synch();

    // compare the results
    T delta = (T)0.00001;

    for (size_t n = 0; n < s.sDim().n(); ++n) {
        size_t offset = n * s.sDim().xyzPadded();
        // skip the padded area, it can contain garbage data
        for (size_t z = 0; z < s.sDim().z(); ++z) {
            for (size_t y = 0; y < s.sDim().y(); ++y) {
                for (size_t x = 0; x < s.sDim().x(); ++x) {
                    size_t index = offset + z * s.sDim().xyPadded() + y * s.sDim().xPadded() + x;
                    ASSERT_NEAR(ref[index], inOut[index] / s.sDim().xyz(), delta);
                }
            }
        }
    }

    delete[] inOut;
    delete[] ref;
    if ((void*)inOut != (void*)fd) {
        delete[] fd;
    }
}

template<typename T, typename F>
void generateAndTest(F condition, bool bothDirections = false) {
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
      return ((l.sDim().x() < r.sDim().x())
              || (l.sDim().y() < r.sDim().y())
              || (l.sDim().z() < r.sDim().z())
              || (l.sDim().n() < r.sDim().n())
              || (l.batch() < r.batch()));
    };
    auto tested = std::set<FFTSettingsNew<T>,decltype(settingsComparator)>(settingsComparator);

    int seed = 42;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<> dist(0, 4097);
    GPU gpu(0);
    T availableBytes = gpu.lastFreeBytes();
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
            T totalBytes = CudaFFT<T>::estimateTotalBytes(settings);
            if (availableBytes < totalBytes) {
                skippedSize++;
                continue;
            }
            // make sure we did not test this before
            auto result = tested.insert(settings);
            if ( ! result.second) continue;

//            auto dir = bothDirections ? "both" : (isForward ? "fft" : "ifft");
//            printf("Testing %lu %lu %lu %lu %lu %s %s\n",
//                    x, y, z, n, b, inPlace ? "inPlace" : "outOfPlace", dir);
            if (bothDirections) {
                testFFTIFFT(settings);
            } else {
                if (isForward) {
                    testFFTInpulseOrigin(settings);
                    testFFTInpulseShifted(settings);
                } else {
                    testIFFTInpulseOrigin(settings);
                }
            }


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
    return (z == 1) && (y != 1) && (x != 1);
};

auto is3D = [] (size_t x, size_t y, size_t z) {
    return (z != 1) && (y != 1) && (x != 1);
};

auto isBatchMultiple = [] (size_t n, size_t batch) {
    return (batch != 1) && (0 == (n % batch));
};

auto isNotBatchMultiple = [] (size_t n, size_t batch) {
    return (batch != 1) && (0 != (n % batch));
};

auto isNBatch = [] (size_t n, size_t batch) {
    return (batch != 1) && (n == batch);
};

//***********************************************
//              Out of place FFT tests
//***********************************************

TYPED_TEST_P( CudaFFTTest, fft_OOP_Single)
{
    // test a forward, out-of-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is3D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_Batch1)
{
    // test a forward, out-of-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is3D(x, y, z) && isNBatch(n, batch);
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_Batch2)
{
    // test a forward transform of many signals, out-of-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, fft_OOP_Batch3)
{
    // test a forward transform of many signals, out-of-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && (!inPlace) && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

//***********************************************
//              In place FFT tests
//***********************************************

TYPED_TEST_P( CudaFFTTest, fft_IP_Single)
{
    // test a forward, in-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is3D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_Batch1)
{
    // test a forward, in-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is3D(x, y, z) && isNBatch(n, batch);
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_Batch2)
{
    // test a forward transform of many signals, in-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, fft_IP_Batch3)
{
    // test a forward transform of many signals, in-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return isForward && inPlace && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

//***********************************************
//              Out of place IFFT tests
//***********************************************

TYPED_TEST_P( CudaFFTTest, ifft_OOP_Single)
{
    // test an inverse, out-of-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is3D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, ifft_OOP_Batch1)
{
    // test an inverse, out-of-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is3D(x, y, z) && isNBatch(n, batch);
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, ifft_OOP_Batch2)
{
    // test an inverse transform of many signals, out-of-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, ifft_OOP_Batch3)
{
    // test an inverse transform of many signals, out-of-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && (!inPlace) && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

//***********************************************
//              In place IFFT tests
//***********************************************

TYPED_TEST_P( CudaFFTTest, ifft_IP_Single)
{
    // test an inverse, in-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is3D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, ifft_IP_Batch1)
{
    // test an inverse, in-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is3D(x, y, z) && isNBatch(n, batch);
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, ifft_IP_Batch2)
{
    // test an inverse transform of many signals, in-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

TYPED_TEST_P( CudaFFTTest, ifft_IP_Batch3)
{
    // test an inverse transform of many signals, in-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!isForward) && inPlace && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D);
    generateAndTest<TypeParam>(condition2D);
    generateAndTest<TypeParam>(condition3D);
}

//***********************************************
//              Out of place FFT + IFFT tests
//***********************************************

TYPED_TEST_P( CudaFFTTest, OOP_Single)
{
    // test forward and inverse, out-of-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is3D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition1D, true);
    generateAndTest<TypeParam>(condition2D, true);
    generateAndTest<TypeParam>(condition3D, true);
}

TYPED_TEST_P( CudaFFTTest, OOP_Batch1)
{
    // test forward and inverse, out-of-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is3D(x, y, z) && isNBatch(n, batch);
    };
    generateAndTest<TypeParam>(condition1D, true);
    generateAndTest<TypeParam>(condition2D, true);
    generateAndTest<TypeParam>(condition3D, true);
}

TYPED_TEST_P( CudaFFTTest, OOP_Batch2)
{
    // test forward and inverse transform of many signals, out-of-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D, true);
    generateAndTest<TypeParam>(condition2D, true);
    generateAndTest<TypeParam>(condition3D, true);
}

TYPED_TEST_P( CudaFFTTest, OOP_Batch3)
{
    // test forward and inverse transform of many signals, out-of-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return (!inPlace) && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D, true);
    generateAndTest<TypeParam>(condition2D, true);
    generateAndTest<TypeParam>(condition3D, true);
}

//***********************************************
//              In place FFT + IFFT tests
//***********************************************

TYPED_TEST_P( CudaFFTTest, IP_Single)
{
    // test forward and inverse, in-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is3D(x, y, z) && (1 == n);
    };
    generateAndTest<TypeParam>(condition1D, true);
    generateAndTest<TypeParam>(condition2D, true);
    generateAndTest<TypeParam>(condition3D, true);
}

TYPED_TEST_P( CudaFFTTest, IP_Batch1)
{
    // test forward and inverse, in-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is3D(x, y, z) && isNBatch(n, batch);
    };
    generateAndTest<TypeParam>(condition1D, true);
    generateAndTest<TypeParam>(condition2D, true);
    generateAndTest<TypeParam>(condition3D, true);
}

TYPED_TEST_P( CudaFFTTest, IP_Batch2)
{
    // test forward and inverse transform of many signals, in-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D, true);
    generateAndTest<TypeParam>(condition2D, true);
    generateAndTest<TypeParam>(condition3D, true);
}

TYPED_TEST_P( CudaFFTTest, IP_Batch3)
{
    // test forward and inverse transform of many signals, in-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, size_t inPlace, size_t isForward) {
        return inPlace && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    generateAndTest<TypeParam>(condition1D, true);
    generateAndTest<TypeParam>(condition2D, true);
    generateAndTest<TypeParam>(condition3D, true);
}


REGISTER_TYPED_TEST_CASE_P(CudaFFTTest,
    // FFT out-of-place
    fft_OOP_Single,
    fft_OOP_Batch1,
    fft_OOP_Batch2,
    fft_OOP_Batch3,
    // FFT in-place
    fft_IP_Single,
    fft_IP_Batch1,
    fft_IP_Batch2,
    fft_IP_Batch3,
    // IFFT out-of-place
    ifft_OOP_Single,
    ifft_OOP_Batch1,
    ifft_OOP_Batch2,
    ifft_OOP_Batch3,
    // IFFT in-place
    ifft_IP_Single,
    ifft_IP_Batch1,
    ifft_IP_Batch2,
    ifft_IP_Batch3,
    // FFT + IFFT out-of-place
    OOP_Single,
    OOP_Batch1,
    OOP_Batch2,
    OOP_Batch3,
    // FFT + IFFT in-place
    IP_Single,
    IP_Batch1,
    IP_Batch2,
    IP_Batch3
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cuda, CudaFFTTest, TestTypes);
