#include <gtest/gtest.h>
#include <random>
#include <set>

#include "../../../libraries/data/aft.h"

template<typename T>
class AFT_Test : public ::testing::Test {
public:
    SETUP

    void TearDown() {
        delete ft;
    }

    SETUPTESTCASE

    static void TearDownTestCase() {
        delete hw;
    }

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

        ft->init(*hw, s);
        ft->fft(in, out);
        hw->synch();

        T delta = (T)0.00001;
        for (size_t i = 0; i < s.fDim().size(); ++i) {
            // ... will result in constant magnitude
            T re = out[i].real();
            T im = out[i].imag();
            T mag = (re * re) + (im * im);
            ASSERT_NEAR((T)1, std::sqrt(mag), delta) << " at " << i;
        }

        delete[] in;
        if ((void*)in != (void*)out) {
            delete[] out;
        }
    }

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

        ft->init(*hw, s);
        ft->fft(in, out);
        hw->synch();

        T delta = (T)0.00001;
        for (size_t i = 0; i < s.fDim().size(); ++i) {
            // ... will result in constant real value, and no imag value
            ASSERT_NEAR((T)1, out[i].real(), delta) << " at " << i;
            ASSERT_NEAR((T)0, out[i].imag(), delta) << " at " << i;
        }

        delete[] in;
        if ((void*)in != (void*)out) {
            delete[] out;
        }
    }

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

        ft->init(*hw, s);
        ft->ifft(in, out);
        hw->synch();

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
                            ASSERT_NEAR((T)1, out[index] / s.sDim().xyz(), delta) << "at " << index;
                        } else {
                            // ... and zeros elsewhere
                            ASSERT_NEAR((T)0, out[index] / s.sDim().xyz(), delta) << "at " << index;
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
        ft->init(*hw, forward);
        ft->fft(inOut, fd);
        ft->init(*hw, inverse);
        ft->ifft(fd, inOut);
        hw->synch();

        // compare the results
        T delta = (T)0.00001;

        for (size_t n = 0; n < s.sDim().n(); ++n) {
            size_t offset = n * s.sDim().xyzPadded();
            // skip the padded area, it can contain garbage data
            for (size_t z = 0; z < s.sDim().z(); ++z) {
                for (size_t y = 0; y < s.sDim().y(); ++y) {
                    for (size_t x = 0; x < s.sDim().x(); ++x) {
                        size_t index = offset + z * s.sDim().xyPadded() + y * s.sDim().xPadded() + x;
                        ASSERT_NEAR(ref[index], inOut[index] / s.sDim().xyz(), delta) << " at " << index;
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

    template<typename F>
    void generateAndTest(F condition, bool bothDirections = false) {
        size_t executed = 0;
        size_t skippedSize = 0;
        size_t skippedCondition = 0;
        TEST_VALUES
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
        while ((executed < EXECUTIONS)
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
                size_t totalBytes = ft->estimateTotalBytes(settings);
                // since version 9.1 // FIXME DS check
                // forward transformation can use much less memory than inverse one
                if (bothDirections) {
                    totalBytes = std::max(totalBytes, ft->estimateTotalBytes(settings.createInverse()));
                }
                hw->updateMemoryInfo();
                size_t availableBytes = hw->lastFreeBytes();
                if (availableBytes < totalBytes) {
                    skippedSize++;
                    continue;
                }
                // make sure we did not test this before
                auto result = tested.insert(settings);
                if ( ! result.second) continue;

//                auto dir = bothDirections ? "both" : (isForward ? "fft" : "ifft");
//                printf("Testing %lu %lu %lu %lu %lu %s %s\n",
//                        x, y, z, n, b, inPlace ? "inPlace" : "outOfPlace", dir);
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

private:
    AFT<T> *ft;
    static HW *hw;

};
TYPED_TEST_CASE_P(AFT_Test);

template<typename T>
HW* AFT_Test<T>::hw;


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

TYPED_TEST_P( AFT_Test, fft_OOP_Single)
{
    // test a forward, out-of-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is3D(x, y, z) && (1 == n);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, fft_OOP_Batch1)
{
    // test a forward, out-of-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is3D(x, y, z) && isNBatch(n, batch);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, fft_OOP_Batch2)
{
    // test a forward transform of many signals, out-of-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, fft_OOP_Batch3)
{
    // test a forward transform of many signals, out-of-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && (!inPlace) && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

//***********************************************
//              In place FFT tests
//***********************************************

TYPED_TEST_P( AFT_Test, fft_IP_Single)
{
    // test a forward, in-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is3D(x, y, z) && (1 == n);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, fft_IP_Batch1)
{
    // test a forward, in-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is3D(x, y, z) && isNBatch(n, batch);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, fft_IP_Batch2)
{
    // test a forward transform of many signals, in-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, fft_IP_Batch3)
{
    // test a forward transform of many signals, in-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return isForward && inPlace && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

//***********************************************
//              Out of place IFFT tests
//***********************************************

TYPED_TEST_P( AFT_Test, ifft_OOP_Single)
{
    // test an inverse, out-of-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is3D(x, y, z) && (1 == n);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, ifft_OOP_Batch1)
{
    // test an inverse, out-of-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is3D(x, y, z) && isNBatch(n, batch);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, ifft_OOP_Batch2)
{
    // test an inverse transform of many signals, out-of-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, ifft_OOP_Batch3)
{
    // test an inverse transform of many signals, out-of-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && (!inPlace) && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

//***********************************************
//              In place IFFT tests
//***********************************************

TYPED_TEST_P( AFT_Test, ifft_IP_Single)
{
    // test an inverse, in-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is3D(x, y, z) && (1 == n);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, ifft_IP_Batch1)
{
    // test an inverse, in-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is3D(x, y, z) && isNBatch(n, batch);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, ifft_IP_Batch2)
{
    // test an inverse transform of many signals, in-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

TYPED_TEST_P( AFT_Test, ifft_IP_Batch3)
{
    // test an inverse transform of many signals, in-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!isForward) && inPlace && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D);
    AFT_Test<TypeParam>::generateAndTest(condition2D);
    AFT_Test<TypeParam>::generateAndTest(condition3D);
}

//***********************************************
//              Out of place FFT + IFFT tests
//***********************************************

TYPED_TEST_P( AFT_Test, OOP_Single)
{
    // test forward and inverse, out-of-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is3D(x, y, z) && (1 == n);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D, true);
    AFT_Test<TypeParam>::generateAndTest(condition2D, true);
    AFT_Test<TypeParam>::generateAndTest(condition3D, true);
}

TYPED_TEST_P( AFT_Test, OOP_Batch1)
{
    // test forward and inverse, out-of-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is3D(x, y, z) && isNBatch(n, batch);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D, true);
    AFT_Test<TypeParam>::generateAndTest(condition2D, true);
    AFT_Test<TypeParam>::generateAndTest(condition3D, true);
}

TYPED_TEST_P( AFT_Test, OOP_Batch2)
{
    // test forward and inverse transform of many signals, out-of-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D, true);
    AFT_Test<TypeParam>::generateAndTest(condition2D, true);
    AFT_Test<TypeParam>::generateAndTest(condition3D, true);
}

TYPED_TEST_P( AFT_Test, OOP_Batch3)
{
    // test forward and inverse transform of many signals, out-of-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return (!inPlace) && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D, true);
    AFT_Test<TypeParam>::generateAndTest(condition2D, true);
    AFT_Test<TypeParam>::generateAndTest(condition3D, true);
}

//***********************************************
//              In place FFT + IFFT tests
//***********************************************

TYPED_TEST_P( AFT_Test, IP_Single)
{
    // test forward and inverse, in-place transform of a single signal,
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is1D(x, y, z) && (1 == n);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is2D(x, y, z) && (1 == n);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is3D(x, y, z) && (1 == n);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D, true);
    AFT_Test<TypeParam>::generateAndTest(condition2D, true);
    AFT_Test<TypeParam>::generateAndTest(condition3D, true);
}

TYPED_TEST_P( AFT_Test, IP_Batch1)
{
    // test forward and inverse, in-place transform of many signals,
    // check that n == batch works properly
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is1D(x, y, z) && isNBatch(n, batch);
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is2D(x, y, z) && isNBatch(n, batch);
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is3D(x, y, z) && isNBatch(n, batch);
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D, true);
    AFT_Test<TypeParam>::generateAndTest(condition2D, true);
    AFT_Test<TypeParam>::generateAndTest(condition3D, true);
}

TYPED_TEST_P( AFT_Test, IP_Batch2)
{
    // test forward and inverse transform of many signals, in-place
    // test that n mod batch != 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is1D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is2D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is3D(x, y, z) && (isNotBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D, true);
    AFT_Test<TypeParam>::generateAndTest(condition2D, true);
    AFT_Test<TypeParam>::generateAndTest(condition3D, true);
}

TYPED_TEST_P( AFT_Test, IP_Batch3)
{
    // test forward and inverse transform of many signals, in-place
    // test that n mod batch = 0 works
    auto condition1D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is1D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition2D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is2D(x, y, z) && (isBatchMultiple(n, batch));
    };
    auto condition3D = []
            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
        return inPlace && is3D(x, y, z) && (isBatchMultiple(n, batch));
    };
    AFT_Test<TypeParam>::generateAndTest(condition1D, true);
    AFT_Test<TypeParam>::generateAndTest(condition2D, true);
    AFT_Test<TypeParam>::generateAndTest(condition3D, true);
}

//TYPED_TEST_P( AFT_Test, DEBUG)
//{
//    auto condition = []
//            (size_t x, size_t y, size_t z, size_t n, size_t batch, bool inPlace, bool isForward) {
//        return x == 3 && y == 1 && z == 1 && n == 1 && batch == 1 && !inPlace && !isForward;
//    };
//    AFT_Test<TypeParam>::generateAndTest(condition);
//}

REGISTER_TYPED_TEST_CASE_P(AFT_Test,
//    DEBUG,
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
