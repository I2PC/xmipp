#include <gtest/gtest.h>
#include "data/cpu.h"
#include "core/utils/memory_utils.h"
#include <random>
#include "reconstruction/aextrema_finder.h"

using namespace ExtremaFinder;

template<typename T>
class SingleExtremaFinder_Test : public ::testing::Test {
public:
    static void SetUpTestCase() {
        data = memoryUtils::page_aligned_alloc<T>(maxDims.size(), false);
        auto generate = [&](size_t n, int id) {
            // Each thread needs its own mt, otherwise we might generate the same data
            // multiple times.
            // Now it can happen too, but the probability is much lower
            auto mt = std::mt19937(n);
            auto dist = std::normal_distribution<T>((T)0, (T)1);
            const size_t elems = maxDims.sizeSingle();
            const size_t offset = n * elems;
            for (size_t i = 0; i < elems; ++i) {
                data[offset + i] = dist(mt);
            }
        };
        runPerSignal(generate, maxDims);
        SETUPTESTCASE_SPECIFIC
        hw.at(0)->lockMemory(data, maxDims.size() * sizeof(T));
    }

    static void TearDownTestCase() {
        hw.at(0)->unlockMemory(data);
        for (auto device : hw) {
            delete device;
        }
        hw.clear();
        delete finder;
        free(data);
        data = nullptr;
    }

    void test(const ExtremaFinderSettings &s) {
//        std::cout << s.dims << ", batch " << s.batch << "\n";
        if (s.dims.size() > maxDims.size()) {
            printf("skipping current setting (dimensions are bigger than preallocated data)\n");
            return;
        }
        assert(ResultType::Both == s.resultType); // we will always test both results
        finder->init(s, true);
        finder->find(data);
        switch(s.searchType) {
            case SearchType::Max : return check(std::greater<T>(), s);
            case SearchType::Lowest : return check(std::less<T>(), s);
            case SearchType::MaxAroundCenter : return check2DAroundCenter(std::greater<T>(),s);
            case SearchType::LowestAroundCenter : return check2DAroundCenter(std::less<T>(),s);
            default: FAIL() << "Test check not implemented";
        }
    }

private:
    template<typename F>
    static void runPerSignal(F f, const Dimensions &dims) {
        int threads = std::min((size_t)CPU::findCores(), dims.n());
        auto workers = std::vector<std::thread>();
        int workloadPerWorker = std::ceil(dims.n() / (float)threads);

        auto workload = [&](int id){
            const size_t first = id * workloadPerWorker;
            const size_t last = std::min(first + workloadPerWorker, dims.n());
            for (size_t n = first; n < last; ++n) {
                f(n, id);
            }
        };
        for (size_t w = 0; w < threads; ++w) {
            workers.emplace_back(workload, w);
        }
        for (auto &w : workers) {
            w.join();
        }
    }

    template<typename C>
    void check(const C &comp,
            const ExtremaFinderSettings &s) {
        const auto &tmp = *finder;
        auto actPos = tmp.getPositions();
        auto actVals = tmp.getValues();
        EXPECT_EQ(s.dims.n(), actPos.size());
        EXPECT_EQ(s.dims.n(), actVals.size());

        auto check = [&](size_t n, int id) {
            float expPos = -1;
            T expVal = getStartingValue(s);
            const size_t elems = s.dims.sizeSingle();
            const size_t offset = n * elems;
            for (size_t i = 0; i < elems; ++i) {
                T v = data[offset + i];
                if (comp(v, expVal)) {
                    expVal = v;
                    expPos = i;
                }
            }
            if (actPos.at(n) != expPos) {
                if (std::abs(actVals.at(n) - expVal) < std::numeric_limits<T>::min()) {
                    printf("Position mismatch for signal %lu (exp %.2f, act %.2f), "
                            "but values are very similar: exp %.30f act %.30f. New value will be generated\n",
                            n, expPos, actPos.at(n),
                            expVal, actVals.at(n));
                    return;
                }
            }
            EXPECT_EQ(expPos, actPos.at(n)) << "for signal " << n << "\n";
            ASSERT_EQ(actVals.at(n), expVal) << "for signal " << n << "\n";
        };
        SingleExtremaFinder_Test<T>::runPerSignal(check, s.dims);
    }

    template<typename C>
    void check2DAroundCenterRowColumn(
            const C &comp,
            const std::vector<size_t> &xVals, // only one of these should have more than a single value
            const std::vector<size_t> &yVals, // only one of these should have more than a single value
            const ExtremaFinderSettings &s,
            size_t signalOffset,
            T &val, float &pos) {
        auto center =  s.getCenter();
        float maxDistSq = s.maxDistFromCenter * s.maxDistFromCenter;
//        printf("testing (center %lu %lu, max %f:\n", center.x, center.y, s.maxDistFromCenter);
        for (auto x : xVals) {
            for (auto y : yVals) {
//                printf("x %lu y %lu:", x, y);
                float d = std::pow((float)y-(float)center.y, 2) + std::pow((float)x-(float)center.x, 2);
                if (d <= maxDistSq) {
                    size_t offset = y * s.dims.x() + x;
                    T v = data[signalOffset + offset];
                    if (comp(v, val)) {
//                        printf(" update (%f -> %f)\n", val, v);
                        val = v;
                        pos = offset;
                    } else {
//                        printf(" don't update (%f %f)\n", val, v);
                    }
                } else {
//                    printf(" skip\n");
                }
            }
        }
    }

    void printData(const ExtremaFinderSettings &s) {
        for (size_t n = 0; n < s.dims.n(); ++n) {
            const size_t offsetN = n * s.dims.sizeSingle();
            for (size_t y = 0; y < s.dims.y(); ++y) {
                const size_t offsetY = y * s.dims.x();
                for (size_t x = 0; x < s.dims.x(); ++x) {
                    printf("%.2f\t", data[offsetN + offsetY + x]);
                }
              printf("\n");
            }
            printf("\n");
        }
    }

    T getStartingValue(const ExtremaFinderSettings &s) {
        switch (s.searchType) {
            case SearchType::Max:
            case SearchType::MaxAroundCenter:
            case SearchType::MaxNearCenter:
                return std::numeric_limits<T>::lowest();
            case SearchType::Lowest:
            case SearchType::LowestAroundCenter:
                return std::numeric_limits<T>::max();
            default: std::cout << "NOT IMPLEMENTED\n";
        }
        assert(false); // cannot use FAIL() as we have to return some value
    }

    template<typename C>
    void check2DAroundCenter(const C &comp, const ExtremaFinderSettings &s) {
        const auto &tmp = *finder;
        auto actPos = tmp.getPositions();
        auto actVals = tmp.getValues();
        EXPECT_EQ(s.dims.n(), actPos.size());
        EXPECT_EQ(s.dims.n(), actVals.size());

//        printData(s);

        auto check = [&](size_t n, int id) {
            float expPos = -1;
            T expVal = getStartingValue(s);
            float currDistance = -1;
            auto center =  s.getCenter();
            const size_t elems = s.dims.sizeSingle();
            const size_t signalOffset = n * elems;
            // iterate from the center
            while (currDistance <= s.maxDistFromCenter) {
                currDistance += 1.f;
                auto xVals = std::vector<size_t>(2 * currDistance + 1, -1);
                std::iota(xVals.begin(), xVals.end(), center.x - currDistance);
                auto yVals = std::vector<size_t>(2 * currDistance + 1, -1);
                std::iota(yVals.begin(), yVals.end(), center.y - currDistance);
                // check row above
                check2DAroundCenterRowColumn(comp, xVals, {center.y - (size_t)currDistance}, s, signalOffset, expVal, expPos);
                // check row bellow
                check2DAroundCenterRowColumn(comp, xVals, {center.y + (size_t)currDistance}, s, signalOffset, expVal, expPos);
                // check column left
                check2DAroundCenterRowColumn(comp, {center.x - (size_t)currDistance}, yVals, s, signalOffset, expVal, expPos);
                // check column right
                check2DAroundCenterRowColumn(comp, {center.x + (size_t)currDistance}, yVals, s, signalOffset, expVal, expPos);
            }
            // check result
            if (actPos.at(n) != expPos) {
                if (std::abs(actVals.at(n) - expVal) < std::numeric_limits<T>::min()) {
                    printf("Position mismatch for signal %lu (exp %.2f, act %.2f), "
                            "but values are very similar: exp %.30f act %.30f. New value will be generated\n",
                            n, expPos, actPos.at(n),
                            expVal, actVals.at(n));
                    return;
                }
            }
            EXPECT_EQ(expPos, actPos.at(n)) << "for signal " << n << "\n";
            ASSERT_EQ(actVals.at(n), expVal) << "for signal " << n << "\n";
        };
        SingleExtremaFinder_Test<T>::runPerSignal(check, s.dims);
    }

    static std::mt19937 mt;
    static T * data;
    static Dimensions maxDims;
    static AExtremaFinder<T> *finder;
public:
    static std::vector<HW*> hw; // public to be able to use it in testcases
};
TYPED_TEST_CASE_P(SingleExtremaFinder_Test);

template<typename T>
T *SingleExtremaFinder_Test<T>::data = nullptr;
template<typename T>
AExtremaFinder<T> *SingleExtremaFinder_Test<T>::finder = nullptr;
template<typename T>
std::vector<HW*> SingleExtremaFinder_Test<T>::hw;
template<typename T>
std::mt19937 SingleExtremaFinder_Test<T>::mt(42); // fixed seed to ensure reproducibility
template<typename T>
Dimensions SingleExtremaFinder_Test<T>::maxDims(1000, 1000, 1, 1000);


auto isBatchMultiple = [] (size_t n, size_t batch) {
    return (batch != 1) && (0 == (n % batch));
};

auto isNotBatchMultiple = [] (size_t n, size_t batch) {
    return (batch != 1) && (0 != (n % batch));
};

auto isNBatch = [] (size_t n, size_t batch) {
    return (batch != 1) && (n == batch);
};

size_t randSize(size_t max, std::mt19937 &mt) {
    static std::normal_distribution<float> dist(0, 1);
    float v = std::abs(dist(mt) / 3.f); // get just one half of the distribution, 99.7% of the results should be here
    v = std::min(1.f, v); // crop values over 1;
    return (size_t)(v * max);
}

TYPED_TEST_P( SingleExtremaFinder_Test, findMax1D)
{
    auto mt = std::mt19937(42);
    auto nBatch = std::vector<std::pair<size_t, size_t>>(); // n, batch
    nBatch.emplace_back(1, 1);
    nBatch.emplace_back(5, 5);
    nBatch.emplace_back(10, 5);
    nBatch.emplace_back(10, 6);
    for (auto c : nBatch) {
        for (int i = 0; i < 10; ++i) {
            auto settings = ExtremaFinderSettings();
            settings.batch = c.second;
            settings.dims = Dimensions(randSize(1000000, mt), 1, 1, c.first);
            settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
            settings.resultType = ResultType::Both;
            settings.searchType = SearchType::Max;
            settings.maxDistFromCenter = 0;
            SingleExtremaFinder_Test<TypeParam>::test(settings);
        }
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findMax1DMany)
{
    auto mt = std::mt19937(42);
    std::uniform_int_distribution<> dist(1, 500);
    for (int i = 0; i < 5; ++i) {
        auto settings = ExtremaFinderSettings();
        settings.batch = 41;
        settings.dims = Dimensions(dist(mt), 1, 1, 10000);
        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::Max;
        settings.maxDistFromCenter = 0;
        SingleExtremaFinder_Test<TypeParam>::test(settings);
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findMax2D)
{
    auto mt = std::mt19937(42);
    auto nBatch = std::vector<std::pair<size_t, size_t>>(); // n, batch
    nBatch.emplace_back(1, 1);
    nBatch.emplace_back(5, 5);
    nBatch.emplace_back(10, 5);
    nBatch.emplace_back(10, 6);
    for (auto c : nBatch) {
        for (int i = 0; i < 5; ++i) {
            auto settings = ExtremaFinderSettings();
            settings.batch = c.second;
            settings.dims = Dimensions(randSize(10000, mt), randSize(10000, mt), 1, c.first);
            settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
            settings.resultType = ResultType::Both;
            settings.searchType = SearchType::Max;
            settings.maxDistFromCenter = 0;
            SingleExtremaFinder_Test<TypeParam>::test(settings);
        }
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findMax2DMany)
{
    auto mt = std::mt19937(42);
    std::uniform_int_distribution<> dist(1, 500);
    for (int i = 0; i < 5; ++i) {
        auto settings = ExtremaFinderSettings();
        settings.batch = 83;
        settings.dims = Dimensions(dist(mt), dist(mt), 1, 5051);
        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::Max;
        settings.maxDistFromCenter = 0;
        SingleExtremaFinder_Test<TypeParam>::test(settings);
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findMax3D)
{
    auto mt = std::mt19937(42);
    auto nBatch = std::vector<std::pair<size_t, size_t>>(); // n, batch
    nBatch.emplace_back(1, 1);
    nBatch.emplace_back(5, 5);
    nBatch.emplace_back(10, 5);
    nBatch.emplace_back(10, 6);
    for (auto c : nBatch) {
        for (int i = 0; i < 5; ++i) {
            auto settings = ExtremaFinderSettings();
            settings.batch = c.second;
            settings.dims = Dimensions(randSize(1000, mt), randSize(1000, mt), randSize(1000, mt), c.first);
            settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
            settings.resultType = ResultType::Both;
            settings.searchType = SearchType::Max;
            settings.maxDistFromCenter = 0;
            SingleExtremaFinder_Test<TypeParam>::test(settings);
        }
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findMax3DMany)
{
    auto mt = std::mt19937(42);
    std::uniform_int_distribution<> dist(1, 200);
    for (int i = 0; i < 5; ++i) {
        auto settings = ExtremaFinderSettings();
        settings.batch = 83;
        settings.dims = Dimensions(dist(mt), dist(mt), dist(mt), 1009);
        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::Max;
        settings.maxDistFromCenter = 0;
        SingleExtremaFinder_Test<TypeParam>::test(settings);
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findLowest1D)
{
    auto mt = std::mt19937(42);
    auto nBatch = std::vector<std::pair<size_t, size_t>>(); // n, batch
    nBatch.emplace_back(1, 1);
    nBatch.emplace_back(5, 5);
    nBatch.emplace_back(10, 5);
    nBatch.emplace_back(10, 6);
    for (auto c : nBatch) {
        for (int i = 0; i < 10; ++i) {
            auto settings = ExtremaFinderSettings();
            settings.batch = c.second;
            settings.dims = Dimensions(randSize(1000000, mt), 1, 1, c.first);
            settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
            settings.resultType = ResultType::Both;
            settings.searchType = SearchType::Lowest;
            settings.maxDistFromCenter = 0;
            SingleExtremaFinder_Test<TypeParam>::test(settings);
        }
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findLowest1DMany)
{
    auto mt = std::mt19937(42);
    std::uniform_int_distribution<> dist(1, 500);
    for (int i = 0; i < 5; ++i) {
        auto settings = ExtremaFinderSettings();
        settings.batch = 41;
        settings.dims = Dimensions(dist(mt), 1, 1, 10000);
        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::Lowest;
        settings.maxDistFromCenter = 0;
        SingleExtremaFinder_Test<TypeParam>::test(settings);
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findLowest2D)
{
    auto mt = std::mt19937(42);
    auto nBatch = std::vector<std::pair<size_t, size_t>>(); // n, batch
    nBatch.emplace_back(1, 1);
    nBatch.emplace_back(5, 5);
    nBatch.emplace_back(10, 5);
    nBatch.emplace_back(10, 6);
    for (auto c : nBatch) {
        for (int i = 0; i < 5; ++i) {
            auto settings = ExtremaFinderSettings();
            settings.batch = c.second;
            settings.dims = Dimensions(randSize(10000, mt), randSize(10000, mt), 1, c.first);
            settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
            settings.resultType = ResultType::Both;
            settings.searchType = SearchType::Lowest;
            settings.maxDistFromCenter = 0;
            SingleExtremaFinder_Test<TypeParam>::test(settings);
        }
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findLowest2DMany)
{
    auto mt = std::mt19937(42);
    std::uniform_int_distribution<> dist(1, 500);
    for (int i = 0; i < 5; ++i) {
        auto settings = ExtremaFinderSettings();
        settings.batch = 83;
        settings.dims = Dimensions(dist(mt), dist(mt), 1, 5051);
        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::Lowest;
        settings.maxDistFromCenter = 0;
        SingleExtremaFinder_Test<TypeParam>::test(settings);
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findLowest3D)
{
    auto mt = std::mt19937(42);
    auto nBatch = std::vector<std::pair<size_t, size_t>>(); // n, batch
    nBatch.emplace_back(1, 1);
    nBatch.emplace_back(5, 5);
    nBatch.emplace_back(10, 5);
    nBatch.emplace_back(10, 6);
    for (auto c : nBatch) {
        for (int i = 0; i < 5; ++i) {
            auto settings = ExtremaFinderSettings();
            settings.batch = c.second;
            settings.dims = Dimensions(randSize(1000, mt), randSize(1000, mt), randSize(1000, mt), c.first);
            settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
            settings.resultType = ResultType::Both;
            settings.searchType = SearchType::Lowest;
            settings.maxDistFromCenter = 0;
            SingleExtremaFinder_Test<TypeParam>::test(settings);
        }
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findLowest3DMany)
{
    auto mt = std::mt19937(42);
    std::uniform_int_distribution<> dist(1, 200);
    for (int i = 0; i < 5; ++i) {
        auto settings = ExtremaFinderSettings();
        settings.batch = 83;
        settings.dims = Dimensions(dist(mt), dist(mt), dist(mt), 1009);
        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::Lowest;
        settings.maxDistFromCenter = 0;
        SingleExtremaFinder_Test<TypeParam>::test(settings);
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findMax2DAroundCenter)
{
    XMIPP_TRY
    auto mt = std::mt19937(42);
    auto nBatch = std::vector<std::pair<size_t, size_t>>(); // n, batch
    nBatch.emplace_back(1, 1);
    nBatch.emplace_back(5, 5);
    nBatch.emplace_back(10, 5);
    nBatch.emplace_back(10, 6);
    for (auto c : nBatch) {
        for (int i = 0; i < 10; ++i) {
            auto settings = ExtremaFinderSettings();
            settings.batch = c.second;
            settings.dims = Dimensions(randSize(10000, mt), randSize(10000, mt), 1, c.first);
            settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
            settings.resultType = ResultType::Both;
            settings.searchType = SearchType::MaxAroundCenter;
            settings.maxDistFromCenter = std::min(settings.dims.x(), settings.dims.y()) / 2 - 1;
            SingleExtremaFinder_Test<TypeParam>::test(settings);
        }
    }
    XMIPP_CATCH
}

TYPED_TEST_P( SingleExtremaFinder_Test, findMax2DAroundCenterMany)
{
    auto mt = std::mt19937(42);
    std::uniform_int_distribution<> dist(1, 500);
    for (int i = 0; i < 5; ++i) {
        auto settings = ExtremaFinderSettings();
        settings.batch = 83;
        settings.dims = Dimensions(dist(mt), dist(mt), 1, 5051);
        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::MaxAroundCenter;
        settings.maxDistFromCenter = std::min(settings.dims.x(), settings.dims.y()) / 2 - 1;
        SingleExtremaFinder_Test<TypeParam>::test(settings);
    }
}

TYPED_TEST_P( SingleExtremaFinder_Test, findLowest2DAroundCenter)
{
    XMIPP_TRY
    auto mt = std::mt19937(42);
    auto nBatch = std::vector<std::pair<size_t, size_t>>(); // n, batch
    nBatch.emplace_back(1, 1);
    nBatch.emplace_back(5, 5);
    nBatch.emplace_back(10, 5);
    nBatch.emplace_back(10, 6);
    for (auto c : nBatch) {
        for (int i = 0; i < 10; ++i) {
            auto settings = ExtremaFinderSettings();
            settings.batch = c.second;
            settings.dims = Dimensions(randSize(10000, mt), randSize(10000, mt), 1, c.first);
            settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
            settings.resultType = ResultType::Both;
            settings.searchType = SearchType::LowestAroundCenter;
            settings.maxDistFromCenter = std::min(settings.dims.x(), settings.dims.y()) / 2 - 1;
            SingleExtremaFinder_Test<TypeParam>::test(settings);
        }
    }
    XMIPP_CATCH
}

TYPED_TEST_P( SingleExtremaFinder_Test, findLowest2DAroundCenterMany)
{
    auto mt = std::mt19937(42);
    std::uniform_int_distribution<> dist(1, 500);
    for (int i = 0; i < 5; ++i) {
        auto settings = ExtremaFinderSettings();
        settings.batch = 83;
        settings.dims = Dimensions(dist(mt), dist(mt), 1, 5051);
        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::LowestAroundCenter;
        settings.maxDistFromCenter = std::min(settings.dims.x(), settings.dims.y()) / 2 - 1;
        SingleExtremaFinder_Test<TypeParam>::test(settings);
    }
}

//TYPED_TEST_P( SingleExtremaFinder_Test, debug)
//{
//    XMIPP_TRY
//    auto mt = std::mt19937(42);
//    std::uniform_int_distribution<> dist(1, 500);
//    for (int i = 0; i < 5; ++i) {
//        auto settings = ExtremaFinderSettings();
//        settings.batch = 83;
//        settings.dims = Dimensions(dist(mt), dist(mt), 1, 5051);
//        settings.hw = SingleExtremaFinder_Test<TypeParam>::hw;
//        settings.resultType = ResultType::Both;
//        settings.searchType = SearchType::LowestAroundCenter;
//        settings.maxDistFromCenter = 0;
//        SingleExtremaFinder_Test<TypeParam>::test(settings);
//    }
//    XMIPP_CATCH;
//}

REGISTER_TYPED_TEST_CASE_P(SingleExtremaFinder_Test,
//    debug
    findMax1D,
    findMax1DMany,
    findMax2D,
    findMax2DMany,
    findMax3D,
    findMax3DMany,
    findLowest1D,
    findLowest1DMany,
    findLowest2D,
    findLowest2DMany,
    findLowest3D,
    findLowest3DMany,
    findMax2DAroundCenter,
    findMax2DAroundCenterMany,
    findLowest2DAroundCenter,
    findLowest2DAroundCenterMany
);
