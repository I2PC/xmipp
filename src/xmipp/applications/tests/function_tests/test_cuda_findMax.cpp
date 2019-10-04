/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
#include <gtest/gtest.h>
#include <data/dimensions.h>
#include <numeric>
#include <memory>
#include "reconstruction_cuda/cuda_find_extrema.h"
#include <algorithm>

using namespace ExtremaFinder;

template<typename T>
class TestData {
public:
    TestData(const Dimensions &dims):
        dims(dims) {
        // allocate CPU
        // FIXME DS try to use unique_ptr
        data = memoryUtils::page_aligned_alloc<T>(dims.size(), false);
        resPos = new size_t[dims.n()]();
        resVal = new T[dims.n()]();
    }

    ~TestData() {
        free(data);
        delete[] resPos;
        delete[] resVal;
    }

    // CPU data
    Dimensions dims;
    T *data;
    size_t *resPos;
    T *resVal;
};

template<typename T>
class FindMax_Test : public ::testing::Test {
public:
    void compare1D(const TestData<T> &tc) {
        for (size_t n = 0; n < tc.dims.n(); ++n) {
            size_t offset = n * tc.dims.sizeSingle();
            auto start = tc.data + offset;
            auto max = std::max_element(start, start + tc.dims.sizeSingle());
            auto pos = std::distance(start, max);
            ASSERT_EQ(pos, tc.resPos[n]) << "for signal " << n;
            EXPECT_EQ(*max, tc.resVal[n]) << "for signal " << n;
        }
    }

    void print(const TestData<T> &tc) {
        printf("\n");
        for (size_t n = 0; n < tc.dims.n(); ++n) {
            printf("signal %lu:\n", n);
            size_t offset = n * tc.dims.sizeSingle();
            for (size_t i = 0; i < tc.dims.x(); ++i) {
                printf("%f ", tc.data[i + offset]);
            }
            printf("\n");
            printf("found max: %f pos: %lu\n", tc.resVal[n], tc.resPos[n]);
        }
    }

    void test_1D_increasing(const TestData<T> &tc) {
        // prepare data
        std::iota(tc.data, tc.data + tc.dims.size(), 0);

        auto settings = ExtremaFinderSettings();
        settings.batch = std::max((size_t)1, tc.dims.n() / 4); // FIXME DS do this properly
        settings.dims = tc.dims;
        settings.hw = hw;
        settings.resultType = ResultType::Both;
        settings.searchType = SearchType::Max;

        // test
        finder->init(settings, true);
        hw.at(0)->lockMemory(tc.data, tc.dims.size() * sizeof(T));
        finder->find(tc.data);
        hw.at(0)->unlockMemory(tc.data);

        // get results and compare
        const auto *f = finder;
        std::copy(f->getPositions().begin(),
                f->getPositions().end(),
                tc.resPos);
        std::copy(f->getValues().begin(),
                f->getValues().end(),
                tc.resVal);
        compare1D(tc);
    }

    void test_2D_window_increasing(const TestData<T> &tc) {
//        // prepare data
//        std::iota(tc.data.get(), tc.data.get() + tc.dims.size(), 0);
//
//        // test
//        sFindMax1D<T, false>(*hw, tc.dims, tc.data.get(), tc.resPos.get(), tc.resVal.get());
//
//        // get results and compare
//        compare1D(tc);
    }



    void test1D(const TestData<T> &tc) {
        // run tests
        printf("samples %lu signals %lu\n", tc.dims.sizeSingle(), tc.dims.n());
        test_1D_increasing(tc);
    }

    void test2DWindow(const TestData<T> &tc) {
        // run tests
        printf("samples %lu signals %lu\n", tc.dims.sizeSingle(), tc.dims.n());
        test_2D_window_increasing(tc);
    }

    static void TearDownTestCase() {
        for (auto device : hw) {
            delete device;
        }
        hw.clear();
    }

    static void SetUpTestCase() {
        for (int i = 0; i < 2; ++i) {
            auto g = new GPU();
            g->set();
            hw.push_back(g);
        }
    }

    void SetUp() {
        finder = new CudaExtremaFinder<T>();
    }

    void TearDown() {
        delete finder;
    }

private:
    static std::vector<HW*> hw;
    ExtremaFinder::AExtremaFinder<T> *finder;

};
TYPED_TEST_CASE_P(FindMax_Test);

template<typename T>
std::vector<HW*> FindMax_Test<T>::hw;

TYPED_TEST_P( FindMax_Test, debug)
{
//    XMIPP_TRY
    for (size_t n = 1; n < 100; ++n) {
        for (size_t i = 1000; i >= 1; --i) {
            auto testCase = TestData<TypeParam>(Dimensions(i, 1, 1, n));
            FindMax_Test<TypeParam>::test1D(testCase);
        }
    }
//    XMIPP_CATCH
//    auto testCase = TestData<TypeParam>(Dimensions(20, 1, 1, 2));
//    FindMax_Test<TypeParam>::test1D(testCase);

}

REGISTER_TYPED_TEST_CASE_P(FindMax_Test,
    debug
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(, FindMax_Test, TestTypes);
