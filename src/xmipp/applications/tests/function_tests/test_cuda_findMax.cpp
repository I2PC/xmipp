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
#include "reconstruction_cuda/cuda_find_max.h"
#include <algorithm>

template<typename T>
class TestData {
public:
    TestData(const Dimensions &dims):
        dims(dims) {
        // allocate CPU
        data = std::unique_ptr<T[]>(new T[dims.size()]);
        resPos = std::unique_ptr<T[]>(new T[dims.n()]);
        resVal = std::unique_ptr<T[]>(new T[dims.n()]);
    }

    // CPU data
    Dimensions dims;
    std::unique_ptr<T[]> data;
    std::unique_ptr<T[]> resPos;
    std::unique_ptr<T[]> resVal;
};

template<typename T>
class FindMax_Test : public ::testing::Test {
public:
    void compare(const TestData<T> &tc) {
        for (size_t n = 0; n < tc.dims.n(); ++n) {
            size_t offset = n * tc.dims.sizeSingle();
            auto start = tc.data.get() + offset;
            auto max = std::max_element(start, start + tc.dims.x());
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
            printf("found max: %f pos: %f\n", tc.resVal[n], tc.resPos[n]);
        }
    }

    void test_1D_increasing(const TestData<T> &tc) {
        // prepare data
        std::iota(tc.data.get(), tc.data.get() + tc.dims.size(), 0);

        // test
        sFindMax<T, false>(*hw, tc.dims, tc.data.get(), tc.resPos.get(), tc.resVal.get());

        // get results and compare
        compare(tc);
    }



    void test1D(const TestData<T> &tc) {
        // run tests
        printf("samples %lu signals %lu\n", tc.dims.sizeSingle(), tc.dims.n());
        test_1D_increasing(tc);
    }

    static void TearDownTestCase() {
        delete hw;
    }

    static void SetUpTestCase() {
            hw = new GPU();
            hw->set();
        }

private:
    static GPU *hw;

};
TYPED_TEST_CASE_P(FindMax_Test);

template<typename T>
GPU* FindMax_Test<T>::hw;

TYPED_TEST_P( FindMax_Test, debug)
{
    for (size_t n = 1; n < 100; ++n) {
        for (size_t i = 1; i < 1000; ++i) {
            auto testCase = TestData<TypeParam>(Dimensions(i, 1, 1, n));
            FindMax_Test<TypeParam>::test1D(testCase);
        }
    }
//        auto testCase = TestData<TypeParam>(Dimensions(33, 1, 1, 3));
//        FindMax_Test<TypeParam>::test1D(testCase);
}

REGISTER_TYPED_TEST_CASE_P(FindMax_Test,
    debug
);

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(, FindMax_Test, TestTypes);
