#include "reconstruction/single_extrema_finder.h"

template<typename T>
class SingleExtremaFinder_Test;

#define SETUPTESTCASE_SPECIFIC \
    finder = new ExtremaFinder::SingleExtremaFinder<T>(); \
    hw.emplace_back(new CPU());

#include "asingle_extrema_finder_tests.h"

typedef ::testing::Types<float, double> TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(CPU, SingleExtremaFinder_Test, TestTypes);
