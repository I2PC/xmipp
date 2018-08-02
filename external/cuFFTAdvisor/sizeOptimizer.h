#ifndef CUFFTADVISOR_SIZEOPTIMIZER_H_
#define CUFFTADVISOR_SIZEOPTIMIZER_H_

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>
#include "benchmarker.h"
#include "generalTransform.h"
#include "transformGenerator.h"
#include "utils.h"

namespace cuFFTAdvisor {

class SizeOptimizer {
 private:
  struct Polynom {
    size_t value;
    int invocations;
    int noOfPrimes;
    size_t exponent2;
    size_t exponent3;
    size_t exponent5;
    size_t exponent7;
  };

  struct valueComparator {
    bool asc;
    valueComparator(bool asc) : asc(asc) {};
    bool operator()(const Polynom &l, const Polynom &r) {
      if (asc) return l.value < r.value;
      return l.value > r.value;
    }
  };

  struct kernelCallComparator {
    bool operator()(Triplet<Polynom *> *l, Triplet<Polynom *> *r) {
      int lval = l->fst->invocations + l->snd->invocations + l->rd->invocations;
      int rval = r->fst->invocations + r->snd->invocations + r->rd->invocations;
      return lval < rval;
    }
  };

 public:
  SizeOptimizer(CudaVersion::CudaVersion version, GeneralTransform &tr,
                bool allowTrans);
  std::vector<const Transform *> *optimize(size_t nBest, int maxPercIncrease,
                                           int maxMemMB, bool squareOnly,
                                           bool crop);

 private:
  int getNoOfPrimes(Polynom &poly);
  int getInvocations(int maxPower, size_t num);
  std::vector<Triplet<int> *> optimize(GeneralTransform &tr, size_t nBest,
                                       int maxPercIncrease);

  int getInvocations(Polynom &poly, bool isFloat);
  int getInvocationsV8(Polynom &poly, bool isFloat);
  std::set<Polynom, valueComparator> *filterOptimal(
      std::vector<Polynom> *input, bool crop);
  std::vector<Polynom> *generatePolys(size_t num, bool isFloat, bool crop);
  std::vector<GeneralTransform> *optimizeXYZ(GeneralTransform &tr, size_t nBest,
                                             int maxPercIncrease, bool squareOnly,
                                             bool crop);
  std::vector<const Transform *> *optimizeN(
      std::vector<GeneralTransform> *transforms, size_t maxMem, size_t nBest);
  void collapseBatched(GeneralTransform &gt, size_t maxMem,
                       std::vector<const Transform *> *result);
  static bool perfSort(const Transform *l, const Transform *r);
  bool collapse(GeneralTransform &gt, bool isBatched, size_t N, size_t maxMemMB,
                std::vector<const Transform *> *result);
  size_t getMaxSize(GeneralTransform &tr, int maxPercIncrease, bool squareOnly,
          bool crop);
  size_t getMinSize(GeneralTransform &tr, int maxPercDecrease, bool crop);
  static bool sizeSort(const Transform *l, const Transform *r);

 private:
  std::vector<GeneralTransform> input;
  const CudaVersion::CudaVersion version;

  const double log_2;
  const double log_3;
  const double log_5;
  const double log_7;

  static const int V8_RADIX_2_MAX_SP = 10;
  static const int V8_RADIX_3_MAX_SP = 6;
  static const int V8_RADIX_5_MAX_SP = 3;
  static const int V8_RADIX_7_MAX_SP = 3;
  static const int V8_RADIX_2_MAX_DP = 9;
  static const int V8_RADIX_3_MAX_DP = 5;
  static const int V8_RADIX_5_MAX_DP = 3;
  static const int V8_RADIX_7_MAX_DP = 3;
  static const Polynom UNIT;
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_SIZEOPTIMIZER_H_
