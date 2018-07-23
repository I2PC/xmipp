#ifndef CUFFTADVISOR_BENCHMARKRESULT_H_
#define CUFFTADVISOR_BENCHMARKRESULT_H_

#include <cmath>
#include "transform.h"
#include "utils.h"

namespace cuFFTAdvisor {

class BenchmarkResult {
 public:
  BenchmarkResult(cuFFTAdvisor::Transform const *transform)
      : transform(transform),
        planSizeEstimateB(0),
        planSizeEstimate2B(0),
        planSizeRealB(0),
        planTimeMS(NAN),
        execTimeMS(NAN),
        totalTimeMS(NAN),
        errMsg("") {}
  ~BenchmarkResult() { delete transform; }

  void print(FILE *stream = stdout) const;
  static void printHeader(FILE *stream = stdout);
  static bool execSort(const BenchmarkResult *l, const BenchmarkResult *r);

 public:
  cuFFTAdvisor::Transform const *transform;
  size_t planSizeEstimateB;
  size_t planSizeEstimate2B;
  size_t planSizeRealB;
  float planTimeMS;
  float execTimeMS;
  float totalTimeMS;
  std::string errMsg;

 private:
  float getPerf() const;
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_BENCHMARKRESULT_H_
