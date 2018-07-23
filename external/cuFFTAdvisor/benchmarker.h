#ifndef CUFFTADVISOR_BENCHMARKER_H_
#define CUFFTADVISOR_BENCHMARKER_H_

#include <cstdlib>
#include "benchmarkResult.h"
#include "cudaUtils.h"

namespace cuFFTAdvisor {

class Benchmarker {
 public:
  static BenchmarkResult const *benchmark(Transform const *tr);
  static void estimatePlanSize(BenchmarkResult *res);

 private:
  static void checkMemory(BenchmarkResult *res);
  static void estimatePlanSize(cufftHandle &plan, BenchmarkResult *res);
  static void createPlan(cufftHandle &plan, Transform const *tr,
                         BenchmarkResult *res);
  static float execute(cufftHandle &plan, void *d_in, void *d_out,
                       Transform const *tr);
  static void execute(cufftHandle &plan, Transform const *tr,
                      BenchmarkResult *res);
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_BENCHMARKER_H_
