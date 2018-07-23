#ifndef CUFFTADVISOR_ADVISOR_H
#define CUFFTADVISOR_ADVISOR_H

#include "benchmarker.h"
#include "sizeOptimizer.h"
#include "validator.h"

namespace cuFFTAdvisor {

class Advisor {
 public:
  static std::vector<BenchmarkResult const *> *benchmark(
      int device, int x, int y = 1, int z = 1, int n = 1,
      Tristate::Tristate isBatched = Tristate::TRUE,
      Tristate::Tristate isFloat = Tristate::TRUE,
      Tristate::Tristate isForward = Tristate::TRUE,
      Tristate::Tristate isInPlace = Tristate::TRUE,
      Tristate::Tristate isReal = Tristate::TRUE);

  static std::vector<Transform const *> *recommend(
      int howMany, int device, int x, int y = 1, int z = 1, int n = 1,
      Tristate::Tristate isBatched = Tristate::TRUE,
      Tristate::Tristate isFloat = Tristate::TRUE,
      Tristate::Tristate isForward = Tristate::TRUE,
      Tristate::Tristate isInPlace = Tristate::TRUE,
      Tristate::Tristate isReal = Tristate::TRUE, int maxSignalInc = INT_MAX,
      int maxMemory = INT_MAX,
      bool allowTransposition = false,
      bool squareOnly = false);

  static std::vector<BenchmarkResult const *> *find(
      int howMany, int device, int x, int y = 1, int z = 1, int n = 1,
      Tristate::Tristate isBatched = Tristate::TRUE,
      Tristate::Tristate isFloat = Tristate::TRUE,
      Tristate::Tristate isForward = Tristate::TRUE,
      Tristate::Tristate isInPlace = Tristate::TRUE,
      Tristate::Tristate isReal = Tristate::TRUE, int maxSignalInc = INT_MAX,
      int maxMemory = INT_MAX,
      bool allowTransposition = false,
      bool squareOnly = false);

 private:
  static int getMaxMemory(int device, int size);

  static inline std::vector<BenchmarkResult const *> *benchmark(
      std::vector<Transform const *> &transforms);
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_ADVISOR_H
