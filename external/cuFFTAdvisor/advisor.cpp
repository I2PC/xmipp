#include "advisor.h"

namespace cuFFTAdvisor {

std::vector<BenchmarkResult const *> *Advisor::benchmark(
    int device, int x, int y, int z, int n, Tristate::Tristate isBatched,
    Tristate::Tristate isFloat, Tristate::Tristate isForward,
    Tristate::Tristate isInPlace, Tristate::Tristate isReal) {
  Validator::validate(x, y, z, n, device);
  std::vector<Transform const *> transforms;
  TransformGenerator generator;
  generator.generate(device, x, y, z, n, isBatched, isFloat, isForward,
                     isInPlace, isReal, transforms);
  std::vector<BenchmarkResult const *> *result = benchmark(transforms);
  resetDevice();
  return result;
}

int Advisor::getMaxMemory(int device, int size) {
  if (size == INT_MAX) {
    return std::ceil(toMB(getTotalMemory(device)));
  }
  return size;
}

std::vector<Transform const *> *Advisor::recommend(
    int howMany, int device, int x, int y, int z, int n,
    Tristate::Tristate isBatched, Tristate::Tristate isFloat,
    Tristate::Tristate isForward, Tristate::Tristate isInPlace,
    Tristate::Tristate isReal, int maxSignalInc, int maxMemory,
    bool allowTransposition, bool squareOnly) {
  Validator::validate(device);
  maxMemory = getMaxMemory(device, maxMemory);
  Validator::validate(x, y, z, n, device, maxSignalInc, maxMemory, allowTransposition, squareOnly);
  GeneralTransform tr = GeneralTransform(device, x, y, z, n, isBatched, isFloat,
                                         isForward, isInPlace, isReal);

  SizeOptimizer optimizer(CudaVersion::V_8, tr, allowTransposition);
  std::vector<const Transform *> *result =
      optimizer.optimize(howMany, maxSignalInc, maxMemory, squareOnly);
  resetDevice();
  return result;
}

std::vector<BenchmarkResult const *> *Advisor::find(
    int howMany, int device, int x, int y, int z, int n,
    Tristate::Tristate isBatched, Tristate::Tristate isFloat,
    Tristate::Tristate isForward, Tristate::Tristate isInPlace,
    Tristate::Tristate isReal, int maxSignalInc, int maxMemory,
    bool allowTransposition, bool squareOnly) {
  std::vector<Transform const *> *candidates =
      recommend(howMany, device, x, y, z, n, isBatched, isFloat, isForward,
                isInPlace, isReal, maxSignalInc, maxMemory, allowTransposition, squareOnly);
  std::vector<BenchmarkResult const *> *result = benchmark(*candidates);
  std::sort(result->begin(), result->end(), BenchmarkResult::execSort);
  delete candidates;
  resetDevice();
  return result;
}

std::vector<BenchmarkResult const *> *Advisor::benchmark(
    std::vector<Transform const *> &transforms) {
  std::vector<BenchmarkResult const *> *results =
      new std::vector<BenchmarkResult const *>();
  int size = transforms.size();
  for (int i = 0; i < size; i++) {
    results->push_back(Benchmarker::benchmark(transforms.at(i)));
  }
  return results;
}

}  // namespace cuFFTAdvisor
