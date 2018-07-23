#ifndef CUFFTADVISOR_CUDAUTILS_H_
#define CUFFTADVISOR_CUDAUTILS_H_

#include <cuda_runtime.h>
#include "cudaAsserts.h"
#include "utils.h"

namespace cuFFTAdvisor {

static inline int getDeviceCount() {
  int deviceCount = 0;
  gpuErrchk(cudaGetDeviceCount(&deviceCount));
  return deviceCount;
}

static inline size_t getFreeMemory(int dev) {
  gpuErrchk(cudaSetDevice(dev));
  size_t free, total;
  gpuErrchk(cudaMemGetInfo(&free, &total));
  return free;
}

static inline size_t getTotalMemory(int dev) {
  gpuErrchk(cudaSetDevice(dev));
  size_t free, total;
  gpuErrchk(cudaMemGetInfo(&free, &total));
  return total;
}

static inline cufftHandle createPlan() {
  cufftHandle plan;
  gpuErrchkFFT(cufftCreate(&plan));
  return plan;
}

static inline void resetDevice() { gpuErrchk(cudaDeviceReset()); }

static inline void destroyPlan(cufftHandle &plan) {
  gpuErrchkFFT(cufftDestroy(plan));
}

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_CUDAUTILS_H_
