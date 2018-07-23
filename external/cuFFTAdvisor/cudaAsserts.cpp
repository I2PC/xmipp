#include "cudaAsserts.h"

namespace cuFFTAdvisor {

void gpuErrchk(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    char buffer[100];
    snprintf(buffer, 100, "GPUassert: %s %s %d", cudaGetErrorString(code), file,
             line);
    if (abort) throw std::runtime_error(std::string(buffer));
  }
}

void gpuErrchkFFT(cufftResult_t code, const char *file, int line, bool abort) {
  if (code != CUFFT_SUCCESS) {
    char buffer[100];
    snprintf(buffer, 100, "GPUassertFFT: %s %s %d", _cudaGetErrorEnum(code),
             file, line);
    if (abort) throw std::runtime_error(std::string(buffer));
  }
}

}  // namespace cuFFTAdvisor
