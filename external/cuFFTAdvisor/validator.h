#ifndef CUFFTADVISOR_VALIDATOR_H_
#define CUFFTADVISOR_VALIDATOR_H_

#include <cmath>
#include <stdexcept>
#include "cudaUtils.h"

namespace cuFFTAdvisor {

class Validator {
 public:
  static void validate(int x, int y, int z, int n, int device);
  static void validate(int x, int y, int z, int n, int device, int maxSignalInc,
                       int maxMemMB, bool allowTrans, bool squareOnly);
  static void validate(int device);
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_VALIDATOR_H_
