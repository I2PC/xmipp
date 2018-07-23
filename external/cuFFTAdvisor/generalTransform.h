#ifndef CUFFTADVISOR_GENERALTRANSFORM_H_
#define CUFFTADVISOR_GENERALTRANSFORM_H_

#include "utils.h"

namespace cuFFTAdvisor {

class GeneralTransform {
 public:
  GeneralTransform(int device, int X, int Y, int Z, int N,
                   Tristate::Tristate isBatched, Tristate::Tristate isFloat,
                   Tristate::Tristate isForward, Tristate::Tristate isInPlace,
                   Tristate::Tristate isReal);

  GeneralTransform(int X, int Y, int Z, const GeneralTransform &tr);
  GeneralTransform(const GeneralTransform &tr);
  GeneralTransform &operator=(const GeneralTransform &tr);

  size_t getDimSize() { return (size_t)X * Y * Z; }

  int device;
  // requested size of the transform
  int X;
  int Y;
  int Z;
  int N;  // no of images to process (not necessary in batch)
  // additional transform properties
  Tristate::Tristate isBatched;
  Tristate::Tristate isFloat;    // otherwise double
  Tristate::Tristate isForward;  // otherwise inverse
  Tristate::Tristate isInPlace;  // otherwise out-of-place
  Tristate::Tristate isReal;     // otherwise C2C
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_GENERALTRANSFORM_H_
