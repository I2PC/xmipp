#include "generalTransform.h"

namespace cuFFTAdvisor {

GeneralTransform::GeneralTransform(int device, int X, int Y, int Z, int N,
                                   Tristate::Tristate isBatched,
                                   Tristate::Tristate isFloat,
                                   Tristate::Tristate isForward,
                                   Tristate::Tristate isInPlace,
                                   Tristate::Tristate isReal)
    : device(device),
      X(X),
      Y(Y),
      Z(Z),
      N(N),
      isBatched(isBatched),
      isFloat(isFloat),
      isForward(isForward),
      isInPlace(isInPlace),
      isReal(isReal) {}

GeneralTransform::GeneralTransform(int X, int Y, int Z,
                                   const GeneralTransform &tr)
    : device(tr.device),
      X(X),
      Y(Y),
      Z(Z),
      N(tr.N),
      isBatched(tr.isBatched),
      isFloat(tr.isFloat),
      isForward(tr.isForward),
      isInPlace(tr.isInPlace),
      isReal(tr.isReal) {}

GeneralTransform::GeneralTransform(const GeneralTransform &tr) { *this = tr; }

GeneralTransform &GeneralTransform::operator=(const GeneralTransform &tr) {
  if (this != &tr) {
    this->device = tr.device;
    this->X = tr.X;
    this->Y = tr.Y;
    this->Z = tr.Z;
    this->N = tr.N;
    this->isBatched = tr.isBatched;
    this->isFloat = tr.isFloat;
    this->isForward = tr.isForward;
    this->isInPlace = tr.isInPlace;
    this->isReal = tr.isReal;
  }
  return *this;
}

}  // namespace cuFFTAdvisor
