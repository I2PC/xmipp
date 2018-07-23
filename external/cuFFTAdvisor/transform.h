#ifndef CUFFTADVISOR_TRANSFORM_H_
#define CUFFTADVISOR_TRANSFORM_H_

#include <cufft.h>
#include <cstdio>
#include "utils.h"

namespace cuFFTAdvisor {

class Transform {
 public:
  void print(FILE *stream = stdout) const;
  static void printHeader(FILE *stream = stdout);

  enum Rank { RANK_1D = 1, RANK_2D = 2, RANK_3D = 3 };

  Transform(int device, int X, int Y, int Z, int N, bool isBatched,
            bool isFloat, bool isForward, bool isInPlace, bool isReal)
      : device(device),
        X(X),
        Y(Y),
        Z(Z),
        N(N),
        isBatched(isBatched),
        isInPlace(isInPlace),
        isReal(isReal),
        isFloat(isFloat),
        isForward(isForward) {
    // preserve order of these methods!
    setRankInfo();
    setTypeInfo();
    setSizeInfo();
    validate();
  }

  int device;
  // requested size of the transform
  int X;
  int Y;
  int Z;
  int N;  // no of images to process (not necessary in batch)
  // additional transform properties
  bool isBatched;
  bool isInPlace;  // otherwise out-of-place
  bool isReal;     // otherwise C2C
  bool isFloat;    // otherwise double
  bool isForward;  // otherwise inverse

  // derived
  Rank rank;
  size_t elems;  // of the transform // FIXME remove
  size_t inTypeSize;
  size_t outTypeSize;
  size_t inElems;    // actual length of the input signal
  size_t outElems;   // actual length of the output signal
  size_t dataSizeB;  // no. of bytes needed for transform
  cufftType type;
  int xIn;   // actual X dim of the input of the transform with padding etc
  int xOut;  // actual X dim of the output of the transform with padding etc
  int repetitions;  // how many plan executions should be called

  // 'many' specific
  int nr[3];
  // FIXME implement strides. Inspiration:
  // http://docs.nvidia.com/cuda/cufft/index.html#twod-advanced-data-layout-use
  int istride, ostride;
  int idist;
  int odist;

 private:
  /**
   * Will throw std::logic_error exception with problem, if any is found
   */
  void validate();

  /**
   * Returns X dim of the signal with padding, if necessary
   */
  size_t getXDim();

  /**
   * Sets fields that have sth to do with transform sizes
   */
  void setSizeInfo();

  /**
   * Sets field that describe transform data types
   */
  void setTypeInfo();

  /**
   * Sets fields that describe rank (dimensionality)
   */
  void setRankInfo();
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_TRANSFORM_H_
