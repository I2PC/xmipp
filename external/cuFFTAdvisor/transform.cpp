#include "transform.h"

namespace cuFFTAdvisor {

void Transform::print(FILE *stream) const {
  if (NULL == stream) return;
  std::string rankStr =
      isBatched ? (std::string("many")) : (numToString(rank) + "D");
  std::string type = (isReal ? std::string("R2R") : std::string("C2C")) + " " +
                     (isForward ? "forward" : "inverse");
  fprintf(stream,
          "%s%s%s%s%s"         // rank, praceness, type
          "%s%s"               // use floats
          "%s%d%s%d%s%d%s%d",  // dims
          rankStr.c_str(),
          SEP, isInPlace ? ("in-place") : ("out-of-place"), SEP, type.c_str(),
          SEP, isFloat ? "float" : "double", SEP, X, SEP, Y, SEP, Z, SEP,
          N / repetitions);
}

void Transform::printHeader(FILE *stream) {
  if (NULL == stream) return;
  fprintf(stream,
          "rank%splaceness%stype"
          "%sdata"
          "%sX%sY%sZ%sbatch",
          SEP, SEP, SEP, SEP, SEP, SEP, SEP);
}

void Transform::validate() {
  if (X <= 0) {
    throw std::logic_error(
        "X value is not positive. Wrong input or int overflow.\n");
  }
  if (Y <= 0) {
    throw std::logic_error(
        "Y value is not positive. Wrong input or int overflow.\n");
  }
  if (Z <= 0) {
    throw std::logic_error(
        "Z value is not positive. Wrong input or int overflow.\n");
  }
  if (N <= 0) {
    throw std::logic_error(
        "N value is not positive. Wrong input or int overflow.\n");
  }
  if (isBatched && ((idist <= 0) || (odist <= 0))) {
    throw std::logic_error(
        "idist and/or odist value is not positive. Wrong input or int "
        "overflow (cuFFT is not able to process so big input)\n");
  }
}

size_t Transform::getXDim() {
  size_t padded = (X / 2 + 1) * 2;  // padding to complex numbers
  bool shouldPadd = isReal && isInPlace;
  return shouldPadd ? padded : X;
}

void Transform::setSizeInfo() {
  // set X dim
  int xFFT = (X / 2 + 1);
  if (isForward) {
    xIn = getXDim();
    // see http://docs.nvidia.com/cuda/cufft/index.html#data-layout
    xOut =
        isReal ? xFFT : xIn;  // for complex, input size is the same as output
  } else {
    // see http://docs.nvidia.com/cuda/cufft/index.html#data-layout
    xIn = isReal ? xFFT : X;
    xOut = getXDim();
  }

  // set repetitions
  if (isBatched || (RANK_1D == rank)) {
    repetitions = 1;  // N is directly included in plan
  } else {
    repetitions = N;
  }

  // set no. of elems
  inElems = (size_t)xIn * Y * Z * (N / repetitions);
  outElems = (size_t)xOut * Y * Z * (N / repetitions);
  elems = (size_t)X * Y * Z * N;
  size_t iSizeBytes = inElems * inTypeSize;
  size_t oSizeBytes = outElems * outTypeSize;
  dataSizeB =
      isInPlace ? std::max(iSizeBytes, oSizeBytes) : iSizeBytes + oSizeBytes;

  // set 'many' specific stuff
  istride = ostride = 1;
  idist = xIn * Y * Z;
  odist = xOut * Y * Z;
}

void Transform::setTypeInfo() {
  if (isFloat) {
    if (isReal) {
      if (isForward) {
        type = CUFFT_R2C;
        inTypeSize = sizeof(cufftReal);
        outTypeSize = sizeof(cufftComplex);
      } else {
        type = CUFFT_C2R;
        inTypeSize = sizeof(cufftComplex);
        outTypeSize = sizeof(cufftReal);
      }
    } else {
      type = CUFFT_C2C;
      inTypeSize = outTypeSize = sizeof(cufftComplex);
    }
  } else {  // double
    if (isReal) {
      if (isForward) {
        type = CUFFT_D2Z;
        inTypeSize = sizeof(cufftDoubleReal);
        outTypeSize = sizeof(cufftDoubleComplex);
      } else {
        type = CUFFT_Z2D;
        outTypeSize = sizeof(cufftDoubleReal);
        inTypeSize = sizeof(cufftDoubleComplex);
      }
    } else {
      type = CUFFT_Z2Z;
      inTypeSize = outTypeSize = sizeof(cufftDoubleComplex);
    }
  }
}

void Transform::setRankInfo() {
  rank = RANK_3D;
  nr[0] = Z;
  nr[1] = Y;
  nr[2] = X;
  if (1 == Z) {
    if (1 == Y) {
      rank = RANK_1D;
      nr[0] = X;
    } else {
      rank = RANK_2D;
      nr[0] = Y;
      nr[1] = X;
    }
  }
}

}  // namespace cuFFTAdvisor
