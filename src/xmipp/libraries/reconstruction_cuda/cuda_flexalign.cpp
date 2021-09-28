#include "libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp"
#include "libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp"
#include "reconstruction_cuda/cuda_asserts.h"
#include <iostream>
#include <reconstruction_cuda/cuda_flexalign.h>

using namespace umpalumpa::data;
using namespace umpalumpa;
// namespace ft = umpalumpa::fourier_transformation;
// using namespace umpalumpa::fourier_processing;
// using namespace umpalumpa::data;

template <typename T>
void performFFTAndScale(T *inOutData, int noOfImgs, int inX, int inY,
                        int batchSize, int outFFTX, int outY, T *filter) {
  //   ft::Locality locality = ft::Locality::kOutOfPlace;
  //   Settings settings(locality);
  //   std::cout << settings.GetApplyFilter();

  auto inSize = Size(inX, inY, 1, noOfImgs);
  auto outSize = Size(outFFTX, outY, 1, noOfImgs);

  auto fftSettings = fourier_transformation::Settings(
      fourier_transformation::Locality::kOutOfPlace,
      fourier_transformation::Direction::kForward);
  auto cropSettings = fourier_processing::Settings(
      fourier_transformation::Locality::kOutOfPlace);
  auto ldSpatial = FourierDescriptor(inSize, inSize);
  auto pdSpatial = PhysicalDescriptor(ldSpatial.GetPaddedSize().total *
                                          Sizeof(DataType::kFloat),
                                      DataType::kFloat);

  auto ldFrequency = FourierDescriptor(
      inSize, inSize, FourierDescriptor::FourierSpaceDescriptor());
  auto frequencySizeInBytes =
      ldFrequency.GetPaddedSize().total * Sizeof(DataType::kFloat) * 2;
  auto pdFrequency = PhysicalDescriptor(frequencySizeInBytes, DataType::kFloat);

  auto ldOut = FourierDescriptor(outSize, outSize,
                                 FourierDescriptor::FourierSpaceDescriptor());
  auto outSizeInBytes =
      ldOut.GetPaddedSize().total * Sizeof(DataType::kFloat) * 2;
  auto pdOut = PhysicalDescriptor(outSizeInBytes, DataType::kFloat);

  auto inFullP = Payload(inOutData, ldSpatial, pdSpatial, "Input data (full)");
  auto outFFTFullP =
      Payload(nullptr, ldFrequency, pdFrequency, "Result FFT data (full)");
  auto outFullP = Payload(inOutData, ldOut, pdOut, "Result data (full)");

  auto ldFilter =
      LogicalDescriptor(outSize.CopyFor(1), outSize.CopyFor(1), "Filter");
  auto pdFilter = PhysicalDescriptor(ldFilter.GetPaddedSize().total *
                                     Sizeof(DataType::kFloat), DataType::kFloat);
  auto filterP = Payload(filter, ldFilter, pdFilter, "Filter");

  void *dataFreq = nullptr;
  gpuErrchk(cudaMallocManaged(&dataFreq,
                              outFFTFullP.Subset(0, batchSize).dataInfo.bytes));

  auto fftTransformer = fourier_transformation::FFTCUDA(0);
  auto cropTransformer = fourier_processing::FP_CUDA(0);
  bool isFirstIter = true;
  for (size_t offset = 0; offset < noOfImgs; offset += batchSize) {
    auto in = fourier_transformation::AFFT::InputData(
        inFullP.Subset(offset, batchSize));
    auto tmp1 = outFFTFullP.Subset(offset, batchSize);
    tmp1.ptr = dataFreq;
    auto tmp2 = tmp1;
    auto outFT = fourier_transformation::AFFT::ResultData(std::move(tmp1));
    auto inFT = fourier_processing::AFP::InputData(std::move(tmp2), std::move(filterP));
    auto out = fourier_processing::AFP::OutputData(
        std::move(outFullP.Subset(offset, batchSize)));
    if (isFirstIter) {
      isFirstIter = false;
      assert(fftTransformer.Init(outFT, in, fftSettings));
      assert(cropTransformer.Init(out, inFT, cropSettings));
    }
    fftTransformer.Execute(outFT, in);
    cropTransformer.Execute(out, inFT);
  }
  gpuErrchk(cudaFree(dataFreq));
}

template void performFFTAndScale<float>(float *inOutData, int noOfImgs, int inX,
                                        int inY, int inBatch, int outFFTX,
                                        int outY, float *filter);