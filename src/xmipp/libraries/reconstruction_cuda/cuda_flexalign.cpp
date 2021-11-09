#include "libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp"
#include "libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp"
#include "reconstruction_cuda/cuda_asserts.h"
#include <iostream>
#include <reconstruction_cuda/cuda_flexalign.h>

using namespace umpalumpa::data;
using namespace umpalumpa;

template <typename T> auto createFilterPayload(T *ptrCPU, const Size &size) {
  auto ld = LogicalDescriptor(size);
  auto bytes = ld.GetPaddedSize().total * Sizeof(DataType::kFloat);
  void *ptr = nullptr;
  gpuErrchk(cudaMalloc(&ptr, bytes));
  gpuErrchk(cudaMemcpy(ptr, ptrCPU, bytes, cudaMemcpyDefault));
  auto pd =
      PhysicalDescriptor(ptr, bytes, DataType::kFloat, ManagedBy::CUDA, 0);
  return Payload(ld, pd, "Filter");
}

template <typename T>
void performFFTAndScale(T *inOutData, int noOfImgs, int inX, int inY,
                        int batchSize, int outX, int outY, T *filter) {
  // define sizes
  auto sizeInFull = Size(inX, inY, 1, noOfImgs);
  auto sizeInBatch = Size(inX, inY, 1, batchSize);
  auto sizeOutFull = Size(outX, outY, 1, noOfImgs);
  auto sizeFilter =
      Size(outX / 2 + 1, outY, 1,
           1); // FIXME maybe create method to get the hermitian half size?

  // define settings
  auto fftSettings = fourier_transformation::Settings(
      fourier_transformation::Locality::kOutOfPlace,
      fourier_transformation::Direction::kForward);

  auto cropSettings = fourier_processing::Settings(
      fourier_transformation::Locality::kOutOfPlace);
  cropSettings.SetApplyFilter(true);
  cropSettings.SetNormalize(true);

  // create input Payload
  auto ldInFull = FourierDescriptor(sizeInFull, PaddingDescriptor());
  auto pdInFull = PhysicalDescriptor(
      inOutData, ldInFull.GetPaddedSize().total * Sizeof(DataType::kFloat),
      DataType::kFloat, ManagedBy::CUDA, 0);
  auto inFull = Payload(ldInFull, pdInFull, "Input data");

  // create intermediary Payload
  auto ldBatch = FourierDescriptor(sizeInBatch, PaddingDescriptor(),
                                   FourierDescriptor::FourierSpaceDescriptor());
  printf("ldBatch size in: %lu %lu %lu %lu\n", sizeInBatch.x, sizeInBatch.y,
         sizeInBatch.z, sizeInBatch.n);
  auto bytesBatch =
      ldBatch.GetPaddedSize().total * Sizeof(DataType::kComplexFloat);
  printf("ldBatch size in: %lu %lu %lu %lu\n", ldBatch.GetPaddedSize().x,
         ldBatch.GetPaddedSize().y, ldBatch.GetPaddedSize().z,
         ldBatch.GetPaddedSize().n);
  printf("bytesBatch: %lu\n", bytesBatch);
  void *batchPtr = nullptr;
  gpuErrchk(cudaMalloc(&batchPtr, bytesBatch));
  auto pdBatch = PhysicalDescriptor(
      batchPtr, bytesBatch, DataType::kComplexFloat, ManagedBy::CUDA, 0);
  auto outBatch = Payload(ldBatch, pdBatch, "Batch data");

  // create output Payload
  // FIXME make sure that we cannot overwrite the input data by the resulting
  // cropped data
  auto ldOutFull =
      FourierDescriptor(sizeOutFull, PaddingDescriptor(),
                        FourierDescriptor::FourierSpaceDescriptor());
  auto bytesOutFull =
      ldOutFull.GetPaddedSize().total * Sizeof(DataType::kComplexFloat);
  auto pdOutFull = PhysicalDescriptor(
      inOutData, bytesOutFull, DataType::kComplexFloat, ManagedBy::CUDA, 0);
  auto outFull = Payload(ldOutFull, pdOutFull, "Output data");

  // create filter Payload
  auto filterP = createFilterPayload(filter, sizeFilter);

  // create transformers
  auto fftTransformer = fourier_transformation::FFTCUDA(0);
  auto cropTransformer = fourier_processing::FPCUDA(0);

  bool doInit = true;
  for (size_t offset = 0; offset < noOfImgs; offset += batchSize) {
    auto inFFT = fourier_transformation::AFFT::InputData(
        inFull.Subset(offset, batchSize));
    // trying to prefetch data to avoid page faults
    gpuErrchk(cudaMemPrefetchAsync(inFFT.GetData().GetPtr(),
                                   inFFT.GetData().dataInfo.GetBytes(), 0));

    auto batch = inFFT.GetData().info.GetSize().n;
    std::cout << "Processing images " << offset << "-" << offset + batch << std::endl;
    // start at 0 to reuse the temporal storage, set N according to what is left
    auto outFFT =
        fourier_transformation::AFFT::OutputData(outBatch.Subset(0, batch));
    auto inCrop = fourier_processing::AFP::InputData(outBatch.Subset(0, batch),
                                                     std::move(filterP));
    auto outCrop =
        fourier_processing::AFP::OutputData(outFull.Subset(offset, batch));
    // printf("inFFT: %p %p\n", inFFT.GetData().GetPtr(), inOutData + (offset *
    // inX * inY)); printf("outFFT: %p %p\n", outFFT.GetData().GetPtr(),
    // batchPtr); printf("inCrop: %p %p\n", inCrop.GetData().GetPtr(),
    // batchPtr); printf("outCrop: %p %p\n", outCrop.GetData().GetPtr(),
    //        inOutData + (offset * (outX / 2 + 1) * outY));

    if (batchSize != batch) {
      std::cout << "Calling cleanup" << std::endl;
      fftTransformer.Cleanup();
      cropTransformer.Cleanup();

      doInit = true;
    }
    if (doInit) {
      std::cout << "Calling init" << std::endl;
      assert(fftTransformer.Init(outFFT, inFFT, fftSettings));
      assert(cropTransformer.Init(outCrop, inCrop, cropSettings));
      doInit = false;
    }
    std::cout << "Calling execute" << std::endl;
    // cudaMemset(outFFT.GetData().GetPtr(), 0, pdBatch.bytes);
    assert(fftTransformer.Execute(outFFT, inFFT));

    // auto &s = outFFT.GetData().info.GetSize();
    // auto *tmp = new std::complex<float>[s.total];
    // printf("%lu %lu %lu %lu -> %lu %lu\n", s.x, s.y, s.z, s.n, s.total,
    //        outFFT.GetData().dataInfo.bytes);
    // cudaMemcpy(tmp, outFFT.GetData().GetPtr(),
    // outFFT.GetData().dataInfo.bytes * 2,
    //            cudaMemcpyDeviceToHost);
    // Image<float> img(s.x, s.y, s.z, s.n);
    // for (size_t i = 0; i < s.total; ++i) {
    //   img.GetData().data[i] = tmp[i].real();
    // }
    // img.write("fft_new_" + std::to_string(offset) + ".mrc");
    // delete[] tmp;
    // gpuErrchk(cudaPeekAtLastError());

    // cudaMemset(inFFT.GetData().GetPtr(), 0, inFFT.GetData().dataInfo.bytes);

    assert(cropTransformer.Execute(outCrop, inCrop));
    std::cout << "iteration done" << std::endl;
  }

  fftTransformer.Synchronize();
  cropTransformer.Synchronize();
  gpuErrchk(cudaFree(batchPtr));
  gpuErrchk(cudaFree(filterP.GetPtr()));
}

template void performFFTAndScale<float>(float *inOutData, int noOfImgs, int inX,
                                        int inY, int inBatch, int outFFTX,
                                        int outY, float *filter);