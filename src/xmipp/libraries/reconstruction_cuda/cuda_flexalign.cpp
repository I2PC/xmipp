#include "libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp"
#include "libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp"
#include "reconstruction_cuda/cuda_asserts.h"
#include <iostream>
#include <reconstruction_cuda/cuda_flexalign.h>

using namespace umpalumpa::data;
using namespace umpalumpa;

template <typename T> auto createFilterPayload(T *ptrCPU, const Size &size) {
  auto ld = LogicalDescriptor(size, size, "Filter");
  auto pd = PhysicalDescriptor(
      ld.GetPaddedSize().total * Sizeof(DataType::kFloat), DataType::kFloat);
  void *ptr = nullptr;
  gpuErrchk(cudaMalloc(&ptr, pd.bytes));
  gpuErrchk(cudaMemcpy(ptr, ptrCPU, pd.bytes, cudaMemcpyDefault));
  return Payload(ptr, ld, pd, "Filter");
}

template <typename T>
void performFFT(T *inOutData, int noOfImgs, int inX, int inY, int batchSize,
                int outX, int outY, T *filter) {
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
  auto ldInFull = FourierDescriptor(sizeInFull, sizeInFull);
  auto pdInFull =
      PhysicalDescriptor(ldInFull.GetPaddedSize().total *
                             Sizeof(umpalumpa::data::DataType::kFloat),
                         umpalumpa::data::DataType::kFloat);
  auto inFull = Payload(inOutData, ldInFull, pdInFull, "Input data");

  // create intermediary Payload
  auto ldBatch = FourierDescriptor(sizeInBatch, sizeInBatch,
                                   FourierDescriptor::FourierSpaceDescriptor());
  printf("ldBatch size in: %lu %lu %lu %lu\n", sizeInBatch.x, sizeInBatch.y,
         sizeInBatch.z, sizeInBatch.n);
  auto bytesBatch = ldBatch.GetPaddedSize().total *
                    Sizeof(umpalumpa::data::DataType::kFloat) * 2;
  printf("ldBatch size in: %lu %lu %lu %lu\n", ldBatch.GetPaddedSize().x,
         ldBatch.GetPaddedSize().y, ldBatch.GetPaddedSize().z,
         ldBatch.GetPaddedSize().n);
  printf("bytesBatch: %lu\n", bytesBatch);
  auto pdBatch =
      PhysicalDescriptor(bytesBatch, umpalumpa::data::DataType::kFloat);
  void *batchPtr = nullptr;
  gpuErrchk(cudaMalloc(&batchPtr, pdBatch.bytes));
  auto outBatch = Payload(batchPtr, ldBatch, pdBatch, "Batch data");

  // create output Payload
  // FIXME make sure that we cannot overwrite the input data by the resulting
  // cropped data
  auto ldOutFull = FourierDescriptor(
      sizeOutFull, sizeOutFull, FourierDescriptor::FourierSpaceDescriptor());
  auto bytesOutFull = ldOutFull.GetPaddedSize().total *
                      Sizeof(umpalumpa::data::DataType::kFloat) * 2;
  auto pdOutFull =
      PhysicalDescriptor(bytesOutFull, umpalumpa::data::DataType::kFloat);
  auto outFull = Payload(inOutData, ldOutFull, pdOutFull, "Output data");

  // create filter Payload
  auto filterP = createFilterPayload(filter, sizeFilter);

  // create transformers
  auto fftTransformer = fourier_transformation::FFTCUDA(0);
  auto cropTransformer = fourier_processing::FP_CUDA(0);

  bool doInit = true;
  for (size_t offset = 0; offset < noOfImgs; offset += batchSize) {
    auto inFFT = fourier_transformation::AFFT::InputData(
        inFull.Subset(offset, batchSize));
    // trying to prefetch data to avoid page faults
    gpuErrchk(
        cudaMemPrefetchAsync(inFFT.data.ptr, inFFT.data.dataInfo.bytes, 0));

    auto batch = inFFT.data.info.GetSize().n;
    std::cout << "Processing images " << offset << "-" << batch << std::endl;
    // start at 0 to reuse the temporal storage, set N according to what is left
    auto outFFT =
        fourier_transformation::AFFT::ResultData(outBatch.Subset(0, batch));
    auto inCrop = fourier_processing::AFP::InputData(outBatch.Subset(0, batch),
                                                     std::move(filterP));
    auto outTmp = outFull.Subset(offset, batch);
    outTmp.ptr = inOutData + (outTmp.info.GetSize().single * offset * 2);
    auto outCrop = fourier_processing::AFP::OutputData(std::move(outTmp));
    // printf("inFFT: %p %p\n", inFFT.data.ptr, inOutData + (offset * inX *
    // inY)); printf("outFFT: %p %p\n", outFFT.data.ptr, batchPtr);
    // printf("inCrop: %p %p\n", inCrop.data.ptr, batchPtr);
    // printf("outCrop: %p %p\n", outCrop.data.ptr,
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
    // cudaMemset(outFFT.data.ptr, 0, pdBatch.bytes);
    assert(fftTransformer.Execute(outFFT, inFFT));

    // auto &s = outFFT.data.info.GetSize();
    // auto *tmp = new std::complex<float>[s.total];
    // printf("%lu %lu %lu %lu -> %lu %lu\n", s.x, s.y, s.z, s.n, s.total,
    //        outFFT.data.dataInfo.bytes);
    // cudaMemcpy(tmp, outFFT.data.ptr, outFFT.data.dataInfo.bytes * 2,
    //            cudaMemcpyDeviceToHost);
    // Image<float> img(s.x, s.y, s.z, s.n);
    // for (size_t i = 0; i < s.total; ++i) {
    //   img.data.data[i] = tmp[i].real();
    // }
    // img.write("fft_new_" + std::to_string(offset) + ".mrc");
    // delete[] tmp;
    // gpuErrchk(cudaPeekAtLastError());

    // cudaMemset(inFFT.data.ptr, 0, inFFT.data.dataInfo.bytes);

    assert(cropTransformer.Execute(outCrop, inCrop));
    std::cout << "iteration done" << std::endl;
  }

  fftTransformer.Synchronize();
  cropTransformer.Synchronize();
  gpuErrchk(cudaFree(batchPtr));
  gpuErrchk(cudaFree(filterP.ptr));
}

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
  auto pdSpatial =
      PhysicalDescriptor(ldSpatial.GetPaddedSize().total *
                             Sizeof(umpalumpa::data::DataType::kFloat),
                         umpalumpa::data::DataType::kFloat);

  auto ldFrequency = FourierDescriptor(
      inSize, inSize, FourierDescriptor::FourierSpaceDescriptor());
  auto frequencySizeInBytes = ldFrequency.GetPaddedSize().total *
                              Sizeof(umpalumpa::data::DataType::kFloat) * 2;
  auto pdFrequency = PhysicalDescriptor(frequencySizeInBytes,
                                        umpalumpa::data::DataType::kFloat);

  auto ldOut = FourierDescriptor(outSize, outSize,
                                 FourierDescriptor::FourierSpaceDescriptor());
  auto outSizeInBytes = ldOut.GetPaddedSize().total *
                        Sizeof(umpalumpa::data::DataType::kFloat) * 2;
  auto pdOut =
      PhysicalDescriptor(outSizeInBytes, umpalumpa::data::DataType::kFloat);

  auto inFullP = Payload(inOutData, ldSpatial, pdSpatial, "Input data (full)");
  auto outFFTFullP =
      Payload(nullptr, ldFrequency, pdFrequency, "Result FFT data (full)");
  auto outFullP = Payload(inOutData, ldOut, pdOut, "Result data (full)");

  auto ldFilter =
      LogicalDescriptor(outSize.CopyFor(1), outSize.CopyFor(1), "Filter");
  auto pdFilter =
      PhysicalDescriptor(ldFilter.GetPaddedSize().total *
                             Sizeof(umpalumpa::data::DataType::kFloat),
                         umpalumpa::data::DataType::kFloat);
  auto filterP = Payload(filter, ldFilter, pdFilter, "Filter");

  void *dataFreq = nullptr;
  gpuErrchk(cudaMallocManaged(&dataFreq,
                              outFFTFullP.Subset(0, batchSize).dataInfo.bytes));
  printf("managed memory: %p\n", dataFreq);

  auto fftTransformer = fourier_transformation::FFTCUDA(0);
  auto cropTransformer = fourier_processing::FP_CUDA(0);
  bool isFirstIter = true;
  std::cout << "Before the loop" << std::endl;
  for (size_t offset = 0; offset < noOfImgs; offset += batchSize) {
    auto in = fourier_transformation::AFFT::InputData(
        inFullP.Subset(offset, batchSize));
    auto tmp1 = outFFTFullP.Subset(offset, batchSize);
    tmp1.ptr = dataFreq;
    auto tmp2 = tmp1;
    auto outFT = fourier_transformation::AFFT::ResultData(std::move(tmp1));
    auto inFT =
        fourier_processing::AFP::InputData(std::move(tmp2), std::move(filterP));
    auto out = fourier_processing::AFP::OutputData(
        std::move(outFullP.Subset(offset, batchSize)));
    if (isFirstIter) {
      isFirstIter = false;
      std::cout << "About to initialize transformers" << std::endl;
      assert(fftTransformer.Init(outFT, in, fftSettings));
      assert(cropTransformer.Init(out, inFT, cropSettings));
      std::cout << "transformers initialized" << std::endl;
    }
    printf("managed memory in the loop: %p\n", dataFreq);
    fflush(stdout);
    fftTransformer.Execute(outFT, in);
    fftTransformer.Synchronize();
    std::cout << "fft done" << std::endl;
    cropTransformer.Execute(out, inFT);
    cropTransformer.Synchronize();
    std::cout << "crop done" << std::endl;
  }
  printf("managed memory before synch and release: %p\n", dataFreq);
  fftTransformer.Synchronize();
  cropTransformer.Synchronize();
  gpuErrchk(cudaFree(dataFreq));
}

template void performFFTAndScale<float>(float *inOutData, int noOfImgs, int inX,
                                        int inY, int inBatch, int outFFTX,
                                        int outY, float *filter);
template void performFFT<float>(float *inOutData, int noOfImgs, int inX,
                                int inY, int inBatch, int outFFTX, int outY,
                                float *filter);