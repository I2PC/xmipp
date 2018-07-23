#include "benchmarker.h"

namespace cuFFTAdvisor {

size_t getXDim(Transform const *input) {
  size_t padded = (input->X / 2 + 1) * 2;  // padding to complex numbers
  bool shouldPadd = input->isReal && input->isInPlace;
  return shouldPadd ? padded : input->X;
}

void Benchmarker::checkMemory(BenchmarkResult *res) {
  size_t freeMem = getFreeMemory(res->transform->device);
  // check if you have sufficient memory
  size_t planSize = std::max(res->planSizeEstimateB, res->planSizeEstimate2B);
  size_t totalSizeBytes = res->transform->dataSizeB + planSize;
  if (0 == planSize) {
    throw std::runtime_error(
        "Plan estimation probably failed, as the estimated size is 0(zero)");
  }
  if (freeMem <= totalSizeBytes) {
    char buffer[100];
    snprintf(buffer, 100,
             "Insufficient memory (required %0.1fMB: plan %0.1fMB data "
             "%0.1fMB, available %0.1fMB)",
             toMB(totalSizeBytes), toMB(planSize),
             toMB(res->transform->dataSizeB), toMB(freeMem));
    throw std::runtime_error(std::string(buffer));
  }
}

void Benchmarker::estimatePlanSize(cufftHandle &plan, BenchmarkResult *res) {
  // estimate plan size
  // size is always considered to be size of the transform, i.e. 'forward'
  // fastest changing dimension is X, opposite of what cuFFT expects
  Transform const *tr = res->transform;
  if (tr->isBatched) {
    gpuErrchkFFT(cufftEstimateMany(
        tr->rank, const_cast<int *>(tr->nr), NULL, tr->istride, tr->idist, NULL,
        tr->ostride, tr->odist, tr->type, tr->N, &res->planSizeEstimateB));
    gpuErrchkFFT(cufftGetSizeMany(plan, tr->rank, const_cast<int *>(tr->nr),
                                  NULL, tr->istride, tr->idist, NULL,
                                  tr->ostride, tr->odist, tr->type, tr->N,
                                  &res->planSizeEstimate2B));
  } else {
    switch (tr->rank) {
      case (Transform::RANK_1D):
        gpuErrchkFFT(
            cufftEstimate1d(tr->X, tr->type, tr->N, &res->planSizeEstimateB));
        gpuErrchkFFT(cufftGetSize1d(plan, tr->X, tr->type, tr->N,
                                    &res->planSizeEstimate2B));
        break;
      case (Transform::RANK_2D):
        gpuErrchkFFT(
            cufftEstimate2d(tr->Y, tr->X, tr->type, &res->planSizeEstimateB));
        gpuErrchkFFT(cufftGetSize2d(plan, tr->Y, tr->X, tr->type,
                                    &res->planSizeEstimate2B));
        break;
      case (Transform::RANK_3D):
        gpuErrchkFFT(cufftEstimate3d(tr->Z, tr->Y, tr->X, tr->type,
                                     &res->planSizeEstimateB));
        gpuErrchkFFT(cufftGetSize3d(plan, tr->Z, tr->Y, tr->X, tr->type,
                                    &res->planSizeEstimate2B));
        break;
      default:
        throw std::invalid_argument("No such rank of transform");
    }
  }
}

void Benchmarker::estimatePlanSize(BenchmarkResult *res) {
  gpuErrchk(cudaSetDevice(res->transform->device));
  cufftHandle plan = cuFFTAdvisor::createPlan();
  estimatePlanSize(plan, res);
  cuFFTAdvisor::destroyPlan(plan);
}

void Benchmarker::createPlan(cufftHandle &plan, Transform const *tr,
                             BenchmarkResult *res) {
  clock_t begin = clock();
  // size is always considered to be size of the transform, i.e. 'forward'
  // fastest changing dimension is X, so we need to 'swap' them
  if (tr->isBatched) {
    gpuErrchkFFT(cufftPlanMany(&plan, tr->rank, const_cast<int *>(tr->nr), NULL,
                               tr->istride, tr->idist, NULL, tr->ostride,
                               tr->odist, tr->type, tr->N));
  } else {
    switch (tr->rank) {
      case (Transform::RANK_1D):
        gpuErrchkFFT(cufftPlan1d(&plan, tr->X, tr->type, tr->N));
        break;
      case (Transform::RANK_2D):
        gpuErrchkFFT(cufftPlan2d(&plan, tr->Y, tr->X, tr->type));
        break;
      case (Transform::RANK_3D):
        gpuErrchkFFT(cufftPlan3d(&plan, tr->Z, tr->Y, tr->X, tr->type));
        break;
      default:
        throw std::invalid_argument("No such rank of transform");
    }
  }
  gpuErrchk(cudaDeviceSynchronize());

  // measure plan creation
  res->planTimeMS = float(clock() - begin) / CLOCKS_PER_MS;
  gpuErrchkFFT(cufftGetSize(plan, &res->planSizeRealB));

#ifdef DEBUG
  printf(
      "batched: %d rank %d\n"
      "in: %lu (%d x %d x %d x %d * %lu) "
      "out: %lu (%d x %d x %d x %d * %lu)\n",
      tr->isBatched, tr->rank, tr->inElems, tr->xIn, tr->Y, tr->Z, tr->N,
      tr->inTypeSize, tr->outElems, tr->xOut, tr->Y, tr->Z, tr->N,
      tr->outTypeSize);
#endif
}

float Benchmarker::execute(cufftHandle &plan, void *d_in, void *d_out,
                           Transform const *tr) {
  clock_t begin = clock();
  if (tr->isFloat) {
    if (tr->isReal) {
      if (tr->isForward) {
        gpuErrchkFFT(
            cufftExecR2C(plan, (cufftReal *)d_in, (cufftComplex *)d_out));
      } else {
        gpuErrchkFFT(
            cufftExecC2R(plan, (cufftComplex *)d_in, (cufftReal *)d_out));
      }
    } else {
      gpuErrchkFFT(cufftExecC2C(plan, (cufftComplex *)d_in,
                                (cufftComplex *)d_out,
                                tr->isForward ? CUFFT_FORWARD : CUFFT_INVERSE));
    }
  } else {
    if (tr->isReal) {
      if (tr->isForward) {
        gpuErrchkFFT(cufftExecD2Z(plan, (cufftDoubleReal *)d_in,
                                  (cufftDoubleComplex *)d_out));
      } else {
        gpuErrchkFFT(cufftExecZ2D(plan, (cufftDoubleComplex *)d_in,
                                  (cufftDoubleReal *)d_out));
      }
    } else {
      gpuErrchkFFT(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_in,
                                (cufftDoubleComplex *)d_out,
                                tr->isForward ? CUFFT_FORWARD : CUFFT_INVERSE));
    }
  }
  gpuErrchk(cudaDeviceSynchronize());
  return float(clock() - begin) / CLOCKS_PER_MS;
}

void Benchmarker::execute(cufftHandle &plan, Transform const *tr,
                          BenchmarkResult *res) {
  size_t iBytes = tr->inElems * tr->inTypeSize;
  size_t oBytes = tr->outElems * tr->outTypeSize;
  size_t inPlaceBytes = std::max(iBytes, oBytes);
  void *d_in, *d_out;
  void *h_in = malloc(iBytes);

  // allocate GPU data
  if (tr->isInPlace) {
    gpuErrchk(cudaMalloc((void **)&d_in, inPlaceBytes));
    d_out = d_in;
  } else {
    gpuErrchk(cudaMalloc((void **)&d_in, iBytes));
    gpuErrchk(cudaMalloc((void **)&d_out, oBytes));
  }

#ifdef DEBUG
  printf(
      "execute:\n"
      "%p %lu(%lu x %lu)\n"
      "%p %lu(%lu x %lu), %d reps, isForward %d\n",
      d_in, iBytes, tr->inElems, tr->inTypeSize, d_out, oBytes, tr->outElems,
      tr->outTypeSize, tr->repetitions, tr->isForward);
#endif

  float execTime = 0.f;
  srand(time(NULL));
  for (int r = 0; r < 3; r++) {
    // copy to GPU to remove data from cache
    gpuErrchk(cudaMemcpy(d_in, h_in, iBytes, cudaMemcpyHostToDevice));
    // execute plan
    execTime += execute(plan, d_in, d_out, tr);
  }
  execTime /= 3.;  // average run time

  // free data
  gpuErrchk(cudaFree(d_in));
  if (!tr->isInPlace) gpuErrchk(cudaFree(d_out));
  free(h_in);

  // update results
  res->execTimeMS =
      execTime /
      (tr->N / tr->repetitions);  // normalize exec time to one signal
  res->totalTimeMS =
      execTime * tr->repetitions;
}

cuFFTAdvisor::BenchmarkResult const *Benchmarker::benchmark(
    cuFFTAdvisor::Transform const *tr) {
  gpuErrchk(cudaSetDevice(tr->device));

  BenchmarkResult *result = new BenchmarkResult(tr);

  cufftHandle plan = cuFFTAdvisor::createPlan();
  estimatePlanSize(plan, result);
  try {
    checkMemory(result);
    createPlan(plan, tr, result);
    execute(plan, tr, result);
  } catch (std::runtime_error &e) {
    result->errMsg = e.what();
  }
  cuFFTAdvisor::destroyPlan(plan);
  return result;
}

}  // namespace cuFFTAdvisor
