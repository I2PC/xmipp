// Xmipp includes
#include "cuda_forward_art_zernike3d.h"
// Standard includes
#include <iostream>


// Common functions
template<typename T>
cudaError cudaMallocAndCopy(T **target, const T *source, size_t numberOfElements, size_t memSize = 0) {
    size_t elemSize = numberOfElements * sizeof(T);
    memSize = memSize == 0 ? elemSize : memSize * sizeof(T);

    cudaError err = cudaSuccess;
    if ((err = cudaMalloc(target, memSize)) != cudaSuccess) {
        *target = NULL;
        return err;
    }

    if ((err = cudaMemcpy(*target, source, elemSize, cudaMemcpyHostToDevice)) != cudaSuccess) {
        cudaFree(*target);
        *target = NULL;
    }

    if (memSize > elemSize) {
        cudaMemset((*target) + numberOfElements, 0, memSize - elemSize);
    }

    return err;
}

void processCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(err));
        exit(err);
    }
}

// Copies data from CPU to the GPU and at the same time transforms from
// type 'U' to type 'T'. Works only for numeric types
template<typename Target, typename Source>
void transformData(Target **dest, Source *source, size_t n, bool mallocMem = true) {
    std::vector <Target> tmp(source, source + n);

    if (mallocMem) {
        if (cudaMalloc(dest, sizeof(Target) * n) != cudaSuccess) {
            processCudaError();
        }
    }

    if (cudaMemcpy(*dest, tmp.data(), sizeof(Target) * n, cudaMemcpyHostToDevice) != cudaSuccess) {
        processCudaError();
    }
}

// CUDAForwardArtZernike3D methods

template<typename PrecisionType>
CUDAForwardArtZernike3D<PrecisionType>::CUDAForwardArtZernike3D(
        const CUDAForwardArtZernike3D<PrecisionType>::ConstantParameters parameters) {
    (void) parameters;
}

template<typename PrecisionType>
CUDAForwardArtZernike3D<PrecisionType>::~CUDAForwardArtZernike3D() {

}

template<typename PrecisionType>
template<bool usesZernike>
void CUDAForwardArtZernike3D<PrecisionType>::runForwardKernel(
        const std::vector<PrecisionType> &clnm,
        std::vector<Image<PrecisionType>> &P,
        std::vector<Image<PrecisionType>> &W) {
    if (usesZernike) {
        return;
    }
}

template<typename PrecisionType>
template<bool usesZernike>
void CUDAForwardArtZernike3D<PrecisionType>::runBackwardKernel(const std::vector<PrecisionType> &clnm,
                                                               const Image<PrecisionType> &Idiff) {
    if (usesZernike) {
        return;
    }
}