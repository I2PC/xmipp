#include "core/metadata_label.h"
#include "core/xmipp_random_mode.h"
#include "reconstruction_adapt_cuda/volume_deform_sph_gpu.h"
#include "cuda_volume_deform_sph.h"
#include "core/matrix1d.h"
#include <iterator>
#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <exception>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include "cuda_volume_deform_sph.cu"


// Not everything will be needed to transfer every time.
// Some parameters stay the same for the whole time.
//----------------------------------------------------
// constant params:
//      Rmax2, iRmax, VI, VR, vL1, vN, vL2, vM,
//      volumesI, volumesR
//----------------------------------------------------
// changing params:
//      steps, clnm, 
//----------------------------------------------------
// parameters that can be initialized at gpu:
//      outputs(diff2,sumVD,modg,Ncount) = 0
//      VO().initZeros(VR()).setXmippOrigin()
//      Gx().initZeros(VR()).setXmippOrigin(), Gy..., Gz...
//----------------------------------------------------
// applyTransformation is true only in the very last call.
// saveDeformation is true only in the very last call and only when analyzeStrain is true

// explicit instantiations
//template class VolumeDeformSph<float>;
template class VolumeDeformSph<ComputationDataType>;

// Common functions
template<typename T>
cudaError cudaMallocAndCopy(T** target, const T* source, size_t numberOfElements, size_t memSize = 0) 
{
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

void printCudaError() 
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(err));
}

// Copies data from CPU to the GPU and at the same time transforms from
// type 'U' to type 'T'. Works only for numeric types
template<typename Target, typename Source>
void transformData(Target** dest, Source* source, size_t n, bool mallocMem = true)
{
    std::vector<Target> tmp(source, source + n);

    if (mallocMem){
        if (cudaMalloc(dest, sizeof(Target) * n) != cudaSuccess){
            printCudaError();
        }
    }

    if (cudaMemcpy(*dest, tmp.data(), sizeof(Target) * n, cudaMemcpyHostToDevice) != cudaSuccess){
        printCudaError();
    }
}
/*
// Copies data from CPU to the GPU and at the same time transforms from
// type 'U' to type 'T'. Works only for numeric types
template<typename Target, typename Source>
void transformData(Target** dest, Source* source, size_t n, bool mallocMem = true)
{
    size_t aligned = n + COPY_BLOCK_X_DIM - (n % COPY_BLOCK_X_DIM);

    Source* gpuSource;
    if (cudaMallocAndCopy(&gpuSource, source, n, aligned) != cudaSuccess) {
        printCudaError();
    }

    if (mallocMem){
        if (cudaMalloc(dest, aligned * sizeof(Target)) != cudaSuccess) {
            printCudaError();
        }
    }

    transformAndCopyKernel<<<aligned / COPY_BLOCK_X_DIM, COPY_BLOCK_X_DIM>>>(*dest, gpuSource);
    cudaDeviceSynchronize();

    cudaFree(gpuSource);
}
*/
// VolumeDeformSph methods

template<typename T>
VolumeDeformSph<T>::~VolumeDeformSph() 
{
    freeImage(images.VI);
    freeImage(images.VR);
    freeImage(images.VO);

    cudaFree(zshparams.vL1);
    cudaFree(zshparams.vL2);
    cudaFree(zshparams.vN);
    cudaFree(zshparams.vM);

    for (size_t i = 0; i < volumes.size; i++) {
        freeImage(justForFreeR[i]);
        freeImage(justForFreeI[i]);
    }
    cudaFree(volumes.R);
    cudaFree(volumes.I);

    cudaFree(steps);
    cudaFree(clnm);

    cudaFree(outputs);

    freeImage(deformImages.Gx);
    freeImage(deformImages.Gy);
    freeImage(deformImages.Gz);
}

template<typename T>
void VolumeDeformSph<T>::freeImage(ImageData<T> &im) 
{
    if (im.data != nullptr)
        cudaFree(im.data);
}

template<typename T>
void VolumeDeformSph<T>::associateWith(ProgVolumeDeformSphGpu* prog) 
{
    program = prog;
}

template<typename T>
void VolumeDeformSph<T>::setupConstantParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    this->Rmax2 = program->Rmax * program->Rmax;
    this->iRmax = 1 / program->Rmax;
    setupImage(program->VI, images.VI);
    setupImage(program->VR, images.VR);
    setupZSHparams();
    setupVolumes();
}

template<typename T>
void VolumeDeformSph<T>::setupChangingParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    unsigned stepsSize = program->steps_cp.size() * sizeof(T);
    unsigned clnmSize = program->clnm.size() * sizeof(T);

    if (this->steps == nullptr)
        if (cudaMalloc(&(this->steps), stepsSize) != cudaSuccess)
            printCudaError();
    if (this->clnm == nullptr)
        if (cudaMalloc(&(this->clnm), clnmSize) != cudaSuccess)
            printCudaError();

/*
    if (cudaMemcpy(this->steps, program->steps_cp.vdata, stepsSize, cudaMemcpyHostToDevice) != cudaSuccess)
        printCudaError();
    if (cudaMemcpy(this->clnm, program->clnm.vdata, clnmSize, cudaMemcpyHostToDevice) != cudaSuccess)
        printCudaError();
*/
    transformData(&(this->steps), program->steps_cp.vdata, program->steps_cp.size(), false);
    transformData(&(this->clnm), program->clnm.vdata, program->clnm.size(), false);

    this->applyTransformation = program->applyTransformation;
    this->saveDeformation = program->saveDeformation;

    if (applyTransformation) {
        setupImage(images.VR, images.VO);
    }
    if (saveDeformation) {
        setupImage(images.VR, deformImages.Gx);
        setupImage(images.VR, deformImages.Gy);
        setupImage(images.VR, deformImages.Gz);
    }
}

template<typename T>
KernelOutputs<T> VolumeDeformSph<T>::getOutputs() 
{
    return exOuts;
}

template<typename T>
void VolumeDeformSph<T>::transferImageData(Image<double>& outputImage, ImageData<T>& inputData) 
{
    size_t elements = inputData.xDim * inputData.yDim * inputData.zDim;
    std::vector<T> tVec(elements);
    cudaMemcpy(tVec.data(), inputData.data, sizeof(T) * elements, cudaMemcpyDeviceToHost);
    std::vector<double> dVec(tVec.begin(), tVec.end());
    memcpy(outputImage().data, dVec.data(), sizeof(double) * elements);
    /*
    size_t size = inputData.xDim * inputData.yDim * inputData.zDim * sizeof(T);
    cudaMemcpy(outputImage().data, inputData.data, size, cudaMemcpyDeviceToHost);
    */
    /*
    double* tmp;
    transformData(&tmp, inputData.data, elements);
    cudaMemcpy(outputImage().data, tmp, elements * sizeof(double), cudaMemcpyDeviceToHost);
    */
}

template<typename T>
void VolumeDeformSph<T>::runKernel() 
{
    // Does not work in general case, but test data have nice sizes
    dim3 grid;
    grid.x = images.VR.xDim / BLOCK_X_DIM;
    grid.y = images.VR.yDim / BLOCK_Y_DIM;
    grid.z = images.VR.zDim / BLOCK_Z_DIM;

    dim3 block;
    block.x = BLOCK_X_DIM;
    block.y = BLOCK_Y_DIM;
    block.z = BLOCK_Z_DIM;

    // thrust experiment
    int TOTAL_GRID_SIZE = grid.x * grid.y * grid.z;
    thrust::device_vector<T> t_out(TOTAL_GRID_SIZE * 4, 0.0);

    computeDeform<<< grid, block >>>(Rmax2, iRmax,
            images, zshparams, steps, clnm,
            volumes, deformImages, applyTransformation, saveDeformation, t_out.data());
    cudaDeviceSynchronize();

    auto diff2It = t_out.begin();
    auto sumVDIt = diff2It + TOTAL_GRID_SIZE;
    auto modgIt = sumVDIt + TOTAL_GRID_SIZE;
    auto NcountIt = modgIt + TOTAL_GRID_SIZE;

    exOuts.diff2 = thrust::reduce(diff2It, sumVDIt);
    exOuts.sumVD = thrust::reduce(sumVDIt, modgIt);
    exOuts.modg = thrust::reduce(modgIt, NcountIt);
    exOuts.Ncount = thrust::reduce(NcountIt, t_out.end());
}

template<typename T>
void VolumeDeformSph<T>::transferResults() 
{
    if (applyTransformation) {
        transferImageData(program->VO, images.VO);
    }
    if (saveDeformation) {
        transferImageData(program->Gx, deformImages.Gx);
        transferImageData(program->Gy, deformImages.Gy);
        transferImageData(program->Gz, deformImages.Gz);
    }
}

template<typename T>
void VolumeDeformSph<T>::setupZSHparams()
{
    zshparams.size = program->vL1.size();

    if (cudaMallocAndCopy(&zshparams.vL1, program->vL1.vdata, zshparams.size) != cudaSuccess)
        printCudaError();
    if (cudaMallocAndCopy(&zshparams.vL2, program->vL2.vdata, zshparams.size) != cudaSuccess)
        printCudaError();
    if (cudaMallocAndCopy(&zshparams.vN, program->vN.vdata, zshparams.size) != cudaSuccess)
        printCudaError();
    if (cudaMallocAndCopy(&zshparams.vM, program->vM.vdata, zshparams.size) != cudaSuccess)
        printCudaError();
}

template<typename T>
void VolumeDeformSph<T>::setupVolumes()
{
    volumes.size = program->volumesR.size();

    justForFreeR.resize(volumes.size);
    justForFreeI.resize(volumes.size);

    for (size_t i = 0; i < volumes.size; i++) {
        setupImage(program->volumesR[i], justForFreeR[i]);
        setupImage(program->volumesI[i], justForFreeI[i]);
    }

    if (cudaMallocAndCopy(&volumes.R, justForFreeR.data(), volumes.size) != cudaSuccess)
        printCudaError();
    if (cudaMallocAndCopy(&volumes.I, justForFreeI.data(), volumes.size) != cudaSuccess)
        printCudaError();
}

template<typename T>
void VolumeDeformSph<T>::setupImage(Image<double>& inputImage, ImageData<T>& outputImageData) 
{
    auto& mda = inputImage();

    outputImageData.xShift = mda.xinit;
    outputImageData.yShift = mda.yinit;
    outputImageData.zShift = mda.zinit;
    outputImageData.xDim = mda.xdim;
    outputImageData.yDim = mda.ydim;
    outputImageData.zDim = mda.zdim;

    // if T is smaller than double -> error
    // might be replaced with cudaMallocAndCopy, but there are different sizes: T vs double
    /*
    int size = outputImageData.xDim * outputImageData.yDim * outputImageData.zDim * sizeof(T);
    if (cudaMalloc(&outputImageData.data, size) != cudaSuccess)
        printCudaError();
    if (cudaMemcpy(outputImageData.data, mda.data, size, cudaMemcpyHostToDevice) != cudaSuccess)
        printCudaError();
    */
    transformData(&outputImageData.data, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

template<typename T>
void VolumeDeformSph<T>::setupImage(ImageData<T>& inputImage, ImageData<T>& outputImageData, bool copyData) 
{
    outputImageData.xShift = inputImage.xShift;
    outputImageData.yShift = inputImage.yShift;
    outputImageData.zShift = inputImage.zShift;
    outputImageData.xDim = inputImage.xDim;
    outputImageData.yDim = inputImage.yDim;
    outputImageData.zDim = inputImage.zDim;

    size_t size = inputImage.xDim * inputImage.yDim * inputImage.zDim * sizeof(T);
    if (cudaMalloc(&outputImageData.data, size) != cudaSuccess)
        printCudaError();

    if (copyData) {
        if (cudaMemcpy(outputImageData.data, inputImage.data, size, cudaMemcpyHostToDevice) != cudaSuccess)
            printCudaError();
    }
}
