// Xmipp includes
#include "core/metadata_label.h"
#include "core/xmipp_random_mode.h"
#include "core/matrix1d.h"
#include "reconstruction_adapt_cuda/angular_sph_alignment_gpu.h"
#include "cuda_angular_sph_alignment.h"
#include "cuda_angular_sph_alignment.cu"
#include "cuda_volume_deform_sph_defines.h"//TODO
// Standard includes
#include <iterator>
#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <exception>


namespace AngularAlignmentGpu {

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

#define processCudaError() (_processCudaError(__FILE__, __LINE__))
void _processCudaError(const char* file, int line) 
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "File: %s: line %d\nCuda error: %s\n", file, line, cudaGetErrorString(err));
        exit(err);
    }
}

// Copies data from CPU to the GPU and at the same time transforms from
// type 'U' to type 'T'. Works only for numeric types
template<typename Target, typename Source>
void transformData(Target** dest, Source* source, size_t n, bool mallocMem = true)
{
    std::vector<Target> tmp(source, source + n);

    if (mallocMem){
        if (cudaMalloc(dest, sizeof(Target) * n) != cudaSuccess){
            processCudaError();
        }
    }

    if (cudaMemcpy(*dest, tmp.data(), sizeof(Target) * n, cudaMemcpyHostToDevice) != cudaSuccess){
        processCudaError();
    }
}

// AngularSphAlignment methods

AngularSphAlignment::AngularSphAlignment()
{
}

AngularSphAlignment::~AngularSphAlignment()
{
    cudaFree(dVolData);
    cudaFree(dRotation);
    cudaFree(dZshParams);
    cudaFree(dClnm);
    cudaFree(dVolMask);
    cudaFree(dProjectionPlane);
    cudaFree(reductionArray);
    cudaFreeHost(outputs);
}

void AngularSphAlignment::associateWith(ProgAngularSphAlignmentGpu* prog) 
{
    program = prog;
}

static dim3 grid;
static dim3 block;

void AngularSphAlignment::setupConstantParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("AngularSphAlignment not associated with the program!");

    // kernel arguments
    this->Rmax2 = program->RmaxDef * program->RmaxDef;
    this->iRmax = 1.0 / program->RmaxDef;
    setupImageMetaData(program->V);

    setupVolumeData();
    setupVolumeMask();
    setupZSHparams();
    setupOutputs();

    // kernel dimension
    block.x = BLOCK_X_DIM;
    block.y = BLOCK_Y_DIM;
    block.z = BLOCK_Z_DIM;
    grid.x = ((imageMetaData.xDim + block.x - 1) / block.x);
    grid.y = ((imageMetaData.yDim + block.y - 1) / block.y);
    grid.z = ((imageMetaData.zDim + block.z - 1) / block.z);

    totalGridSize = grid.x * grid.y * grid.z;

    setupOutputArray();

    // Dynamic shared memory
    constantSharedMemSize = 0;
}

void AngularSphAlignment::setupChangingParameters()
{
    if (program == nullptr)
        throw new std::runtime_error("AngularSphAlignment not associated with the program!");

    setupClnm();
    setupRotation();
    setupProjectionPlane();

    steps = program->onesInSteps;

    changingSharedMemSize = 0;
    changingSharedMemSize += sizeof(int4) * steps;
    changingSharedMemSize += sizeof(PrecisionType3) * steps;
}

void AngularSphAlignment::setupClnm()
{
    clnmVec.resize(program->vL1.size());

    for (unsigned i = 0; i < program->vL1.size(); ++i) {
        clnmVec[i].x = program->clnm[i];
        clnmVec[i].y = program->clnm[i + program->vL1.size()];
        clnmVec[i].z = program->clnm[i + program->vL1.size() * 2];
    }

    if (dClnm == nullptr) {
        if (cudaMallocAndCopy(&dClnm, clnmVec.data(), clnmVec.size()) != cudaSuccess)
            processCudaError();
    } else {
        if (cudaMemcpy(dClnm, clnmVec.data(), clnmVec.size() * sizeof(PrecisionType3),
                    cudaMemcpyHostToDevice) != cudaSuccess)
            processCudaError();
    }
}

void AngularSphAlignment::setupOutputs()
{
    if (cudaMallocHost(&outputs, sizeof(KernelOutputs)) != cudaSuccess)
        processCudaError();
}

void AngularSphAlignment::setupOutputArray()
{
    if (cudaMalloc(&reductionArray, 3 * totalGridSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();
}

KernelOutputs AngularSphAlignment::getOutputs() 
{
    return *outputs;
}

void AngularSphAlignment::transferImageData(Image<double>& outputImage, PrecisionType* inputData) 
{
    size_t elements = imageMetaData.xDim * imageMetaData.yDim * imageMetaData.zDim;
    std::vector<PrecisionType> tVec(elements);
    cudaMemcpy(tVec.data(), inputData, sizeof(PrecisionType) * elements, cudaMemcpyDeviceToHost);
    std::vector<double> dVec(tVec.begin(), tVec.end());
    memcpy(outputImage().data, dVec.data(), sizeof(double) * elements);
}

void AngularSphAlignment::setupVolumeData() 
{
    const auto& vol = program->V();
    transformData(&dVolData, vol.data, vol.zyxdim, dVolData == nullptr);
}

void AngularSphAlignment::setupRotation() 
{
    transformData(&dRotation, program->R.mdata, program->R.mdim, dRotation == nullptr);
}

void AngularSphAlignment::setupVolumeMask() 
{
    if (dVolMask == nullptr) {
        if (cudaMallocAndCopy(&dVolMask, program->V_mask.data, program->V_mask.getSize())
                != cudaSuccess)
            processCudaError();
    } else {
        if (cudaMemcpy(dVolMask, program->V_mask.data, program->V_mask.getSize() * sizeof(int),
                    cudaMemcpyHostToDevice) != cudaSuccess)
            processCudaError();
    }
}

void AngularSphAlignment::setupProjectionPlane() 
{
    const auto& projPlane = program->P();
    if (dProjectionPlane == nullptr) {
        if (cudaMalloc(&dProjectionPlane, projPlane.yxdim * sizeof(PrecisionType)) != cudaSuccess)
            processCudaError();
    }
    cudaMemset(dProjectionPlane, 0, projPlane.yxdim * sizeof(PrecisionType));
}

void AngularSphAlignment::runKernel() 
{
    // Run kernel
    projectionKernel<<<grid, block, constantSharedMemSize + changingSharedMemSize>>>(
            Rmax2,
            iRmax,
            imageMetaData,
            dVolData,
            dRotation,
            steps,
            dZshParams,
            dClnm,
            dVolMask,
            dProjectionPlane,
            reductionArray
            );

    PrecisionType* countPtr = reductionArray;
    PrecisionType* sumVDPtr = countPtr + totalGridSize;
    PrecisionType* modgPtr = sumVDPtr + totalGridSize;

    reduceDiff.reduceDeviceArrayAsync(countPtr, totalGridSize, &outputs->count);
    reduceSumVD.reduceDeviceArrayAsync(sumVDPtr, totalGridSize, &outputs->sumVD);
    reduceModg.reduceDeviceArrayAsync(modgPtr, totalGridSize, &outputs->modg);

    if (cudaDeviceSynchronize() != cudaSuccess)
        processCudaError();
}

void AngularSphAlignment::transferProjectionPlane()
{
    // mozna lepsi nez neustale pretypovavat a kopirovat vectory, to proste ukladat v double na GPU
    // nic se tam nepocita jen se to ulozi (tzn "jedno" pretypovani z float na double)
    std::vector<PrecisionType> tmp(program->P().zyxdim);
    if (cudaMemcpy(tmp.data(), dProjectionPlane, tmp.size() * sizeof(PrecisionType),
            cudaMemcpyDeviceToHost) != cudaSuccess)
        processCudaError();
    std::vector<double> tmpDouble(tmp.begin(), tmp.end());
    memcpy(program->P().data, tmpDouble.data(), tmpDouble.size() * sizeof(double));
}

void AngularSphAlignment::transferResults() 
{
    transferProjectionPlane();
}

void AngularSphAlignment::setupZSHparams()
{
    zshparamsVec.resize(program->vL1.size());

    for (unsigned i = 0; i < zshparamsVec.size(); ++i) {
        zshparamsVec[i].w = program->vL1[i];
        zshparamsVec[i].x = program->vN[i];
        zshparamsVec[i].y = program->vL2[i];
        zshparamsVec[i].z = program->vM[i];
    }

    if (dZshParams == nullptr) {
        if (cudaMallocAndCopy(&dZshParams, zshparamsVec.data(), zshparamsVec.size()) != cudaSuccess)
            processCudaError();
    } else {
        if (cudaMemcpy(dZshParams, zshparamsVec.data(), zshparamsVec.size() * sizeof(int4), 
                    cudaMemcpyHostToDevice) != cudaSuccess)
            processCudaError();
    }
}

void setupImageNew(Image<double>& inputImage, PrecisionType** outputImageData) 
{
    auto& mda = inputImage();
    transformData(outputImageData, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

void AngularSphAlignment::setupImageMetaData(const Image<double>& mda) 
{
    imageMetaData.xShift = mda().xinit;
    imageMetaData.yShift = mda().yinit;
    imageMetaData.zShift = mda().zinit;
    imageMetaData.xDim = mda().xdim;
    imageMetaData.yDim = mda().ydim;
    imageMetaData.zDim = mda().zdim;
}

void AngularSphAlignment::setupImage(Image<double>& inputImage, PrecisionType** outputImageData) 
{
    auto& mda = inputImage();
    transformData(outputImageData, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

void AngularSphAlignment::setupImage(const ImageMetaData& inputImage, PrecisionType** outputImageData) 
{
    size_t size = inputImage.xDim * inputImage.yDim * inputImage.zDim * sizeof(PrecisionType);
    if (cudaMalloc(outputImageData, size) != cudaSuccess)
        processCudaError();
}

} // namespace AngularAlignmentGpu
