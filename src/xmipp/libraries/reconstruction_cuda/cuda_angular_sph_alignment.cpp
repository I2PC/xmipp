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

AngularSphAlignment::AngularSphAlignment(ProgAngularSphAlignmentGpu* prog)
{
    program = prog;
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

namespace {
    dim3 grid;
    dim3 block;
    // Cuda stream used during the data preparation. More streams could be used
    // but at this point preparations on GPU are not as intesive as to require
    // more streams. More streams might be useful for combination of
    // weak GPU and powerful CPU.
    cudaStream_t prepStream;
}

void AngularSphAlignment::setupConstantParameters()
{
    if (program == nullptr)
        throw new std::runtime_error("AngularSphAlignment not associated with the program!");

    // kernel arguments
    this->Rmax2 = program->RmaxDef * program->RmaxDef;
    this->iRmax = 1.0 / program->RmaxDef;
    //setupImageMetaData(program->V);

    //setupVolumeData();
    setupVolumeMask();
    setupZSHparams();
    setupOutputs();

    //setupGpuBlocks();

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

void AngularSphAlignment::setupGpuBlocks()
{
    block.x = BLOCK_X_DIM;
    block.y = BLOCK_Y_DIM;
    block.z = BLOCK_Z_DIM;
    grid.x = ((imageMetaData.xDim + block.x - 1) / block.x);
    grid.y = ((imageMetaData.yDim + block.y - 1) / block.y);
    grid.z = ((imageMetaData.zDim + block.z - 1) / block.z);

    totalGridSize = grid.x * grid.y * grid.z;
    // prepped for warp only reduction
    //totalGridSize = grid.x * grid.y * grid.z * (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM / 32);
}

void AngularSphAlignment::setupClnm()
{
    clnmVec.resize(MAX_COEF_COUNT);

    for (unsigned i = 0; i < program->vL1.size(); ++i) {
        clnmVec[i].x = program->clnm[i];
        clnmVec[i].y = program->clnm[i + program->vL1.size()];
        clnmVec[i].z = program->clnm[i + program->vL1.size() * 2];
    }

    if (cudaMemcpyToSymbol(cClnm, clnmVec.data(), MAX_COEF_COUNT * sizeof(PrecisionType3)) != cudaSuccess)
        processCudaError();
}

void AngularSphAlignment::setupOutputs()
{
    if (outputs == nullptr){//FIXME do this better
        if (cudaMallocHost(&outputs, sizeof(KernelOutputs)) != cudaSuccess)
            processCudaError();
    }
}

void AngularSphAlignment::setupOutputArray()
{
    if (reductionArray == nullptr){//FIXME do this better
        if (cudaMalloc(&reductionArray, 3 * totalGridSize * sizeof(PrecisionType)) != cudaSuccess)
            processCudaError();
    }
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

void AngularSphAlignment::init()
{
    setupImageMetaData(program->V);
    setupGpuBlocks();
    cudaStreamCreate(&prepStream);

    int size = imageMetaData.xDim * imageMetaData.yDim * imageMetaData.zDim * sizeof(double);
    int paddedSize = (imageMetaData.xDim + 2) * (imageMetaData.yDim + 2) * (imageMetaData.zDim + 2) * sizeof(PrecisionType);
    if (cudaMalloc(&dVolData, paddedSize) != cudaSuccess)
        processCudaError();
    if (cudaMalloc(&dPrepVolume, size) != cudaSuccess)
        processCudaError();
}

void AngularSphAlignment::setupVolumeData()
{
    const auto& vol = program->V();
    transformData(&dVolData, vol.data, vol.zyxdim, dVolData == nullptr);
}

void AngularSphAlignment::prepareVolumeData()
{
    prepareVolume<true>(program->V().data, dPrepVolume, dVolData);
}

template<bool PADDING>
void AngularSphAlignment::prepareVolume(const double* mdaData, double* prepVol, PrecisionType* volume)
{
    int size = imageMetaData.xDim * imageMetaData.yDim * imageMetaData.zDim * sizeof(double);
    int paddedSize = (imageMetaData.xDim + 2) * (imageMetaData.yDim + 2) * (imageMetaData.zDim + 2) * sizeof(PrecisionType);
    if (cudaMemsetAsync(volume, 0, paddedSize, prepStream) != cudaSuccess)
        processCudaError();
    if (cudaMemcpyAsync(prepVol, mdaData, size, cudaMemcpyHostToDevice, prepStream) != cudaSuccess)
        processCudaError();
    prepareVolumeKernel<PADDING><<<grid, block, 0, prepStream>>>(volume, prepVol, imageMetaData);
    processCudaError();
}

void AngularSphAlignment::waitToFinishPreparations()
{
    if (cudaStreamSynchronize(prepStream) != cudaSuccess)
        processCudaError();
}

bool once = true;//FIXME just tmp debug solution
void AngularSphAlignment::cleanupPreparations()
{
    if (once) {
        if (cudaFree(dPrepVolume) != cudaSuccess)
            processCudaError();
        once = false;
    }
    //if (cudaStreamDestroy(prepStream) != cudaSuccess)
    //    processCudaError();
}

void AngularSphAlignment::setupRotation()
{
    std::vector<PrecisionType> tmp(program->R.mdata, program->R.mdata + program->R.mdim);
    if (cudaMemcpyToSymbol(cRotation, tmp.data(), 9 * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();
    //transformData(&dRotation, program->R.mdata, program->R.mdim, dRotation == nullptr);
}

void AngularSphAlignment::setupVolumeMask()
{
    if (dVolMask == nullptr) {//FIXME do this better
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
    // Before and after running the kernel is no need for explicit synchronization,
    // because it is being run in the default cuda stream, therefore it is synchronized automatically
    // If the cuda stream of this kernel ever changes explicit synchronization is needed!
    if (program->L1 > 3 || program->L2 > 3) {
        projectionKernel<BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM, 5, 5>
            <<<grid, block, constantSharedMemSize + changingSharedMemSize>>>(
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
    } else {
        projectionKernel<BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM, 3, 3>
            <<<grid, block, constantSharedMemSize + changingSharedMemSize>>>(
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
    }
    // FIXME remove, just for debug
    processCudaError();

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
    std::vector<int4> zshparamsVec(program->vL1.size());

    for (unsigned i = 0; i < zshparamsVec.size(); ++i) {
        zshparamsVec[i].w = program->vL1[i];
        zshparamsVec[i].x = program->vN[i];
        zshparamsVec[i].y = program->vL2[i];
        zshparamsVec[i].z = program->vM[i];
    }

    if (cudaMemcpyToSymbol(cZsh, zshparamsVec.data(), zshparamsVec.size() * sizeof(int4)) != cudaSuccess)
        processCudaError();
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
