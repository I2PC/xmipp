// Xmipp includes
#include "api/dimension_vector.h"
#include "core/metadata_label.h"
#include "core/xmipp_random_mode.h"
#include "core/matrix1d.h"
#include "reconstruction_adapt_cuda/volume_deform_sph_gpu.h"
#include "cuda_volume_deform_sph.h"
#include "cuda_volume_deform_sph.cu"
#include "cuda_volume_deform_sph_defines.h"
// Standard includes
#include <iterator>
#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <exception>
// Thrust includes
#include <thrust/reduce.h>
#include <thrust/device_vector.h>


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

// VolumeDeformSph methods

VolumeDeformSph::VolumeDeformSph()
{
}

void VolumeDeformSph::freeZSHSCATTERED()
{
    cudaFree(zshparamsSCATTERED.vL1);
    cudaFree(zshparamsSCATTERED.vL2);
    cudaFree(zshparamsSCATTERED.vN);
    cudaFree(zshparamsSCATTERED.vM);
}

VolumeDeformSph::~VolumeDeformSph() 
{
    freeImage(images.VI);
    freeImage(images.VR);
    freeImage(images.VO);

    freeZSHSCATTERED();

    for (size_t i = 0; i < volumes.size; i++) {
        freeImage(justForFreeR[i]);
        freeImage(justForFreeI[i]);
    }
    cudaFree(volumes.R);
    cudaFree(volumes.I);

    freeImage(deformImages.Gx);
    freeImage(deformImages.Gy);
    freeImage(deformImages.Gz);
}

void VolumeDeformSph::freeImage(ImageData &im) 
{
    if (im.data != nullptr)
        cudaFree(im.data);
}

void VolumeDeformSph::associateWith(ProgVolumeDeformSphGpu* prog) 
{
    program = prog;
}

//TEMPORARY SOLUTION
static dim3 grid;
static dim3 block;

void VolumeDeformSph::setupConstantParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    // kernel arguments
    this->Rmax2 = program->Rmax * program->Rmax;
    this->iRmax = 1 / program->Rmax;
    setupImage(program->VI, images.VI);
    setupImage(program->VR, images.VR);
    setupZSHparams();
    setupZSHparamsSCATTERED();
    setupVolumes();

    // kernel dimension
    block.x = BLOCK_X_DIM;
    block.y = BLOCK_Y_DIM;
    block.z = BLOCK_Z_DIM;
    grid.x = ((images.VR.xDim + block.x - 1) / block.x);
    grid.y = ((images.VR.yDim + block.y - 1) / block.y);
    grid.z = ((images.VR.zDim + block.z - 1) / block.z);

    totalGridSize = grid.x * grid.y * grid.z;

    // Dynamic shared memory
    constantSharedMemSize = 0;
#if USE_SHARED_VOLUME_METADATA == 1
    constantSharedMemSize += sizeof(ImageData) * volumes.size * 2;
#endif
#if USE_SHARED_VOLUME_DATA == 1
    constantSharedMemSize += sizeof(PrecisionType) * block.x * block.y * block.z * volumes.size * 2;
#endif
}

void VolumeDeformSph::setupChangingParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    setupClnm();
    setupClnmSCATTERED();

    steps = program->onesInSteps;

    changingSharedMemSize = 0;
#if USE_SHARED_MEM_ZSH_CLNM == 1
    changingSharedMemSize += sizeof(int4) * steps;
    changingSharedMemSize += sizeof(PrecisionType3) * steps;
#endif

    // Deformation and transformation booleans
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

void VolumeDeformSph::setupClnm()
{
    clnmVec.resize(program->vL1.size());

    for (unsigned i = 0; i < program->vL1.size(); ++i) {
        clnmVec[i].x = program->clnm[i];
        clnmVec[i].y = program->clnm[i + program->vL1.size()];
        clnmVec[i].z = program->clnm[i + program->vL1.size() * 2];
    }

    if (cudaMallocAndCopy(&dClnm, clnmVec.data(), clnmVec.size()) != cudaSuccess)
        printCudaError();
}

void VolumeDeformSph::setupClnmSCATTERED()
{
    clnmVecSCATTERED.assign(program->clnm.vdata, program->clnm.vdata + program->clnm.size());
    if (cudaMallocAndCopy(&dClnmSCATTERED, clnmVecSCATTERED.data(), clnmVecSCATTERED.size()) != cudaSuccess)
        printCudaError();
}

KernelOutputs VolumeDeformSph::getOutputs() 
{
    return outputs;
}

void VolumeDeformSph::transferImageData(Image<double>& outputImage, ImageData& inputData) 
{
    size_t elements = inputData.xDim * inputData.yDim * inputData.zDim;
    std::vector<PrecisionType> tVec(elements);
    cudaMemcpy(tVec.data(), inputData.data, sizeof(PrecisionType) * elements, cudaMemcpyDeviceToHost);
    std::vector<double> dVec(tVec.begin(), tVec.end());
    memcpy(outputImage().data, dVec.data(), sizeof(double) * elements);
}

void VolumeDeformSph::runKernel() 
{

    // Define thrust reduction vector
    thrust::device_vector<PrecisionType> thrustVec(totalGridSize * 4, 0.0);

    // Run kernel
    computeDeform<<<grid, block, constantSharedMemSize + changingSharedMemSize>>>(
            Rmax2,
            iRmax,
            images,
            dZshParams,
            dClnm,
            zshparamsSCATTERED,
            dClnmSCATTERED,
            steps,
            volumes,
            deformImages,
            applyTransformation,
            saveDeformation,
            thrust::raw_pointer_cast(thrustVec.data())
            );

    cudaDeviceSynchronize();

    auto diff2It = thrustVec.begin();
    auto sumVDIt = diff2It + totalGridSize;
    auto modgIt = sumVDIt + totalGridSize;
    auto NcountIt = modgIt + totalGridSize;

    outputs.diff2 = thrust::reduce(diff2It, sumVDIt);
    outputs.sumVD = thrust::reduce(sumVDIt, modgIt);
    outputs.modg = thrust::reduce(modgIt, NcountIt);
    outputs.Ncount = thrust::reduce(NcountIt, thrustVec.end());
}

void VolumeDeformSph::transferResults() 
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

void VolumeDeformSph::setupZSHparams()
{
    zshparamsVec.resize(program->vL1.size());

    for (unsigned i = 0; i < zshparamsVec.size(); ++i) {
        zshparamsVec[i].w = program->vL1[i];
        zshparamsVec[i].x = program->vN[i];
        zshparamsVec[i].y = program->vL2[i];
        zshparamsVec[i].z = program->vM[i];
    }

    if (cudaMallocAndCopy(&dZshParams, zshparamsVec.data(), zshparamsVec.size()) != cudaSuccess)
        printCudaError();
}

void VolumeDeformSph::setupZSHparamsSCATTERED()
{
    zshparamsSCATTERED.size = program->vL1.size();

    if (cudaMallocAndCopy(&zshparamsSCATTERED.vL1, program->vL1.vdata, zshparamsSCATTERED.size) != cudaSuccess)
        printCudaError();
    if (cudaMallocAndCopy(&zshparamsSCATTERED.vL2, program->vL2.vdata, zshparamsSCATTERED.size) != cudaSuccess)
        printCudaError();
    if (cudaMallocAndCopy(&zshparamsSCATTERED.vN, program->vN.vdata, zshparamsSCATTERED.size) != cudaSuccess)
        printCudaError();
    if (cudaMallocAndCopy(&zshparamsSCATTERED.vM, program->vM.vdata, zshparamsSCATTERED.size) != cudaSuccess)
        printCudaError();
}

void VolumeDeformSph::setupVolumes()
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

void VolumeDeformSph::setupImage(Image<double>& inputImage, ImageData& outputImageData) 
{
    auto& mda = inputImage();

    outputImageData.xShift = mda.xinit;
    outputImageData.yShift = mda.yinit;
    outputImageData.zShift = mda.zinit;
    outputImageData.xDim = mda.xdim;
    outputImageData.yDim = mda.ydim;
    outputImageData.zDim = mda.zdim;

    transformData(&outputImageData.data, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

void VolumeDeformSph::setupImage(ImageData& inputImage, ImageData& outputImageData, bool copyData) 
{
    outputImageData.xShift = inputImage.xShift;
    outputImageData.yShift = inputImage.yShift;
    outputImageData.zShift = inputImage.zShift;
    outputImageData.xDim = inputImage.xDim;
    outputImageData.yDim = inputImage.yDim;
    outputImageData.zDim = inputImage.zDim;

    size_t size = inputImage.xDim * inputImage.yDim * inputImage.zDim * sizeof(PrecisionType);
    if (cudaMalloc(&outputImageData.data, size) != cudaSuccess)
        printCudaError();

    if (copyData) {
        if (cudaMemcpy(outputImageData.data, inputImage.data, size, cudaMemcpyHostToDevice) != cudaSuccess)
            printCudaError();
    }
}
