// Xmipp includes
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

void processCudaError() 
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(err));
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

// VolumeDeformSph methods

VolumeDeformSph::VolumeDeformSph()
{
}

VolumeDeformSph::~VolumeDeformSph() 
{
    cudaFree(images.VI);
    cudaFree(images.VR);
    cudaFree(images.VO);

    cudaFree(volumes.R);
    cudaFree(volumes.I);

    cudaFree(deformImages.Gx);
    cudaFree(deformImages.Gy);
    cudaFree(deformImages.Gz);
}

void VolumeDeformSph::associateWith(ProgVolumeDeformSphGpu* prog) 
{
    program = prog;
}

static dim3 grid;
static dim3 block;

void VolumeDeformSph::setupConstantParameters() 
{
    if (program == nullptr)
        throw(std::runtime_error("VolumeDeformSph not associated with the program!"));

    // kernel arguments
    this->Rmax2 = program->Rmax * program->Rmax;
    this->iRmax = 1 / program->Rmax;
    setupImage(program->VI, &images.VI);
    //setupImage(program->VR, &images.VR);
    setupImageMetaData(program->VR);
    setupZSHparams();
    setupVolumes();

    // kernel dimension
    block.x = BLOCK_X_DIM;
    block.y = BLOCK_Y_DIM;
    block.z = BLOCK_Z_DIM;
    grid.x = ((imageMetaData.xDim + block.x - 1) / block.x);
    grid.y = ((imageMetaData.yDim + block.y - 1) / block.y);
    grid.z = ((imageMetaData.zDim + block.z - 1) / block.z);

    totalGridSize = grid.x * grid.y * grid.z;

    // Dynamic shared memory
    constantSharedMemSize = 0;
}

void VolumeDeformSph::setupChangingParameters() 
{
    if (program == nullptr)
        throw(std::runtime_error("VolumeDeformSph not associated with the program!"));

    setupClnm();

    steps = program->onesInSteps;

    changingSharedMemSize = 0;
    changingSharedMemSize += sizeof(int4) * steps;
    changingSharedMemSize += sizeof(PrecisionType3) * steps;

    // Deformation and transformation booleans
    this->applyTransformation = program->applyTransformation;
    this->saveDeformation = program->saveDeformation;

    if (applyTransformation) {
        setupImage(imageMetaData, &images.VO);
    }
    if (saveDeformation) {
        setupImage(imageMetaData, &deformImages.Gx);
        setupImage(imageMetaData, &deformImages.Gy);
        setupImage(imageMetaData, &deformImages.Gz);
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
        processCudaError();
}

KernelOutputs VolumeDeformSph::getOutputs() 
{
    return outputs;
}

void VolumeDeformSph::transferImageData(Image<double>& outputImage, PrecisionType* inputData) 
{
    size_t elements = imageMetaData.xDim * imageMetaData.yDim * imageMetaData.zDim;
    std::vector<PrecisionType> tVec(elements);
    cudaMemcpy(tVec.data(), inputData, sizeof(PrecisionType) * elements, cudaMemcpyDeviceToHost);
    std::vector<double> dVec(tVec.begin(), tVec.end());
    memcpy(outputImage().data, dVec.data(), sizeof(double) * elements);
}

void VolumeDeformSph::runKernel() 
{

    // Define thrust reduction vector
    thrust::device_vector<PrecisionType> thrustVec(totalGridSize * 3, 0.0);

    // Run kernel
    computeDeform<<<grid, block, constantSharedMemSize + changingSharedMemSize>>>(
            Rmax2,
            iRmax,
            images,
            dZshParams,
            dClnm,
            steps,
            imageMetaData,
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

    outputs.diff2 = thrust::reduce(diff2It, sumVDIt);
    outputs.sumVD = thrust::reduce(sumVDIt, modgIt);
    outputs.modg = thrust::reduce(modgIt, thrustVec.end());
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
        processCudaError();
}

void setupImageNew(Image<double>& inputImage, PrecisionType** outputImageData) 
{
    auto& mda = inputImage();
    transformData(outputImageData, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

void VolumeDeformSph::setupVolumes()
{
    volumes.count = program->volumesR.size();
    volumes.volumeSize = program->VR().getSize();

    if (cudaMalloc(&volumes.I, volumes.count * volumes.volumeSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();
    if (cudaMalloc(&volumes.R, volumes.count * volumes.volumeSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();

    for (size_t i = 0; i < volumes.count; i++) {
        PrecisionType* tmpI = volumes.I + i * volumes.volumeSize;
        PrecisionType* tmpR = volumes.R + i * volumes.volumeSize;
        transformData(&tmpI, program->volumesI[i]().data, volumes.volumeSize, false);
        transformData(&tmpR, program->volumesR[i]().data, volumes.volumeSize, false);
    }
}

void VolumeDeformSph::setupImageMetaData(const Image<double>& mda) 
{

    imageMetaData.xShift = mda().xinit;
    imageMetaData.yShift = mda().yinit;
    imageMetaData.zShift = mda().zinit;
    imageMetaData.xDim = mda().xdim;
    imageMetaData.yDim = mda().ydim;
    imageMetaData.zDim = mda().zdim;
}

void VolumeDeformSph::setupImage(Image<double>& inputImage, PrecisionType** outputImageData) 
{
    auto& mda = inputImage();
    transformData(outputImageData, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

void VolumeDeformSph::setupImage(const ImageMetaData& inputImage, PrecisionType** outputImageData) 
{
    size_t size = inputImage.xDim * inputImage.yDim * inputImage.zDim * sizeof(PrecisionType);
    if (cudaMalloc(outputImageData, size) != cudaSuccess)
        processCudaError();
}
