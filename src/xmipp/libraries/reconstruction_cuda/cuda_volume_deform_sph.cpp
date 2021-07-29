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

// VolumeDeformSph methods

VolumeDeformSph::VolumeDeformSph(ProgVolumeDeformSphGpu* program)
{
    this->program = program;
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

    cudaFreeHost(outputs);
}

namespace
{
    dim3 grid;
    dim3 block;
    // Cuda stream used during the data preparation. More streams could be used
    // but at this point preparations on GPU are not as intesive as to require
    // more streams. More streams might be useful for combination of
    // weak GPU and powerful CPU.
    cudaStream_t prepStream;
};

void VolumeDeformSph::setupConstantParameters()
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    // kernel arguments
    this->Rmax2 = program->Rmax * program->Rmax;
    this->iRmax = 1 / program->Rmax;
    setupZSHparams();

    setupOutputArray();
    setupOutputs();

    // Dynamic shared memory
    constantSharedMemSize = 0;
}

void VolumeDeformSph::setupGpuBlocks()
{
    block.x = BLOCK_X_DIM;
    block.y = BLOCK_Y_DIM;
    block.z = BLOCK_Z_DIM;
    grid.x = ((imageMetaData.xDim + block.x - 1) / block.x);
    grid.y = ((imageMetaData.yDim + block.y - 1) / block.y);
    grid.z = ((imageMetaData.zDim + block.z - 1) / block.z);

    totalGridSize = grid.x * grid.y * grid.z * (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM / 32);
}

void VolumeDeformSph::setupChangingParameters()
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    setupClnm();
    steps = program->onesInSteps;

    changingSharedMemSize = 0;

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

// maybe this requires too much memory...
void VolumeDeformSph::initVolumes()
{
    setupImageMetaData(program->VR);
    setupGpuBlocks();
    cudaStreamCreate(&prepStream);

    volumes.count = 1;
    if (program->sigma.size() != 1 || program->sigma[0] != 0) {
        volumes.count += program->sigma.size();
    }
    volumes.volumeSize = program->VR().xdim * program->VR().ydim * program->VR().zdim;
    volumes.volumePaddedSize = (program->VR().xdim + 2) *
        (program->VR().ydim + 2) * (program->VR().zdim + 2);

    prepVolumes.count = volumes.count;
    prepVolumes.volumeSize = volumes.volumeSize;
    prepVolumes.volumePaddedSize = volumes.volumePaddedSize;

    if (cudaMalloc(&prepVolumes.R, prepVolumes.count * prepVolumes.volumeSize * sizeof(double)) != cudaSuccess)
        processCudaError();
    if (cudaMalloc(&prepVolumes.I, prepVolumes.count * prepVolumes.volumeSize * sizeof(double)) != cudaSuccess)
        processCudaError();

    if (cudaMalloc(&volumes.R, volumes.count * volumes.volumeSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();
    if (cudaMalloc(&volumes.I, volumes.count * volumes.volumePaddedSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();

    if (cudaMalloc(&dTmpVI, volumes.volumeSize * sizeof(double)) != cudaSuccess)
        processCudaError();
    if (cudaMalloc(&images.VI, volumes.volumePaddedSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();

    if (cudaMemsetAsync(volumes.I, 0, volumes.count * volumes.volumePaddedSize * sizeof(PrecisionType), prepStream) != cudaSuccess)
        processCudaError();
    if (cudaMemsetAsync(images.VI, 0, volumes.volumePaddedSize * sizeof(PrecisionType), prepStream) != cudaSuccess)
        processCudaError();
}

void VolumeDeformSph::prepareInputVolume(const MultidimArray<double>& vol)
{
    prepareVolume<true>
        (vol.data, prepVolumes.I + posI * prepVolumes.volumeSize, volumes.I + posI * volumes.volumePaddedSize);
    posI++;
}

void VolumeDeformSph::prepareReferenceVolume(const MultidimArray<double>& vol)
{
    prepareVolume<false>
        (vol.data, prepVolumes.R + posR * prepVolumes.volumeSize, volumes.R + posR * volumes.volumeSize);
    posR++;
}

template<bool PADDING>
void VolumeDeformSph::prepareVolume(const double* mdaData, double* prepVol, PrecisionType* volume)
{
    int size = prepVolumes.volumeSize * sizeof(double);
    if (cudaMemcpyAsync(prepVol, mdaData, size, cudaMemcpyHostToDevice, prepStream) != cudaSuccess)
        processCudaError();
    prepareVolumes<PADDING><<<grid, block, 0, prepStream>>>(volume, prepVol, imageMetaData);
}

void VolumeDeformSph::prepareVI()
{
    prepareVolume<true>(program->VI().data, dTmpVI, images.VI);
}

void VolumeDeformSph::waitToFinishPreparations()
{
    if (cudaStreamSynchronize(prepStream) != cudaSuccess)
        processCudaError();
}

void VolumeDeformSph::cleanupPreparations()
{
    if (cudaFree(prepVolumes.I) != cudaSuccess)
        processCudaError();
    if (cudaFree(prepVolumes.R) != cudaSuccess)
        processCudaError();
    if (cudaFree(dTmpVI) != cudaSuccess)
        processCudaError();
    if (cudaStreamDestroy(prepStream) != cudaSuccess)
        processCudaError();
}

void VolumeDeformSph::setupOutputs()
{
    if (cudaMallocHost(&outputs, sizeof(KernelOutputs)) != cudaSuccess)
        processCudaError();
}

void VolumeDeformSph::setupOutputArray()
{
    if (cudaMalloc(&reductionArray, 3 * totalGridSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();
}

void VolumeDeformSph::setupClnm()
{
    std::vector<PrecisionType3> tmp(MAX_COEF_COUNT);
    for (unsigned i = 0; i < program->vL1.size(); ++i) {
        tmp[i].x = program->clnm[i];
        tmp[i].y = program->clnm[i + program->vL1.size()];
        tmp[i].z = program->clnm[i + program->vL1.size() * 2];
    }
    if (cudaMemcpyToSymbol(clnmShared, tmp.data(), 56 * sizeof(PrecisionType3)) != cudaSuccess)
        processCudaError();
}

KernelOutputs VolumeDeformSph::getOutputs()
{
    return *outputs;
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
    // Before and after running the kernel is no need for explicit synchronization,
    // because it is being run in the default cuda stream, therefore it is synchronized automatically
    // If the cuda stream of this kernel ever changes explicit synchronization is needed!
    if (program->L1 > 3 || program->L2 > 3) {
        computeDeform<BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM, 5, 5>
            <<<grid, block, constantSharedMemSize + changingSharedMemSize>>>(
                    Rmax2,
                    iRmax,
                    images,
                    steps,
                    imageMetaData,
                    volumes,
                    deformImages,
                    applyTransformation,
                    saveDeformation,
                    reductionArray
                    );
    } else {
        computeDeform<BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM, 3, 3>
            <<<grid, block, constantSharedMemSize + changingSharedMemSize>>>(
                    Rmax2,
                    iRmax,
                    images,
                    steps,
                    imageMetaData,
                    volumes,
                    deformImages,
                    applyTransformation,
                    saveDeformation,
                    reductionArray
                    );
    }

    PrecisionType* diff2Ptr = reductionArray;
    PrecisionType* sumVDPtr = diff2Ptr + totalGridSize;
    PrecisionType* modgPtr = sumVDPtr + totalGridSize;

    reduceDiff.reduceDeviceArrayAsync(diff2Ptr, totalGridSize, &outputs->diff2);
    reduceSumVD.reduceDeviceArrayAsync(sumVDPtr, totalGridSize, &outputs->sumVD);
    reduceModg.reduceDeviceArrayAsync(modgPtr, totalGridSize, &outputs->modg);

    cudaDeviceSynchronize();
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
    std::vector<int4> zshparamsVec(program->vL1.size());

    for (unsigned i = 0; i < zshparamsVec.size(); ++i) {
        zshparamsVec[i].w = program->vL1[i];
        zshparamsVec[i].x = program->vN[i];
        zshparamsVec[i].y = program->vL2[i];
        zshparamsVec[i].z = program->vM[i];
    }

    if (cudaMemcpyToSymbol(zshShared, zshparamsVec.data(),
                zshparamsVec.size() * sizeof(int4)) != cudaSuccess)
        processCudaError();
}

void setupImageNew(Image<double>& inputImage, PrecisionType** outputImageData)
{
    auto& mda = inputImage();
    transformData(outputImageData, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

void makePadded(const MultidimArray<double>& orig, void* dest, size_t size)
{
    MultidimArray<PrecisionType> tmpMA;
    typeCast(orig, tmpMA);
    tmpMA.selfWindow(STARTINGZ(tmpMA) - 1, STARTINGY(tmpMA) - 1, STARTINGX(tmpMA) - 1,
            FINISHINGZ(tmpMA) + 1, FINISHINGY(tmpMA) + 1, FINISHINGX(tmpMA) + 1);
    if (cudaMemcpy(dest, tmpMA.data, size * sizeof(PrecisionType), cudaMemcpyHostToDevice) != cudaSuccess)
        processCudaError();
}

//FIXME VR should not be padded (it is not necessary)
void VolumeDeformSph::setupVolumes()
{
    //volumes.count = program->volumesR.size();
    //volumes.volumeSize = (program->VR().xdim + 2) *
    //    (program->VR().ydim + 2) * (program->VR().zdim + 2);

    //if (cudaMalloc(&volumes.I, volumes.count * volumes.volumeSize * sizeof(PrecisionType)) != cudaSuccess)
    //    processCudaError();
    //if (cudaMalloc(&volumes.R, volumes.count * volumes.volumeSize * sizeof(PrecisionType)) != cudaSuccess)
    //    processCudaError();

    //FIXME should be working now, but can be faster -> transfer non-padded data, malloc space for padded data,
    //make kernel that places the data correctly in the padded memory (plus it will be done in async!)
    for (size_t i = 0; i < volumes.count; i++) {
        PrecisionType* tmpI = volumes.I + i * volumes.volumeSize;
        PrecisionType* tmpR = volumes.R + i * volumes.volumeSize;
        makePadded(program->volumesI[i](), tmpI, volumes.volumeSize);
        makePadded(program->volumesR[i](), tmpR, volumes.volumeSize);
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
    size_t size = (mda.xdim + 2) * (mda.ydim + 2) * (mda.zdim + 2);
    cudaMalloc(outputImageData, size * sizeof(PrecisionType));
    makePadded(mda, *outputImageData, size);
}

void VolumeDeformSph::setupImage(const ImageMetaData& inputImage, PrecisionType** outputImageData)
{
    size_t size = inputImage.xDim * inputImage.yDim * inputImage.zDim * sizeof(PrecisionType);
    if (cudaMalloc(outputImageData, size) != cudaSuccess)
        processCudaError();
}
