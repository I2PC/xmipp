// Xmipp includes
#include "api/dimension_vector.h"
#include "core/metadata_label.h"
#include "core/xmipp_random_mode.h"
#include "enum/argument_access_type.h"
#include "enum/argument_memory_location.h"
#include "enum/compute_api.h"
#include "enum/logging_level.h"
#include "enum/modifier_action.h"
#include "enum/modifier_dimension.h"
#include "enum/modifier_type.h"
#include "enum/print_format.h"
#include "enum/time_unit.h"
#include "ktt_types.h"
#include "reconstruction_adapt_cuda/volume_deform_sph_gpu.h"
#include "cuda_volume_deform_sph.h"
#include "core/matrix1d.h"
// Standard includes
#include <iterator>
#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <exception>
// Thrust includes
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
// KTT includes
#include "tuner_api.h"


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
// type 'Source' to type 'Target'. Works only for numeric types
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

VolumeDeformSph::VolumeDeformSph() : tuner(0, 0, ktt::ComputeAPI::CUDA)
{
    tuner.setLoggingLevel(ktt::LoggingLevel::Off);
    tuner.setCompilerOptions("-std=c++14 -lineinfo"
#ifdef USE_DOUBLE_PRECISION
    " -DUSE_DOUBLE_PRECISION=1"
#endif
            );
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
    freeZSHSCATTERED();

    for (size_t i = 0; i < volumes.size; i++) {
        cudaFree(justForFreeR[i]);
        cudaFree(justForFreeI[i]);
    }
    cudaFree(volumes.R);
    cudaFree(volumes.I);

    cudaFree(outputImages.Gx);
    cudaFree(outputImages.Gy);
    cudaFree(outputImages.Gz);
    cudaFree(outputImages.VO);
}

void VolumeDeformSph::associateWith(ProgVolumeDeformSphGpu* prog) 
{
    program = prog;
}

void VolumeDeformSph::setupConstantParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    // paths
    if (!program->pathToXmipp.isEmpty()) {
        pathToXmipp = program->pathToXmipp;
    }

    // kernel arguments
    this->Rmax2 = program->Rmax * program->Rmax;
    this->iRmax = 1 / program->Rmax;
    setupImageMetaData(program->VR);
    setupZSHparams();
    setupZSHparamsSCATTERED();
    setupVolumes();
    setupOutputImages();

    // ktt stuff
    Rmax2Id = tuner.addArgumentScalar(Rmax2);
    iRmaxId = tuner.addArgumentScalar(iRmax);
    zshparamsSCATTEREDId = tuner.addArgumentScalar(zshparamsSCATTERED);
    zshparamsId = tuner.addArgumentVector(zshparamsVec, ktt::ArgumentAccessType::ReadOnly);
    imageMetaDataId = tuner.addArgumentScalar(imageMetaData);
    volumesId = tuner.addArgumentScalar(volumes);
    outputImagesId = tuner.addArgumentScalar(outputImages);

    // kernel dimension
    kttBlock.setSizeX(1);
    kttBlock.setSizeY(1);
    kttBlock.setSizeZ(1);
    kttGrid.setSizeX(imageMetaData.xDim);
    kttGrid.setSizeY(imageMetaData.yDim);
    kttGrid.setSizeZ(imageMetaData.zDim);

    // kernel init
    kernelId = tuner.addKernelFromFile(pathToXmipp + "/" + pathToKernel, "computeDeform", kttGrid, kttBlock);

    // tuning block/grid size
    tuner.addParameter(kernelId, BLOCK_X_DIM, /*{ 1, 2, 4, 8, 16, 32, 64, 128}*/{16});
    tuner.addParameter(kernelId, BLOCK_Y_DIM, /*{ 1, 2, 4, 8, 16, 32, 64, 128}*/{8});
    tuner.addParameter(kernelId, BLOCK_Z_DIM, /*{ 1, 2, 4, 8, 16, 32, 64, 128}*/{1});
    // block size modification
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, BLOCK_X_DIM, ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, BLOCK_Y_DIM, ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Z, BLOCK_Z_DIM, ktt::ModifierAction::Multiply);
    // grid size modification
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, BLOCK_X_DIM, ktt::ModifierAction::DivideCeil);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Y, BLOCK_Y_DIM, ktt::ModifierAction::DivideCeil);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Z, BLOCK_Z_DIM, ktt::ModifierAction::DivideCeil);
    // constrains
    tuner.addConstraint(kernelId, { BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM },
            [&metaData = imageMetaData](const std::vector<size_t>& vec)
            {
                return 32 <= vec[0] * vec[1] * vec[2]
                    && vec[0] < metaData.xDim && vec[1] < metaData.yDim && vec[2] < metaData.zDim;
            });

    // kernel parameters
    // simple defines
    tuner.addParameter(kernelId, "L1", {static_cast<unsigned>(program->L1)});
    tuner.addParameter(kernelId, "L2", {static_cast<unsigned>(program->L2)});
    tuner.addParameter(kernelId, "KTT_USED", {1});
    // tuning parameters
    tuner.addParameter(kernelId, "USE_SCATTERED_ZSH_CLNM", {0});
    tuner.addParameter(kernelId, "USE_ZSH_FUNCTION", {0});
    tuner.addParameter(kernelId, "USE_SHARED_MEM_ZSH_CLNM", {1});
    tuner.addParameter(kernelId, "USE_SHARED_VOLUME_DATA", {0});

    // Dynamic shared memory allocation
    sharedMemId = tuner.addArgumentLocal<char>(1);// cannot be zero

    tuner.setLocalMemoryModifier(kernelId, sharedMemId, { BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM, "USE_SHARED_VOLUME_DATA", "USE_SHARED_MEM_ZSH_CLNM" },
            [&vols = volumes.size, &steps = program->onesInSteps](const size_t size, const std::vector<size_t>& vec)
            {
                size_t sharedMemSize = sizeof(PrecisionType*) * vols * 2;
                if (vec[3] == 1) { // volume data
                    sharedMemSize += sizeof(PrecisionType) * vec[0] * vec[1] * vec[2] * vols * 2;
                }
                if (vec[4] == 1) { // zsh, clnm
                    sharedMemSize += sizeof(int4) * steps;
                    sharedMemSize += sizeof(PrecisionType3) * steps;
                }
                return sharedMemSize;
            });
}

void VolumeDeformSph::setupChangingParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    setupClnm();
    setupClnmSCATTERED();

    clnmId = tuner.addArgumentVector(clnmVec, ktt::ArgumentAccessType::ReadOnly);
    clnmSCATTEREDId = tuner.addArgumentVector(clnmVecSCATTERED, ktt::ArgumentAccessType::ReadOnly);
    stepsId = tuner.addArgumentScalar(program->onesInSteps);

    // Deformation and transformation booleans
    this->applyTransformation = program->applyTransformation;
    this->saveDeformation = program->saveDeformation;

    // ktt stuff
    applyTransformationId = tuner.addArgumentScalar(static_cast<int>(applyTransformation));
    saveDeformationId = tuner.addArgumentScalar(static_cast<int>(saveDeformation));
}

void VolumeDeformSph::setupClnm()
{
    clnmVec.resize(program->vL1.size());

    for (unsigned i = 0; i < program->vL1.size(); ++i) {
        clnmVec[i].x = program->clnm[i];
        clnmVec[i].y = program->clnm[i + program->vL1.size()];
        clnmVec[i].z = program->clnm[i + program->vL1.size() * 2];
    }
}

void VolumeDeformSph::setupClnmSCATTERED()
{
    clnmVecSCATTERED.assign(program->clnm.vdata, program->clnm.vdata + program->clnm.size());
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

void VolumeDeformSph::pretuneKernel() 
{
    if (!tuneKernel) {
        return;
    }

    // Define thrust reduction vector
    // During the tuning process it needs to be able to accomodate all the possible
    // variations of block sizes. That is why it is so large.
    thrust::device_vector<PrecisionType> thrustVec(kttGrid.getTotalSize() * 3, 0.0);

    // Add arguments for the kernel
    ktt::ArgumentId thrustVecId = tuner.addArgumentVector<PrecisionType>(static_cast<ktt::UserBuffer>(thrust::raw_pointer_cast(thrustVec.data())), thrustVec.size() * sizeof(PrecisionType), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);

    // Assign arguments to the kernel
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{
            Rmax2Id,
            iRmaxId,
            zshparamsId,
            clnmId,
            zshparamsSCATTEREDId,
            clnmSCATTEREDId,
            stepsId,
            imageMetaDataId,
            volumesId,
            outputImagesId,
            applyTransformationId,
            saveDeformationId,
            thrustVecId,
            sharedMemId
            });

    // Tune kernel
    tuner.tuneKernel(kernelId);
    tuneKernel = false;

    if (!program->kttTuningLog.isEmpty()) {
        tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);
        tuner.printResult(kernelId, program->kttTuningLog, ktt::PrintFormat::CSV);
    }

    bestKernelConfig = tuner.getBestComputationResult(kernelId).getConfiguration();
    for (const auto& pair : bestKernelConfig) {
        if (pair.getName() == BLOCK_X_DIM) {
            tunedGridSize *= ((kttGrid.getSizeX() + pair.getValue() - 1) / pair.getValue());
        }
        if (pair.getName() == BLOCK_Y_DIM) {
            tunedGridSize *= ((kttGrid.getSizeY() + pair.getValue() - 1) / pair.getValue());
        }
        if (pair.getName() == BLOCK_Z_DIM) {
            tunedGridSize *= ((kttGrid.getSizeZ() + pair.getValue() - 1) / pair.getValue());
        }
    }
}

void VolumeDeformSph::runKernel() 
{
    if (tuneKernel) {
        pretuneKernel();
    }

    // Define thrust reduction vector
    thrust::device_vector<PrecisionType> thrustVec(tunedGridSize * 3, 0.0);

    // Add arguments for the kernel
    ktt::ArgumentId thrustVecId = tuner.addArgumentVector<PrecisionType>(static_cast<ktt::UserBuffer>(thrust::raw_pointer_cast(thrustVec.data())), thrustVec.size() * sizeof(PrecisionType), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);

    // Assign arguments to the kernel
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{
            Rmax2Id,
            iRmaxId,
            zshparamsId,
            clnmId,
            zshparamsSCATTEREDId,
            clnmSCATTEREDId,
            stepsId,
            imageMetaDataId,
            volumesId,
            outputImagesId,
            applyTransformationId,
            saveDeformationId,
            thrustVecId,
            sharedMemId
            });

    // Run kernel
    tuner.runKernel(kernelId, bestKernelConfig, {});

    auto diff2It = thrustVec.begin();
    auto sumVDIt = diff2It + tunedGridSize;
    auto modgIt = sumVDIt + tunedGridSize;

    outputs.diff2 = thrust::reduce(diff2It, sumVDIt);
    outputs.sumVD = thrust::reduce(sumVDIt, modgIt);
    outputs.modg = thrust::reduce(modgIt, thrustVec.end());
}

void VolumeDeformSph::transferResults() 
{
    if (applyTransformation) {
        transferImageData(program->VO, outputImages.VO);
    }
    if (saveDeformation) {
        transferImageData(program->Gx, outputImages.Gx);
        transferImageData(program->Gy, outputImages.Gy);
        transferImageData(program->Gz, outputImages.Gz);
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
        setupImage(program->volumesR[i], &justForFreeR[i]);
        setupImage(program->volumesI[i], &justForFreeI[i]);
    }

    if (cudaMallocAndCopy(&volumes.R, justForFreeR.data(), volumes.size) != cudaSuccess)
        printCudaError();
    if (cudaMallocAndCopy(&volumes.I, justForFreeI.data(), volumes.size) != cudaSuccess)
        printCudaError();

}

void VolumeDeformSph::setupOutputImages()
{
    setupImage(&outputImages.Gx);
    setupImage(&outputImages.Gy);
    setupImage(&outputImages.Gz);
    setupImage(&outputImages.VO);
}

void VolumeDeformSph::setupImageMetaData(Image<double>& inputImage) 
{
    auto& mda = inputImage();

    imageMetaData.xShift = mda.xinit;
    imageMetaData.yShift = mda.yinit;
    imageMetaData.zShift = mda.zinit;
    imageMetaData.xDim = mda.xdim;
    imageMetaData.yDim = mda.ydim;
    imageMetaData.zDim = mda.zdim;
}

void VolumeDeformSph::setupImage(PrecisionType** imageData)
{
    if (cudaMalloc(imageData, sizeof(PrecisionType) * imageMetaData.xDim * imageMetaData.yDim * imageMetaData.zDim) != cudaSuccess)
        printCudaError();
}

void VolumeDeformSph::setupImage(Image<double>& inputImage, PrecisionType** outputImageData) 
{
    auto& mda = inputImage();

    transformData(outputImageData, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

