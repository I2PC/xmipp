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
cudaError cudaMallocAndCopy(T** target, const T* source, size_t numberOfElements)
{
    size_t memSize = numberOfElements * sizeof(T);

    cudaError err = cudaSuccess;
    if ((err = cudaMalloc(target, memSize)) != cudaSuccess) {
        *target = NULL;
        return err;
    }

    if ((err = cudaMemcpy(*target, source, memSize, cudaMemcpyHostToDevice)) != cudaSuccess) {
        cudaFree(*target);
        *target = NULL;
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
// type 'Source' to type 'Target'. Works only for numeric types
template<typename Target, typename Source>
void transformData(Target** dest, Source* source, size_t n, bool mallocMem = true)
{
    std::vector<Target> tmp(source, source + n);

    if (mallocMem) {
        if (cudaMalloc(dest, sizeof(Target) * n) != cudaSuccess){
            processCudaError();
        }
    }

    if (cudaMemcpy(*dest, tmp.data(), sizeof(Target) * n, cudaMemcpyHostToDevice) != cudaSuccess){
        processCudaError();
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

VolumeDeformSph::~VolumeDeformSph() 
{
    cudaFree(images.I);
    cudaFree(images.R);
    cudaFree(images.O);

    cudaFree(volumes.R);
    cudaFree(volumes.I);

    cudaFree(outputImages.Gx);
    cudaFree(outputImages.Gy);
    cudaFree(outputImages.Gz);
}

void VolumeDeformSph::associateWith(ProgVolumeDeformSphGpu* prog) 
{
    program = prog;
}

void VolumeDeformSph::setupConstantParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    if (program->pathToXmipp.isEmpty()) {
        throw new std::runtime_error("Path to the Xmipp-bundle in not specified!");
    }

    pathToXmipp = program->pathToXmipp;

    this->Rmax2 = program->Rmax * program->Rmax;
    this->iRmax = 1 / program->Rmax;

    setupImageMetaData(program->VR);
    setupImage(program->VI, &images.I);
    //setupImage(program->VR, &images.R);
    setupImage(&images.O);
    setupZSHparams();
    setupVolumes();
    setupOutputImages();

    setupConstantKtt();

    constantDataReady = true;
}

void VolumeDeformSph::setupKttKernel() 
{
    // kernel dimension
    kttBlock.setSizeX(1);
    kttBlock.setSizeY(1);
    kttBlock.setSizeZ(1);
    kttGrid.setSizeX(imageMetaData.xDim);
    kttGrid.setSizeY(imageMetaData.yDim);
    kttGrid.setSizeZ(imageMetaData.zDim);

    // kernel init
    kernelId = tuner.addKernelFromFile(pathToXmipp + "/" + pathToKernel,
            "computeDeformation", kttGrid, kttBlock);
}

void VolumeDeformSph::setupKttBlockSize() 
{
    // tuning block/grid size
    tuner.addParameter(kernelId, BLOCK_X_DIM, { 1, 2, 4, 8, 16, 32, 64, 128});
    tuner.addParameter(kernelId, BLOCK_Y_DIM, { 1, 2, 4, 8, 16, 32, 64, 128});
    tuner.addParameter(kernelId, BLOCK_Z_DIM, { 1, 2, 4, 8, 16, 32, 64, 128});

    // block size modification
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local,
            ktt::ModifierDimension::X, BLOCK_X_DIM, ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local,
            ktt::ModifierDimension::Y, BLOCK_Y_DIM, ktt::ModifierAction::Multiply);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local,
            ktt::ModifierDimension::Z, BLOCK_Z_DIM, ktt::ModifierAction::Multiply);

    // grid size modification
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global,
            ktt::ModifierDimension::X, BLOCK_X_DIM, ktt::ModifierAction::DivideCeil);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global,
            ktt::ModifierDimension::Y, BLOCK_Y_DIM, ktt::ModifierAction::DivideCeil);
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global,
            ktt::ModifierDimension::Z, BLOCK_Z_DIM, ktt::ModifierAction::DivideCeil);

    // block size constrains
    tuner.addConstraint(kernelId, { BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM },
            [&metaData = imageMetaData](const std::vector<size_t>& vec)
            {
                return 32 <= vec[0] * vec[1] * vec[2]
                    && vec[0] < metaData.xDim
                    && vec[1] < metaData.yDim
                    && vec[2] < metaData.zDim;
            });
}

void VolumeDeformSph::setupKttDefines()
{
    tuner.addParameter(kernelId, "L1", {static_cast<unsigned>(program->L1)});
    tuner.addParameter(kernelId, "L2", {static_cast<unsigned>(program->L2)});
    tuner.addParameter(kernelId, "KTT_USED", {1});
}

void VolumeDeformSph::setupKttTuningParameters()
{

}

void VolumeDeformSph::setupKttConstantKernelArguments()
{
    Rmax2Id = tuner.addArgumentScalar(Rmax2);
    iRmaxId = tuner.addArgumentScalar(iRmax);
    imagesId = tuner.addArgumentScalar(images);
    zshparamsId = tuner.addArgumentVector(zshparamsVec, ktt::ArgumentAccessType::ReadOnly);
    imageMetaDataId = tuner.addArgumentScalar(imageMetaData);
    volumesId = tuner.addArgumentScalar(volumes);
    outputImagesId = tuner.addArgumentScalar(outputImages);
}

void VolumeDeformSph::setupKttSharedMemory()
{
    // Dynamic shared memory allocation
    size_t sharedMemSize = sizeof(PrecisionType*) * volumes.count * 2;
    sharedMemSize += sizeof(int4) * program->vecSize;
    sharedMemSize += sizeof(PrecisionType3) * program->vecSize;
    sharedMemId = tuner.addArgumentLocal<char>(sharedMemSize);
}

void VolumeDeformSph::setupConstantKtt()
{
    setupKttConstantKernelArguments();
    setupKttKernel();
    setupKttBlockSize();

    setupKttDefines();
    setupKttTuningParameters();
    setupKttSharedMemory();
}

void VolumeDeformSph::setupChangingParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    setupClnm();

    this->applyTransformation = program->applyTransformation;
    this->saveDeformation = program->saveDeformation;

    setupChangingKtt();

    changingDataReady = true;
}

void VolumeDeformSph::setupChangingKtt() 
{
    clnmId = tuner.addArgumentVector(clnmVec, ktt::ArgumentAccessType::ReadOnly);
    stepsId = tuner.addArgumentScalar(program->onesInSteps);

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
    thrust::device_vector<PrecisionType> thrustVec(kttGrid.getTotalSize() * 3, 0.0);

    // Add arguments for the kernel
    ktt::ArgumentId thrustVecId = tuner.addArgumentVector<PrecisionType>(
            static_cast<ktt::UserBuffer>(thrust::raw_pointer_cast(thrustVec.data())),
            thrustVec.size() * sizeof(PrecisionType),
            ktt::ArgumentAccessType::ReadWrite,
            ktt::ArgumentMemoryLocation::Device);

    // Assign arguments to the kernel
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{
            Rmax2Id,
            iRmaxId,
            imagesId,
            zshparamsId,
            clnmId,
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

    tunedGridSize = 1; // needed for multiple tuning in single run
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
    if (!constantDataReady || !changingDataReady) {
        throw new std::runtime_error("VolumeDeformSph - runKernel called before data setup!");
    }

    if (tuneKernel) {
        pretuneKernel();
    }

    // Define thrust reduction vector
    thrust::device_vector<PrecisionType> thrustVec(tunedGridSize * 3, 0.0);

    // Add arguments for the kernel
    ktt::ArgumentId thrustVecId = tuner.addArgumentVector<PrecisionType>(
            static_cast<ktt::UserBuffer>(thrust::raw_pointer_cast(thrustVec.data())),
            thrustVec.size() * sizeof(PrecisionType),
            ktt::ArgumentAccessType::ReadWrite,
            ktt::ArgumentMemoryLocation::Device);

    // Assign arguments to the kernel
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{
            Rmax2Id,
            iRmaxId,
            imagesId,
            zshparamsId,
            clnmId,
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
        transferImageData(program->VO, images.O);
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

void VolumeDeformSph::setupVolumes()
{
    volumes.count = program->volumesR.size();
    volumes.volumeSize = program->VR().getSize();

    if (cudaMalloc(&volumes.I, volumes.count * volumes.volumeSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();
    if (cudaMalloc(&volumes.R, volumes.count * volumes.volumeSize * sizeof(PrecisionType)) != cudaSuccess)
        processCudaError();

    for (int i = 0; i < volumes.count; i++) {
        PrecisionType* tmpI = volumes.I + i * volumes.volumeSize;
        PrecisionType* tmpR = volumes.R + i * volumes.volumeSize;
        transformData(&tmpI, program->volumesI[i]().data, volumes.volumeSize, false);
        transformData(&tmpR, program->volumesR[i]().data, volumes.volumeSize, false);
    }
}

void VolumeDeformSph::setupOutputImages()
{
    setupImage(&outputImages.Gx);
    setupImage(&outputImages.Gy);
    setupImage(&outputImages.Gz);
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
        processCudaError();
}

void VolumeDeformSph::setupImage(Image<double>& inputImage, PrecisionType** outputImageData) 
{
    auto& mda = inputImage();
    transformData(outputImageData, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

