// Xmipp includes
#include "api/dimension_vector.h"
#include "core/metadata_label.h"
#include "core/xmipp_random_mode.h"
#include "enum/argument_access_type.h"
#include "enum/argument_memory_location.h"
#include "enum/compute_api.h"
#include "enum/logging_level.h"
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
// Cuda kernel include
//#include "cuda_volume_deform_sph.cu"

// CUDA kernel defines
#define BLOCK_X_DIM 8
#define BLOCK_Y_DIM 4
#define BLOCK_Z_DIM 4
#define TOTAL_BLOCK_SIZE (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM)


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

    // ktt stuff
    Rmax2Id = tuner.addArgumentScalar(Rmax2);
    iRmaxId = tuner.addArgumentScalar(iRmax);
    imagesId = tuner.addArgumentScalar(images);
    zshparamsSCATTEREDId = tuner.addArgumentScalar(zshparamsSCATTERED);
    zshparamsId = tuner.addArgumentVector(zshparamsVec, ktt::ArgumentAccessType::ReadOnly);
    volumesId = tuner.addArgumentScalar(volumes);
    // this one is just a dummy argument, for real it is initilized
    // in the setupChangingParameters, but it has to be initilized here
    // (or elsewhere) for successful kernel call
    deformImagesId = tuner.addArgumentScalar(deformImages);

    // kernel dimension
    kttBlock.setSizeX(BLOCK_X_DIM);
    kttBlock.setSizeY(BLOCK_Y_DIM);
    kttBlock.setSizeZ(BLOCK_Z_DIM);
    kttGrid.setSizeX(images.VR.xDim / BLOCK_X_DIM);
    kttGrid.setSizeY(images.VR.yDim / BLOCK_Y_DIM);
    kttGrid.setSizeZ(images.VR.zDim / BLOCK_Z_DIM);

    // kernel init
    kernelId = tuner.addKernelFromFile(pathToXmipp + pathToKernel, "computeDeform", kttGrid, kttBlock);

    // kernel parameters
    tuner.addParameter(kernelId, "L1", std::vector<size_t>{static_cast<size_t>(program->L1)});
    tuner.addParameter(kernelId, "L2", std::vector<size_t>{static_cast<size_t>(program->L2)});
    tuner.addParameter(kernelId, "USE_SCATTERED_ZSH_CLNM", {0, 1});
    tuner.addParameter(kernelId, "USE_ZSH_FUNCTION", {0, 1});
    tuner.addParameter(kernelId, "USE_NAIVE_BLOCK_REDUCTION", {0, 1});
    tuner.addParameter(kernelId, "USE_SHARED_MEM_ZSH_CLNM", {0, 1});
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

    if (applyTransformation) {
        setupImage(images.VR, images.VO);
        imagesId = tuner.addArgumentScalar(images);
    }
    if (saveDeformation) {
        setupImage(images.VR, deformImages.Gx);
        setupImage(images.VR, deformImages.Gy);
        setupImage(images.VR, deformImages.Gz);
        deformImagesId = tuner.addArgumentScalar(deformImages);
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
}

void VolumeDeformSph::setupClnmSCATTERED()
{
    clnmVecSCATTERED.assign(program->clnm.vdata, program->clnm.vdata + program->clnm.size());
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
    // Does not work in general case, but test data have nice sizes

    // Define thrust reduction vector
    thrust::device_vector<PrecisionType> thrustVec(kttGrid.getTotalSize() * 4, 0.0);

    // Add arguments for the kernel
    ktt::ArgumentId thrustVecId = tuner.addArgumentVector<PrecisionType>(static_cast<ktt::UserBuffer>(thrust::raw_pointer_cast(thrustVec.data())), thrustVec.size() * sizeof(PrecisionType), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);

    // Assign arguments to the kernel
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{
            Rmax2Id,
            iRmaxId,
            imagesId,
            zshparamsId,
            clnmId,
            zshparamsSCATTEREDId,
            clnmSCATTEREDId,
            stepsId,
            volumesId,
            deformImagesId,
            applyTransformationId,
            saveDeformationId,
            thrustVecId
            });

    // Run/tune kernel
    tuner.tuneKernel(kernelId);


    auto diff2It = thrustVec.begin();
    auto sumVDIt = diff2It + kttGrid.getTotalSize();
    auto modgIt = sumVDIt + kttGrid.getTotalSize();
    auto NcountIt = modgIt + kttGrid.getTotalSize();

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
