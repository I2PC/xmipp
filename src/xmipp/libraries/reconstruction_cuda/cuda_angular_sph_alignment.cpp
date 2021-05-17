// Xmipp includes
#include "core/metadata_label.h"
#include "core/xmipp_random_mode.h"
#include "core/matrix1d.h"
#include "reconstruction_adapt_cuda/angular_sph_alignment_gpu.h"
#include "cuda_angular_sph_alignment.h"
//#include "cuda_volume_deform_sph.cu"
#include "cuda_volume_deform_sph_defines.h"//TODO
// Standard includes
#include <iterator>
#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <exception>
// Thrust includes
#include <thrust/reduce.h>
#include <thrust/device_vector.h>


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

// AngularSphAlignment methods

AngularSphAlignment::AngularSphAlignment()
{
}

AngularSphAlignment::~AngularSphAlignment() 
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
    this->Rmax2 = program->Rmax * program->Rmax;
    this->iRmax = 1 / program->Rmax;
    //setupImage(program->VI, &images.VI);
    //setupImage(program->VR, &images.VR);
    //setupImageMetaData(program->VR);
    //setupZSHparams();
    //setupVolumes();

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

void AngularSphAlignment::setupChangingParameters() 
{
    if (program == nullptr)
        throw new std::runtime_error("AngularSphAlignment not associated with the program!");

    //setupClnm();

    steps = program->onesInSteps;

    changingSharedMemSize = 0;
    changingSharedMemSize += sizeof(int4) * steps;
    changingSharedMemSize += sizeof(PrecisionType3) * steps;

    // Deformation and transformation booleans
    //this->applyTransformation = program->applyTransformation;
    //this->saveDeformation = program->saveDeformation;

    /*
    if (applyTransformation) {
        setupImage(imageMetaData, &images.VO);
    }
    if (saveDeformation) {
        setupImage(imageMetaData, &deformImages.Gx);
        setupImage(imageMetaData, &deformImages.Gy);
        setupImage(imageMetaData, &deformImages.Gz);
    }
    */
}

void AngularSphAlignment::setupClnm()
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

KernelOutputs AngularSphAlignment::getOutputs() 
{
    return outputs;
}

void AngularSphAlignment::transferImageData(Image<double>& outputImage, PrecisionType* inputData) 
{
    size_t elements = imageMetaData.xDim * imageMetaData.yDim * imageMetaData.zDim;
    std::vector<PrecisionType> tVec(elements);
    cudaMemcpy(tVec.data(), inputData, sizeof(PrecisionType) * elements, cudaMemcpyDeviceToHost);
    std::vector<double> dVec(tVec.begin(), tVec.end());
    memcpy(outputImage().data, dVec.data(), sizeof(double) * elements);
}

void AngularSphAlignment::runKernel() 
{

    /*
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
    */
}

void AngularSphAlignment::transferResults() 
{
    /*
    if (applyTransformation) {
        transferImageData(program->VO, images.VO);
    }
    if (saveDeformation) {
        transferImageData(program->Gx, deformImages.Gx);
        transferImageData(program->Gy, deformImages.Gy);
        transferImageData(program->Gz, deformImages.Gz);
    }
    */
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

    if (cudaMallocAndCopy(&dZshParams, zshparamsVec.data(), zshparamsVec.size()) != cudaSuccess)
        processCudaError();
}

void setupImageNew(Image<double>& inputImage, PrecisionType** outputImageData) 
{
    auto& mda = inputImage();
    transformData(outputImageData, mda.data, mda.xdim * mda.ydim * mda.zdim);
}

void AngularSphAlignment::setupVolumes()
{
    /*
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
    */
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

void AngularSphAlignment::runKernelTest(
        Matrix1D<double>& clnm,
        size_t idxY0,
        double RmaxF2,
        double iRmaxF,
        Matrix2D<double> R,
        MultidimArray<double> mV,
        Matrix1D<double>& steps_cp,
        Matrix1D<int>& vL1,
        Matrix1D<int>& vN,
        Matrix1D<int>& vL2,
        Matrix1D<int>& vM,
        MultidimArray<int>& V_mask,
        MultidimArray<double>& mP
        )
{
        size_t idxZ0=2*idxY0;
        outputs.sumVD = 0.0;
    outputs.modg = 0.0;
    outputs.count = 0.0;

    Matrix1D<double> pos;
    pos.initZeros(3);

        for (int k=STARTINGZ(mV); k<=FINISHINGZ(mV); k++)
        {
                for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i++)
                {
                        for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j++)
                        {
                ZZ(pos) = k; YY(pos) = i; XX(pos) = j;
                pos = R * pos;
                                double gx=0.0, gy=0.0, gz=0.0;
                                double k2=ZZ(pos)*ZZ(pos);
                                double kr=ZZ(pos)*iRmaxF;
                                double k2i2=k2+YY(pos)*YY(pos);
                                double ir=YY(pos)*iRmaxF;
                                double r2=k2i2+XX(pos)*XX(pos);
                                double jr=XX(pos)*iRmaxF;
                                double rr=sqrt(r2)*iRmaxF;
                                if (r2<RmaxF2) {
                                        for (size_t idx=0; idx<idxY0; idx++) {
                                                if (VEC_ELEM(steps_cp,idx) == 1) {
                                                        double zsph=0.0;
                                                        int l1 = VEC_ELEM(vL1,idx);
                                                        int n = VEC_ELEM(vN,idx);
                                                        int l2 = VEC_ELEM(vL2,idx);
                                                        int m = VEC_ELEM(vM,idx);
                                                        zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
                                                        if (rr>0 || l2==0) {
                                                                gx += VEC_ELEM(clnm,idx)        *(zsph);
                                                                gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
                                                                gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
                                                        }
                                                }
                                        }
                                        int k_mask, i_mask, j_mask;
                                        int voxelI_mask;
                                        k_mask = (int)(ZZ(pos)+gz); i_mask = (int)(YY(pos)+gy); j_mask = (int)(XX(pos)+gx);
                                        if (V_mask.outside(k_mask, i_mask, j_mask)) {
                                                voxelI_mask = 0;
                                        }
                                        else {
                                                voxelI_mask = A3D_ELEM(V_mask, k_mask, i_mask, j_mask);
                                        }
                                        if (voxelI_mask == 1) {
                                                double voxelI=mV.interpolatedElement3D(XX(pos)+gx,YY(pos)+gy,ZZ(pos)+gz);
                                                A2D_ELEM(mP,i,j) += voxelI;
                                                outputs.sumVD += voxelI;
                                                outputs.modg += gx*gx+gy*gy+gz*gz;
                                                outputs.count++;
                                        }
                                }
                        }
                }
        }
}

} // namespace AngularAlignmentGpu
