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
    if (program->useFakeKernel) {
    } else {
        cudaFree(dVolData);
        cudaFree(dRotation);
        cudaFree(dZshParams);
        cudaFree(dClnm);
        cudaFree(dVolMask);
        cudaFree(dProjectionPlane);
    }
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
        throw(std::runtime_error("AngularSphAlignment not associated with the program!"));

    // kernel arguments
    this->Rmax2 = program->RmaxDef * program->RmaxDef;
    this->iRmax = 1.0 / program->RmaxDef;
    setupImageMetaData(program->V);

    if (program->useFakeKernel) {
        setupVolumeDataCpu();
        setupVolumeMaskCpu();
        setupZSHparamsCpu();
    } else {
        setupVolumeData();
        setupVolumeMask();
        setupZSHparams();
    }

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
        throw(std::runtime_error("AngularSphAlignment not associated with the program!"));

    if (program->useFakeKernel) {
        setupClnmCpu();
        setupRotationCpu();
        setupProjectionPlaneCpu();
    } else {
        setupClnm();
        setupRotation();
        setupProjectionPlane();
    }

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

void AngularSphAlignment::setupClnmCpu()
{
    clnmVec.resize(program->vL1.size());

    for (unsigned i = 0; i < program->vL1.size(); ++i) {
        clnmVec[i].x = program->clnm[i];
        clnmVec[i].y = program->clnm[i + program->vL1.size()];
        clnmVec[i].z = program->clnm[i + program->vL1.size() * 2];
    }
    dClnm = clnmVec.data();
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

void AngularSphAlignment::setupVolumeData() 
{
    const auto& vol = program->V();
    transformData(&dVolData, vol.data, vol.zyxdim, dVolData == nullptr);
}

void AngularSphAlignment::setupVolumeDataCpu() 
{
    const auto& vol = program->V();
    volDataVec.assign(vol.data, vol.data + vol.zyxdim);
    dVolData = volDataVec.data();
}

void AngularSphAlignment::setupRotation() 
{
    transformData(&dRotation, program->R.mdata, program->R.mdim, dRotation == nullptr);
}

void AngularSphAlignment::setupRotationCpu() 
{
    rotationVec.assign(program->R.mdata, program->R.mdata + program->R.mdim);
    dRotation = rotationVec.data();
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

void AngularSphAlignment::setupVolumeMaskCpu()
{
    dVolMask = program->V_mask.data;
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

void AngularSphAlignment::setupProjectionPlaneCpu() 
{
    const auto& projPlane = program->P();
    projectionPlaneVec.assign(projPlane.data, projPlane.data + projPlane.yxdim);
    dProjectionPlane = projectionPlaneVec.data();
}

void fakeKernel(
        PrecisionType Rmax2,
        PrecisionType iRmax,
        ImageMetaData volMeta,
        PrecisionType* volData,
        PrecisionType* rotation,
        int steps,
        int4* zshparams,
        PrecisionType3* clnm,
        int* Vmask,
        PrecisionType* projectionPlane,
        KernelOutputs* outputs
        );

void AngularSphAlignment::runKernel() 
{
    if (program->useFakeKernel) {
        fakeKernel(
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
                &outputs);
    } else {
        // Define thrust reduction vector
        thrust::device_vector<PrecisionType> thrustVec(totalGridSize * 3, 0.0);

        // TEST make sure everything is ready before kernel starts
        cudaDeviceSynchronize();

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
                thrust::raw_pointer_cast(thrustVec.data())
                );

        cudaDeviceSynchronize();

        auto countIt = thrustVec.begin();
        auto sumVDIt = countIt + totalGridSize;
        auto modgIt = sumVDIt + totalGridSize;

        outputs.count = thrust::reduce(countIt, sumVDIt);
        outputs.sumVD = thrust::reduce(sumVDIt, modgIt);
        outputs.modg = thrust::reduce(modgIt, thrustVec.end());
    }
}

void AngularSphAlignment::transferProjectionPlane() 
{
    // mozna lepsi nez neustale pretypovavat a kopirovat vectory, to proste ukladat v double na GPU
    // nic se tam nepocita jen se to ulozi (tzn "jedno" pretypovani z float na double)
    std::vector<PrecisionType> tmp(program->P().zyxdim);
    cudaMemcpy(tmp.data(), dProjectionPlane, tmp.size() * sizeof(PrecisionType),
            cudaMemcpyDeviceToHost);
    std::vector<double> tmpDouble(tmp.begin(), tmp.end());
    memcpy(program->P().data, tmpDouble.data(), tmpDouble.size() * sizeof(double));
}

void AngularSphAlignment::transferProjectionPlaneCpu() 
{
    std::vector<double> tmp(projectionPlaneVec.begin(), projectionPlaneVec.end());
    memcpy(program->P().data, tmp.data(), tmp.size() * sizeof(double));
}

void AngularSphAlignment::transferResults() 
{
    if (program->useFakeKernel) {
        transferProjectionPlaneCpu();
    } else {
        transferProjectionPlane();
    }
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

void AngularSphAlignment::setupZSHparamsCpu()
{
    zshparamsVec.resize(program->vL1.size());

    for (unsigned i = 0; i < zshparamsVec.size(); ++i) {
        zshparamsVec[i].w = program->vL1[i];
        zshparamsVec[i].x = program->vN[i];
        zshparamsVec[i].y = program->vL2[i];
        zshparamsVec[i].z = program->vM[i];
    }
    dZshParams = zshparamsVec.data();
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

void AngularSphAlignment::runKernelTest(
        Matrix1D<double>& clnm,
        size_t idxY0,
        double RmaxF2,
        double iRmaxF,
        Matrix2D<double> R,
        const MultidimArray<double>& mV,
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

    /*
    std::cout 
        << "clnm: " << clnm[0] << "," << clnm[1] << "," << clnm[2] << "\n"
        << "Rmax2: " << RmaxF2 << "\n"
        << "iRmax: " << iRmaxF << "\n"
        << "Rotation: " << R(0, 0) << "," << R(0, 1) << "," << R(1, 2) << "\n"
        << "Volume: " << mV(0, 0, 0) << "," << mV(0, 0, 1) << "," << mV(0, 1, 2) << "\n"
        << "" << std::endl;
    */

    for (int k=STARTINGZ(mV); k<=FINISHINGZ(mV); k++) {
        for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i++) {
            for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j++) {
                ZZ(pos) = k; YY(pos) = i; XX(pos) = j;
                pos = R * pos;
                //if (k == 10 && i == 10 && j == 10)
                //    std::cout << "pos("<<pos[0]<<","<<pos[1]<<","<<pos[2]<<")" << std::endl;
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
                            zsph=::ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
                            if (rr>0 || l2==0) {
                                gx += VEC_ELEM(clnm,idx)        *(zsph);
                                gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
                                gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
                            }
                        }
                    }

                    int k_mask, i_mask, j_mask;
                    int voxelI_mask;
                    k_mask = (int)(ZZ(pos)+gz);
                    i_mask = (int)(YY(pos)+gy);
                    j_mask = (int)(XX(pos)+gx);

                    if (V_mask.outside(k_mask, i_mask, j_mask)) {
                        voxelI_mask = 0;
                    } else {
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

void rotate(PrecisionType* pos, PrecisionType* rotation)
{
    PrecisionType tmp[3] = {0};

    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            tmp[i] += rotation[3 * i + j] * pos[j];

    pos[0] = tmp[0];
    pos[1] = tmp[1];
    pos[2] = tmp[2];
}

PrecisionType interpolatedElement3DCpu(PrecisionType* data, ImageMetaData ImD,
        PrecisionType x, PrecisionType y, PrecisionType z,
        PrecisionType outside_value = 0) 
{
        int x0 = FLOOR(x);
        PrecisionType fx = x - x0;
        int x1 = x0 + 1;

        int y0 = FLOOR(y);
        PrecisionType fy = y - y0;
        int y1 = y0 + 1;

        int z0 = FLOOR(z);
        PrecisionType fz = z - z0;
        int z1 = z0 + 1;

        PrecisionType d000 = (IS_OUTSIDE(ImD, z0, y0, x0)) ?
            outside_value : ELEM_3D_SHIFTED(data, ImD, z0, y0, x0);
        PrecisionType d001 = (IS_OUTSIDE(ImD, z0, y0, x1)) ?
            outside_value : ELEM_3D_SHIFTED(data, ImD, z0, y0, x1);
        PrecisionType d010 = (IS_OUTSIDE(ImD, z0, y1, x0)) ?
            outside_value : ELEM_3D_SHIFTED(data, ImD, z0, y1, x0);
        PrecisionType d011 = (IS_OUTSIDE(ImD, z0, y1, x1)) ?
            outside_value : ELEM_3D_SHIFTED(data, ImD, z0, y1, x1);
        PrecisionType d100 = (IS_OUTSIDE(ImD, z1, y0, x0)) ?
            outside_value : ELEM_3D_SHIFTED(data, ImD, z1, y0, x0);
        PrecisionType d101 = (IS_OUTSIDE(ImD, z1, y0, x1)) ?
            outside_value : ELEM_3D_SHIFTED(data, ImD, z1, y0, x1);
        PrecisionType d110 = (IS_OUTSIDE(ImD, z1, y1, x0)) ?
            outside_value : ELEM_3D_SHIFTED(data, ImD, z1, y1, x0);
        PrecisionType d111 = (IS_OUTSIDE(ImD, z1, y1, x1)) ?
            outside_value : ELEM_3D_SHIFTED(data, ImD, z1, y1, x1);

        PrecisionType dx00 = LIN_INTERP(fx, d000, d001);
        PrecisionType dx01 = LIN_INTERP(fx, d100, d101);
        PrecisionType dx10 = LIN_INTERP(fx, d010, d011);
        PrecisionType dx11 = LIN_INTERP(fx, d110, d111);
        PrecisionType dxy0 = LIN_INTERP(fy, dx00, dx10);
        PrecisionType dxy1 = LIN_INTERP(fy, dx01, dx11);

        return LIN_INTERP(fz, dxy0, dxy1);
}

void fakeKernel(
        PrecisionType Rmax2,
        PrecisionType iRmax,
        ImageMetaData volMeta,
        PrecisionType* volData,
        PrecisionType* rotation,
        int steps,
        int4* zshparams,
        PrecisionType3* clnm,
        int* Vmask,
        PrecisionType* projectionPlane,
        KernelOutputs* outputs
        ) 
{
    outputs->sumVD = 0.0;
    outputs->modg = 0.0;
    outputs->count = 0.0;

    PrecisionType pos[3];

    /*
    std::cout 
        << "clnm: " << clnm[0].x << "," << clnm[1].x << "," << clnm[2].x << "\n"
        << "Rmax2: " << Rmax2 << "\n"
        << "iRmax: " << iRmax << "\n"
        << "Rotation: " << rotation[0] << "," << rotation[1] << "," << rotation[5] << "\n"
        << "Volume: " << ELEM_3D_SHIFTED(volData, volMeta, 0, 0, 0) << "," << ELEM_3D_SHIFTED(volData, volMeta, 0, 0, 1) << "," << ELEM_3D_SHIFTED(volData, volMeta, 0, 1, 2) << "\n"
        << "" << std::endl;
    */

    for (int k = P2L_Z_IDX(volMeta, 0); k < P2L_Z_IDX(volMeta, volMeta.zDim); k++) {
        for (int i = P2L_Y_IDX(volMeta, 0); i < P2L_Y_IDX(volMeta, volMeta.yDim); i++) {
            for (int j = P2L_X_IDX(volMeta, 0); j < P2L_X_IDX(volMeta, volMeta.xDim); j++) {
                pos[2] = k; pos[1] = i; pos[0] = j;
                rotate(pos, rotation);
                //if (k == 10 && i == 10 && j == 10)
                //    std::cout << "pos("<<pos[0]<<","<<pos[1]<<","<<pos[2]<<")" << std::endl;
                double gx = 0.0, gy = 0.0, gz = 0.0;
                double k2= pos[2] * pos[2];
                double kr= pos[2] * iRmax;
                double k2i2 =k2 + pos[1] * pos[1];
                double ir= pos[1] * iRmax;
                double r2= k2i2 + pos[0] * pos[0];
                double jr= pos[0] * iRmax;
                double rr = sqrt(r2) * iRmax;

                if (r2 < Rmax2) {
                    for (int idx = 0; idx < steps; idx++) {
                        int l1 = zshparams[idx].w;
                        int n = zshparams[idx].x;
                        int l2 = zshparams[idx].y;
                        int m = zshparams[idx].z;
                        PrecisionType zsph = ::ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
                        if (rr>0 || l2==0) {
                            gx += zsph * clnm[idx].x;
                            gy += zsph * clnm[idx].y;
                            gz += zsph * clnm[idx].z;
                        }
                    }

                    int k_mask, i_mask, j_mask;
                    int voxelI_mask;
                    k_mask = (int)(pos[2] + gz);
                    i_mask = (int)(pos[1] + gy);
                    j_mask = (int)(pos[0] + gx);

                    if (IS_OUTSIDE(volMeta, k_mask, i_mask, j_mask)) {
                        voxelI_mask = 0;
                    } else {
                        voxelI_mask = ELEM_3D_SHIFTED(Vmask, volMeta, k_mask, i_mask, j_mask);
                    }

                    if (voxelI_mask == 1) {
                        double voxelI=interpolatedElement3DCpu(volData, volMeta,
                                pos[0] + gx, pos[1] + gy, pos[2] + gz);
                        ELEM_2D_SHIFTED(projectionPlane, volMeta, i, j) += voxelI;
                        outputs->sumVD += voxelI;
                        outputs->modg += gx*gx + gy*gy + gz*gz;
                        outputs->count++;
                    }
                }
            }// inner for
        }
    }
}

} // namespace AngularAlignmentGpu
