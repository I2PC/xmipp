/***************************************************************************
 *
 * Authors:    David Myska (davidmyska@mail.muni.cz)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

// Xmipp includes
#include "core/metadata_label.h"
#include "core/xmipp_random_mode.h"
#include "core/matrix1d.h"
#include "reconstruction_adapt_cuda/angular_sph_alignment_gpu.h"
#include "cuda_angular_sph_alignment.h"
#include "cuda_angular_sph_alignment.cu"
#include "cuda_volume_deform_sph_defines.h"//TODO
#include "reconstruction_cuda/cuda_asserts.h"
// Standard includes
#include <iterator>
#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <exception>

// Data that can't be in the header file because of compilation scope
namespace {
    dim3 grid;
    dim3 block;
    // Cuda stream used during the data preparation. More streams could be used
    // but at this point preparations on GPU are not as intesive as to require
    // more streams. More streams might be useful for combination of
    // weak GPU and powerful CPU.
    cudaStream_t prepStream;
}

AngularSphAlignment::AngularSphAlignment(ProgAngularSphAlignmentGpu* prog)
{
    program = prog;

    cudaStreamCreate(&prepStream);
}

AngularSphAlignment::~AngularSphAlignment()
{
    cudaFree(dVolData);
    cudaFree(dVolMask);
    cudaFree(dProjectionPlane);
    cudaFree(reductionArray);
    cudaFreeHost(outputs);
    cudaStreamDestroy(prepStream);
}

void AngularSphAlignment::setupConstantParameters()
{
    if (program == nullptr)
        throw new std::runtime_error("AngularSphAlignment not associated with the program!");

    this->Rmax2 = program->RmaxDef * program->RmaxDef;
    this->iRmax = 1.0 / program->RmaxDef;

    setupVolumeMask();
    setupZSHparams();
    setupOutputs();
    setupOutputArray();
}

void AngularSphAlignment::setupChangingParameters()
{
    if (program == nullptr)
        throw new std::runtime_error("AngularSphAlignment not associated with the program!");

    setupClnm();
    setupRotation();
    setupProjectionPlane();

    steps = program->onesInSteps;
}

void AngularSphAlignment::setupGpuBlocks()
{
    block.x = BLOCK_X_DIM;
    block.y = BLOCK_Y_DIM;
    block.z = BLOCK_Z_DIM;
    grid.x = ((volumeMetaData.xDim + block.x - 1) / block.x);
    grid.y = ((volumeMetaData.yDim + block.y - 1) / block.y);
    grid.z = ((volumeMetaData.zDim + block.z - 1) / block.z);

    kernelOutputSize = grid.x * grid.y * grid.z * (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM / 32);
}

void AngularSphAlignment::setupClnm()
{
    clnmPrepVec.resize(MAX_COEF_COUNT);

    for (unsigned i = 0; i < program->vL1.size(); ++i) {
        clnmPrepVec[i].x = program->clnm[i];
        clnmPrepVec[i].y = program->clnm[i + program->vL1.size()];
        clnmPrepVec[i].z = program->clnm[i + program->vL1.size() * 2];
    }

    gpuErrchk(cudaMemcpyToSymbol(cClnm, clnmPrepVec.data(), MAX_COEF_COUNT * sizeof(PrecisionType3)));
}

void AngularSphAlignment::setupOutputs()
{
    if (outputs == nullptr){//FIXME do this better
        gpuErrchk(cudaMallocHost(&outputs, sizeof(KernelOutputs)));
    }
}

void AngularSphAlignment::setupOutputArray()
{
    if (reductionArray == nullptr){//FIXME do this better
        gpuErrchk(cudaMalloc(&reductionArray, 3 * kernelOutputSize * sizeof(PrecisionType)));
    }
}

KernelOutputs AngularSphAlignment::getOutputs()
{
    return *outputs;
}

void AngularSphAlignment::init()
{
    setupVolumeMetaData(program->V);
    setupGpuBlocks();

    int size = volumeMetaData.xDim * volumeMetaData.yDim * volumeMetaData.zDim * sizeof(double);
    int paddedSize = (volumeMetaData.xDim + 2) * (volumeMetaData.yDim + 2) *
        (volumeMetaData.zDim + 2) * sizeof(PrecisionType);

    gpuErrchk(cudaMalloc(&dVolData, paddedSize));
    gpuErrchk(cudaMalloc(&dPrepVolume, size));
}

void AngularSphAlignment::prepareVolumeData()
{
    prepareVolume<true>(program->V().data, dPrepVolume, dVolData);
}

template<bool PADDING>
void AngularSphAlignment::prepareVolume(const double* mdaData, double* prepVol, PrecisionType* volume)
{
    int size = volumeMetaData.xDim * volumeMetaData.yDim * volumeMetaData.zDim * sizeof(double);
    int paddedSize = (volumeMetaData.xDim + 2) * (volumeMetaData.yDim + 2) *
        (volumeMetaData.zDim + 2) * sizeof(PrecisionType);

    gpuErrchk(cudaMemsetAsync(volume, 0, paddedSize, prepStream));
    gpuErrchk(cudaMemcpyAsync(prepVol, mdaData, size, cudaMemcpyHostToDevice, prepStream));
    prepareVolumeKernel<PADDING><<<grid, block, 0, prepStream>>>(volume, prepVol, volumeMetaData);
}

void AngularSphAlignment::waitToFinishPreparations()
{
    gpuErrchk(cudaStreamSynchronize(prepStream));
}

void AngularSphAlignment::cleanupPreparations()
{
    if (dPrepVolume != nullptr) {
        gpuErrchk(cudaFree(dPrepVolume));
        dPrepVolume = nullptr;
    }
}

void AngularSphAlignment::setupRotation()
{
    std::vector<PrecisionType> tmp(program->R.mdata, program->R.mdata + program->R.mdim);
    gpuErrchk(cudaMemcpyToSymbol(cRotation, tmp.data(), 9 * sizeof(PrecisionType)));
}

void AngularSphAlignment::setupVolumeMask()
{
    if (dVolMask == nullptr) {//FIXME do this better
        auto size = program->V_mask.getSize() * sizeof(int);
        gpuErrchk(cudaMalloc(&dVolMask, size));
        gpuErrchk(cudaMemcpy(dVolMask, program->V_mask.data, size, cudaMemcpyHostToDevice));
    } else {
        gpuErrchk(cudaMemcpy(dVolMask, program->V_mask.data, program->V_mask.getSize() * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void AngularSphAlignment::setupProjectionPlane()
{
    const auto& projPlane = program->P();
    if (dProjectionPlane == nullptr) {
        gpuErrchk(cudaMalloc(&dProjectionPlane, projPlane.yxdim * sizeof(double)));
    }
    cudaMemset(dProjectionPlane, 0, projPlane.yxdim * sizeof(double));
}

void AngularSphAlignment::runKernelAsync()
{
    // Before and after running the kernel is no need for explicit synchronization,
    // because it is being run in the default cuda stream, therefore it is synchronized automatically
    // If the cuda stream of this kernel ever changes explicit synchronization is needed!
    if (program->L1 > 3 || program->L2 > 3) {
        projectionKernel<BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM, 5, 5>
            <<<grid, block>>>(
                    Rmax2,
                    iRmax,
                    volumeMetaData,
                    dVolData,
                    steps,
                    dVolMask,
                    dProjectionPlane,
                    reductionArray
                    );
    } else {
        projectionKernel<BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM, 3, 3>
            <<<grid, block>>>(
                    Rmax2,
                    iRmax,
                    volumeMetaData,
                    dVolData,
                    steps,
                    dVolMask,
                    dProjectionPlane,
                    reductionArray
                    );
    }

    // Transfer will start only after projectionKernel ends, because it is being
    // run in the default stream
    transferProjectionPlaneAsync();

    PrecisionType* countPtr = reductionArray;
    PrecisionType* sumVDPtr = countPtr + kernelOutputSize;
    PrecisionType* modgPtr = sumVDPtr + kernelOutputSize;

    // Reduction will start only after projectionKernel ends, because it is being
    // run in the default stream
    reduceCount.reduceDeviceArrayAsync(countPtr, kernelOutputSize, &outputs->count);
    reduceSumVD.reduceDeviceArrayAsync(sumVDPtr, kernelOutputSize, &outputs->sumVD);
    reduceModg.reduceDeviceArrayAsync(modgPtr, kernelOutputSize, &outputs->modg);
}

void AngularSphAlignment::transferProjectionPlaneAsync()
{
    gpuErrchk(cudaMemcpyAsync(program->P().data, dProjectionPlane, program->P().yxdim * sizeof(double),
                cudaMemcpyDeviceToHost, prepStream));
}

void AngularSphAlignment::synchronize()
{
    gpuErrchk(cudaStreamSynchronize(prepStream));
    reduceCount.synchronize();
    reduceSumVD.synchronize();
    reduceModg.synchronize();
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

    gpuErrchk(cudaMemcpyToSymbol(cZsh, zshparamsVec.data(), zshparamsVec.size() * sizeof(int4)));
}

void AngularSphAlignment::setupVolumeMetaData(const Image<double>& mda)
{
    volumeMetaData.xShift = mda().xinit;
    volumeMetaData.yShift = mda().yinit;
    volumeMetaData.zShift = mda().zinit;
    volumeMetaData.xDim = mda().xdim;
    volumeMetaData.yDim = mda().ydim;
    volumeMetaData.zDim = mda().zdim;
}

