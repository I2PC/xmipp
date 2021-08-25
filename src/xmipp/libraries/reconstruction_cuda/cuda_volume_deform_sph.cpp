/***************************************************************************
 *
 * Authors:    David Myska              davidmyska@mail.muni.cz
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

// Xmipp includes
#include "reconstruction_adapt_cuda/volume_deform_sph_gpu.h"
#include "cuda_volume_deform_sph.h"
#include "cuda_volume_deform_sph.cu"
#include "cuda_volume_deform_sph_defines.h"
#include "reconstruction_cuda/cuda_asserts.h"
// Standard includes

// Data that cannot be in the header file, because of compilation scope
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

void VolumeDeformSph::setupConstantParameters()
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    this->Rmax2 = program->Rmax * program->Rmax;
    this->iRmax = 1 / program->Rmax;
    setupZSHparams();

    setupOutputArray();
    setupOutputs();
}

void VolumeDeformSph::setupGpuBlocks()
{
    block.x = BLOCK_X_DIM;
    block.y = BLOCK_Y_DIM;
    block.z = BLOCK_Z_DIM;
    grid.x = ((volMetaData.xDim + block.x - 1) / block.x);
    grid.y = ((volMetaData.yDim + block.y - 1) / block.y);
    grid.z = ((volMetaData.zDim + block.z - 1) / block.z);

    kernelOutputSize = grid.x * grid.y * grid.z * (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM / 32);
}

void VolumeDeformSph::setupChangingParameters()
{
    if (program == nullptr)
        throw new std::runtime_error("VolumeDeformSph not associated with the program!");

    setupClnm();
    steps = program->onesInSteps;

    this->applyTransformation = program->applyTransformation;
    this->saveDeformation = program->saveDeformation;

    if (applyTransformation) {
        setupImage(volMetaData, &images.VO);
    }
    if (saveDeformation) {
        setupImage(volMetaData, &deformImages.Gx);
        setupImage(volMetaData, &deformImages.Gy);
        setupImage(volMetaData, &deformImages.Gz);
    }
}

// maybe this requires too much memory...
void VolumeDeformSph::initVolumes()
{
    setupVolumeMetaData(program->VR);
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

    gpuErrchk(cudaMalloc(&prepVolumes.R, prepVolumes.count * prepVolumes.volumeSize * sizeof(double)));
    gpuErrchk(cudaMalloc(&prepVolumes.I, prepVolumes.count * prepVolumes.volumeSize * sizeof(double)));

    gpuErrchk(cudaMalloc(&volumes.R, volumes.count * volumes.volumeSize * sizeof(PrecisionType)));
    gpuErrchk(cudaMalloc(&volumes.I, volumes.count * volumes.volumePaddedSize * sizeof(PrecisionType)));

    gpuErrchk(cudaMalloc(&dTmpVI, volumes.volumeSize * sizeof(double)));
    gpuErrchk(cudaMalloc(&images.VI, volumes.volumePaddedSize * sizeof(PrecisionType)));

    gpuErrchk(cudaMemsetAsync(volumes.I, 0, volumes.count * volumes.volumePaddedSize * sizeof(PrecisionType), prepStream));
    gpuErrchk(cudaMemsetAsync(images.VI, 0, volumes.volumePaddedSize * sizeof(PrecisionType), prepStream));
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
    gpuErrchk(cudaMemcpyAsync(prepVol, mdaData, size, cudaMemcpyHostToDevice, prepStream));
    prepareVolumes<PADDING><<<grid, block, 0, prepStream>>>(volume, prepVol, volMetaData);
}

void VolumeDeformSph::prepareVI()
{
    prepareVolume<true>(program->VI().data, dTmpVI, images.VI);
}

void VolumeDeformSph::waitToFinishPreparations()
{
    gpuErrchk(cudaStreamSynchronize(prepStream));
}

void VolumeDeformSph::cleanupPreparations()
{
    gpuErrchk(cudaFree(prepVolumes.I));
    gpuErrchk(cudaFree(prepVolumes.R));
    gpuErrchk(cudaFree(dTmpVI));
    gpuErrchk(cudaStreamDestroy(prepStream));
}

void VolumeDeformSph::setupOutputs()
{
    gpuErrchk(cudaMallocHost(&outputs, sizeof(KernelOutputs)));
}

void VolumeDeformSph::setupOutputArray()
{
    gpuErrchk(cudaMalloc(&reductionArray, 3 * kernelOutputSize * sizeof(PrecisionType)));
}

void VolumeDeformSph::setupClnm()
{
    std::vector<PrecisionType3> tmp(MAX_COEF_COUNT);
    for (unsigned i = 0; i < program->vL1.size(); ++i) {
        tmp[i].x = program->clnm[i];
        tmp[i].y = program->clnm[i + program->vL1.size()];
        tmp[i].z = program->clnm[i + program->vL1.size() * 2];
    }
    gpuErrchk(cudaMemcpyToSymbol(cClnm, tmp.data(), 56 * sizeof(PrecisionType3)));
}

KernelOutputs VolumeDeformSph::getOutputs()
{
    return *outputs;
}

void VolumeDeformSph::transferImageData(Image<double>& outputImage, PrecisionType* inputData)
{
    size_t elements = volMetaData.xDim * volMetaData.yDim * volMetaData.zDim;
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
            <<<grid, block>>>(
                    Rmax2,
                    iRmax,
                    images,
                    steps,
                    volMetaData,
                    volumes,
                    deformImages,
                    applyTransformation,
                    saveDeformation,
                    reductionArray
                    );
    } else {
        computeDeform<BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM, 3, 3>
            <<<grid, block>>>(
                    Rmax2,
                    iRmax,
                    images,
                    steps,
                    volMetaData,
                    volumes,
                    deformImages,
                    applyTransformation,
                    saveDeformation,
                    reductionArray
                    );
    }

    PrecisionType* diff2Ptr = reductionArray;
    PrecisionType* sumVDPtr = diff2Ptr + kernelOutputSize;
    PrecisionType* modgPtr = sumVDPtr + kernelOutputSize;

    reduceDiff.reduceDeviceArrayAsync(diff2Ptr, kernelOutputSize, &outputs->diff2);
    reduceSumVD.reduceDeviceArrayAsync(sumVDPtr, kernelOutputSize, &outputs->sumVD);
    reduceModg.reduceDeviceArrayAsync(modgPtr, kernelOutputSize, &outputs->modg);

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

    gpuErrchk(cudaMemcpyToSymbol(cZsh, zshparamsVec.data(),
                zshparamsVec.size() * sizeof(int4)));
}

void VolumeDeformSph::setupVolumeMetaData(const Image<double>& mda)
{
    volMetaData.xShift = mda().xinit;
    volMetaData.yShift = mda().yinit;
    volMetaData.zShift = mda().zinit;
    volMetaData.xDim = mda().xdim;
    volMetaData.yDim = mda().ydim;
    volMetaData.zDim = mda().zdim;
}

void VolumeDeformSph::setupImage(const VolumeMetaData& inputImage, PrecisionType** outputImageData)
{
    size_t size = inputImage.xDim * inputImage.yDim * inputImage.zDim * sizeof(PrecisionType);
    gpuErrchk(cudaMalloc(outputImageData, size));
}
