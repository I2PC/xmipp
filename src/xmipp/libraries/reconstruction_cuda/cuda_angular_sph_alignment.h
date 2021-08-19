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

#pragma once

// Xmipp includes
#include "core/xmipp_image.h"
#include "core/multidim_array.h"
#include "reconstruction_cuda/cuda_reduction.h"
// Standard includes
#include <vector>

// Forward declarations
class ProgAngularSphAlignmentGpu;
struct int4;
struct float3;
struct double3;

#ifdef USE_DOUBLE_PRECISION
using PrecisionType = double;
using PrecisionType3 = double3;
#else
using PrecisionType = float;
using PrecisionType3 = float3;
#endif

struct VolumeMetaData
{
    int xShift = 0;
    int yShift = 0;
    int zShift = 0;

    int xDim = 0;
    int yDim = 0;
    int zDim = 0;
};

struct Volumes
{
    PrecisionType* I = nullptr;
    PrecisionType* R = nullptr;
    unsigned count = 0;
    unsigned volumeSize = 0;
};

struct IROimages
{
    PrecisionType* VI;
    PrecisionType* VR;
    PrecisionType* VO;
};

struct DeformImages
{
    PrecisionType* Gx;
    PrecisionType* Gy;
    PrecisionType* Gz;
};

struct KernelOutputs
{
    PrecisionType sumVD = 0.0;
    PrecisionType modg = 0.0;
    PrecisionType count = 0.0;
};

class AngularSphAlignment
{
public:
    void init();
    void prepareVolumeData();
    void waitToFinishPreparations();
    void cleanupPreparations();

    void setupConstantParameters();
    void setupChangingParameters();

    void runKernel();

    void transferResults();
    KernelOutputs getOutputs();

    AngularSphAlignment(ProgAngularSphAlignmentGpu* prog);
    ~AngularSphAlignment();

private:
    ProgAngularSphAlignmentGpu* program = nullptr;

    GpuReduction<PrecisionType> reduceCount;
    GpuReduction<PrecisionType> reduceModg;
    GpuReduction<PrecisionType> reduceSumVD;
    PrecisionType* reductionArray = nullptr;

    size_t kernelOutputSize;

    // Variables transfered to the GPU memory

    PrecisionType Rmax2;

    PrecisionType iRmax;

    VolumeMetaData volumeMetaData;

    double* dPrepVolume = nullptr;
    PrecisionType* dVolData = nullptr;

    int steps;

    std::vector<PrecisionType3> clnmPrepVec;

    int* dVolMask = nullptr;

    PrecisionType* dProjectionPlane = nullptr;

    KernelOutputs* outputs = nullptr;

    // helper methods for simplifying and transfering data to gpu

    void setupRotation();
    void setupVolumeMask();
    void setupProjectionPlane();
    void setupOutputArray();
    void setupOutputs();
    void setupZSHparams();
    void setupClnm();
    void setupGpuBlocks();

    template<bool PADDING = false>
    void prepareVolume(const double* mdaData, double* prepVol, PrecisionType* volume);
    void setupVolumeMetaData(const Image<double>& inputImage);

    void transferProjectionPlane();
};
