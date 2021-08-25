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

#ifndef VOLUME_DEFORM_SPH_H
#define VOLUME_DEFORM_SPH_H
// Xmipp includes
#include "core/xmipp_image.h"
#include "core/multidim_array.h"
#include "reconstruction_cuda/cuda_reduction.h"
// Standard includes
#include <vector>

// Forward declarations
class ProgVolumeDeformSphGpu;
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

    int padding = 0;
};

template<typename T>
struct Volumes
{
    T* I = nullptr;
    T* R = nullptr;
    unsigned count = 0;
    unsigned volumeSize = 0;
    unsigned volumePaddedSize = 0;
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
    PrecisionType diff2 = 0.0;
    PrecisionType sumVD = 0.0;
    PrecisionType modg = 0.0;
};

class VolumeDeformSph
{
public:
    void initVolumes();
    void prepareVI();
    void prepareInputVolume(const MultidimArray<double>& vol);
    void prepareReferenceVolume(const MultidimArray<double>& vol);
    void waitToFinishPreparations();
    void cleanupPreparations();

    void setupConstantParameters();
    void setupChangingParameters();

    void pretuneKernel();
    void runKernel();
    void transferResults();

    KernelOutputs getOutputs();

    VolumeDeformSph(ProgVolumeDeformSphGpu* prog);
    ~VolumeDeformSph();

private:
    ProgVolumeDeformSphGpu* program = nullptr;

    GpuReduction<PrecisionType> reduceDiff;
    GpuReduction<PrecisionType> reduceModg;
    GpuReduction<PrecisionType> reduceSumVD;
    PrecisionType* reductionArray = nullptr;


    size_t kernelOutputSize;

    // Variables transfered to the GPU memory

    PrecisionType Rmax2;

    PrecisionType iRmax;

    int steps;

    bool applyTransformation;

    bool saveDeformation;

    IROimages images;
    double* dTmpVI;

    DeformImages deformImages;

    VolumeMetaData volMetaData;

    Volumes<PrecisionType> volumes;
    Volumes<double> prepVolumes;
    int posR;
    int posI;

    KernelOutputs* outputs;

    // helper methods for simplifying and transfering data to gpu

    void setupImage(const VolumeMetaData& inputImage, PrecisionType** outputImageData);
    void setupVolumeMetaData(const Image<double>& inputImage);

    void setupVolumes();
    template<bool PADDING = false>
    void prepareVolume(const double* mdaData, double* prepVol, PrecisionType* volume);

    void setupZSHparams();

    void setupClnm();
    void transferImageData(Image<double>& outputImage, PrecisionType* inputData);
    void setupOutputArray();
    void setupOutputs();

    void setupGpuBlocks();
};

#endif// VOLUME_DEFORM_SPH_H
