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

namespace AngularAlignmentGpu {

#ifdef USE_DOUBLE_PRECISION
using PrecisionType = double;
using PrecisionType3 = double3;
#else
using PrecisionType = float;
using PrecisionType3 = float3;
#endif

struct ImageMetaData
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
    void setupConstantParameters();
    void setupChangingParameters();

    void pretuneKernel();
    void runKernel();

    void transferResults();
    KernelOutputs getOutputs();

    AngularSphAlignment(ProgAngularSphAlignmentGpu* prog);
    ~AngularSphAlignment();

private:
    ProgAngularSphAlignmentGpu* program = nullptr;

    GpuReduction<PrecisionType> reduceDiff;
    GpuReduction<PrecisionType> reduceModg;
    GpuReduction<PrecisionType> reduceSumVD;
    PrecisionType* reductionArray = nullptr;

    // Kernel stuff
    size_t constantSharedMemSize;
    size_t changingSharedMemSize;

    //FIXME better naming, it is not really grid size, but size of output arrays, kernelOutputSize??
    size_t totalGridSize;

    // Variables transfered to the GPU memory

    PrecisionType Rmax2;

    PrecisionType iRmax;

    ImageMetaData imageMetaData;

    PrecisionType* dVolData = nullptr;
    std::vector<PrecisionType> volDataVec;

    PrecisionType* dRotation = nullptr;
    std::vector<PrecisionType> rotationVec;

    int steps;

    int4* dZshParams = nullptr;
    std::vector<int4> zshparamsVec;

    PrecisionType3* dClnm = nullptr;
    std::vector<PrecisionType3> clnmVec;

    int* dVolMask = nullptr;

    PrecisionType* dProjectionPlane = nullptr;
    std::vector<PrecisionType> projectionPlaneVec;

    KernelOutputs* outputs;

    // helper methods for simplifying and transfering data to gpu

    void setupVolumeData();
    void setupRotation();
    void setupVolumeMask();
    void setupProjectionPlane();
    void setupOutputArray();
    void setupOutputs();
    void setupZSHparams();
    void setupClnm();
    void setupGpuBlocks();

    void setupImage(Image<double>& inputImage, PrecisionType** outputImageData);
    void setupImage(const ImageMetaData& inputImage, PrecisionType** outputImageData);
    void setupImageMetaData(const Image<double>& inputImage);

    void transferProjectionPlane();

    void transferImageData(Image<double>& outputImage, PrecisionType* inputData);
};

} // namespace AngularAlignmentGpu
