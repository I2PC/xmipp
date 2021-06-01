#pragma once

// Xmipp includes
#include "core/xmipp_image.h"
#include "core/multidim_array.h"
// Standard includes
#include <vector>

//#define USE_DOUBLE_PRECISION 1

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
    void associateWith(ProgAngularSphAlignmentGpu* prog);
    void setupConstantParameters();
    void setupChangingParameters();

    void pretuneKernel();
    void runKernel();
    void runKernelTest(
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
        MultidimArray<double>& mP);
    void transferResults();

    KernelOutputs getOutputs();

    AngularSphAlignment();
    ~AngularSphAlignment();

private:
    ProgAngularSphAlignmentGpu* program = nullptr;

    // Kernel stuff
    size_t constantSharedMemSize;
    size_t changingSharedMemSize;

    // Kernel dimensions
    //dim3 block;
    //dim3 grid;

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

    KernelOutputs outputs;

    // helper methods for simplifying and transfering data to gpu

    void setupVolumeData();
    void setupRotation();
    void setupVolumeMask();
    void setupProjectionPlane();

    void setupVolumeDataCpu();
    void setupRotationCpu();
    void setupVolumeMaskCpu();
    void setupProjectionPlaneCpu();

    void setupImage(Image<double>& inputImage, PrecisionType** outputImageData);
    void setupImage(const ImageMetaData& inputImage, PrecisionType** outputImageData);
    void setupImageMetaData(const Image<double>& inputImage);

    void setupZSHparams();
    void setupClnm();

    void setupZSHparamsCpu();
    void setupClnmCpu();

    void transferProjectionPlane();
    void transferProjectionPlaneCpu();

    void transferImageData(Image<double>& outputImage, PrecisionType* inputData);
};

} // namespace AngularAlignmentGpu
