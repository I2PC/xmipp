#pragma once

// Xmipp includes
#include "core/xmipp_image.h"
#include "core/multidim_array.h"
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
    PrecisionType diff2 = 0.0;
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
        MultidimArray<double> mV,
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

    int steps;

    PrecisionType3* dClnm;
    std::vector<PrecisionType3> clnmVec;

    bool applyTransformation;

    bool saveDeformation;

    // Inside pointers point to the GPU memory

    IROimages images;

    DeformImages deformImages;

    int4* dZshParams;
    std::vector<int4> zshparamsVec;

    ImageMetaData imageMetaData;

    Volumes volumes;

    KernelOutputs outputs;

    // helper methods for simplifying and transfering data to gpu

    void setupImage(Image<double>& inputImage, PrecisionType** outputImageData);
    void setupImage(const ImageMetaData& inputImage, PrecisionType** outputImageData);
    void setupImageMetaData(const Image<double>& inputImage);

    void setupVolumes();

    void setupZSHparams();

    void setupClnm();
    void transferImageData(Image<double>& outputImage, PrecisionType* inputData);
};

} // namespace AngularAlignmentGpu
