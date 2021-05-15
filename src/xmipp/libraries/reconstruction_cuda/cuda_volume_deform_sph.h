#ifndef VOLUME_DEFORM_SPH_H
#define VOLUME_DEFORM_SPH_H
// Xmipp includes
#include "core/xmipp_image.h"
#include "core/multidim_array.h"
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

struct ImageMetaData
{
    int xShift = 0;
    int yShift = 0;
    int zShift = 0;

    int xDim = 0;
    int yDim = 0;
    int zDim = 0;
};

struct ImageData
{
    int xShift = 0;
    int yShift = 0;
    int zShift = 0;

    int xDim = 0;
    int yDim = 0;
    int zDim = 0;

    PrecisionType* data = nullptr;
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
    ImageData VI;
    ImageData VR;
    ImageData VO;
};

struct DeformImages 
{
    ImageData Gx;
    ImageData Gy;
    ImageData Gz;
};

struct KernelOutputs 
{
    PrecisionType diff2 = 0.0;
    PrecisionType sumVD = 0.0;
    PrecisionType modg = 0.0;
    PrecisionType Ncount = 0.0;
};

class VolumeDeformSph
{
public:
    void associateWith(ProgVolumeDeformSphGpu* prog);
    void setupConstantParameters();
    void setupChangingParameters();

    void pretuneKernel();
    void runKernel();
    void transferResults();

    KernelOutputs getOutputs();
    void transferImageData(Image<double>& outputImage, ImageData& inputData);

    VolumeDeformSph();
    ~VolumeDeformSph();

private:
    ProgVolumeDeformSphGpu* program = nullptr;

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

    void setupImage(Image<double>& inputImage, ImageData& outputImageData);
    void setupImage(ImageData& inputImage, ImageData& outputImageData, bool copyData = false);
    void setupImageMetaData(const Image<double>& inputImage);

    void freeImage(ImageData &im);

    void simplifyVec(std::vector<Image<double>>& vec, std::vector<ImageData>& res);

    void setupVolumes();

    void setupZSHparams();

    void setupClnm();
};

#endif// VOLUME_DEFORM_SPH_H
