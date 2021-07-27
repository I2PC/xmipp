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

struct ImageMetaData
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

    // Kernel stuff
    size_t constantSharedMemSize;
    size_t changingSharedMemSize;

    //FIXME better naming, it is not really grid size, but size of output arrays
    size_t totalGridSize;

    // Variables transfered to the GPU memory

    PrecisionType Rmax2;

    PrecisionType iRmax;

    int steps;

    PrecisionType3* dClnm = nullptr;
    std::vector<PrecisionType3> clnmVec;
    PrecisionType3* mClnm = nullptr;

    bool applyTransformation;

    bool saveDeformation;

    // Inside pointers point to the GPU memory

    IROimages images;

    DeformImages deformImages;

    int4* dZshParams;
    std::vector<int4> zshparamsVec;

    ImageMetaData imageMetaData;

    Volumes<PrecisionType> volumes;
    Volumes<double> prepVolumes;
    int posR;
    void* streamR;
    int posI;
    void* streamI;

    KernelOutputs* outputs;

    // helper methods for simplifying and transfering data to gpu

    void setupImage(Image<double>& inputImage, PrecisionType** outputImageData);
    void setupImage(const ImageMetaData& inputImage, PrecisionType** outputImageData);
    void setupImageMetaData(const Image<double>& inputImage);

    void setupVolumes();

    void setupZSHparams();

    void setupClnm();
    void fillClnm();
    void transferImageData(Image<double>& outputImage, PrecisionType* inputData);
    void setupOutputArray();
    void setupOutputs();

    void setupGpuBlocks();
};

#endif// VOLUME_DEFORM_SPH_H
