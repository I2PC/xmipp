#ifndef VOLUME_DEFORM_SPH_H
#define VOLUME_DEFORM_SPH_H

#include <vector>
#include "core/xmipp_image.h"
#include "core/multidim_array.h"

#ifdef COMP_DOUBLE
using ComputationDataType = double;
#else
using ComputationDataType = float;
#endif

// Forward declarations
class ProgVolumeDeformSphGpu;

template<typename T>
struct ImageData
{
    int xShift = 0;
    int yShift = 0;
    int zShift = 0;

    int xDim = 0;
    int yDim = 0;
    int zDim = 0;

    T* data = nullptr;
};

struct ZSHparams 
{
    int* vL1 = nullptr;
    int* vN = nullptr;
    int* vL2 = nullptr;
    int* vM = nullptr;
    unsigned size = 0;
};

template<typename T>
struct Volumes 
{
    ImageData<T>* I = nullptr;
    ImageData<T>* R = nullptr;
    unsigned size = 0;
};

template<typename T>
struct IROimages 
{
    ImageData<T> VI;
    ImageData<T> VR;
    ImageData<T> VO;
};

template<typename T>
struct DeformImages 
{
    ImageData<T> Gx;
    ImageData<T> Gy;
    ImageData<T> Gz;
};

template<typename T>
struct KernelOutputs 
{
    T diff2 = 0.0;
    T sumVD = 0.0;
    T modg = 0.0;
    T Ncount = 0.0;
};

template<typename T>
class VolumeDeformSph
{
public:
    void associateWith(ProgVolumeDeformSphGpu* prog);
    void setupConstantParameters();
    void setupChangingParameters();

    void runKernel();
    void transferResults();

    KernelOutputs<T> getOutputs();
    void transferImageData(Image<double>& outputImage, ImageData<T>& inputData);

    ~VolumeDeformSph();

private:
    ProgVolumeDeformSphGpu* program = nullptr;

    // Variables transfered to the GPU memory
    T Rmax2;

    T iRmax;

    T* steps = nullptr;

    T* clnm = nullptr;

    bool applyTransformation;

    bool saveDeformation;

    // Inside pointers point to the GPU memory

    IROimages<T> images;

    DeformImages<T> deformImages;

    ZSHparams zshparams;

    Volumes<T> volumes;
    // because of the stupid design... :(
    std::vector<ImageData<T>> justForFreeI;
    std::vector<ImageData<T>> justForFreeR;

    KernelOutputs<T> *outputs = nullptr;
    KernelOutputs<T> exOuts;

    // helper methods for simplifying and transfering data to gpu

    void setupImage(Image<double>& inputImage, ImageData<T>& outputImageData);
    void setupImage(ImageData<T>& inputImage, ImageData<T>& outputImageData, bool copyData = false);

    void freeImage(ImageData<T> &im);

    void simplifyVec(std::vector<Image<double>>& vec, std::vector<ImageData<T>>& res);

    void setupVolumes();

    void setupZSHparams();
};

#endif// VOLUME_DEFORM_SPH_H
