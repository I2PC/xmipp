#ifndef VOLUME_DEFORM_SPH_H
#define VOLUME_DEFORM_SPH_H
// Xmipp includes
#include "api/dimension_vector.h"
#include "api/parameter_pair.h"
#include "core/xmipp_image.h"
#include "core/multidim_array.h"
// Standard includes
#include <vector>
// KTT includes
#include "ktt_types.h"
#include "tuner_api.h"

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

struct ZSHparams 
{
    int* vL1 = nullptr;
    int* vN = nullptr;
    int* vL2 = nullptr;
    int* vM = nullptr;
    unsigned size = 0;
};

struct Volumes 
{
    ImageData* I = nullptr;
    ImageData* R = nullptr;
    unsigned size = 0;
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

    // KTT tuner and kernel
    ktt::Tuner tuner;
    ktt::KernelId kernelId;
    std::vector<ktt::ParameterPair> bestKernelConfig;
    bool tuneKernel = true;

    // Kernel dimensions
    ktt::DimensionVector kttBlock;
    ktt::DimensionVector kttGrid;
    int tunedGridSize = 1;
    // Kernel dimensions tuning
    const std::string BLOCK_X_DIM = "BLOCK_X_DIM";
    const std::string BLOCK_Y_DIM = "BLOCK_Y_DIM";
    const std::string BLOCK_Z_DIM = "BLOCK_Z_DIM";

    // Kernel path
    std::string pathToXmipp = "/home/david/thesis/xmipp-bundle/";//default
    std::string pathToKernel = "src/xmipp/libraries/reconstruction_cuda/cuda_volume_deform_sph.cu";

    // Variables transfered to the GPU memory

    ktt::ArgumentId Rmax2Id;
    PrecisionType Rmax2;

    ktt::ArgumentId iRmaxId;
    PrecisionType iRmax;

    ktt::ArgumentId stepsId;

    ktt::ArgumentId clnmId;
    std::vector<PrecisionType3> clnmVec;

    ktt::ArgumentId clnmSCATTEREDId;
    std::vector<PrecisionType> clnmVecSCATTERED;

    ktt::ArgumentId applyTransformationId;
    bool applyTransformation;

    ktt::ArgumentId saveDeformationId;
    bool saveDeformation;

    // Inside pointers point to the GPU memory

    ktt::ArgumentId imagesId;
    IROimages images;

    ktt::ArgumentId deformImagesId;
    DeformImages deformImages;

    ktt::ArgumentId zshparamsId;
    std::vector<int4> zshparamsVec;

    ktt::ArgumentId zshparamsSCATTEREDId;
    ZSHparams zshparamsSCATTERED;

    ktt::ArgumentId volumesId;
    Volumes volumes;
    // because of the stupid design... :(
    std::vector<ImageData> justForFreeI;
    std::vector<ImageData> justForFreeR;

    KernelOutputs outputs;

    // helper methods for simplifying and transfering data to gpu

    void reduceResults();

    void setupImage(Image<double>& inputImage, ImageData& outputImageData);
    void setupImage(ImageData& inputImage, ImageData& outputImageData, bool copyData = false);

    void freeImage(ImageData &im);
    void freeZSHSCATTERED();

    void simplifyVec(std::vector<Image<double>>& vec, std::vector<ImageData>& res);

    void setupVolumes();

    void setupZSHparams();
    void setupZSHparamsSCATTERED();

    void setupClnm();
    void setupClnmSCATTERED();
};

#endif// VOLUME_DEFORM_SPH_H
