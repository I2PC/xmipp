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

struct ImageMetaData
{
    int xShift;
    int yShift;
    int zShift;

    int xDim;
    int yDim;
    int zDim;
};

struct IROimages 
{
    PrecisionType* I = nullptr;
    PrecisionType* R = nullptr;
    PrecisionType* O = nullptr;
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
    PrecisionType* I = nullptr;
    PrecisionType* R = nullptr;
    unsigned count = 0;
    unsigned volumeSize = 0;
};

struct OutputImages 
{
    PrecisionType* Gx = nullptr;
    PrecisionType* Gy = nullptr;
    PrecisionType* Gz = nullptr;
};

struct KernelOutputs 
{
    PrecisionType diff2 = 0.0;
    PrecisionType sumVD = 0.0;
    PrecisionType modg = 0.0;
};

/**
 * Class that takes care of the data preparation, data transfer to the GPU,
 * calling kernel and returning kernel outputs.
 */
class VolumeDeformSph
{
public:

    /**
     * Method associates VolumeDeformSph class with the ProgVolumeDeformSphGpu.
     *
     * This method has to be called before any other method of the VolumeDeformSph class.
     *
     * \param prog Pointer to the program class.
     */
    void associateWith(ProgVolumeDeformSphGpu* prog);

    /**
     * Method has to be called after the class has been associated with a program by
     * calling associateWith method.
     *
     * Calling this method prepares and transfers data which need to be
     * transferred to the GPU only once per runtime.
     *
     * Constant parameters are: Rmax2, iRmax, volumesR, volumesI, ...
     */
    void setupConstantParameters();

    /**
     * Method has to be called after the class has been associated with a program by
     * calling associateWith method.
     *
     * Calling this method prepares and transfers data which need to be updated
     * before each kernel call.
     *
     * Changing parameters are: clnm, steps
     */
    void setupChangingParameters();

    /**
     * All the data need to be ready before calling this method.
     *
     * Calling this method invokes a GPU kernel.
     */
    void runKernel();

    /**
     * Transfers results from GPU to the CPU.
     */
    void transferResults();

    /**
     * Returns results computed by the GPU kernel.
     */
    KernelOutputs getOutputs();

    /**
     * Constructs an object of VolumeDeformSph class
     */
    VolumeDeformSph();

    /**
     * Destroys an object of VolumeDeformSph class
     */
    ~VolumeDeformSph();

private:
    ProgVolumeDeformSphGpu* program = nullptr;

    // KTT tuner and kernel
    ktt::Tuner tuner;
    ktt::KernelId kernelId;
    std::vector<ktt::ParameterPair> bestKernelConfig;
    bool tuneKernel = true;
    bool constantDataReady = false;
    bool changingDataReady = false;

    ktt::ArgumentId sharedMemId;

    // Kernel dimensions
    ktt::DimensionVector kttBlock;
    ktt::DimensionVector kttGrid;
    int tunedGridSize = 1;
    // Kernel dimensions tuning
    const std::string BLOCK_X_DIM = "BLOCK_X_DIM";
    const std::string BLOCK_Y_DIM = "BLOCK_Y_DIM";
    const std::string BLOCK_Z_DIM = "BLOCK_Z_DIM";

    // Kernel path
    std::string pathToXmipp;
    std::string pathToKernel = "src/xmipp/libraries/reconstruction_cuda/cuda_volume_deform_sph.cu";

    // Variables transfered to the GPU memory

    ktt::ArgumentId Rmax2Id;
    PrecisionType Rmax2;

    ktt::ArgumentId iRmaxId;
    PrecisionType iRmax;

    ktt::ArgumentId imagesId;
    IROimages images;

    ktt::ArgumentId stepsId;

    ktt::ArgumentId clnmId;
    std::vector<PrecisionType3> clnmVec;

    ktt::ArgumentId applyTransformationId;
    bool applyTransformation;

    ktt::ArgumentId saveDeformationId;
    bool saveDeformation;

    ktt::ArgumentId outputImagesId;
    OutputImages outputImages;

    ktt::ArgumentId zshparamsId;
    std::vector<int4> zshparamsVec;

    ktt::ArgumentId imageMetaDataId;
    ImageMetaData imageMetaData;

    ktt::ArgumentId volumesId;
    Volumes volumes;
    std::vector<PrecisionType*> justForFreeI;
    std::vector<PrecisionType*> justForFreeR;

    KernelOutputs outputs;

    // helper methods for simplifying and transfering data to gpu

    void transferImageData(Image<double>& outputImage, PrecisionType* inputData);

    void pretuneKernel();

    void reduceResults();

    void setupImageMetaData(Image<double>& inputImage);
    void setupImage(Image<double>& inputImage, PrecisionType** outputImageData);
    void setupImage(PrecisionType** imageData);

    void setupVolumes();
    void setupOutputImages();

    void setupZSHparams();

    void setupClnm();

    void setupConstantKtt();
    void setupChangingKtt();

    void setupKttKernel();
    void setupKttBlockSize();
    void setupKttDefines();
    void setupKttTuningParameters();
    void setupKttSharedMemory();
    void setupKttConstantKernelArguments();
};

#endif// VOLUME_DEFORM_SPH_H
