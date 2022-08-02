#ifndef CUDA_FORWARD_ART_ZERNIKE3D_H
#define CUDA_FORWARD_ART_ZERNIKE3D_H

// Xmipp includes
#include <core/xmipp_image.h>
#include <core/multidim_array.h>
#include <core/xmipp_image.h>
#include <core/matrix1d.h>
#include <core/multidim_array.h>
// Standard includes
#include <vector>

struct float3;
struct double3;

template<typename PrecisionType = float>
class CUDAForwardArtZernike3D
{
    static_assert(std::is_floating_point<PrecisionType>::value, "Floating point type is required.");

    using PrecisionType3 = std::conditional<std::is_same<PrecisionType, float>::value, float3, double3>;

public:
    /// Constant parameters for the computation
    struct ConstantParameters {
        Image<PrecisionType> &Vrefined;
        MultidimArray<int> &VRecMask, &sphMask;
        Matrix1D<int> &vL1, &vN, &vL2, &vM;
        std::vector<PrecisionType> &sigma;
        int RmaxDef;
        PrecisionType rot, tilt, psi;
    };

public:

    template<bool usesZernike>
    void runForwardKernel(const std::vector<PrecisionType> &clnm,
                          std::vector<Image<PrecisionType>> &P,
                          std::vector<Image<PrecisionType>> &W);

    template<bool usesZernike>
    void runBackwardKernel(const std::vector<PrecisionType> &clnm,
                           const Image<PrecisionType> &Idiff);

    CUDAForwardArtZernike3D(const ConstantParameters parameters);
    ~CUDAForwardArtZernike3D();
};

#endif// CUDA_FORWARD_ART_ZERNIKE3D_H
