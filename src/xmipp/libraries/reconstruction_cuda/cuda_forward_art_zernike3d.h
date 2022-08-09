#ifndef CUDA_FORWARD_ART_ZERNIKE3D_H
#define CUDA_FORWARD_ART_ZERNIKE3D_H

// Xmipp includes
#include <core/xmipp_image.h>
#include <core/multidim_array.h>
#include <core/xmipp_image.h>
#include <core/matrix1d.h>
#include <core/matrix2d.h>
#include <core/multidim_array.h>
// Standard includes
#include <vector>
#include <memory>
#include <atomic>

struct float3;
struct double3;

template<typename T>
struct MultidimArrayCuda {
    size_t xdim;
    size_t ydim;
    size_t yxdim;
    int xinit;
    int yinit;
    int zinit;
    T* data;
};

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
        int loopStep;
        size_t Xdim;
    };

    struct AngleParameters {
        PrecisionType rot, tilt, psi;
    };

    struct DynamicParameters {
        const std::vector<PrecisionType> &clnm;
        std::vector<Image<PrecisionType>> &P;
        std::vector<Image<PrecisionType>> &W;
        const Image<PrecisionType> &Idiff;
        struct AngleParameters angles;
    };

public:

    template<bool usesZernike>
    void runForwardKernel(struct DynamicParameters &parameters);

    template<bool usesZernike>
    void runBackwardKernel(struct DynamicParameters &parameters);

    explicit CUDAForwardArtZernike3D(const ConstantParameters parameters) noexcept;
    ~CUDAForwardArtZernike3D();

private:

    const MultidimArrayCuda<PrecisionType> V;

    const MultidimArrayCuda<int> VRecMask, sphMask;

    const int RmaxDef;

    const int loopStep;

    const int lastX, lastY, lastZ;

    const Matrix1D<int> vL1, vN, vL2, vM;

    const std::vector<PrecisionType> sigma;

    // Atomic mutex
    std::vector<std::unique_ptr<std::atomic<PrecisionType*>>> p_busy_elem;
    std::vector<std::unique_ptr<std::atomic<PrecisionType*>>> w_busy_elem;

private:

    /// Move data from MultidimArray to struct usable by CUDA kernel
    template<typename T>
    MultidimArrayCuda<T> initializeMultidimArray(const MultidimArray<T> &multidimArray) const;

    /// Function inspired by std::find with support for CUDA allowed data types
    size_t findCuda(const PrecisionType *begin, size_t size, PrecisionType value) const;

    void splattingAtPos(PrecisionType pos_x, PrecisionType pos_y, PrecisionType weight,
                        MultidimArrayCuda<PrecisionType> &mP, MultidimArrayCuda<PrecisionType> &mW,
                        std::unique_ptr<std::atomic<PrecisionType *>> *p_busy_elem_cuda,
                        std::unique_ptr<std::atomic<PrecisionType *>> *w_busy_elem_cuda) const;

    Matrix2D<PrecisionType> createRotationMatrix(struct AngleParameters angles) const;

    PrecisionType interpolatedElement2DCuda(PrecisionType x,
                                            PrecisionType y,
                                            MultidimArrayCuda<PrecisionType> &diffImage) const;
};

#endif// CUDA_FORWARD_ART_ZERNIKE3D_H
