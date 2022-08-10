// Xmipp includes
#include "cuda_forward_art_zernike3d.h"
#include <core/geometry.h>
#include "data/numerical_tools.h"
#include <cassert>

// Macros
#define SQRT sqrtf

#define IS_OUTSIDE2D(ImD,i,j) \
    ((j) < STARTINGX((ImD)) || (j) > FINISHINGX((ImD)) || \
     (i) < STARTINGY((ImD)) || (i) > FINISHINGY((ImD)))

template<typename PrecisionType>
CUDAForwardArtZernike3D<PrecisionType>::CUDAForwardArtZernike3D(
        const CUDAForwardArtZernike3D<PrecisionType>::ConstantParameters parameters) noexcept
    : V(initializeMultidimArray(parameters.Vrefined())),
      VRecMask(initializeMultidimArray(parameters.VRecMask)),
      sphMask(initializeMultidimArray(parameters.sphMask)),
      sigma(parameters.sigma),
      RmaxDef(parameters.RmaxDef),
      lastZ(FINISHINGZ(parameters.Vrefined())),
      lastY(FINISHINGY(parameters.Vrefined())),
      lastX(FINISHINGX(parameters.Vrefined())),
      loopStep(parameters.loopStep),
      cudaVL1(parameters.vL1.vdata),
      cudaVL2(parameters.vL2.vdata),
      cudaVN(parameters.vN.vdata),
      cudaVM(parameters.vM.vdata) {
   auto Xdim = parameters.Xdim;
   p_busy_elem.resize(Xdim*Xdim);
   for (auto& p : p_busy_elem) {
       p = std::unique_ptr<std::atomic<PrecisionType*>>(new std::atomic<PrecisionType*>(nullptr));
   }

   w_busy_elem.resize(Xdim*Xdim);
   for (auto& p : w_busy_elem) {
       p = std::unique_ptr<std::atomic<PrecisionType*>>(new std::atomic<PrecisionType*>(nullptr));
   }
}

template<typename PrecisionType>
CUDAForwardArtZernike3D<PrecisionType>::~CUDAForwardArtZernike3D() {

}

template<typename PrecisionType>
template<bool usesZernike>
struct CUDAForwardArtZernike3D<PrecisionType>::CommonKernelParameters CUDAForwardArtZernike3D<PrecisionType>::setCommonArgumentsKernel(struct DynamicParameters &parameters) {
    auto clnm = parameters.clnm;
    auto angles = parameters.angles;

    // We can't set idxY0 to 0 because the compiler
    // would give irrelevant warnings.
    assert(usesZernike || clnm.size() == 0);
    const size_t idxY0 = clnm.size() / 3;
    const size_t idxZ0 = usesZernike ? (2 * idxY0) : 0;
    const PrecisionType RmaxF = usesZernike ? RmaxDef : 0;
    const PrecisionType iRmaxF = usesZernike ? (1.0f / RmaxF) : 0;

    // Rotation Matrix (has to pass the whole Matrix2D so it is not automatically deallocated)
    const Matrix2D<PrecisionType> R = createRotationMatrix(angles);

    auto cudaClnm = clnm.data();

    CommonKernelParameters output = {
        .idxY0 = idxY0,
        .idxZ0 = idxZ0,
        .iRmaxF = iRmaxF,
        .cudaClnm = cudaClnm,
        .R = R
    };

    return output;
}

template<typename PrecisionType>
MultidimArrayCuda<PrecisionType> *CUDAForwardArtZernike3D<PrecisionType>::setVectorMultidimArrayCuda(std::vector<Image<PrecisionType>> &image, std::vector<MultidimArrayCuda<PrecisionType>> &output) {
    for (int m = 0; m < image.size(); m++)
    {
        output.push_back(initializeMultidimArray(image[m]()));
    }   
    return output.data();
}

template<typename PrecisionType>
template<bool usesZernike>
void CUDAForwardArtZernike3D<PrecisionType>::runForwardKernel(struct DynamicParameters &parameters) {
    // Unique parameters 
    std::vector<MultidimArrayCuda<PrecisionType>> outputP;
    std::vector<MultidimArrayCuda<PrecisionType>> outputW;
    auto cudaP = setVectorMultidimArrayCuda(parameters.P, outputP);
    auto cudaW = setVectorMultidimArrayCuda(parameters.W, outputW);
    auto p_busy_elem_cuda = p_busy_elem.data();
    auto w_busy_elem_cuda = w_busy_elem.data();
    auto sigma_size = sigma.size();
    const auto cudaSigma = sigma.data();
    const int step = loopStep;

    // Common parameters
    auto commonParameters = setCommonArgumentsKernel<usesZernike>(parameters);
    auto idxY0 = commonParameters.idxY0;
    auto idxZ0 = commonParameters.idxZ0;
    auto iRmaxF = commonParameters.iRmaxF;
    auto cudaR = commonParameters.R.mdata;
    auto cudaClnm = commonParameters.cudaClnm;

    for (int k = STARTINGZ(V); k <= lastZ; k += step)
    {
        for (int i = STARTINGY(V); i <= lastY; i += step)
        {
            for (int j = STARTINGX(V); j <= lastX; j += step)
            {
                // Future CUDA code
                PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;
                if (A3D_ELEM(VRecMask, k, i, j) != 0)
                {
                    int img_idx = 0;
                    if (sigma_size > 1)
                    {
                        PrecisionType sigma_mask = A3D_ELEM(VRecMask, k, i, j);
                        img_idx = findCuda(cudaSigma, sigma_size, sigma_mask);
                    }
                    auto &mP = cudaP[img_idx];
                    auto &mW = cudaW[img_idx];
                    if (usesZernike)
                    {
                        auto k2 = k * k;
                        auto kr = k * iRmaxF;
                        auto k2i2 = k2 + i * i;
                        auto ir = i * iRmaxF;
                        auto r2 = k2i2 + j * j;
                        auto jr = j * iRmaxF;
                        auto rr = SQRT(r2) * iRmaxF;
                        for (size_t idx = 0; idx < idxY0; idx++)
                        {
                            auto l1 = cudaVL1[idx];
                            auto n = cudaVN[idx];
                            auto l2 = cudaVL2[idx];
                            auto m = cudaVM[idx];
                            if (rr > 0 || l2 == 0)
                            {
                                PrecisionType zsph = ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
                                gx += cudaClnm[idx] * (zsph);
                                gy += cudaClnm[idx + idxY0] * (zsph);
                                gz += cudaClnm[idx + idxZ0] * (zsph);
                            }
                        }
                    }

                    auto r_x = j + gx;
                    auto r_y = i + gy;
                    auto r_z = k + gz;

                    auto pos_x = cudaR[0] * r_x + cudaR[1] * r_y + cudaR[2] * r_z;
                    auto pos_y = cudaR[3] * r_x + cudaR[4] * r_y + cudaR[5] * r_z;
                    PrecisionType voxel_mV = A3D_ELEM(V, k, i, j);
                    splattingAtPos(pos_x, pos_y, voxel_mV, mP, mW, p_busy_elem_cuda, w_busy_elem_cuda);
                }
                // End of future CUDA code
            }
        }
    }
}

template<typename PrecisionType>
template<bool usesZernike>
void CUDAForwardArtZernike3D<PrecisionType>::runBackwardKernel(struct DynamicParameters &parameters) {
    // Unique parameters
    auto &mId = parameters.Idiff();
    auto cudaMId = initializeMultidimArray(mId);
    const int step = 1;

    // Common parameters
    auto commonParameters = setCommonArgumentsKernel<usesZernike>(parameters);
    auto idxY0 = commonParameters.idxY0;
    auto idxZ0 = commonParameters.idxZ0;
    auto iRmaxF = commonParameters.iRmaxF;
    auto cudaR = commonParameters.R.mdata;
    auto cudaClnm = commonParameters.cudaClnm;

    for (int k = STARTINGZ(V); k <= lastZ; k += step)
    {
        for (int i = STARTINGY(V); i <= lastY; i += step)
        {
            for (int j = STARTINGX(V); j <= lastX; j += step)
            {
                PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;
                if (A3D_ELEM(sphMask, k, i, j) != 0)
                {
                    if (usesZernike)
                    {
                        auto k2 = k * k;
                        auto kr = k * iRmaxF;
                        auto k2i2 = k2 + i * i;
                        auto ir = i * iRmaxF;
                        auto r2 = k2i2 + j * j;
                        auto jr = j * iRmaxF;
                        auto rr = SQRT(r2) * iRmaxF;
                        for (size_t idx = 0; idx < idxY0; idx++)
                        {
                            auto l1 = cudaVL1[idx];
                            auto n = cudaVN[idx];
                            auto l2 = cudaVL2[idx];
                            auto m = cudaVM[idx];
                            if (rr > 0 || l2 == 0)
                            {
                                PrecisionType zsph = ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
                                gx += cudaClnm[idx] * (zsph);
                                gy += cudaClnm[idx + idxY0] * (zsph);
                                gz += cudaClnm[idx + idxZ0] * (zsph);
                            }
                        }
                    }

                    auto r_x = j + gx;
                    auto r_y = i + gy;
                    auto r_z = k + gz;

                    auto pos_x = cudaR[0] * r_x + cudaR[1] * r_y + cudaR[2] * r_z;
                    auto pos_y = cudaR[3] * r_x + cudaR[4] * r_y + cudaR[5] * r_z;
                    PrecisionType voxel = interpolatedElement2DCuda(pos_x, pos_y, cudaMId);
                    A3D_ELEM(V, k, i, j) += voxel;
                }
            }
        }
    }
}

template<typename PrecisionType>
template<typename T>
MultidimArrayCuda<T> CUDAForwardArtZernike3D<PrecisionType>::initializeMultidimArray(const MultidimArray<T> &multidimArray) const {
    struct MultidimArrayCuda<T> cudaArray = {
            .xdim = multidimArray.xdim,
            .ydim = multidimArray.ydim,
            .yxdim = multidimArray.yxdim,
            .xinit = multidimArray.xinit,
            .yinit = multidimArray.yinit,
            .zinit = multidimArray.zinit,
            .data = multidimArray.data
    };
    return cudaArray;
}

template<typename PrecisionType>
void CUDAForwardArtZernike3D<PrecisionType>::splattingAtPos(PrecisionType pos_x, PrecisionType pos_y, PrecisionType weight,
                                                MultidimArrayCuda<PrecisionType> &mP, MultidimArrayCuda<PrecisionType> &mW,
                                                std::unique_ptr<std::atomic<PrecisionType *>> *p_busy_elem_cuda,
                                                std::unique_ptr<std::atomic<PrecisionType *>> *w_busy_elem_cuda) const
{
    int i = round(pos_y);
    int j = round(pos_x);
    if(!IS_OUTSIDE2D(mP, i, j))
    {
        int idy = (i)-STARTINGY(mP);
        int idx = (j)-STARTINGX(mP);
        int idn = (idy) * (mP).xdim + (idx);
        // Not sure if std::unique_ptr and std::atomic can be used in CUDA code
        while ((*p_busy_elem_cuda[idn]) == &A2D_ELEM(mP, i, j));
        (*p_busy_elem_cuda[idn]).exchange(&A2D_ELEM(mP, i, j));
        (*w_busy_elem_cuda[idn]).exchange(&A2D_ELEM(mW, i, j));
        A2D_ELEM(mP, i, j) += weight;
        A2D_ELEM(mW, i, j) += 1.0;
        (*p_busy_elem_cuda[idn]).exchange(nullptr);
        (*w_busy_elem_cuda[idn]).exchange(nullptr);
    }
}

template<typename PrecisionType>
size_t CUDAForwardArtZernike3D<PrecisionType>::findCuda(const PrecisionType *begin, size_t size, PrecisionType value) const
{
    if (size <= 0)
    {
        return 0;
    }
    for (size_t i = 0; i < size; i++)
    {
        if (begin[i] == value)
        {
            return i;
        }
    }
    return size - 1;
}

template<typename PrecisionType>
Matrix2D<PrecisionType> CUDAForwardArtZernike3D<PrecisionType>::createRotationMatrix(struct AngleParameters angles) const {
    auto rot = angles.rot;
    auto tilt = angles.tilt;
    auto psi = angles.psi;
    constexpr size_t matrixSize = 3;
    auto tmp = Matrix2D<PrecisionType>();
    tmp.initIdentity(matrixSize);
    Euler_angles2matrix(rot, tilt, psi, tmp, false);
    return tmp;
}

template<typename PrecisionType>
PrecisionType CUDAForwardArtZernike3D<PrecisionType>::interpolatedElement2DCuda(PrecisionType x,
                                                                                PrecisionType y,
                                                                                MultidimArrayCuda<PrecisionType> &diffImage) const
{
    int x0 = floor(x);
    PrecisionType fx = x - x0;
    int x1 = x0 + 1;
    int y0 = floor(y);
    PrecisionType fy = y - y0;
    int y1 = y0 + 1;

    int i0=STARTINGY(diffImage);
    int j0=STARTINGX(diffImage);
    int iF=FINISHINGY(diffImage);
    int jF=FINISHINGX(diffImage);

#define ASSIGNVAL2DCUDA(d,i,j) \
     if ((j) < j0 || (j) > jF || (i) < i0 || (i) > iF) \
      d=(PrecisionType) 0;\
        else \
         d=A2D_ELEM(diffImage, i, j);

    PrecisionType d00, d10, d11, d01;
    ASSIGNVAL2DCUDA(d00,y0,x0);
    ASSIGNVAL2DCUDA(d01,y0,x1);
    ASSIGNVAL2DCUDA(d10,y1,x0);
    ASSIGNVAL2DCUDA(d11,y1,x1);

    PrecisionType d0 = LIN_INTERP(fx, d00, d01);
    PrecisionType d1 = LIN_INTERP(fx, d10, d11);
    return LIN_INTERP(fy, d0, d1);
}
// Cuda memory helper function
namespace {

    template<typename T>
    cudaError cudaMallocAndCopy(T **target, const T *source, size_t numberOfElements, size_t memSize = 0) {
        size_t elemSize = numberOfElements * sizeof(T);
        memSize = memSize == 0 ? elemSize : memSize * sizeof(T);

        cudaError err = cudaSuccess;
        if ((err = cudaMalloc(target, memSize)) != cudaSuccess) {
            *target = NULL;
            return err;
        }

        if ((err = cudaMemcpy(*target, source, elemSize, cudaMemcpyHostToDevice)) != cudaSuccess) {
            cudaFree(*target);
            *target = NULL;
        }

        if (memSize > elemSize) {
            cudaMemset((*target) + numberOfElements, 0, memSize - elemSize);
        }

        return err;
    }

    void processCudaError() {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(err));
            exit(err);
        }
    }

// Copies data from CPU to the GPU and at the same time transforms from
// type 'U' to type 'T'. Works only for numeric types
    template<typename Target, typename Source>
    void transformData(Target **dest, Source *source, size_t n, bool mallocMem = true) {
        std::vector <Target> tmp(source, source + n);

        if (mallocMem) {
            if (cudaMalloc(dest, sizeof(Target) * n) != cudaSuccess) {
                processCudaError();
            }
        }

        if (cudaMemcpy(*dest, tmp.data(), sizeof(Target) * n, cudaMemcpyHostToDevice) != cudaSuccess) {
            processCudaError();
        }
    }

    template<typename T>
    void setupMultidimArray(MultidimArray<T>& inputArray, T** outputArrayData)
    {
        transformData(outputArrayData, inputArray.data, inputArray.xdim * inputArray.ydim * inputArray.zdim);
    }

    template<typename T>
    void setupVectorOfMultidimArray(std::vector<MultidimArrayCuda<T>>& inputVector, MultidimArrayCuda<T>** outputVectorData)
    {
        if (cudaMallocAndCopy(&outputVectorData, inputVector.data(), inputVector.size()) != cudaSuccess)
            processCudaError();
    }

    template<typename T>
    void setupMatrix1D(Matrix1D<T>& inputVector, T** outputVector)
    {
        transformData(outputVector, inputVector.vdata, inputVector.vdim);
    }

    template<typename T>
    void setupStdVector(std::vector<T>& inputVector, T** outputVector)
    {
        transformData(outputVector, inputVector.data(), inputVector.size());
    }

    template<typename T>
    void setupMatrix2D(Matrix2D<T>& inputMatrix, T** outputMatrixData)
    {
        transformData(outputMatrixData, inputMatrix.mdata, inputMatrix.mdim);
    }
}

// explicit template instantiation
template class CUDAForwardArtZernike3D<float>;
template class CUDAForwardArtZernike3D<double>;
template void CUDAForwardArtZernike3D<float>::runForwardKernel<true>(struct DynamicParameters&);
template void CUDAForwardArtZernike3D<float>::runForwardKernel<false>(struct DynamicParameters&);
template void CUDAForwardArtZernike3D<double>::runForwardKernel<true>(struct DynamicParameters&);
template void CUDAForwardArtZernike3D<double>::runForwardKernel<false>(struct DynamicParameters&);
template void CUDAForwardArtZernike3D<float>::runBackwardKernel<true>(struct DynamicParameters&);
template void CUDAForwardArtZernike3D<float>::runBackwardKernel<false>(struct DynamicParameters&);
template void CUDAForwardArtZernike3D<double>::runBackwardKernel<true>(struct DynamicParameters&);
template void CUDAForwardArtZernike3D<double>::runBackwardKernel<false>(struct DynamicParameters&);
