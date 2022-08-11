#ifndef CUDA_FORWARD_ART_ZERNIKE3D_CU
#define CUDA_FORWARD_ART_ZERNIKE3D_CU

#include "cuda_forward_art_zernike3d_defines.h"
#include "cuda_forward_art_zernike3d.h"

// Compilation settings

// Constants
#define _PI_ (3.1415926535897f)
// Functions
#define SQRT sqrtf
#define ATAN2 atan2f
#define COS cosf
#define SIN sinf
#define CUDA_FLOOR floorf

// Index to global memory
#define GET_IDX(ImD,k,i,j) \
    ((ImD).xDim * (ImD).yDim * (k) + (ImD).xDim * (i) + (j))

// Logical index = Physical index + shift
#define P2L_X_IDX(ImD,j) \
    ((j) + (ImD).xShift)

#define P2L_Y_IDX(ImD,i) \
    ((i) + (ImD).yShift)

#define P2L_Z_IDX(ImD,k) \
    ((k) + (ImD).zShift)

// Physical index = Logical index - shift
#define L2P_X_IDX(ImD,j) \
    ((j) - (ImD).xShift)

#define L2P_Y_IDX(ImD,i) \
    ((i) - (ImD).yShift)

#define L2P_Z_IDX(ImD,k) \
    ((k) - (ImD).zShift)

// Element access
#define ELEM_3D(ImD,meta,k,i,j) \
    ((ImD)[GET_IDX((meta), (k), (i), (j))])

#define ELEM_3D_SHIFTED(ImD,meta,k,i,j) \
    (ELEM_3D((ImD), (meta), (k) - (meta).zShift, (i) - (meta).yShift, (j) - (meta).xShift))

#define IS_OUTSIDE2D(ImD,i,j) \
    ((j) < STARTINGX((ImD)) || (j) > FINISHINGX((ImD)) || \
     (i) < STARTINGY((ImD)) || (i) > FINISHINGY((ImD)))

// Smart casting to selected precision (at compile time)
// ...just shorter static_cast
#define CST(num) (static_cast<PrecisionType>((num)))

#define FLOOR(x) (((x) == (int)(x)) ? (int)(x):(((x) > 0) ? (int)(x) : \
                  (int)((x) - 1)))
#define LIN_INTERP(a, l, h) ((l) + ((h) - (l)) * (a))

namespace device {

    template<typename PrecisionType>
    __forceinline__ __device__ PrecisionType ZernikeSphericalHarmonics(int l1, int n, int l2, int m, PrecisionType xr, PrecisionType yr, PrecisionType zr, PrecisionType rr)
    {
        // General variables
        PrecisionType r2 = rr * rr, xr2 = xr * xr, yr2 = yr * yr,
                zr2 = zr * zr;

    #if L2 >= 5
        // Variables needed for l2 >= 5
        PrecisionType tht = CST(0.0), phi = CST(0.0), cost = CST(0.0),
                sint = CST(0.0), cost2 = CST(0.0), sint2 = CST(0.0);
        if (l2 >= 5) {
            tht = ATAN2(yr, xr);
            phi = ATAN2(zr, SQRT(xr2 + yr2));
            sint = SIN(phi);
            cost = COS(tht);
            sint2 = sint * sint;
            cost2 = cost * cost;
        }
    #endif// L2 >= 5

        // Zernike polynomial
        PrecisionType R = CST(0.0);

        switch (l1) {
            case 0:
                R = SQRT(CST(3));
                break;
            case 1:
                R = SQRT(CST(5)) * rr;
                break;
            case 2:
                switch (n) {
                    case 0:
                        R = CST(-0.5) * SQRT(CST(7)) *
                            (CST(2.5) * (1 - 2 * r2) + CST(0.5));
                        break;
                    case 2:
                        R = SQRT(CST(7)) * r2;
                        break;
                }
                break;
    #if L1 >= 3
            case 3:
                switch (n) {
                    case 1:
                        R = CST(-1.5) * rr * (CST(3.5) * (1 - 2 * r2) + CST(1.5));
                        break;
                    case 3:
                        R = 3 * r2 * rr;
                }
                break;
    #endif// L1 >= 3
    #if L1 >= 4
            case 4:
                switch (n) {
                    case 0:
                        R = SQRT(CST(11)) *
                            ((63 * r2 * r2 / 8) - (35 * r2 / 4) + (CST(15) / CST(8)));
                        break;
                    case 2:
                        R = CST(-0.5) * SQRT(CST(11)) * r2 *
                            (CST(4.5) * (1 - 2 * r2) + CST(2.5));
                        break;
                    case 4:
                        R = SQRT(CST(11)) * r2 * r2;
                        break;
                }
                break;
    #endif// L1 >= 4
    #if L1 >= 5
            case 5:
                switch (n) {
                    case 1:
                        R = SQRT(CST(13)) * rr *
                            ((99 * r2 * r2 / 8) - (63 * r2 / 4) + (CST(35) / CST(8)));
                        break;
                    case 3:
                        R = CST(-0.5) * SQRT(CST(13)) * r2 * rr *
                            (CST(5.5) * (1 - 2 * r2) + CST(3.5));
                        break;
                }
                break;
    #endif// L1 >= 5
        }

        // Spherical harmonic
        PrecisionType Y = CST(0.0);

        switch (l2) {
            case 0:
                Y = (CST(1.0) / CST(2.0)) * SQRT((PrecisionType)CST(1.0) / _PI_);
                break;
            case 1:
                switch (m) {
                    case -1:
                        Y = SQRT(CST(3.0) / (CST(4.0) * _PI_)) * yr;
                        break;
                    case 0:
                        Y = SQRT(CST(3.0) / (CST(4.0) * _PI_)) * zr;
                        break;
                    case 1:
                        Y = SQRT(CST(3.0) / (CST(4.0) * _PI_)) * xr;
                        break;
                }
                break;
            case 2:
                switch (m) {
                    case -2:
                        Y = SQRT(CST(15.0) / (CST(4.0) * _PI_)) * xr * yr;
                        break;
                    case -1:
                        Y = SQRT(CST(15.0) / (CST(4.0) * _PI_)) * zr * yr;
                        break;
                    case 0:
                        Y = SQRT(CST(5.0) / (CST(16.0) * _PI_)) *
                            (-xr2 - yr2 + CST(2.0) * zr2);
                        break;
                    case 1:
                        Y = SQRT(CST(15.0) / (CST(4.0) * _PI_)) * xr * zr;
                        break;
                    case 2:
                        Y = SQRT(CST(15.0) / (CST(16.0) * _PI_)) * (xr2 - yr2);
                        break;
                }
                break;
    #if L2 >= 3
            case 3:
                switch (m) {
                    case -3:
                        Y = SQRT(CST(35.0) / (CST(16.0) * CST(2.0) * _PI_)) * yr *
                            (CST(3.0) * xr2 - yr2);
                        break;
                    case -2:
                        Y = SQRT(CST(105.0) / (CST(4.0) * _PI_)) * zr * yr * xr;
                        break;
                    case -1:
                        Y = SQRT(CST(21.0) / (CST(16.0) * CST(2.0) * _PI_)) * yr *
                            (CST(4.0) * zr2 - xr2 - yr2);
                        break;
                    case 0:
                        Y = SQRT(CST(7.0) / (CST(16.0) * _PI_)) * zr *
                            (CST(2.0) * zr2 - CST(3.0) * xr2 - CST(3.0) * yr2);
                        break;
                    case 1:
                        Y = SQRT(CST(21.0) / (CST(16.0) * CST(2.0) * _PI_)) * xr *
                            (CST(4.0) * zr2 - xr2 - yr2);
                        break;
                    case 2:
                        Y = SQRT(CST(105.0) / (CST(16.0) * _PI_)) * zr * (xr2 - yr2);
                        break;
                    case 3:
                        Y = SQRT(CST(35.0) / (CST(16.0) * CST(2.0) * _PI_)) * xr *
                            (xr2 - CST(3.0) * yr2);
                        break;
                }
                break;
    #endif// L2 >= 3
    #if L2 >= 4
            case 4:
                switch (m) {
                    case -4:
                        Y = SQRT((CST(35.0) * CST(9.0)) / (CST(16.0) * _PI_)) * yr *
                            xr * (xr2 - yr2);
                        break;
                    case -3:
                        Y = SQRT((CST(9.0) * CST(35.0)) /
                                 (CST(16.0) * CST(2.0) * _PI_)) *
                            yr * zr * (CST(3.0) * xr2 - yr2);
                        break;
                    case -2:
                        Y = SQRT((CST(9.0) * CST(5.0)) / (CST(16.0) * _PI_)) * yr * xr *
                            (CST(7.0) * zr2 - (xr2 + yr2 + zr2));
                        break;
                    case -1:
                        Y = SQRT((CST(9.0) * CST(5.0)) /
                                 (CST(16.0) * CST(2.0) * _PI_)) *
                            yr * zr * (CST(7.0) * zr2 - CST(3.0) * (xr2 + yr2 + zr2));
                        break;
                    case 0:
                        Y = SQRT(CST(9.0) / (CST(16.0) * CST(16.0) * _PI_)) *
                            (CST(35.0) * zr2 * zr2 - CST(30.0) * zr2 + CST(3.0));
                        break;
                    case 1:
                        Y = SQRT((CST(9.0) * CST(5.0)) /
                                 (CST(16.0) * CST(2.0) * _PI_)) *
                            xr * zr * (CST(7.0) * zr2 - CST(3.0) * (xr2 + yr2 + zr2));
                        break;
                    case 2:
                        Y = SQRT((CST(9.0) * CST(5.0)) / (CST(8.0) * CST(8.0) * _PI_)) *
                            (xr2 - yr2) * (CST(7.0) * zr2 - (xr2 + yr2 + zr2));
                        break;
                    case 3:
                        Y = SQRT((CST(9.0) * CST(35.0)) /
                                 (CST(16.0) * CST(2.0) * _PI_)) *
                            xr * zr * (xr2 - CST(3.0) * yr2);
                        break;
                    case 4:
                        Y = SQRT((CST(9.0) * CST(35.0)) /
                                 (CST(16.0) * CST(16.0) * _PI_)) *
                            (xr2 * (xr2 - CST(3.0) * yr2) -
                             yr2 * (CST(3.0) * xr2 - yr2));
                        break;
                }
                break;
    #endif// L2 >= 4
    #if L2 >= 5
            case 5:
                switch (m) {
                    case -5:
                        Y = (CST(3.0) / CST(16.0)) *
                            SQRT(CST(77.0) / (CST(2.0) * _PI_)) * sint2 * sint2 * sint *
                            SIN(CST(5.0) * phi);
                        break;
                    case -4:
                        Y = (CST(3.0) / CST(8.0)) *
                            SQRT(CST(385.0) / (CST(2.0) * _PI_)) * sint2 * sint2 *
                            SIN(CST(4.0) * phi);
                        break;
                    case -3:
                        Y = (CST(1.0) / CST(16.0)) *
                            SQRT(CST(385.0) / (CST(2.0) * _PI_)) * sint2 * sint *
                            (CST(9.0) * cost2 - CST(1.0)) * SIN(CST(3.0) * phi);
                        break;
                    case -2:
                        Y = (CST(1.0) / CST(4.0)) *
                            SQRT(CST(1155.0) / (CST(4.0) * _PI_)) * sint2 *
                            (CST(3.0) * cost2 * cost - cost) * SIN(CST(2.0) * phi);
                        break;
                    case -1:
                        Y = (CST(1.0) / CST(8.0)) *
                            SQRT(CST(165.0) / (CST(4.0) * _PI_)) * sint *
                            (CST(21.0) * cost2 * cost2 - CST(14.0) * cost2 + 1) *
                            SIN(phi);
                        break;
                    case 0:
                        Y = (CST(1.0) / CST(16.0)) * SQRT(CST(11.0) / _PI_) *
                            (CST(63.0) * cost2 * cost2 * cost -
                             CST(70.0) * cost2 * cost + CST(15.0) * cost);
                        break;
                    case 1:
                        Y = (CST(1.0) / CST(8.0)) *
                            SQRT(CST(165.0) / (CST(4.0) * _PI_)) * sint *
                            (CST(21.0) * cost2 * cost2 - CST(14.0) * cost2 + 1) *
                            COS(phi);
                        break;
                    case 2:
                        Y = (CST(1.0) / CST(4.0)) *
                            SQRT(CST(1155.0) / (CST(4.0) * _PI_)) * sint2 *
                            (CST(3.0) * cost2 * cost - cost) * COS(CST(2.0) * phi);
                        break;
                    case 3:
                        Y = (CST(1.0) / CST(16.0)) *
                            SQRT(CST(385.0) / (CST(2.0) * _PI_)) * sint2 * sint *
                            (CST(9.0) * cost2 - CST(1.0)) * COS(CST(3.0) * phi);
                        break;
                    case 4:
                        Y = (CST(3.0) / CST(8.0)) *
                            SQRT(CST(385.0) / (CST(2.0) * _PI_)) * sint2 * sint2 *
                            COS(CST(4.0) * phi);
                        break;
                    case 5:
                        Y = (CST(3.0) / CST(16.0)) *
                            SQRT(CST(77.0) / (CST(2.0) * _PI_)) * sint2 * sint2 * sint *
                            COS(CST(5.0) * phi);
                        break;
                }
                break;
    #endif// L2 >= 5
        }

        return R * Y;
    }


    template<typename PrecisionType>
    __device__ void splattingAtPos(PrecisionType pos_x, PrecisionType pos_y, PrecisionType weight,
                                   MultidimArrayCuda<PrecisionType> &mP, MultidimArrayCuda<PrecisionType> &mW)
    {
        int i = round(pos_y);
        int j = round(pos_x);
        if(!IS_OUTSIDE2D(mP, i, j))
        {
            A2D_ELEM(mP, i, j) += weight;
            A2D_ELEM(mW, i, j) += 1.0;
        }
    }

    template<typename PrecisionType>
    __device__ size_t findCuda(const PrecisionType *begin, size_t size, PrecisionType value)
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
}
/*
 * The first beast
 */
template<typename PrecisionType, bool usesZernike>
__global__ void forwardKernel(
        MultidimArrayCuda<PrecisionType> cudaMV,
        MultidimArrayCuda<int> cudaVRecMask,
        MultidimArrayCuda<PrecisionType> *cudaP,
        MultidimArrayCuda<PrecisionType> *cudaW,
        const int lastZ,
        const int lastY,
        const int lastX,
        const int step,
        const size_t sigma_size,
        const PrecisionType *cudaSigma,
        const PrecisionType iRmaxF,
        const size_t idxY0,
        const size_t idxZ0,
        const int *cudaVL1,
        const int *cudaVN,
        const int *cudaVL2,
        const int *cudaVM,
        const PrecisionType *cudaClnm,
        const PrecisionType *cudaR)
{
    for (int k = STARTINGZ(cudaMV); k <= lastZ; k += step)
    {
        for (int i = STARTINGY(cudaMV); i <= lastY; i += step)
        {
            for (int j = STARTINGX(cudaMV); j <= lastX; j += step)
            {
                // Future CUDA code
                PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;
                if (A3D_ELEM(cudaVRecMask, k, i, j) != 0)
                {
                    int img_idx = 0;
                    if (sigma_size > 1)
                    {
                        PrecisionType sigma_mask = A3D_ELEM(cudaVRecMask, k, i, j);
                        img_idx = device::findCuda(cudaSigma, sigma_size, sigma_mask);
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
                                PrecisionType zsph = device::ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
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
                    PrecisionType voxel_mV = A3D_ELEM(cudaMV, k, i, j);
                    device::splattingAtPos(pos_x, pos_y, voxel_mV, mP, mW);
                }
            }
        }
    }
}

#endif //CUDA_FORWARD_ART_ZERNIKE3D_CU