/***************************************************************************
 *
 * Authors:    David Myska              davidmyska@mail.muni.cz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

#ifndef CUDA_VOLUME_DEFORM_SPH_CU
#define CUDA_VOLUME_DEFORM_SPH_CU

// Compilation settings

#ifndef KTT_USED
#include "cuda_volume_deform_sph_defines.h"
#include "cuda_volume_deform_sph.h"
#endif

#if USE_DOUBLE_PRECISION == 1
// Types
using PrecisionType = double;
using PrecisionType3 = double3;
// Constants
#define _PI_ (3.1415926535897931e+0)
// Functions
#define SQRT sqrt
#define ATAN2 atan2
#define COS cos
#define SIN sin
#define CUDA_FLOOR floor

#else
// Types
using PrecisionType = float;
using PrecisionType3 = float3;
// Constants
#define _PI_ (3.1415926535897f)
// Functions
#define SQRT sqrtf
#define ATAN2 atan2f
#define COS cosf
#define SIN sinf
#define CUDA_FLOOR floorf

#endif// USE_DOUBLE_PRECISION

// Compilation settings - end

// Define data structures
#ifdef KTT_USED

struct VolumeMetaData
{
    int xShift;
    int yShift;
    int zShift;

    int xDim;
    int yDim;
    int zDim;

    int padding;
}

template<typename T>
struct Volumes 
{
    T* I;
    T* R;
    unsigned count;
    unsigned volumeSize;
    unsigned volumePaddedSize;
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
#endif// KTT_USED

// CUDA kernel defines
#define BLOCK_SIZE (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM)

// ImageData macros

// Index to global memory
#define GET_IDX(ImD,k,i,j) \
    ((ImD).xDim * (ImD).yDim * (k) + (ImD).xDim * (i) + (j))

#define GET_IDX_PADDED(ImD,k,i,j) \
    (((ImD).xDim + 2) * ((ImD).yDim + 2) * ((k) + 1) + ((ImD).xDim + 2) * ((i) + 1) + ((j) + 1))

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

#define ELEM_3D_PADDED(ImD,meta,k,i,j) \
    ((ImD)[GET_IDX_PADDED((meta), (k), (i), (j))])

#define ELEM_3D_SHIFTED(ImD,meta,k,i,j) \
    (ELEM_3D((ImD), (meta), (k) - (meta).zShift, (i) - (meta).yShift, (j) - (meta).xShift))

#define ELEM_3D_SHIFTED_PADDED(ImD,meta,k,i,j) \
    (ELEM_3D_PADDED((ImD), (meta), (k) - (meta).zShift, (i) - (meta).yShift, (j) - (meta).xShift))

// Utility macros
#define IS_OUTSIDE(ImD,k,i,j) \
    ((j) < (ImD).xShift || (j) > (ImD).xShift + (ImD).xDim - 1 || \
     (i) < (ImD).yShift || (i) > (ImD).yShift + (ImD).yDim - 1 || \
     (k) < (ImD).zShift || (k) > (ImD).zShift + (ImD).zDim - 1)

#define IS_OUTSIDE_PADDED(ImD,k,i,j) \
    ((j) < (ImD).xShift - 1 || (j) > (ImD).xShift + (ImD).xDim - 1 || \
     (i) < (ImD).yShift - 1 || (i) > (ImD).yShift + (ImD).yDim - 1 || \
     (k) < (ImD).zShift - 1 || (k) > (ImD).zShift + (ImD).zDim - 1)

#define IS_OUTSIDE_PHYS(ImD,k,i,j) \
    ((j) < 0 || (ImD).xDim <= (j) || \
     (i) < 0 || (ImD).yDim <= (i) || \
     (k) < 0 || (ImD).zDim <= (k))

// Smart casting to selected precision (at compile time)
// ...just shorter static_cast
#define CST(num) (static_cast<PrecisionType>((num)))

#define LIN_INTERP(a, l, h) ((l) + ((h) - (l)) * (a))

// Forward declarations
template<int _L1 = 5, int _L2 = 5>
__forceinline__ __device__ PrecisionType ZernikeSphericalHarmonics(int l1, int n, int l2, int m,
        PrecisionType xr, PrecisionType yr, PrecisionType zr, PrecisionType r);

__device__ PrecisionType interpolate(
        PrecisionType* ImD, VolumeMetaData volMetaData,
        PrecisionType x, PrecisionType y, PrecisionType z);

// For the current supported degrees L1, L2, the max is 56 coeficients
// if there is added support for higher degrees of L1, L2 then the
// max number of coeficient NEEDS to be recalculated and updated
#ifndef MAX_COEF_COUNT
#define MAX_COEF_COUNT 56
#endif

__constant__ PrecisionType3 cClnm[MAX_COEF_COUNT];
__constant__ int4 cZsh[MAX_COEF_COUNT];

template<int _BLOCK_SIZE = BLOCK_SIZE, int _L1 = 5, int _L2 = 5>
__global__ void computeDeform(
        PrecisionType Rmax2,
        PrecisionType iRmax,
        IROimages images,
        unsigned steps,
        VolumeMetaData volMetaData,
        Volumes<PrecisionType> volumes,
        DeformImages deformImages,
        bool applyTransformation,
        bool saveDeformation,
        PrecisionType* outArrayGlobal
        )
{

    // Thread index in a block
    unsigned tIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // Get physical indexes
    int kPhys = blockIdx.z * blockDim.z + threadIdx.z;
    int iPhys = blockIdx.y * blockDim.y + threadIdx.y;
    int jPhys = blockIdx.x * blockDim.x + threadIdx.x;

    // Update to logical indexes (calculations expect logical indexing)
    int k = P2L_Z_IDX(volMetaData, kPhys);
    int i = P2L_Y_IDX(volMetaData, iPhys);
    int j = P2L_X_IDX(volMetaData, jPhys);

    // Define and compute necessary values
    PrecisionType r2 = k*k + i*i + j*j;
    PrecisionType rr = SQRT(r2) * iRmax;
    PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;

    if (r2 < Rmax2) {
        for (unsigned idx = 0; idx < steps; idx++) {
            int l1 = cZsh[idx].w;
            int n = cZsh[idx].x;
            int l2 = cZsh[idx].y;
            int m = cZsh[idx].z;

            PrecisionType zsph = ZernikeSphericalHarmonics<_L1, _L2>(l1, n, l2, m,
                    j * iRmax, i * iRmax, k * iRmax, rr);

            if (rr > 0 || l2 == 0) {
                gx += zsph * cClnm[idx].x;
                gy += zsph * cClnm[idx].y;
                gz += zsph * cClnm[idx].z;
            }
        }
    }

    PrecisionType voxelI, voxelR;
    PrecisionType diff;

    PrecisionType localDiff2 = 0.0, localSumVD = 0.0, localModg = 0.0;

    bool isOutside = IS_OUTSIDE_PHYS(volMetaData, kPhys, iPhys, jPhys);
    PrecisionType kDef = k + gz;
    PrecisionType iDef = i + gy;
    PrecisionType jDef = j + gx;

    if (applyTransformation && !isOutside) {
        // Logical indexes used to check whether the point is in the matrix
        if (!IS_OUTSIDE_PADDED(volMetaData, kDef, iDef, jDef)) {
            voxelI = interpolate(images.VI, volMetaData, jDef, iDef, kDef);
        } else {
            voxelI = 0.0;
        }

        ELEM_3D(images.VO, volMetaData, kPhys, iPhys, jPhys) = voxelI;
    }

    if (!isOutside) {
        for (unsigned idv = 0; idv < volumes.count; idv++) {
            voxelR = ELEM_3D(volumes.R + idv * volumes.volumeSize,
                    volMetaData, kPhys, iPhys, jPhys);
            if (!IS_OUTSIDE_PADDED(volMetaData, kDef, iDef, jDef)) {
                voxelI = interpolate(volumes.I + idv * volumes.volumePaddedSize,
                        volMetaData, jDef, iDef, kDef);
            } else {
                voxelI = 0.0;
            }

            if (voxelI >= 0.0)
                localSumVD += voxelI;

            diff = voxelR - voxelI;
            localDiff2 += diff * diff;
        }
        localModg += volumes.count * (gx*gx + gy*gy + gz*gz);
    }

    // Reduce warp
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        localDiff2 += __shfl_down_sync(0xFFFFFFFF, localDiff2, offset);
        localSumVD += __shfl_down_sync(0xFFFFFFFF, localSumVD, offset);
        localModg += __shfl_down_sync(0xFFFFFFFF, localModg, offset);
    }

    bool isFirstThreadInWarp = (tIdx & 31) == 0;// tIdx % 32, modulo is slow

    // Save values to the global memory for later
    if (isFirstThreadInWarp) {
        unsigned warpsInBlock = _BLOCK_SIZE >> 5;// _BLOCK_SIZE / 32, division is slow
        unsigned warpInCurrentBlock = tIdx >> 5;// tIdx / 32, division is slow
        unsigned bIdx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
        unsigned wIdx = bIdx * warpsInBlock + warpInCurrentBlock;
        unsigned WARP_GRID_SIZE = gridDim.x * gridDim.y * gridDim.z * warpsInBlock;
        outArrayGlobal[wIdx] = localDiff2;
        outArrayGlobal[wIdx + WARP_GRID_SIZE] = localSumVD;
        outArrayGlobal[wIdx + WARP_GRID_SIZE * 2] = localModg;
    }

    if (saveDeformation && !isOutside) {
        ELEM_3D(deformImages.Gx, volMetaData, kPhys, iPhys, jPhys) = gx;
        ELEM_3D(deformImages.Gy, volMetaData, kPhys, iPhys, jPhys) = gy;
        ELEM_3D(deformImages.Gz, volMetaData, kPhys, iPhys, jPhys) = gz;
    }
}

/*
 * Linear interpolation
 */
__device__ PrecisionType interpolate(
        PrecisionType* ImD, VolumeMetaData volMetaData,
        PrecisionType x, PrecisionType y, PrecisionType z)
{
        int x0 = (int)CUDA_FLOOR(x);
        PrecisionType fx = x - x0;
        int x1 = x0 + 1;

        int y0 = (int)CUDA_FLOOR(y);
        PrecisionType fy = y - y0;
        int y1 = y0 + 1;

        int z0 = (int)CUDA_FLOOR(z);
        PrecisionType fz = z - z0;
        int z1 = z0 + 1;

        PrecisionType d000 = ELEM_3D_SHIFTED_PADDED(ImD, volMetaData, z0, y0, x0);
        PrecisionType d001 = ELEM_3D_SHIFTED_PADDED(ImD, volMetaData, z0, y0, x1);
        PrecisionType d010 = ELEM_3D_SHIFTED_PADDED(ImD, volMetaData, z0, y1, x0);
        PrecisionType d011 = ELEM_3D_SHIFTED_PADDED(ImD, volMetaData, z0, y1, x1);
        PrecisionType d100 = ELEM_3D_SHIFTED_PADDED(ImD, volMetaData, z1, y0, x0);
        PrecisionType d101 = ELEM_3D_SHIFTED_PADDED(ImD, volMetaData, z1, y0, x1);
        PrecisionType d110 = ELEM_3D_SHIFTED_PADDED(ImD, volMetaData, z1, y1, x0);
        PrecisionType d111 = ELEM_3D_SHIFTED_PADDED(ImD, volMetaData, z1, y1, x1);

        PrecisionType dx00 = LIN_INTERP(fx, d000, d001);
        PrecisionType dx01 = LIN_INTERP(fx, d100, d101);
        PrecisionType dx10 = LIN_INTERP(fx, d010, d011);
        PrecisionType dx11 = LIN_INTERP(fx, d110, d111);
        PrecisionType dxy0 = LIN_INTERP(fy, dx00, dx10);
        PrecisionType dxy1 = LIN_INTERP(fy, dx01, dx11);

        return LIN_INTERP(fz, dxy0, dxy1);
}

template<int _L1, int _L2>
__forceinline__ __device__ PrecisionType ZernikeSphericalHarmonics(int l1, int n, int l2, int m,
        PrecisionType xr, PrecisionType yr, PrecisionType zr, PrecisionType rr)
{
    // General variables
    PrecisionType r2 = rr * rr, xr2 = xr * xr, yr2 = yr * yr,
                  zr2 = zr * zr;


    // Variables needed for l2 >= 5
    PrecisionType tht = CST(0.0), phi = CST(0.0), cost = CST(0.0),
                  sint = CST(0.0), cost2 = CST(0.0), sint2 = CST(0.0);
    if (_L2 >= 5) {
        if (l2 >= 5) {
            tht = ATAN2(yr, xr);
            phi = ATAN2(zr, SQRT(xr2 + yr2));
            sint = SIN(phi);
            cost = COS(tht);
            sint2 = sint * sint;
            cost2 = cost * cost;
        }
    }

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
            if (_L1 >= 3) {
                case 3:
                    switch (n) {
                        case 1:
                            R = CST(-1.5) * rr * (CST(3.5) * (1 - 2 * r2) + CST(1.5));
                            break;
                        case 3:
                            R = 3 * r2 * rr;
                    }
                    break;
            }
            if (_L1 >= 4) {
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
            }
            if (_L1 >= 5) {
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
            }
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
            if (_L2 >= 3) {
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
            }
            if (_L2 >= 4) {
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
            }
            if (_L2 >= 5) {
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
            }
    }

    return R * Y;
}

// Cast input volume to the result type. Depending on template parameter may add padding.
template<bool PADDING = false>
__global__ void prepareVolumes(PrecisionType* output, double* input, VolumeMetaData metaData)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // TODO maybe more work per thread would be better
    if (!IS_OUTSIDE_PHYS(metaData, z, y, x)) {
        if (PADDING) {
            ELEM_3D_PADDED(output, metaData, z, y, x) = ELEM_3D(input, metaData, z, y, x);
        } else {
            ELEM_3D(output, metaData, z, y, x) = ELEM_3D(input, metaData, z, y, x);
        }
    }
}

#endif //CUDA_VOLUME_DEFORM_SPH_CU
