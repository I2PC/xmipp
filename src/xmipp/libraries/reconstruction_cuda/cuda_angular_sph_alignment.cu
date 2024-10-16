#pragma once

// Compilation settings

#ifndef KTT_USED
#include "cuda_volume_deform_sph_defines.h"
#include "cuda_angular_sph_alignment.h"
#include "cuda_compatibility.h"
#endif

namespace AngularAlignmentGpu {
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

__device__ __inline__ PrecisionType shfl_down(PrecisionType val, int offset) {
#if (CUDART_VERSION >= 9000)
    return __shfl_down_sync(0xFFFFFFFF, val, offset);
#else
    return __shfl_down(val, offset);
#endif
}

// Compilation settings - end

// Define data structures
#ifdef KTT_USED

struct ImageMetaData
{
    int xShift;
    int yShift;
    int zShift;

    int xDim;
    int yDim;
    int zDim;
}

#endif// KTT_USED

// CUDA kernel defines
#define BLOCK_SIZE (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM)

// ImageData macros

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

#define ELEM_2D(data,ImD,i,j) \
    ((data)[GET_IDX((ImD), 0, (i), (j))])

// TEST PRO ATOMIC_ADD
#define ELEM_2D_ADDR(data,ImD,i,j) \
    ((data) + GET_IDX((ImD), 0, (i), (j)))

#define ELEM_3D_SHIFTED(ImD,meta,k,i,j) \
    (ELEM_3D((ImD), (meta), (k) - (meta).zShift, (i) - (meta).yShift, (j) - (meta).xShift))

#define ELEM_2D_SHIFTED(data,ImD,i,j) \
    (ELEM_2D((data), (ImD), (i) - (ImD).yShift, (j) - (ImD).xShift))

// TEST PRO ATOMIC_ADD
#define ELEM_2D_SHIFTED_ADDR(data,ImD,i,j) \
    (ELEM_2D_ADDR((data), (ImD), (i) - (ImD).yShift, (j) - (ImD).xShift))

// Utility macros
#define IS_OUTSIDE(ImD,k,i,j) \
    ((j) < (ImD).xShift || (j) > (ImD).xShift + (ImD).xDim - 1 || \
     (i) < (ImD).yShift || (i) > (ImD).yShift + (ImD).yDim - 1 || \
     (k) < (ImD).zShift || (k) > (ImD).zShift + (ImD).zDim - 1)

#define IS_OUTSIDE_PHYS(ImD,k,i,j) \
    ((j) < 0 || (ImD).xDim <= (j) || \
     (i) < 0 || (ImD).yDim <= (i) || \
     (k) < 0 || (ImD).zDim <= (k))

// Smart casting to selected precision (at compile time)
// ...just shorter static_cast
#define CST(num) (static_cast<PrecisionType>((num)))

#define LIN_INTERP(a, l, h) ((l) + ((h) - (l)) * (a))

// Forward declarations
__forceinline__ __device__ PrecisionType ZernikeSphericalHarmonics(int l1, int n, int l2, int m,
        PrecisionType xr, PrecisionType yr, PrecisionType zr, PrecisionType r);

__device__ PrecisionType interpolatedElement3D(
        const PrecisionType* ImD, ImageMetaData imgMeta,
        PrecisionType x, PrecisionType y, PrecisionType z,
        PrecisionType doutside_value = 0);

__forceinline__ __device__ void rotateCoordinates(PrecisionType* pos, const PrecisionType* rotation)
{
    PrecisionType tmp[3] = {0};

    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            tmp[i] += rotation[3 * i + j] * pos[j];

    pos[0] = tmp[0];
    pos[1] = tmp[1];
    pos[2] = tmp[2];
}

/*
 * The beast
 */
extern "C" __global__ void projectionKernel(
        PrecisionType Rmax2,
        PrecisionType iRmax,
        ImageMetaData volMeta,
        const PrecisionType* volData,
        const PrecisionType* rotation,
        unsigned steps,
        const int4* zshparams,
        const PrecisionType3* clnm,
        const int* volMask,
        PrecisionType* projectionPlane,
        PrecisionType* outArrayGlobal
        ) 
{
    extern __shared__ char sharedBuffer[];
    unsigned sharedBufferOffset = 0;

    // Thread index in a block
    unsigned tIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // Get physical indexes
    int kPhys = blockIdx.z * blockDim.z + threadIdx.z;
    int iPhys = blockIdx.y * blockDim.y + threadIdx.y;
    int jPhys = blockIdx.x * blockDim.x + threadIdx.x;

    // Update to logical indexes (calculations expect logical indexing)
    PrecisionType pos[3];
    pos[2] = P2L_Z_IDX(volMeta, kPhys);
    pos[1] = P2L_Y_IDX(volMeta, iPhys);
    pos[0] = P2L_X_IDX(volMeta, jPhys);

    rotateCoordinates(pos, rotation);

    int4* zshShared = (int4*)(sharedBuffer + sharedBufferOffset);
    sharedBufferOffset += sizeof(int4) * steps;

    PrecisionType3* clnmShared = (PrecisionType3*)(sharedBuffer + sharedBufferOffset);
    sharedBufferOffset += sizeof(PrecisionType3) * steps;

    // Load zsh, clnm parameters to the shared memory
    if (steps <= BLOCK_SIZE) {
        if (tIdx < steps) {
            zshShared[tIdx] = zshparams[tIdx];
            clnmShared[tIdx] = clnm[tIdx];
        }
    } else {
        if (tIdx == 0) {
            for (unsigned idx = 0; idx < steps; idx++) {
                zshShared[idx] = zshparams[idx];
                clnmShared[idx] = clnm[idx];
            }
        }
    }

    __syncthreads();

    // Define and compute necessary values
    PrecisionType r2 = pos[2]*pos[2] + pos[1]*pos[1] + pos[0]*pos[0];
    PrecisionType rr = SQRT(r2) * iRmax;
    PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;

    if (r2 < Rmax2) {
        for (unsigned idx = 0; idx < steps; idx++) {
            int l1 = zshShared[idx].w;
            int n = zshShared[idx].x;
            int l2 = zshShared[idx].y;
            int m = zshShared[idx].z;

            PrecisionType zsph = ZernikeSphericalHarmonics(l1, n, l2, m,
                    pos[0] * iRmax, pos[1] * iRmax, pos[2] * iRmax, rr);

            if (rr > 0 || l2 == 0) {
                gx += zsph * clnmShared[idx].x;
                gy += zsph * clnmShared[idx].y;
                gz += zsph * clnmShared[idx].z;
            }
        }
    }

    int maskVoxel;
    int kMask = (int)(pos[2] + gz);
    int iMask = (int)(pos[1] + gy);
    int jMask = (int)(pos[0] + gx);

    if (IS_OUTSIDE(volMeta, kMask, iMask, jMask)) {
        maskVoxel = 0;
    } else {
        maskVoxel = ELEM_3D_SHIFTED(volMask, volMeta, kMask, iMask, jMask);
    }

    PrecisionType localCount = 0.0, localSumVD = 0.0, localModg = 0.0;

    if (maskVoxel == 1) {
        PrecisionType voxelI = interpolatedElement3D(volData, volMeta,
                pos[0] + gx, pos[1] + gy, pos[2] + gz);
        //ELEM_2D_SHIFTED(projectionPlane, volMeta,
        //        P2L_Y_IDX(volMeta, iPhys), P2L_X_IDX(volMeta, jPhys)) += voxelI;
        atomicAdd(ELEM_2D_SHIFTED_ADDR(projectionPlane, volMeta,
                P2L_Y_IDX(volMeta, iPhys), P2L_X_IDX(volMeta, jPhys)), voxelI);
        localSumVD += voxelI;
        localModg += gx*gx + gy*gy + gz*gz;
        localCount++;
    }

    __shared__ PrecisionType countShared[BLOCK_SIZE];
    __shared__ PrecisionType sumVDShared[BLOCK_SIZE];
    __shared__ PrecisionType modfgShared[BLOCK_SIZE];

    countShared[tIdx] = localCount;
    sumVDShared[tIdx] = localSumVD;
    modfgShared[tIdx] = localModg;

    __syncthreads();

    // First level of conditions are evaluated during compilation
    if (BLOCK_SIZE >= 1024) {
        if (tIdx < 512) {
            countShared[tIdx] += countShared[tIdx + 512];
            sumVDShared[tIdx] += sumVDShared[tIdx + 512];
            modfgShared[tIdx] += modfgShared[tIdx + 512];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 512) {
        if (tIdx < 256) {
            countShared[tIdx] += countShared[tIdx + 256];
            sumVDShared[tIdx] += sumVDShared[tIdx + 256];
            modfgShared[tIdx] += modfgShared[tIdx + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tIdx < 128) {
            countShared[tIdx] += countShared[tIdx + 128];
            sumVDShared[tIdx] += sumVDShared[tIdx + 128];
            modfgShared[tIdx] += modfgShared[tIdx + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tIdx < 64) {
            countShared[tIdx] += countShared[tIdx + 64];
            sumVDShared[tIdx] += sumVDShared[tIdx + 64];
            modfgShared[tIdx] += modfgShared[tIdx + 64];
        }
        __syncthreads();
    }
    // Last warp reduction
    if (tIdx < 32) {
        localCount = countShared[tIdx];
        localSumVD = sumVDShared[tIdx];
        localModg = modfgShared[tIdx];
        if (BLOCK_SIZE >= 64) {
            localCount += countShared[tIdx + 32];
            localSumVD += sumVDShared[tIdx + 32];
            localModg += modfgShared[tIdx + 32];
        }
        // Reduce warp
        for (int offset = 32 / 2; offset > 0; offset >>= 1) {
            localCount += shfl_down(localCount, offset);
            localSumVD += shfl_down(localSumVD, offset);
            localModg += shfl_down(localModg, offset);
        }
    }

    // Save values to the global memory for later
    if (tIdx == 0) {
        unsigned bIdx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
        unsigned GRID_SIZE = gridDim.x * gridDim.y * gridDim.z;
        // Resulting values are in variables local* => no need to go into shared mem
        outArrayGlobal[bIdx] = localCount;
        outArrayGlobal[bIdx + GRID_SIZE] = localSumVD;
        outArrayGlobal[bIdx + GRID_SIZE * 2] = localModg;
    }
}

/*
 * Linear interpolation
 */
__device__ PrecisionType interpolatedElement3D(
        const PrecisionType* ImD, ImageMetaData imgMeta,
        PrecisionType x, PrecisionType y, PrecisionType z,
        PrecisionType outside_value) 
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

        PrecisionType d000 = (IS_OUTSIDE(imgMeta, z0, y0, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, imgMeta, z0, y0, x0);
        PrecisionType d001 = (IS_OUTSIDE(imgMeta, z0, y0, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, imgMeta, z0, y0, x1);
        PrecisionType d010 = (IS_OUTSIDE(imgMeta, z0, y1, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, imgMeta, z0, y1, x0);
        PrecisionType d011 = (IS_OUTSIDE(imgMeta, z0, y1, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, imgMeta, z0, y1, x1);
        PrecisionType d100 = (IS_OUTSIDE(imgMeta, z1, y0, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, imgMeta, z1, y0, x0);
        PrecisionType d101 = (IS_OUTSIDE(imgMeta, z1, y0, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, imgMeta, z1, y0, x1);
        PrecisionType d110 = (IS_OUTSIDE(imgMeta, z1, y1, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, imgMeta, z1, y1, x0);
        PrecisionType d111 = (IS_OUTSIDE(imgMeta, z1, y1, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, imgMeta, z1, y1, x1);

        PrecisionType dx00 = LIN_INTERP(fx, d000, d001);
        PrecisionType dx01 = LIN_INTERP(fx, d100, d101);
        PrecisionType dx10 = LIN_INTERP(fx, d010, d011);
        PrecisionType dx11 = LIN_INTERP(fx, d110, d111);
        PrecisionType dxy0 = LIN_INTERP(fy, dx00, dx10);
        PrecisionType dxy1 = LIN_INTERP(fy, dx01, dx11);

        return LIN_INTERP(fz, dxy0, dxy1);
}

/*
 * ZSH
 */
__forceinline__ __device__ PrecisionType ZernikeSphericalHarmonics(int l1, int n, int l2, int m,
        PrecisionType xr, PrecisionType yr, PrecisionType zr, PrecisionType rr)
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

} // namespace AngularAlignmentGpu
