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

#endif// USE_DOUBLE_PRECISION

// Compilation settings - end

// Define data structures
#ifdef KTT_USED
struct ImageData
{
    int xShift;
    int yShift;
    int zShift;

    int xDim;
    int yDim;
    int zDim;

    PrecisionType* data;
};

struct ZSHparams
{ 
    int* vL1;
    int* vN;
    int* vL2;
    int* vM;
    unsigned size;
};

struct Volumes 
{
    ImageData* I;
    ImageData* R;
    unsigned size;
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
#endif// KTT_USED

// CUDA kernel defines
#ifndef BLOCK_X_DIM
#define BLOCK_X_DIM 8
#endif
#ifndef BLOCK_Y_DIM
#define BLOCK_Y_DIM 4
#endif
#ifndef BLOCK_Z_DIM
#define BLOCK_Z_DIM 4
#endif
#define BLOCK_SIZE (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM)

// ImageData macros

// Index to global memory
#define GET_IDX(ImD,k,i,j) \
    ((ImD).xDim * (ImD).yDim * (k) + (ImD).xDim * (i) + (j))

// Index to shared memory
#define GET_IDX_SHARED(k,i,j) \
    (blockDim.x * blockDim.y * ((k) % blockDim.z) + blockDim.x * ((i) % blockDim.y) + ((j) % blockDim.x))

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
#define ELEM_3D(ImD,k,i,j) \
    ((ImD).data[GET_IDX((ImD), (k), (i), (j))])

#define ELEM_3D_SHIFTED(ImD,k,i,j) \
    (ELEM_3D((ImD), (k) - (ImD).zShift, (i) - (ImD).yShift, (j) - (ImD).xShift))

#define ELEM_3D_SHARED(VolData,k,i,j) \
    ((VolData)[GET_IDX_SHARED((k), (i), (j))])

// Utility macros
#define IS_OUTSIDE(ImD,k,i,j) \
    ((j) < (ImD).xShift || (j) > (ImD).xShift + (ImD).xDim - 1 || \
     (i) < (ImD).yShift || (i) > (ImD).yShift + (ImD).yDim - 1 || \
     (k) < (ImD).zShift || (k) > (ImD).zShift + (ImD).zDim - 1)

#define IS_OUTSIDE_PHYS(ImD,k,i,j) \
    ((j) < 0 || (ImD).xDim <= (j) || \
     (i) < 0 || (ImD).yDim <= (i) || \
     (k) < 0 || (ImD).zDim <= (k))

#define IS_OUTSIDE_SHARED(k,i,j) \
    ((j) < blockIdx.x * blockDim.x || blockIdx.x * blockDim.x + blockDim.x <= (j) || \
     (i) < blockIdx.y * blockDim.y || blockIdx.y * blockDim.y + blockDim.y <= (i) || \
     (k) < blockIdx.z * blockDim.z || blockIdx.z * blockDim.z + blockDim.z <= (k))

// Smart casting to selected precision (at compile time)
// ...just shorter static_cast
#define CST(num) (static_cast<PrecisionType>((num)))

#define FLOOR(x) (((x) == (int)(x)) ? (int)(x):(((x) > 0) ? (int)(x) : \
                  (int)((x) - 1)))
#define LIN_INTERP(a, l, h) ((l) + ((h) - (l)) * (a))

// Forward declarations
__device__ PrecisionType ZernikeSphericalHarmonics(int l1, int n, int l2, int m,
        PrecisionType xr, PrecisionType yr, PrecisionType zr, PrecisionType r);

__device__ PrecisionType interpolatedElement3D(ImageData ImD,
        PrecisionType x, PrecisionType y, PrecisionType z,
        PrecisionType doutside_value = 0);

/*
 * The beast
 */
extern "C" __global__ void computeDeform(
        PrecisionType Rmax2,
        PrecisionType iRmax,
        IROimages images,
        int4* zshparams,
        PrecisionType3* clnm,
        int steps,
        Volumes volumes,
        DeformImages deformImages,
        bool applyTransformation,
        bool saveDeformation,
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
    int k = P2L_Z_IDX(images.VR, kPhys);
    int i = P2L_Y_IDX(images.VR, iPhys);
    int j = P2L_X_IDX(images.VR, jPhys);

#if USE_SHARED_VOLUME_METADATA == 1
    ImageData* volRMetaShared = (ImageData*)(sharedBuffer + sharedBufferOffset);
    sharedBufferOffset += sizeof(ImageData) * volumes.size;

    ImageData* volIMetaShared = (ImageData*)(sharedBuffer + sharedBufferOffset);
    sharedBufferOffset += sizeof(ImageData) * volumes.size;

    // Load metadata about volumes to the shared memory
    if (volumes.size <= BLOCK_SIZE) {
        if (tIdx < volumes.size) {
            volRMetaShared[tIdx] = volumes.R[tIdx];
            volIMetaShared[tIdx] = volumes.I[tIdx];
        }
    } else {
        if (tIdx == 0) {
            for (unsigned idx = 0; idx < volumes.size; idx++) {
                volRMetaShared[idx] = volumes.R[idx];
                volIMetaShared[idx] = volumes.I[idx];
            }
        }
    }

#endif// USE_SHARED_VOLUME_METADATA

#if USE_SHARED_MEM_ZSH_CLNM == 1 && USE_SCATTERED_ZSH_CLNM == 0
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
#endif

#if USE_SHARED_MEM_ZSH_CLNM + USE_SHARED_VOLUME_METADATA + USE_SHARED_VOLUME_DATA > 0
    __syncthreads();
#endif

    // Define and compute necessary values
    PrecisionType r2 = k*k + i*i + j*j;
    PrecisionType rr = SQRT(r2) * iRmax;
    PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;

    if (r2 < Rmax2) {
        for (int idx = 0; idx < steps; idx++) {
#if USE_SHARED_MEM_ZSH_CLNM == 1
            int l1 = zshShared[idx].w;
            int n = zshShared[idx].x;
            int l2 = zshShared[idx].y;
            int m = zshShared[idx].z;
#else
            int l1 = zshparams[idx].w;
            int n = zshparams[idx].x;
            int l2 = zshparams[idx].y;
            int m = zshparams[idx].z;
#endif// USE_SHARED_MEM_ZSH_CLNM

#if USE_ZSH_FUNCTION == 1
            PrecisionType zsph = ZernikeSphericalHarmonics(l1, n, l2, m,
                    j * iRmax, i * iRmax, k * iRmax, rr);
#else
            PrecisionType xr = j * iRmax, yr = i * iRmax, zr = k * iRmax;

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

            PrecisionType zsph = R * Y;
#endif// USE_ZSH_FUNCTION

            if (rr > 0 || l2 == 0) {
#if USE_SHARED_MEM_ZSH_CLNM == 1
                gx += zsph * clnmShared[idx].x;
                gy += zsph * clnmShared[idx].y;
                gz += zsph * clnmShared[idx].z;
#else
                gx += zsph * clnm[idx].x;
                gy += zsph * clnm[idx].y;
                gz += zsph * clnm[idx].z;
#endif// USE_SHARED_MEM_ZSH_CLNM
            }
        }
    }

    PrecisionType voxelI, voxelR;
    PrecisionType diff;

    PrecisionType localDiff2 = 0.0, localSumVD = 0.0, localModg = 0.0, localNcount = 0.0;

    bool isOutside = IS_OUTSIDE_PHYS(images.VR, kPhys, iPhys, jPhys);

    if (applyTransformation && !isOutside) {
        // Indexing requires physical indexes
        voxelR = ELEM_3D(images.VR, kPhys, iPhys, jPhys);
        // Logical indexes used to check whether the point is in the matrix
        voxelI = interpolatedElement3D(images.VI, j + gx, i + gy, k + gz);

        if (voxelI >= 0.0)
            localSumVD += voxelI;

        ELEM_3D(images.VO, kPhys, iPhys, jPhys) = voxelI;
        diff = voxelR - voxelI;
        localDiff2 += diff * diff;
        localModg += gx*gx + gy*gy + gz*gz;
        localNcount++;
    }

    if (!isOutside) {
        for (unsigned idv = 0; idv < volumes.size; idv++) {
#if USE_SHARED_VOLUME_METADATA == 1
            voxelR = ELEM_3D(volRMetaShared[idv], kPhys, iPhys, jPhys);
            voxelI = interpolatedElement3D(volIMetaShared[idv], j + gx, i + gy, k + gz);
#else
            voxelR = ELEM_3D(volumes.R[idv], kPhys, iPhys, jPhys);
            voxelI = interpolatedElement3D(volumes.I[idv], j + gx, i + gy, k + gz);
#endif// USE_SHARED_VOLUME_METADATA

            if (voxelI >= 0.0)
                localSumVD += voxelI;

            diff = voxelR - voxelI;
            localDiff2 += diff * diff;
            localModg += gx*gx + gy*gy + gz*gz;
            localNcount++;
        }
    }

    __shared__ PrecisionType diff2Shared[BLOCK_SIZE];
    __shared__ PrecisionType sumVDShared[BLOCK_SIZE];
    __shared__ PrecisionType modfgShared[BLOCK_SIZE];
    __shared__ PrecisionType countShared[BLOCK_SIZE];

    diff2Shared[tIdx] = localDiff2;
    sumVDShared[tIdx] = localSumVD;
    modfgShared[tIdx] = localModg;
    countShared[tIdx] = localNcount;

    __syncthreads();

    // First level of conditions are evaluated during compilation
    if (BLOCK_SIZE >= 1024) {
        if (tIdx < 512) {
            diff2Shared[tIdx] += diff2Shared[tIdx + 512];
            sumVDShared[tIdx] += sumVDShared[tIdx + 512];
            modfgShared[tIdx] += modfgShared[tIdx + 512];
            countShared[tIdx] += countShared[tIdx + 512];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 512) {
        if (tIdx < 256) {
            diff2Shared[tIdx] += diff2Shared[tIdx + 256];
            sumVDShared[tIdx] += sumVDShared[tIdx + 256];
            modfgShared[tIdx] += modfgShared[tIdx + 256];
            countShared[tIdx] += countShared[tIdx + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tIdx < 128) {
            diff2Shared[tIdx] += diff2Shared[tIdx + 128];
            sumVDShared[tIdx] += sumVDShared[tIdx + 128];
            modfgShared[tIdx] += modfgShared[tIdx + 128];
            countShared[tIdx] += countShared[tIdx + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tIdx < 64) {
            diff2Shared[tIdx] += diff2Shared[tIdx + 64];
            sumVDShared[tIdx] += sumVDShared[tIdx + 64];
            modfgShared[tIdx] += modfgShared[tIdx + 64];
            countShared[tIdx] += countShared[tIdx + 64];
        }
        __syncthreads();
    }
    // Last warp reduction
    if (tIdx < 32) {
        localDiff2 = diff2Shared[tIdx];
        localSumVD = sumVDShared[tIdx];
        localModg = modfgShared[tIdx];
        localNcount = countShared[tIdx];
        if (BLOCK_SIZE >= 64) {
            localDiff2 += diff2Shared[tIdx + 32];
            localSumVD += sumVDShared[tIdx + 32];
            localModg += modfgShared[tIdx + 32];
            localNcount += countShared[tIdx + 32];
        }
        // Reduce warp
        for (int offset = 32 / 2; offset > 0; offset >>= 1) {
            localDiff2 += __shfl_down_sync(0xFFFFFFFF, localDiff2, offset);
            localSumVD += __shfl_down_sync(0xFFFFFFFF, localSumVD, offset);
            localModg += __shfl_down_sync(0xFFFFFFFF, localModg, offset);
            localNcount += __shfl_down_sync(0xFFFFFFFF, localNcount, offset);
        }
    }

    // Save values to the global memory for later
    if (tIdx == 0) {
        unsigned bIdx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
        unsigned GRID_SIZE = gridDim.x * gridDim.y * gridDim.z;
        // Resulting values are in variables local* => no need to go into shared mem
        outArrayGlobal[bIdx] = localDiff2;
        outArrayGlobal[bIdx + GRID_SIZE] = localSumVD;
        outArrayGlobal[bIdx + GRID_SIZE * 2] = localModg;
        outArrayGlobal[bIdx + GRID_SIZE * 3] = localNcount;
    }

    if (saveDeformation && !isOutside) {
        ELEM_3D(deformImages.Gx, kPhys, iPhys, jPhys) = gx;
        ELEM_3D(deformImages.Gy, kPhys, iPhys, jPhys) = gy;
        ELEM_3D(deformImages.Gz, kPhys, iPhys, jPhys) = gz;
    }
}

/*
 * Linear interpolation
 */
__device__ PrecisionType interpolatedElement3Dshared(ImageData ImD, PrecisionType* volData,
        PrecisionType x, PrecisionType y, PrecisionType z,
        PrecisionType outside_value) 
{
        int x0 = FLOOR(x);
        PrecisionType fx = x - x0;
        int x1 = x0 + 1;

        int y0 = FLOOR(y);
        PrecisionType fy = y - y0;
        int y1 = y0 + 1;

        int z0 = FLOOR(z);
        PrecisionType fz = z - z0;
        int z1 = z0 + 1;

        PrecisionType d000;
        if (IS_OUTSIDE_PHYS(ImD, z0, y0, x0)) {
            d000 = outside_value;
        } else {
            if (IS_OUTSIDE_SHARED(z0, y0, x0)) {
                d000 = ELEM_3D(ImD, z0, y0, x0);
            } else {
                d000 = ELEM_3D_SHARED(volData, z0, y0, x0);
            }
        }

        PrecisionType d001;
        if (IS_OUTSIDE_PHYS(ImD, z0, y0, x1)) {
            d001 = outside_value;
        } else {
            if (IS_OUTSIDE_SHARED(z0, y0, x1)) {
                d001 = ELEM_3D(ImD, z0, y0, x1);
            } else {
                d001 = ELEM_3D_SHARED(volData, z0, y0, x1);
            }
        }

        PrecisionType d010;
        if (IS_OUTSIDE_PHYS(ImD, z0, y1, x0)) {
            d010 = outside_value;
        } else {
            if (IS_OUTSIDE_SHARED(z0, y1, x0)) {
                d010 = ELEM_3D(ImD, z0, y1, x0);
            } else {
                d010 = ELEM_3D_SHARED(volData, z0, y1, x0);
            }
        }

        PrecisionType d011;
        if (IS_OUTSIDE_PHYS(ImD, z0, y1, x1)) {
            d011 = outside_value;
        } else {
            if (IS_OUTSIDE_SHARED(z0, y1, x1)) {
                d011 = ELEM_3D(ImD, z0, y1, x1);
            } else {
                d011 = ELEM_3D_SHARED(volData, z0, y1, x1);
            }
        }

        PrecisionType d100;
        if (IS_OUTSIDE_PHYS(ImD, z1, y0, x0)) {
            d100 = outside_value;
        } else {
            if (IS_OUTSIDE_SHARED(z1, y0, x0)) {
                d100 = ELEM_3D(ImD, z1, y0, x0);
            } else {
                d100 = ELEM_3D_SHARED(volData, z1, y0, x0);
            }
        }

        PrecisionType d101;
        if (IS_OUTSIDE_PHYS(ImD, z1, y0, x1)) {
            d101 = outside_value;
        } else {
            if (IS_OUTSIDE_SHARED(z1, y0, x1)) {
                d101 = ELEM_3D(ImD, z1, y0, x1);
            } else {
                d101 = ELEM_3D_SHARED(volData, z1, y0, x1);
            }
        }

        PrecisionType d110;
        if (IS_OUTSIDE_PHYS(ImD, z1, y1, x0)) {
            d110 = outside_value;
        } else {
            if (IS_OUTSIDE_SHARED(z1, y1, x0)) {
                d110 = ELEM_3D(ImD, z1, y1, x0);
            } else {
                d110 = ELEM_3D_SHARED(volData, z1, y1, x0);
            }
        }

        PrecisionType d111;
        if (IS_OUTSIDE_PHYS(ImD, z1, y1, x1)) {
            d111 = outside_value;
        } else {
            if (IS_OUTSIDE_SHARED(z1, y1, x1)) {
                d111 = ELEM_3D(ImD, z1, y1, x1);
            } else {
                d111 = ELEM_3D_SHARED(volData, z1, y1, x1);
            }
        }

        PrecisionType dx00 = LIN_INTERP(fx, d000, d001);
        PrecisionType dx01 = LIN_INTERP(fx, d100, d101);
        PrecisionType dx10 = LIN_INTERP(fx, d010, d011);
        PrecisionType dx11 = LIN_INTERP(fx, d110, d111);
        PrecisionType dxy0 = LIN_INTERP(fy, dx00, dx10);
        PrecisionType dxy1 = LIN_INTERP(fy, dx01, dx11);

        return LIN_INTERP(fz, dxy0, dxy1);
}

/*
 * Linear interpolation
 */
__device__ PrecisionType interpolatedElement3D(ImageData ImD,
        PrecisionType x, PrecisionType y, PrecisionType z,
        PrecisionType outside_value) 
{
        int x0 = FLOOR(x);
        PrecisionType fx = x - x0;
        int x1 = x0 + 1;

        int y0 = FLOOR(y);
        PrecisionType fy = y - y0;
        int y1 = y0 + 1;

        int z0 = FLOOR(z);
        PrecisionType fz = z - z0;
        int z1 = z0 + 1;

        PrecisionType d000 = (IS_OUTSIDE(ImD, z0, y0, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z0, y0, x0);
        PrecisionType d001 = (IS_OUTSIDE(ImD, z0, y0, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z0, y0, x1);
        PrecisionType d010 = (IS_OUTSIDE(ImD, z0, y1, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z0, y1, x0);
        PrecisionType d011 = (IS_OUTSIDE(ImD, z0, y1, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z0, y1, x1);
        PrecisionType d100 = (IS_OUTSIDE(ImD, z1, y0, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z1, y0, x0);
        PrecisionType d101 = (IS_OUTSIDE(ImD, z1, y0, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z1, y0, x1);
        PrecisionType d110 = (IS_OUTSIDE(ImD, z1, y1, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z1, y1, x0);
        PrecisionType d111 = (IS_OUTSIDE(ImD, z1, y1, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z1, y1, x1);

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
__device__ PrecisionType ZernikeSphericalHarmonics(int l1, int n, int l2, int m, PrecisionType xr, PrecisionType yr, PrecisionType zr, PrecisionType r)
{
	// General variables
	PrecisionType r2=r*r,xr2=xr*xr,yr2=yr*yr,zr2=zr*zr;

	//Variables needed for l>=5
	PrecisionType tht=CST(0.0),phi=CST(0.0),cost=CST(0.0),sint=CST(0.0),cost2=CST(0.0),sint2=CST(0.0);
	if (l2>=5)
	{
		tht = ATAN2(yr,xr);
		phi = ATAN2(zr,SQRT(xr2 + yr2));
		sint = SIN(phi); cost = COS(tht);
		sint2 = sint*sint; cost2 = cost*cost;
	}

	// Zernike polynomial
	PrecisionType R=CST(0.0);

	switch (l1)
	{
	case 0:
		R = SQRT(CST(3));
		break;
	case 1:
		R = SQRT(CST(5))*r;
		break;
	case 2:
		switch (n)
		{
		case 0:
			R = CST(-0.5)*SQRT(CST(7))*(CST(2.5)*(1-2*r2)+CST(0.5));
			break;
		case 2:
			R = SQRT(CST(7))*r2;
			break;
		} break;
	case 3:
		switch (n)
		{
		case 1:
			R = CST(-1.5)*r*(CST(3.5)*(1-2*r2)+CST(1.5));
			break;
		case 3:
			R = 3*r2*r;
		} break;
	case 4:
		switch (n)
		{
		case 0:
			R = SQRT(CST(11))*((63*r2*r2/8)-(35*r2/4)+(CST(15)/CST(8)));
			break;
		case 2:
			R = CST(-0.5)*SQRT(CST(11))*r2*(CST(4.5)*(1-2*r2)+CST(2.5));
			break;
		case 4:
			R = SQRT(CST(11))*r2*r2;
			break;
		} break;
	case 5:
		switch (n)
		{
		case 1:
			R = SQRT(CST(13))*r*((99*r2*r2/8)-(63*r2/4)+(CST(35)/CST(8)));
			break;
		case 3:
			R = CST(-0.5)*SQRT(CST(13))*r2*r*(CST(5.5)*(1-2*r2)+CST(3.5));
			break;
		} break;
	}

	// Spherical harmonic
	PrecisionType Y=CST(0.0);

	switch (l2)
	{
	case 0:
		Y = (CST(1.0)/CST(2.0))*SQRT((PrecisionType) CST(1.0)/_PI_);
		break;
	case 1:
		switch (m)
		{
		case -1:
			Y = SQRT(CST(3.0)/(CST(4.0)*_PI_))*yr;
			break;
		case 0:
			Y = SQRT(CST(3.0)/(CST(4.0)*_PI_))*zr;
			break;
		case 1:
			Y = SQRT(CST(3.0)/(CST(4.0)*_PI_))*xr;
			break;
		} break;
	case 2:
		switch (m)
		{
		case -2:
			Y = SQRT(CST(15.0)/(CST(4.0)*_PI_))*xr*yr;
			break;
		case -1:
			Y = SQRT(CST(15.0)/(CST(4.0)*_PI_))*zr*yr;
			break;
		case 0:
			Y = SQRT(CST(5.0)/(CST(16.0)*_PI_))*(-xr2-yr2+CST(2.0)*zr2);
			break;
		case 1:
			Y = SQRT(CST(15.0)/(CST(4.0)*_PI_))*xr*zr;
			break;
		case 2:
			Y = SQRT(CST(15.0)/(CST(16.0)*_PI_))*(xr2-yr2);
			break;
		} break;
	case 3:
		switch (m)
		{
		case -3:
			Y = SQRT(CST(35.0)/(CST(16.0)*CST(2.0)*_PI_))*yr*(CST(3.0)*xr2-yr2);
			break;
		case -2:
			Y = SQRT(CST(105.0)/(CST(4.0)*_PI_))*zr*yr*xr;
			break;
		case -1:
			Y = SQRT(CST(21.0)/(CST(16.0)*CST(2.0)*_PI_))*yr*(CST(4.0)*zr2-xr2-yr2);
			break;
		case 0:
			Y = SQRT(CST(7.0)/(CST(16.0)*_PI_))*zr*(CST(2.0)*zr2-CST(3.0)*xr2-CST(3.0)*yr2);
			break;
		case 1:
			Y = SQRT(CST(21.0)/(CST(16.0)*CST(2.0)*_PI_))*xr*(CST(4.0)*zr2-xr2-yr2);
			break;
		case 2:
			Y = SQRT(CST(105.0)/(CST(16.0)*_PI_))*zr*(xr2-yr2);
			break;
		case 3:
			Y = SQRT(CST(35.0)/(CST(16.0)*CST(2.0)*_PI_))*xr*(xr2-CST(3.0)*yr2);
			break;
		} break;
	case 4:
		switch (m)
		{
		case -4:
			Y = SQRT((CST(35.0)*CST(9.0))/(CST(16.0)*_PI_))*yr*xr*(xr2-yr2);
			break;
		case -3:
			Y = SQRT((CST(9.0)*CST(35.0))/(CST(16.0)*CST(2.0)*_PI_))*yr*zr*(CST(3.0)*xr2-yr2);
			break;
		case -2:
			Y = SQRT((CST(9.0)*CST(5.0))/(CST(16.0)*_PI_))*yr*xr*(CST(7.0)*zr2-(xr2+yr2+zr2));
			break;
		case -1:
			Y = SQRT((CST(9.0)*CST(5.0))/(CST(16.0)*CST(2.0)*_PI_))*yr*zr*(CST(7.0)*zr2-CST(3.0)*(xr2+yr2+zr2));
			break;
		case 0:
			Y = SQRT(CST(9.0)/(CST(16.0)*CST(16.0)*_PI_))*(CST(35.0)*zr2*zr2-CST(30.0)*zr2+CST(3.0));
			break;
		case 1:
			Y = SQRT((CST(9.0)*CST(5.0))/(CST(16.0)*CST(2.0)*_PI_))*xr*zr*(CST(7.0)*zr2-CST(3.0)*(xr2+yr2+zr2));
			break;
		case 2:
			Y = SQRT((CST(9.0)*CST(5.0))/(CST(8.0)*CST(8.0)*_PI_))*(xr2-yr2)*(CST(7.0)*zr2-(xr2+yr2+zr2));
			break;
		case 3:
			Y = SQRT((CST(9.0)*CST(35.0))/(CST(16.0)*CST(2.0)*_PI_))*xr*zr*(xr2-CST(3.0)*yr2);
			break;
		case 4:
			Y = SQRT((CST(9.0)*CST(35.0))/(CST(16.0)*CST(16.0)*_PI_))*(xr2*(xr2-CST(3.0)*yr2)-yr2*(CST(3.0)*xr2-yr2));
			break;
		} break;
	case 5:
		switch (m)
		{
		case -5:
			Y = (CST(3.0)/CST(16.0))*SQRT(CST(77.0)/(CST(2.0)*_PI_))*sint2*sint2*sint*SIN(CST(5.0)*phi);
			break;
		case -4:
			Y = (CST(3.0)/CST(8.0))*SQRT(CST(385.0)/(CST(2.0)*_PI_))*sint2*sint2*SIN(CST(4.0)*phi);
			break;
		case -3:
			Y = (CST(1.0)/CST(16.0))*SQRT(CST(385.0)/(CST(2.0)*_PI_))*sint2*sint*(CST(9.0)*cost2-CST(1.0))*SIN(CST(3.0)*phi);
			break;
		case -2:
			Y = (CST(1.0)/CST(4.0))*SQRT(CST(1155.0)/(CST(4.0)*_PI_))*sint2*(CST(3.0)*cost2*cost-cost)*SIN(CST(2.0)*phi);
			break;
		case -1:
			Y = (CST(1.0)/CST(8.0))*SQRT(CST(165.0)/(CST(4.0)*_PI_))*sint*(CST(21.0)*cost2*cost2-CST(14.0)*cost2+1)*SIN(phi);
			break;
		case 0:
			Y = (CST(1.0)/CST(16.0))*SQRT(CST(11.0)/_PI_)*(CST(63.0)*cost2*cost2*cost-CST(70.0)*cost2*cost+CST(15.0)*cost);
			break;
		case 1:
			Y = (CST(1.0)/CST(8.0))*SQRT(CST(165.0)/(CST(4.0)*_PI_))*sint*(CST(21.0)*cost2*cost2-CST(14.0)*cost2+1)*COS(phi);
			break;
		case 2:
			Y = (CST(1.0)/CST(4.0))*SQRT(CST(1155.0)/(CST(4.0)*_PI_))*sint2*(CST(3.0)*cost2*cost-cost)*COS(CST(2.0)*phi);
			break;
		case 3:
			Y = (CST(1.0)/CST(16.0))*SQRT(CST(385.0)/(CST(2.0)*_PI_))*sint2*sint*(CST(9.0)*cost2-CST(1.0))*COS(CST(3.0)*phi);
			break;
		case 4:
			Y = (CST(3.0)/CST(8.0))*SQRT(CST(385.0)/(CST(2.0)*_PI_))*sint2*sint2*COS(CST(4.0)*phi);
			break;
		case 5:
			Y = (CST(3.0)/CST(16.0))*SQRT(CST(77.0)/(CST(2.0)*_PI_))*sint2*sint2*sint*COS(CST(5.0)*phi);
			break;
		}break;
	}

	return R*Y;
}


#endif //CUDA_VOLUME_DEFORM_SPH_CU
