#ifndef CUDA_VOLUME_DEFORM_SPH_CU
#define CUDA_VOLUME_DEFORM_SPH_CU
//#include "cuda_volume_deform_sph.h"

// Compilation settings

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

#if USE_SCATTERED_ZSH_CLNM == 1
using ClnmType = PrecisionType*;
using ZshParamsType =
    struct ZSHparams { int *vL1, *vN, *vL2, *vM; unsigned size; };
#else
using ClnmType = PrecisionType3*;
using ZshParamsType = int4*;
#endif// USE_SCATTERED_ZSH_CLNM

// Compilation settings - end

// Define used data structures

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

// CUDA kernel defines
#define BLOCK_X_DIM 8
#define BLOCK_Y_DIM 4
#define BLOCK_Z_DIM 4
#define TOTAL_BLOCK_SIZE (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM)

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
#define ELEM_3D(ImD,k,i,j) \
    ((ImD).data[GET_IDX((ImD), (k), (i), (j))])

#define ELEM_3D_SHIFTED(ImD,k,i,j) \
    (ELEM_3D((ImD), (k) - (ImD).zShift, (i) - (ImD).yShift, (j) - (ImD).xShift))

// Utility macros
#define MY_OUTSIDE(ImD,k,i,j) \
    ((j) < (ImD).xShift || (j) > (ImD).xShift + (ImD).xDim - 1 || \
     (i) < (ImD).yShift || (i) > (ImD).yShift + (ImD).yDim - 1 || \
     (k) < (ImD).zShift || (k) > (ImD).zShift + (ImD).zDim - 1)

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
        ZshParamsType zshparams,
        ClnmType clnm,
        ZshParamsType zshparamsSCATTERED,// just for tuning
        ClnmType clnmSCATTERED,// just for tuning
        int steps,
        Volumes volumes,
        DeformImages deformImages,
        bool applyTransformation,
        bool saveDeformation,
        PrecisionType* g_outArr
        ) 
{
    __shared__ PrecisionType sumArray[TOTAL_BLOCK_SIZE * 4];

    // Compute thread index in a block
    unsigned tIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // Get physical indexes
    int kPhys = blockIdx.z * blockDim.z + threadIdx.z;
    int iPhys = blockIdx.y * blockDim.y + threadIdx.y;
    int jPhys = blockIdx.x * blockDim.x + threadIdx.x;

    // Update to logical indexes (calculations expect logical indexing)
    int k = P2L_Z_IDX(images.VR, kPhys);
    int i = P2L_Y_IDX(images.VR, iPhys);
    int j = P2L_X_IDX(images.VR, jPhys);

    PrecisionType r2 = k*k + i*i + j*j;
    PrecisionType rr = SQRT(r2) * iRmax;
    PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;

    if (r2 < Rmax2) {
        for (int idx = 0; idx < steps; idx++) {
#if USE_SCATTERED_ZSH_CLNM == 1
            int l1 = zshparamsSCATTERED.vL1[idx];
            int n = zshparamsSCATTERED.vN[idx];
            int l2 = zshparamsSCATTERED.vL2[idx];
            int m = zshparamsSCATTERED.vM[idx];
#else
            int l1 = zshparams[idx].w;
            int n = zshparams[idx].x;
            int l2 = zshparams[idx].y;
            int m = zshparams[idx].z;
#endif
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
#if USE_SCATTERED_ZSH_CLNM == 1
                gx += zsph * clnmSCATTERED[idx];
                gy += zsph * clnmSCATTERED[idx + zshparamsSCATTERED.size];
                gz += zsph * clnmSCATTERED[idx + zshparamsSCATTERED.size * 2];
#else
                gx += zsph * clnm[idx].x;
                gy += zsph * clnm[idx].y;
                gz += zsph * clnm[idx].z;
#endif
            }
        }
    }

    PrecisionType voxelI, voxelR;
    PrecisionType diff;

    PrecisionType localDiff2 = 0.0, localSumVD = 0.0, localModg = 0.0, localNcount = 0.0;

    if (applyTransformation) {
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

    for (unsigned idv = 0; idv < volumes.size; idv++) {
        voxelR = ELEM_3D(volumes.R[idv], kPhys, iPhys, jPhys);
        voxelI = interpolatedElement3D(volumes.I[idv], j + gx, i + gy, k + gz);

        if (voxelI >= 0.0)
            localSumVD += voxelI;

        diff = voxelR - voxelI;
        localDiff2 += diff * diff;
        localModg += gx*gx + gy*gy + gz*gz;
        localNcount++;
    }

    sumArray[tIdx] = localDiff2;
    sumArray[tIdx + TOTAL_BLOCK_SIZE] = localSumVD;
    sumArray[tIdx + TOTAL_BLOCK_SIZE * 2] = localModg;
    sumArray[tIdx + TOTAL_BLOCK_SIZE * 3] = localNcount;

    __syncthreads();
#if USE_NAIVE_BLOCK_REDUCTION == 1
    // Block reduction   
    for (unsigned s = TOTAL_BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (tIdx < s) {
            sumArray[tIdx] += sumArray[tIdx + s];
            sumArray[tIdx + TOTAL_BLOCK_SIZE] += sumArray[tIdx + TOTAL_BLOCK_SIZE + s];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 2] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 2 + s];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 3] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 3 + s];
        }
        __syncthreads();
    }
#else
    //TODO preprocess variable sizes

    // Block reductions
    if (TOTAL_BLOCK_SIZE >= 512) {
        if (tIdx < 256) {
            sumArray[tIdx] += sumArray[tIdx + 256];
            sumArray[tIdx + TOTAL_BLOCK_SIZE] += sumArray[tIdx + TOTAL_BLOCK_SIZE + 256];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 2] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 2 + 256];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 3] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 3 + 256];
        }
        __syncthreads();
    }
    if (TOTAL_BLOCK_SIZE >= 256) {
        if (tIdx < 128) {
            sumArray[tIdx] += sumArray[tIdx + 128];
            sumArray[tIdx + TOTAL_BLOCK_SIZE] += sumArray[tIdx + TOTAL_BLOCK_SIZE + 128];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 2] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 2 + 128];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 3] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 3 + 128];
        }
        __syncthreads();
    }
    if (TOTAL_BLOCK_SIZE >= 128) {
        if (tIdx < 64) {
            sumArray[tIdx] += sumArray[tIdx + 64];
            sumArray[tIdx + TOTAL_BLOCK_SIZE] += sumArray[tIdx + TOTAL_BLOCK_SIZE + 64];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 2] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 2 + 64];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 3] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 3 + 64];
        }
        __syncthreads();
    }
    // Last warp reduction
    if (tIdx < 32) {
        // Recycle registers
        localDiff2 = sumArray[tIdx] + sumArray[tIdx + 32];
        localSumVD = sumArray[tIdx + TOTAL_BLOCK_SIZE] + sumArray[tIdx + TOTAL_BLOCK_SIZE + 32];
        localModg = sumArray[tIdx + TOTAL_BLOCK_SIZE * 2] + sumArray[tIdx + TOTAL_BLOCK_SIZE * 2 + 32];
        localNcount = sumArray[tIdx + TOTAL_BLOCK_SIZE * 3] + sumArray[tIdx + TOTAL_BLOCK_SIZE * 3 + 32];
        // Reduce warp
        for (int offset = 32 / 2; offset > 0; offset >>= 1) {
            localDiff2 += __shfl_down_sync(0xFFFFFFFF, localDiff2, offset);
            localSumVD += __shfl_down_sync(0xFFFFFFFF, localSumVD, offset);
            localModg += __shfl_down_sync(0xFFFFFFFF, localModg, offset);
            localNcount += __shfl_down_sync(0xFFFFFFFF, localNcount, offset);
        }
    }
#endif

    // Save values to the global memory for later
    if (tIdx == 0) {
        unsigned bIdx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
        unsigned TOTAL_GRID_SIZE = gridDim.x * gridDim.y * gridDim.z;
#if USE_NAIVE_BLOCK_REDUCTION == 1
        g_outArr[bIdx] = sumArray[0];
        g_outArr[bIdx + TOTAL_GRID_SIZE] = sumArray[TOTAL_BLOCK_SIZE];
        g_outArr[bIdx + TOTAL_GRID_SIZE * 2] = sumArray[TOTAL_BLOCK_SIZE * 2];
        g_outArr[bIdx + TOTAL_GRID_SIZE * 3] = sumArray[TOTAL_BLOCK_SIZE * 3];
#else
        // Resulting values are in registers local* => no need to go into shared mem
        g_outArr[bIdx] = localDiff2;
        g_outArr[bIdx + TOTAL_GRID_SIZE] = localSumVD;
        g_outArr[bIdx + TOTAL_GRID_SIZE * 2] = localModg;
        g_outArr[bIdx + TOTAL_GRID_SIZE * 3] = localNcount;
#endif
    }

    if (saveDeformation) {
        ELEM_3D(deformImages.Gx, kPhys, iPhys, jPhys) = gx;
        ELEM_3D(deformImages.Gy, kPhys, iPhys, jPhys) = gy;
        ELEM_3D(deformImages.Gz, kPhys, iPhys, jPhys) = gz;
    }
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

        PrecisionType d000 = (MY_OUTSIDE(ImD, z0, y0, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z0, y0, x0);
        PrecisionType d001 = (MY_OUTSIDE(ImD, z0, y0, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z0, y0, x1);
        PrecisionType d010 = (MY_OUTSIDE(ImD, z0, y1, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z0, y1, x0);
        PrecisionType d011 = (MY_OUTSIDE(ImD, z0, y1, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z0, y1, x1);
        PrecisionType d100 = (MY_OUTSIDE(ImD, z1, y0, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z1, y0, x0);
        PrecisionType d101 = (MY_OUTSIDE(ImD, z1, y0, x1)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z1, y0, x1);
        PrecisionType d110 = (MY_OUTSIDE(ImD, z1, y1, x0)) ?
            outside_value : ELEM_3D_SHIFTED(ImD, z1, y1, x0);
        PrecisionType d111 = (MY_OUTSIDE(ImD, z1, y1, x1)) ?
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
