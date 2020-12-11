#ifndef CUDA_VOLUME_DEFORM_SPH_CU
#define CUDA_VOLUME_DEFORM_SPH_CU
#include "cuda_volume_deform_sph.h"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include <math_constants.h>

#include <thrust/device_vector.h>

// CUDA kernel defines
#define BLOCK_X_DIM 8
#define BLOCK_Y_DIM 4
#define BLOCK_Z_DIM 4
#define TOTAL_BLOCK_SIZE (BLOCK_X_DIM * BLOCK_Y_DIM * BLOCK_Z_DIM)

// CUDA transform and copy kernel defines
#define COPY_BLOCK_X_DIM 32

// ImageData macros

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

#define ELEM_3D(ImD,k,i,j) \
    ((ImD).data[GET_IDX((ImD), (k), (i), (j))])

#define ELEM_3D_SHIFTED(ImD,k,i,j) \
    (ELEM_3D((ImD), (k) - (ImD).zShift, (i) - (ImD).yShift, (j) - (ImD).xShift))

#define MY_OUTSIDE(ImD,k,i,j) \
    ((j) < (ImD).xShift || (j) > (ImD).xShift + (ImD).xDim - 1 || \
     (i) < (ImD).yShift || (i) > (ImD).yShift + (ImD).yDim - 1 || \
     (k) < (ImD).zShift || (k) > (ImD).zShift + (ImD).zDim - 1)

// For certain reason this could not be done via template
#ifdef COMP_DOUBLE
__device__ double ZernikeSphericalHarmonics(int l1, int n, int l2, int m, double xr, double yr, double zr, double r);
#define my_PI CUDART_PI
#else
__device__ float ZernikeSphericalHarmonics(int l1, int n, int l2, int m, float xr, float yr, float zr, float r);
#define my_PI CUDART_PI_F
#endif

template<typename T>
__device__ T interpolatedElement3D(ImageData<T> ImD,
        T x, T y, T z, T doutside_value = 0) 
{
        int x0 = FLOOR(x);
        T fx = x - x0;
        int x1 = x0 + 1;

        int y0 = FLOOR(y);
        T fy = y - y0;
        int y1 = y0 + 1;

        int z0 = FLOOR(z);
        T fz = z - z0;
        int z1 = z0 + 1;

        T d000 = (MY_OUTSIDE(ImD, z0, y0, x0)) ?
            doutside_value : ELEM_3D_SHIFTED(ImD, z0, y0, x0);
        T d001 = (MY_OUTSIDE(ImD, z0, y0, x1)) ?
            doutside_value : ELEM_3D_SHIFTED(ImD, z0, y0, x1);
        T d010 = (MY_OUTSIDE(ImD, z0, y1, x0)) ?
            doutside_value : ELEM_3D_SHIFTED(ImD, z0, y1, x0);
        T d011 = (MY_OUTSIDE(ImD, z0, y1, x1)) ?
            doutside_value : ELEM_3D_SHIFTED(ImD, z0, y1, x1);
        T d100 = (MY_OUTSIDE(ImD, z1, y0, x0)) ?
            doutside_value : ELEM_3D_SHIFTED(ImD, z1, y0, x0);
        T d101 = (MY_OUTSIDE(ImD, z1, y0, x1)) ?
            doutside_value : ELEM_3D_SHIFTED(ImD, z1, y0, x1);
        T d110 = (MY_OUTSIDE(ImD, z1, y1, x0)) ?
            doutside_value : ELEM_3D_SHIFTED(ImD, z1, y1, x0);
        T d111 = (MY_OUTSIDE(ImD, z1, y1, x1)) ?
            doutside_value : ELEM_3D_SHIFTED(ImD, z1, y1, x1);

        T dx00 = LIN_INTERP(fx, d000, d001);
        T dx01 = LIN_INTERP(fx, d100, d101);
        T dx10 = LIN_INTERP(fx, d010, d011);
        T dx11 = LIN_INTERP(fx, d110, d111);
        T dxy0 = LIN_INTERP(fy, dx00, dx10);
        T dxy1 = LIN_INTERP(fy, dx01, dx11);

        return LIN_INTERP(fz, dxy0, dxy1);
}

template<typename T>
__global__ void computeDeform(T Rmax2, T iRmax, IROimages<T> images,
        ZSHparams zshparams, T* steps_cp, T* clnm,
        Volumes<T> volumes, DeformImages<T> deformImages, bool applyTransformation,
        bool saveDeformation, thrust::device_ptr<T> g_outArr) 
{
    __shared__ T sumArray[TOTAL_BLOCK_SIZE * 4];

    // Compute thread index in a block
    int tIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // Get physical indexes
    int kPhys = blockIdx.z * blockDim.z + threadIdx.z;
    int iPhys = blockIdx.y * blockDim.y + threadIdx.y;
    int jPhys = blockIdx.x * blockDim.x + threadIdx.x;

    // Update to logical indexes (calculations expect logical indexing)
    int k = P2L_Z_IDX(images.VR, kPhys);
    int i = P2L_Y_IDX(images.VR, iPhys);
    int j = P2L_X_IDX(images.VR, jPhys);

    T r2 = k*k + i*i + j*j;
    T rr = sqrt(r2) * iRmax;
    T gx = 0.0, gy = 0.0, gz = 0.0;

    // Try to reduce conditions (some threads in warp may end up slacking)
    if (r2 < Rmax2) {
        for (unsigned idx = 0; idx < zshparams.size; idx++) {
            if (steps_cp[idx] == 1) {
                // Save parameters of the same index next to each other
                // Constant memory? all threads in warp use the same values
                int l1 = zshparams.vL1[idx];
                int n = zshparams.vN[idx];
                int l2 = zshparams.vL2[idx];
                int m = zshparams.vM[idx];
                T zsph = ZernikeSphericalHarmonics(l1, n, l2, m,
                        j * iRmax, i * iRmax, k * iRmax, rr);

                if (rr > 0 || l2 == 0) {
                    gx += zsph * clnm[idx];
                    gy += zsph * clnm[idx + zshparams.size];
                    gz += zsph * clnm[idx + zshparams.size * 2];
                }
            }
        }
    }

    T voxelI, voxelR;
    T diff;

    T localDiff2 = 0.0, localSumVD = 0.0, localModg = 0.0, localNcount = 0.0;

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
    
    // Block reduction   
    for (int s = TOTAL_BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (tIdx < s) {
            sumArray[tIdx] += sumArray[tIdx + s];
            sumArray[tIdx + TOTAL_BLOCK_SIZE] += sumArray[tIdx + TOTAL_BLOCK_SIZE + s];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 2] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 2 + s];
            sumArray[tIdx + TOTAL_BLOCK_SIZE * 3] += sumArray[tIdx + TOTAL_BLOCK_SIZE * 3 + s];
        }
        __syncthreads();
    }

    // Save values to the global memory for later
    if (tIdx == 0) {
        int bIdx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
        int TOTAL_GRID_SIZE = gridDim.x * gridDim.y * gridDim.z;
        g_outArr[bIdx] = sumArray[0];
        g_outArr[bIdx + TOTAL_GRID_SIZE] = sumArray[TOTAL_BLOCK_SIZE];
        g_outArr[bIdx + TOTAL_GRID_SIZE * 2] = sumArray[TOTAL_BLOCK_SIZE * 2];
        g_outArr[bIdx + TOTAL_GRID_SIZE * 3] = sumArray[TOTAL_BLOCK_SIZE * 3];
    }

    if (saveDeformation) {
        ELEM_3D(deformImages.Gx, kPhys, iPhys, jPhys) = gx;
        ELEM_3D(deformImages.Gy, kPhys, iPhys, jPhys) = gy;
        ELEM_3D(deformImages.Gz, kPhys, iPhys, jPhys) = gz;
    }
}



// There is nothing interesting down here.



// Function redefinition
#ifdef COMP_DOUBLE

__device__ double ZernikeSphericalHarmonics(int l1, int n, int l2, int m, double xr, double yr, double zr, double r)
{
	// General variables
	double r2=r*r,xr2=xr*xr,yr2=yr*yr,zr2=zr*zr;

	//Variables needed for l>=5
	double tht=0.0,phi=0.0,cost=0.0,sint=0.0,cost2=0.0,sint2=0.0;
	if (l2>=5)
	{
		tht = atan2(yr,xr);
		phi = atan2(zr,sqrt(xr2 + yr2));
		sint = sin(phi); cost = cos(tht);
		sint2 = sint*sint; cost2 = cost*cost;
	}

	// Zernike polynomial
	double R=0.0;

	switch (l1)
	{
	case 0:
		R = sqrt((double) 3);
		break;
	case 1:
		R = sqrt((double) 5)*r;
		break;
	case 2:
		switch (n)
		{
		case 0:
			R = -0.5*sqrt((double) 7)*(2.5*(1-2*r2)+0.5);
			break;
		case 2:
			R = sqrt((double) 7)*r2;
			break;
		} break;
	case 3:
		switch (n)
		{
		case 1:
			R = -1.5*r*(3.5*(1-2*r2)+1.5);
			break;
		case 3:
			R = 3*r2*r;
		} break;
	case 4:
		switch (n)
		{
		case 0:
			R = sqrt((double) 11)*((63*r2*r2/8)-(35*r2/4)+(15/8));
			break;
		case 2:
			R = -0.5*sqrt((double) 11)*r2*(4.5*(1-2*r2)+2.5);
			break;
		case 4:
			R = sqrt((double) 11)*r2*r2;
			break;
		} break;
	case 5:
		switch (n)
		{
		case 1:
			R = sqrt((double) 13)*r*((99*r2*r2/8)-(63*r2/4)+(35/8));
			break;
		case 3:
			R = -0.5*sqrt((double) 13)*r2*r*(5.5*(1-2*r2)+3.5);
			break;
		} break;
	}

	// Spherical harmonic
	double Y=0.0;

	switch (l2)
	{
	case 0:
		Y = (1.0/2.0)*sqrt((double) 1.0/my_PI);
		break;
	case 1:
		switch (m)
		{
		case -1:
			Y = sqrt(3.0/(4.0*my_PI))*yr;
			break;
		case 0:
			Y = sqrt(3.0/(4.0*my_PI))*zr;
			break;
		case 1:
			Y = sqrt(3.0/(4.0*my_PI))*xr;
			break;
		} break;
	case 2:
		switch (m)
		{
		case -2:
			Y = sqrt(15.0/(4.0*my_PI))*xr*yr;
			break;
		case -1:
			Y = sqrt(15.0/(4.0*my_PI))*zr*yr;
			break;
		case 0:
			Y = sqrt(5.0/(16.0*my_PI))*(-xr2-yr2+2.0*zr2);
			break;
		case 1:
			Y = sqrt(15.0/(4.0*my_PI))*xr*zr;
			break;
		case 2:
			Y = sqrt(15.0/(16.0*my_PI))*(xr2-yr2);
			break;
		} break;
	case 3:
		switch (m)
		{
		case -3:
			Y = sqrt(35.0/(16.0*2.0*my_PI))*yr*(3.0*xr2-yr2);
			break;
		case -2:
			Y = sqrt(105.0/(4.0*my_PI))*zr*yr*xr;
			break;
		case -1:
			Y = sqrt(21.0/(16.0*2.0*my_PI))*yr*(4.0*zr2-xr2-yr2);
			break;
		case 0:
			Y = sqrt(7.0/(16.0*my_PI))*zr*(2.0*zr2-3.0*xr2-3.0*yr2);
			break;
		case 1:
			Y = sqrt(21.0/(16.0*2.0*my_PI))*xr*(4.0*zr2-xr2-yr2);
			break;
		case 2:
			Y = sqrt(105.0/(16.0*my_PI))*zr*(xr2-yr2);
			break;
		case 3:
			Y = sqrt(35.0/(16.0*2.0*my_PI))*xr*(xr2-3.0*yr2);
			break;
		} break;
	case 4:
		switch (m)
		{
		case -4:
			Y = sqrt((35.0*9.0)/(16.0*my_PI))*yr*xr*(xr2-yr2);
			break;
		case -3:
			Y = sqrt((9.0*35.0)/(16.0*2.0*my_PI))*yr*zr*(3.0*xr2-yr2);
			break;
		case -2:
			Y = sqrt((9.0*5.0)/(16.0*my_PI))*yr*xr*(7.0*zr2-(xr2+yr2+zr2));
			break;
		case -1:
			Y = sqrt((9.0*5.0)/(16.0*2.0*my_PI))*yr*zr*(7.0*zr2-3.0*(xr2+yr2+zr2));
			break;
		case 0:
			Y = sqrt(9.0/(16.0*16.0*my_PI))*(35.0*zr2*zr2-30.0*zr2+3.0);
			break;
		case 1:
			Y = sqrt((9.0*5.0)/(16.0*2.0*my_PI))*xr*zr*(7.0*zr2-3.0*(xr2+yr2+zr2));
			break;
		case 2:
			Y = sqrt((9.0*5.0)/(8.0*8.0*my_PI))*(xr2-yr2)*(7.0*zr2-(xr2+yr2+zr2));
			break;
		case 3:
			Y = sqrt((9.0*35.0)/(16.0*2.0*my_PI))*xr*zr*(xr2-3.0*yr2);
			break;
		case 4:
			Y = sqrt((9.0*35.0)/(16.0*16.0*my_PI))*(xr2*(xr2-3.0*yr2)-yr2*(3.0*xr2-yr2));
			break;
		} break;
	case 5:
		switch (m)
		{
		case -5:
			Y = (3.0/16.0)*sqrt(77.0/(2.0*my_PI))*sint2*sint2*sint*sin(5.0*phi);
			break;
		case -4:
			Y = (3.0/8.0)*sqrt(385.0/(2.0*my_PI))*sint2*sint2*sin(4.0*phi);
			break;
		case -3:
			Y = (1.0/16.0)*sqrt(385.0/(2.0*my_PI))*sint2*sint*(9.0*cost2-1.0)*sin(3.0*phi);
			break;
		case -2:
			Y = (1.0/4.0)*sqrt(1155.0/(4.0*my_PI))*sint2*(3.0*cost2*cost-cost)*sin(2.0*phi);
			break;
		case -1:
			Y = (1.0/8.0)*sqrt(165.0/(4.0*my_PI))*sint*(21.0*cost2*cost2-14.0*cost2+1)*sin(phi);
			break;
		case 0:
			Y = (1.0/16.0)*sqrt(11.0/my_PI)*(63.0*cost2*cost2*cost-70.0*cost2*cost+15.0*cost);
			break;
		case 1:
			Y = (1.0/8.0)*sqrt(165.0/(4.0*my_PI))*sint*(21.0*cost2*cost2-14.0*cost2+1)*cos(phi);
			break;
		case 2:
			Y = (1.0/4.0)*sqrt(1155.0/(4.0*my_PI))*sint2*(3.0*cost2*cost-cost)*cos(2.0*phi);
			break;
		case 3:
			Y = (1.0/16.0)*sqrt(385.0/(2.0*my_PI))*sint2*sint*(9.0*cost2-1.0)*cos(3.0*phi);
			break;
		case 4:
			Y = (3.0/8.0)*sqrt(385.0/(2.0*my_PI))*sint2*sint2*cos(4.0*phi);
			break;
		case 5:
			Y = (3.0/16.0)*sqrt(77.0/(2.0*my_PI))*sint2*sint2*sint*cos(5.0*phi);
			break;
		}break;
	}

	return R*Y;
}

#else

__device__ float ZernikeSphericalHarmonics(int l1, int n, int l2, int m, float xr, float yr, float zr, float r)
{
	// General variables
	float r2=r*r,xr2=xr*xr,yr2=yr*yr,zr2=zr*zr;

	//Variables needed for l>=5
	float tht=0.0f,phi=0.0f,cost=0.0f,sint=0.0f,cost2=0.0f,sint2=0.0f;
	if (l2>=5)
	{
		tht = atan2f(yr,xr);
		phi = atan2f(zr,sqrtf(xr2 + yr2));
		sint = sinf(phi); cost = cosf(tht);
		sint2 = sint*sint; cost2 = cost*cost;
	}

	// Zernike polynomial
	float R=0.0f;

	switch (l1)
	{
	case 0:
		R = sqrtf((float) 3);
		break;
	case 1:
		R = sqrtf((float) 5)*r;
		break;
	case 2:
		switch (n)
		{
		case 0:
			R = -0.5f*sqrtf((float) 7)*(2.5f*(1-2*r2)+0.5f);
			break;
		case 2:
			R = sqrtf((float) 7)*r2;
			break;
		} break;
	case 3:
		switch (n)
		{
		case 1:
			R = -1.5f*r*(3.5f*(1-2*r2)+1.5f);
			break;
		case 3:
			R = 3*r2*r;
		} break;
	case 4:
		switch (n)
		{
		case 0:
			R = sqrtf((float) 11)*((63*r2*r2/8)-(35*r2/4)+(15/8));
			break;
		case 2:
			R = -0.5f*sqrtf((float) 11)*r2*(4.5f*(1-2*r2)+2.5f);
			break;
		case 4:
			R = sqrtf((float) 11)*r2*r2;
			break;
		} break;
	case 5:
		switch (n)
		{
		case 1:
			R = sqrtf((float) 13)*r*((99*r2*r2/8)-(63*r2/4)+(35/8));
			break;
		case 3:
			R = -0.5f*sqrtf((float) 13)*r2*r*(5.5f*(1-2*r2)+3.5f);
			break;
		} break;
	}

	// Spherical harmonic
	float Y=0.0f;

	switch (l2)
	{
	case 0:
		Y = (1.0f/2.0f)*sqrtf((float) 1.0f/my_PI);
		break;
	case 1:
		switch (m)
		{
		case -1:
			Y = sqrtf(3.0f/(4.0f*my_PI))*yr;
			break;
		case 0:
			Y = sqrtf(3.0f/(4.0f*my_PI))*zr;
			break;
		case 1:
			Y = sqrtf(3.0f/(4.0f*my_PI))*xr;
			break;
		} break;
	case 2:
		switch (m)
		{
		case -2:
			Y = sqrtf(15.0f/(4.0f*my_PI))*xr*yr;
			break;
		case -1:
			Y = sqrtf(15.0f/(4.0f*my_PI))*zr*yr;
			break;
		case 0:
			Y = sqrtf(5.0f/(16.0f*my_PI))*(-xr2-yr2+2.0f*zr2);
			break;
		case 1:
			Y = sqrtf(15.0f/(4.0f*my_PI))*xr*zr;
			break;
		case 2:
			Y = sqrtf(15.0f/(16.0f*my_PI))*(xr2-yr2);
			break;
		} break;
	case 3:
		switch (m)
		{
		case -3:
			Y = sqrtf(35.0f/(16.0f*2.0f*my_PI))*yr*(3.0f*xr2-yr2);
			break;
		case -2:
			Y = sqrtf(105.0f/(4.0f*my_PI))*zr*yr*xr;
			break;
		case -1:
			Y = sqrtf(21.0f/(16.0f*2.0f*my_PI))*yr*(4.0f*zr2-xr2-yr2);
			break;
		case 0:
			Y = sqrtf(7.0f/(16.0f*my_PI))*zr*(2.0f*zr2-3.0f*xr2-3.0f*yr2);
			break;
		case 1:
			Y = sqrtf(21.0f/(16.0f*2.0f*my_PI))*xr*(4.0f*zr2-xr2-yr2);
			break;
		case 2:
			Y = sqrtf(105.0f/(16.0f*my_PI))*zr*(xr2-yr2);
			break;
		case 3:
			Y = sqrtf(35.0f/(16.0f*2.0f*my_PI))*xr*(xr2-3.0f*yr2);
			break;
		} break;
	case 4:
		switch (m)
		{
		case -4:
			Y = sqrtf((35.0f*9.0f)/(16.0f*my_PI))*yr*xr*(xr2-yr2);
			break;
		case -3:
			Y = sqrtf((9.0f*35.0f)/(16.0f*2.0f*my_PI))*yr*zr*(3.0f*xr2-yr2);
			break;
		case -2:
			Y = sqrtf((9.0f*5.0f)/(16.0f*my_PI))*yr*xr*(7.0f*zr2-(xr2+yr2+zr2));
			break;
		case -1:
			Y = sqrtf((9.0f*5.0f)/(16.0f*2.0f*my_PI))*yr*zr*(7.0f*zr2-3.0f*(xr2+yr2+zr2));
			break;
		case 0:
			Y = sqrtf(9.0f/(16.0f*16.0f*my_PI))*(35.0f*zr2*zr2-30.0f*zr2+3.0f);
			break;
		case 1:
			Y = sqrtf((9.0f*5.0f)/(16.0f*2.0f*my_PI))*xr*zr*(7.0f*zr2-3.0f*(xr2+yr2+zr2));
			break;
		case 2:
			Y = sqrtf((9.0f*5.0f)/(8.0f*8.0f*my_PI))*(xr2-yr2)*(7.0f*zr2-(xr2+yr2+zr2));
			break;
		case 3:
			Y = sqrtf((9.0f*35.0f)/(16.0f*2.0f*my_PI))*xr*zr*(xr2-3.0f*yr2);
			break;
		case 4:
			Y = sqrtf((9.0f*35.0f)/(16.0f*16.0f*my_PI))*(xr2*(xr2-3.0f*yr2)-yr2*(3.0f*xr2-yr2));
			break;
		} break;
	case 5:
		switch (m)
		{
		case -5:
			Y = (3.0f/16.0f)*sqrtf(77.0f/(2.0f*my_PI))*sint2*sint2*sint*sinf(5.0f*phi);
			break;
		case -4:
			Y = (3.0f/8.0f)*sqrtf(385.0f/(2.0f*my_PI))*sint2*sint2*sinf(4.0f*phi);
			break;
		case -3:
			Y = (1.0f/16.0f)*sqrtf(385.0f/(2.0f*my_PI))*sint2*sint*(9.0f*cost2-1.0f)*sinf(3.0f*phi);
			break;
		case -2:
			Y = (1.0f/4.0f)*sqrtf(1155.0f/(4.0f*my_PI))*sint2*(3.0f*cost2*cost-cost)*sinf(2.0f*phi);
			break;
		case -1:
			Y = (1.0f/8.0f)*sqrtf(165.0f/(4.0f*my_PI))*sint*(21.0f*cost2*cost2-14.0f*cost2+1)*sinf(phi);
			break;
		case 0:
			Y = (1.0f/16.0f)*sqrtf(11.0f/my_PI)*(63.0f*cost2*cost2*cost-70.0f*cost2*cost+15.0f*cost);
			break;
		case 1:
			Y = (1.0f/8.0f)*sqrtf(165.0f/(4.0f*my_PI))*sint*(21.0f*cost2*cost2-14.0f*cost2+1)*cosf(phi);
			break;
		case 2:
			Y = (1.0f/4.0f)*sqrtf(1155.0f/(4.0f*my_PI))*sint2*(3.0f*cost2*cost-cost)*cosf(2.0f*phi);
			break;
		case 3:
			Y = (1.0f/16.0f)*sqrtf(385.0f/(2.0f*my_PI))*sint2*sint*(9.0f*cost2-1.0f)*cosf(3.0f*phi);
			break;
		case 4:
			Y = (3.0f/8.0f)*sqrtf(385.0f/(2.0f*my_PI))*sint2*sint2*cosf(4.0f*phi);
			break;
		case 5:
			Y = (3.0f/16.0f)*sqrtf(77.0f/(2.0f*my_PI))*sint2*sint2*sint*cosf(5.0f*phi);
			break;
		}break;
	}

	return R*Y;
}

#endif//COMP_DOUBLE


#endif //CUDA_VOLUME_DEFORM_SPH_CU
