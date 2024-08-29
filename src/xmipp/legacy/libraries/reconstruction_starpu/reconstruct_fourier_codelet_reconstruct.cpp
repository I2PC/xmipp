/***************************************************************************
 *
 * Authors:     David Strelak (davidstrelak@gmail.com)
 *              Jan Polak (456647@mail.muni.cz)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include <atomic>
#include <core/multidim_array.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <reconstruction/reconstruct_fourier_projection_traverse_space.h>

#include <reconstruction_cuda/cuda_basic_math.h>
#include <reconstruction_cuda/cuda_xmipp_utils.h>
#include <reconstruction_cuda/cuda_asserts.h>

#include <cuda_runtime_api.h>
#include <starpu.h>

#include "reconstruct_fourier_codelets.h"
#include "reconstruct_fourier_defines.h"

// ======================================= Fields and their manipulation =====================================

#if SHARED_BLOB_TABLE
__shared__ float BLOB_TABLE[BLOB_TABLE_SIZE_SQRT];
#endif

#if SHARED_IMG
__shared__ Point3D<float> SHARED_AABB[2];
extern __shared__ float2 IMG[];
#endif

// NOTE(jp): These are all used exclusively in this file. It was originally in constant storage, so it has been kept there.
// StarPU would probably like to put it in codelet arguments, but that would uglify the code a lot and I am not sure about performance impact of that.
struct CodeletConstants {
	int cMaxVolumeIndexX;
	int cMaxVolumeIndexYZ;
	float cBlobRadius;
	float cOneOverBlobRadiusSqr;
	float cBlobAlpha;
	float cIw0;
	float cIDeltaSqrt;
	float cOneOverBessiOrderAlpha;
};

__device__ __constant__ CodeletConstants gpuC;
CodeletConstants cpuC;

static void cuda_set_constants(void *gpuConstants) {
	gpuErrchk(cudaMemcpyToSymbol(gpuC, gpuConstants, sizeof(CodeletConstants)));
}

void reconstruct_cuda_initialize_constants(
		int maxVolIndexX, int maxVolIndexYZ,
		float blobRadius, float blobAlpha,
		float iDeltaSqrt, float iw0, float oneOverBessiOrderAlpha) {
	CodeletConstants constants = {0};
	constants.cMaxVolumeIndexX = maxVolIndexX;
	constants.cMaxVolumeIndexYZ = maxVolIndexYZ;
	constants.cBlobRadius = blobRadius;
	constants.cOneOverBlobRadiusSqr = 1.f / (blobRadius * blobRadius);
	constants.cBlobAlpha = blobAlpha;
	constants.cIw0 = iw0;
	constants.cIDeltaSqrt = iDeltaSqrt;
	constants.cOneOverBessiOrderAlpha = oneOverBessiOrderAlpha;

	// Fill GPU side
	// http://starpu.gforge.inria.fr/doc/html/FrequentlyAskedQuestions.html#HowToInitializeAComputationLibraryOnceForEachWorker
	starpu_execute_on_each_worker(&cuda_set_constants, &constants, STARPU_CUDA);

	// Fill CPU side
	memcpy(&cpuC, &constants, sizeof(CodeletConstants));
}

// ========================================= Bessi Kaiser math  ====================================

__host__ __device__
float bessi0Fast(float x) { // X must be <= 15
	// stable rational minimax approximations to the modified bessel functions, blair, edwards
	// from table 5
	float x2 = x*x;
	float num = -0.8436825781374849e-19f; // p11
	num = fmaf(num, x2, -0.93466495199548700e-17f); // p10
	num = fmaf(num, x2, -0.15716375332511895e-13f); // p09
	num = fmaf(num, x2, -0.42520971595532318e-11f); // p08
	num = fmaf(num, x2, -0.13704363824102120e-8f);  // p07
	num = fmaf(num, x2, -0.28508770483148419e-6f);  // p06
	num = fmaf(num, x2, -0.44322160233346062e-4f);  // p05
	num = fmaf(num, x2, -0.46703811755736946e-2f);  // p04
	num = fmaf(num, x2, -0.31112484643702141e-0f);  // p03
	num = fmaf(num, x2, -0.11512633616429962e+2f);  // p02
	num = fmaf(num, x2, -0.18720283332732112e+3f);  // p01
	num = fmaf(num, x2, -0.75281108169006924e+3f);  // p00

	float den = 1.f; // q01
	den = fmaf(den, x2, -0.75281109410939403e+3f); // q00

	return num/den;
}

__host__ __device__
float bessi0(float x) {
	float y, ax, ans;
	if ((ax = fabsf(x)) < 3.75f)
	{
		y = x / 3.75f;
		y *= y;
		ans = 1.f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
		                                                     + y * (0.2659732f + y * (0.360768e-1f + y * 0.45813e-2f)))));
	}
	else
	{
		y = 3.75f / ax;
		ans = (expf(ax) * rsqrtf(ax)) * (0.39894228f + y * (0.1328592e-1f
		                                                    + y * (0.225319e-2f + y * (-0.157565e-2f + y * (0.916281e-2f
		                                                                                                    + y * (-0.2057706e-1f + y * (0.2635537e-1f + y * (-0.1647633e-1f
		                                                                                                                                                      + y * 0.392377e-2f))))))));
	}
	return ans;
}

__host__ __device__
float bessi1(float x) {
	float ax, ans;
	float y;
	if ((ax = fabsf(x)) < 3.75f)
	{
		y = x / 3.75f;
		y *= y;
		ans = ax * (0.5f + y * (0.87890594f + y * (0.51498869f + y * (0.15084934f
		                                                              + y * (0.2658733e-1f + y * (0.301532e-2f + y * 0.32411e-3f))))));
	}
	else
	{
		y = 3.75f / ax;
		ans = 0.2282967e-1f + y * (-0.2895312e-1f + y * (0.1787654e-1f
		                                                 - y * 0.420059e-2f));
		ans = 0.39894228f + y * (-0.3988024e-1f + y * (-0.362018e-2f
		                                               + y * (0.163801e-2f + y * (-0.1031555e-1f + y * ans))));
		ans *= (expf(ax) * rsqrtf(ax));
	}
	return x < 0.0 ? -ans : ans;
}

__host__ __device__
float bessi2(float x) {
	return (x == 0) ? 0 : bessi0(x) - ((2*1) / x) * bessi1(x);
}

__host__ __device__
float bessi3(float x) {
	return (x == 0) ? 0 : bessi1(x) - ((2*2) / x) * bessi2(x);
}

__host__ __device__
float bessi4(float x) {
	return (x == 0) ? 0 : bessi2(x) - ((2*3) / x) * bessi3(x);
}

template<int order>
__host__ __device__
float kaiserValue(float r, float a) {
	const CodeletConstants& c =
#ifdef __CUDA_ARCH__
		gpuC;
#else
		cpuC;
#endif

	float w;
	float rda = r / a;
	if (rda <= 1.f)
	{
		float rdas = rda * rda;
		float arg = c.cBlobAlpha * sqrtf(1.f - rdas);
		if (order == 0)
		{
			w = bessi0(arg) * c.cOneOverBessiOrderAlpha;
		}
		else if (order == 1)
		{
			w = sqrtf (1.f - rdas);
			w *= bessi1(arg) * c.cOneOverBessiOrderAlpha;
		}
		else if (order == 2)
		{
			w = sqrtf (1.f - rdas);
			w = w * w;
			w *= bessi2(arg) * c.cOneOverBessiOrderAlpha;
		}
		else if (order == 3)
		{
			w = sqrtf (1.f - rdas);
			w = w * w * w;
			w *= bessi3(arg) * c.cOneOverBessiOrderAlpha;
		}
		else if (order == 4)
		{
			w = sqrtf (1.f - rdas);
			w = w * w * w *w;
			w *= bessi4(arg) * c.cOneOverBessiOrderAlpha;
		}
		else {
			printf("order (%d) out of range in kaiser_value(): %s, %d\n", order, __FILE__, __LINE__);
			w = 0.f;
		}
	}
	else
		w = 0.f;

	return w;
}

__host__ __device__
float kaiserValueFast(float distSqr) {
	const CodeletConstants& c =
#ifdef __CUDA_ARCH__
			gpuC;
#else
			cpuC;
#endif

	float arg = c.cBlobAlpha * sqrtf(1.f - (distSqr * c.cOneOverBlobRadiusSqr)); // alpha * sqrt(1-(dist/blobRadius^2))
	return bessi0Fast(arg) * c.cOneOverBessiOrderAlpha * c.cIw0;
}

// ========================================= Math utilities ====================================



/** Calculates Z coordinate of the point [x, y] on the plane defined by p0 (origin) and normal */
__host__ __device__
float getZ(float x, float y, const Point3D<float>& n, const Point3D<float>& p0) {
	// from a(x-x0)+b(y-y0)+c(z-z0)=0
	return (-n.x*(x-p0.x)-n.y*(y-p0.y))/n.z + p0.z;
}

/** Calculates Y coordinate of the point [x, z] on the plane defined by p0 (origin) and normal */
__host__ __device__
float getY(float x, float z, const Point3D<float>& n, const Point3D<float>& p0){
	// from a(x-x0)+b(y-y0)+c(z-z0)=0
	return (-n.x*(x-p0.x)-n.z*(z-p0.z))/n.y + p0.y;
}


/** Calculates X coordinate of the point [y, z] on the plane defined by p0 (origin) and normal */
__host__ __device__
float getX(float y, float z, const Point3D<float>& n, const Point3D<float>& p0){
	// from a(x-x0)+b(y-y0)+c(z-z0)=0
	return (-n.y*(y-p0.y)-n.z*(z-p0.z))/n.x + p0.x;
}

/** Do 3x3 x 1x3 matrix-vector multiplication */
__host__ __device__
void multiply(const float transform[3][3], Point3D<float>& inOut) {
	float tmp0 = transform[0][0] * inOut.x + transform[0][1] * inOut.y + transform[0][2] * inOut.z;
	float tmp1 = transform[1][0] * inOut.x + transform[1][1] * inOut.y + transform[1][2] * inOut.z;
	float tmp2 = transform[2][0] * inOut.x + transform[2][1] * inOut.y + transform[2][2] * inOut.z;
	inOut.x = tmp0;
	inOut.y = tmp1;
	inOut.z = tmp2;
}

/**
 * Method will rotate box using transformation matrix around center of the
 * working space
 */
__host__ __device__
void rotate(Point3D<float> box[8], const float transform[3][3]) {
	const CodeletConstants& c =
#ifdef __CUDA_ARCH__
			gpuC;
#else
			cpuC;
#endif

	for (int i = 0; i < 8; i++) {
		Point3D<float> imgPos;
		// transform current point to center
		imgPos.x = box[i].x - c.cMaxVolumeIndexX/2;
		imgPos.y = box[i].y - c.cMaxVolumeIndexYZ/2;
		imgPos.z = box[i].z - c.cMaxVolumeIndexYZ/2;
		// rotate around center
		multiply(transform, imgPos);
		// transform back just Y coordinate, since X now matches to picture and Z is irrelevant
		imgPos.y += c.cMaxVolumeIndexYZ / 2;

		box[i] = imgPos;
	}
}

// ========================================= AABBs ====================================

/** Compute Axis Aligned Bounding Box of given cuboid */
__host__ __device__
void computeAABB(Point3D<float> AABB[2], const Point3D<float> cuboid[8]) {
	AABB[0].x = AABB[0].y = AABB[0].z = INFINITY;
	AABB[1].x = AABB[1].y = AABB[1].z = -INFINITY;
	for (int i = 0; i < 8; i++) {
		Point3D<float> tmp = cuboid[i];
		if (AABB[0].x > tmp.x) AABB[0].x = tmp.x;
		if (AABB[0].y > tmp.y) AABB[0].y = tmp.y;
		if (AABB[0].z > tmp.z) AABB[0].z = tmp.z;
		if (AABB[1].x < tmp.x) AABB[1].x = tmp.x;
		if (AABB[1].y < tmp.y) AABB[1].y = tmp.y;
		if (AABB[1].z < tmp.z) AABB[1].z = tmp.z;
	}
	AABB[0].x = ceilf(AABB[0].x);
	AABB[0].y = ceilf(AABB[0].y);
	AABB[0].z = ceilf(AABB[0].z);

	AABB[1].x = floorf(AABB[1].x);
	AABB[1].y = floorf(AABB[1].y);
	AABB[1].z = floorf(AABB[1].z);
}

#if SHARED_IMG
/**
 * Method calculates an Axis Aligned Bounding Box in the image space.
 * AABB is guaranteed to be big enough that all threads in the block,
 * while processing the traverse space, will not read image data outside
 * of the AABB
 */
__device__
void calculateAABB(const RecFourierProjectionTraverseSpace* tSpace, Point3D<float> dest[2]) {
	const CodeletConstants& c =
#ifdef __CUDA_ARCH__
		gpuC;
#else
		cpuC;
#endif

	Point3D<float> box[8];
	// calculate AABB for the whole working block
	if (tSpace->XY == tSpace->dir) { // iterate XY plane
		box[0].x = box[3].x = box[4].x = box[7].x = blockIdx.x*blockDim.x - c.cBlobRadius;
		box[1].x = box[2].x = box[5].x = box[6].x = (blockIdx.x+1)*blockDim.x + c.cBlobRadius - 1.f;

		box[2].y = box[3].y = box[6].y = box[7].y = (blockIdx.y+1)*blockDim.y + c.cBlobRadius - 1.f;
		box[0].y = box[1].y = box[4].y = box[5].y = blockIdx.y*blockDim.y- c.cBlobRadius;

		box[0].z = getZ(box[0].x, box[0].y, tSpace->unitNormal, tSpace->bottomOrigin);
		box[4].z = getZ(box[4].x, box[4].y, tSpace->unitNormal, tSpace->topOrigin);

		box[3].z = getZ(box[3].x, box[3].y, tSpace->unitNormal, tSpace->bottomOrigin);
		box[7].z = getZ(box[7].x, box[7].y, tSpace->unitNormal, tSpace->topOrigin);

		box[2].z = getZ(box[2].x, box[2].y, tSpace->unitNormal, tSpace->bottomOrigin);
		box[6].z = getZ(box[6].x, box[6].y, tSpace->unitNormal, tSpace->topOrigin);

		box[1].z = getZ(box[1].x, box[1].y, tSpace->unitNormal, tSpace->bottomOrigin);
		box[5].z = getZ(box[5].x, box[5].y, tSpace->unitNormal, tSpace->topOrigin);
	} else if (tSpace->XZ == tSpace->dir) { // iterate XZ plane
		box[0].x = box[3].x = box[4].x = box[7].x = blockIdx.x*blockDim.x - c.cBlobRadius;
		box[1].x = box[2].x = box[5].x = box[6].x = (blockIdx.x+1)*blockDim.x + c.cBlobRadius - 1.f;

		box[2].z = box[3].z = box[6].z = box[7].z = (blockIdx.y+1)*blockDim.y + c.cBlobRadius - 1.f;
		box[0].z = box[1].z = box[4].z = box[5].z = blockIdx.y*blockDim.y - c.cBlobRadius;

		box[0].y = getY(box[0].x, box[0].z, tSpace->unitNormal, tSpace->bottomOrigin);
		box[4].y = getY(box[4].x, box[4].z, tSpace->unitNormal, tSpace->topOrigin);

		box[3].y = getY(box[3].x, box[3].z, tSpace->unitNormal, tSpace->bottomOrigin);
		box[7].y = getY(box[7].x, box[7].z, tSpace->unitNormal, tSpace->topOrigin);

		box[2].y = getY(box[2].x, box[2].z, tSpace->unitNormal, tSpace->bottomOrigin);
		box[6].y = getY(box[6].x, box[6].z, tSpace->unitNormal, tSpace->topOrigin);

		box[1].y = getY(box[1].x, box[1].z, tSpace->unitNormal, tSpace->bottomOrigin);
		box[5].y = getY(box[5].x, box[5].z, tSpace->unitNormal, tSpace->topOrigin);
	} else { // iterate YZ plane
		box[0].y = box[3].y = box[4].y = box[7].y = blockIdx.x*blockDim.x - c.cBlobRadius;
		box[1].y = box[2].y = box[5].y = box[6].y = (blockIdx.x+1)*blockDim.x + c.cBlobRadius - 1.f;

		box[2].z = box[3].z = box[6].z = box[7].z = (blockIdx.y+1)*blockDim.y + c.cBlobRadius - 1.f;
		box[0].z = box[1].z = box[4].z = box[5].z = blockIdx.y*blockDim.y- c.cBlobRadius;

		box[0].x = getX(box[0].y, box[0].z, tSpace->unitNormal, tSpace->bottomOrigin);
		box[4].x = getX(box[4].y, box[4].z, tSpace->unitNormal, tSpace->topOrigin);

		box[3].x = getX(box[3].y, box[3].z, tSpace->unitNormal, tSpace->bottomOrigin);
		box[7].x = getX(box[7].y, box[7].z, tSpace->unitNormal, tSpace->topOrigin);

		box[2].x = getX(box[2].y, box[2].z, tSpace->unitNormal, tSpace->bottomOrigin);
		box[6].x = getX(box[6].y, box[6].z, tSpace->unitNormal, tSpace->topOrigin);

		box[1].x = getX(box[1].y, box[1].z, tSpace->unitNormal, tSpace->bottomOrigin);
		box[5].x = getX(box[5].y, box[5].z, tSpace->unitNormal, tSpace->topOrigin);
	}
	// transform AABB to the image domain
	rotate(box, tSpace->transformInv);
	// AABB is projected on image. Create new AABB that will encompass all vertices
	computeAABB(dest, box);
}

/** Method returns true if AABB lies within the image boundaries */
__device__
bool isWithin(Point3D<float> AABB[2], int imgXSize, int imgYSize) {
	return (AABB[0].x < imgXSize)
	       && (AABB[1].x >= 0)
	       && (AABB[0].y < imgYSize)
	       && (AABB[1].y >= 0);
}
#endif

// ========================================= Actual processing ====================================

/**
 * Method will map one voxel from the temporal
 * spaces to the given projection and update temporal spaces
 * using the pixel value of the projection.
 */
__device__
void processVoxel(
		float2* tempVolumeGPU, float* tempWeightsGPU,
		int x, int y, int z,
		int xSize, int ySize,
		const float2* __restrict__ FFT,
		const RecFourierProjectionTraverseSpace* const space)
{
	Point3D<float> imgPos;
	float wBlob = 1.f;

	float dataWeight = space->weight;

	// transform current point to center
	imgPos.x = x - gpuC.cMaxVolumeIndexX/2;
	imgPos.y = y - gpuC.cMaxVolumeIndexYZ/2;
	imgPos.z = z - gpuC.cMaxVolumeIndexYZ/2;
	if (imgPos.x*imgPos.x + imgPos.y*imgPos.y + imgPos.z*imgPos.z > space->maxDistanceSqr) {
		return; // discard iterations that would access pixel with too high frequency
	}
	// rotate around center
	multiply(space->transformInv, imgPos);
	if (imgPos.x < 0.f) return; // reading outside of the image boundary. Z is always correct and Y is checked by the condition above

	// transform back and round
	// just Y coordinate needs adjusting, since X now matches to picture and Z is irrelevant
	int imgX = clamp((int)(imgPos.x + 0.5f), 0, xSize - 1);
	int imgY = clamp((int)(imgPos.y + 0.5f + gpuC.cMaxVolumeIndexYZ / 2), 0, ySize - 1);

	int index3D = z * (gpuC.cMaxVolumeIndexYZ+1) * (gpuC.cMaxVolumeIndexX+1) + y * (gpuC.cMaxVolumeIndexX+1) + x;
	int index2D = imgY * xSize + imgX;

	float weight = wBlob * dataWeight;

	// use atomic as two blocks can write to same voxel
	atomicAdd(&tempVolumeGPU[index3D].x, FFT[index2D].x * weight);
	atomicAdd(&tempVolumeGPU[index3D].y, FFT[index2D].y * weight);
	atomicAdd(&tempWeightsGPU[index3D], weight);
}

/**
 * Method will map one voxel from the temporal
 * spaces to the given projection and update temporal spaces
 * using the pixel values of the projection withing the blob distance.
 */
template<int blobOrder, bool useFastKaiser>
__device__
void processVoxelBlob(
		float2* tempVolumeGPU, float *tempWeightsGPU,
		const int x, const int y, const int z,
		const int xSize, const int ySize,
		const float2* __restrict__ FFT,
		const RecFourierProjectionTraverseSpace* const space,
		const float* blobTableSqrt,
		const int imgCacheDim)
{
	Point3D<float> imgPos;
	// transform current point to center
	imgPos.x = x - gpuC.cMaxVolumeIndexX/2;
	imgPos.y = y - gpuC.cMaxVolumeIndexYZ/2;
	imgPos.z = z - gpuC.cMaxVolumeIndexYZ/2;
	if ((imgPos.x*imgPos.x + imgPos.y*imgPos.y + imgPos.z*imgPos.z) > space->maxDistanceSqr) {
		return; // discard iterations that would access pixel with too high frequency
	}
	// rotate around center
	multiply(space->transformInv, imgPos);
	if (imgPos.x < -gpuC.cBlobRadius) return; // reading outside of the image boundary. Z is always correct and Y is checked by the condition above
	// transform back just Y coordinate, since X now matches to picture and Z is irrelevant
	imgPos.y += gpuC.cMaxVolumeIndexYZ / 2;

	// check that we don't want to collect data from far far away ...
	float radiusSqr = gpuC.cBlobRadius * gpuC.cBlobRadius;
	float zSqr = imgPos.z * imgPos.z;
	if (zSqr > radiusSqr) return;

	// create blob bounding box
	int minX = ceilf(imgPos.x - gpuC.cBlobRadius);
	int maxX = floorf(imgPos.x + gpuC.cBlobRadius);
	int minY = ceilf(imgPos.y - gpuC.cBlobRadius);
	int maxY = floorf(imgPos.y + gpuC.cBlobRadius);
	minX = fmaxf(minX, 0);
	minY = fmaxf(minY, 0);
	maxX = fminf(maxX, xSize-1);
	maxY = fminf(maxY, ySize-1);

	int index3D = z * (gpuC.cMaxVolumeIndexYZ+1) * (gpuC.cMaxVolumeIndexX+1) + y * (gpuC.cMaxVolumeIndexX+1) + x;
	float2 vol;
	float w;
	vol.x = vol.y = w = 0.f;
	float dataWeight = space->weight;

	// check which pixel in the vicinity should contribute
	for (int i = minY; i <= maxY; i++) {
		float ySqr = (imgPos.y - i) * (imgPos.y - i);
		float yzSqr = ySqr + zSqr;
		if (yzSqr > radiusSqr) continue;
		for (int j = minX; j <= maxX; j++) {
			float xD = imgPos.x - j;
			float distanceSqr = xD*xD + yzSqr;
			if (distanceSqr > radiusSqr) continue;

#if SHARED_IMG
			int index2D = (i - SHARED_AABB[0].y) * imgCacheDim + (j-SHARED_AABB[0].x); // position in img - offset of the AABB
#else
			int index2D = i * xSize + j;
#endif

#if PRECOMPUTE_BLOB_VAL
			int aux = (int) ((distanceSqr * gpuC.cIDeltaSqrt + 0.5f));
#if SHARED_BLOB_TABLE
			float wBlob = BLOB_TABLE[aux];
#else
			float wBlob = blobTableSqrt[aux];
#endif
#else
			float wBlob;
				if (useFastKaiser) {
					wBlob = kaiserValueFast(distanceSqr);
				}
				else {
					wBlob = kaiserValue<blobOrder>(sqrtf(distanceSqr), gpuC.cBlobRadius) * gpuC.cIw0;
				}
#endif
			float weight = wBlob * dataWeight;
			w += weight;
#if SHARED_IMG
			vol += IMG[index2D] * weight;
#else
			vol += FFT[index2D] * weight;
#endif
		}
	}

	// use atomic as two blocks can write to same voxel
	atomicAdd(&tempVolumeGPU[index3D].x, vol.x);
	atomicAdd(&tempVolumeGPU[index3D].y, vol.y);
	atomicAdd(&tempWeightsGPU[index3D], w);
}

/**
  * Method will process one projection image and add result to temporal
  * spaces.
  */
template<bool useFast, int blobOrder, bool useFastKaiser>
__device__
void processProjection(
		float2* tempVolumeGPU, float *tempWeightsGPU,
		const int xSize, const int ySize,
		const float2* __restrict__ FFT,
		const RecFourierProjectionTraverseSpace* const tSpace,
		const float* blobTableSqrt,
		const int imgCacheDim)
{
	// map thread to each (2D) voxel
#if TILE > 1
	int id = threadIdx.y * blockDim.x + threadIdx.x;
	int tidX = threadIdx.x % TILE + (id / (blockDim.y * TILE)) * TILE;
	int tidY = (id / TILE) % blockDim.y;
	int idx = blockIdx.x * blockDim.x + tidX;
	int idy = blockIdx.y * blockDim.y + tidY;
#else
	// map thread to each (2D) voxel
	volatile int idx = blockIdx.x*blockDim.x + threadIdx.x;
	volatile int idy = blockIdx.y*blockDim.y + threadIdx.y;
#endif

	if (tSpace->XY == tSpace->dir) { // iterate XY plane
		if (idy >= tSpace->minY && idy <= tSpace->maxY) {
			if (idx >= tSpace->minX && idx <= tSpace->maxX) {
				if (useFast) {
					float hitZ = getZ(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
					int z = (int)(hitZ + 0.5f); // rounding
					processVoxel(tempVolumeGPU, tempWeightsGPU, idx, idy, z, xSize, ySize, FFT, tSpace);
				} else {
					float z1 = getZ(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin); // lower plane
					float z2 = getZ(idx, idy, tSpace->unitNormal, tSpace->topOrigin); // upper plane
					z1 = clamp(z1, 0, gpuC.cMaxVolumeIndexYZ);
					z2 = clamp(z2, 0, gpuC.cMaxVolumeIndexYZ);
					int lower = static_cast<int>(floorf(fminf(z1, z2)));
					int upper = static_cast<int>(ceilf(fmaxf(z1, z2)));
					for (int z = lower; z <= upper; z++) {
						processVoxelBlob<blobOrder, useFastKaiser>(tempVolumeGPU, tempWeightsGPU, idx, idy, z, xSize, ySize, FFT, tSpace, blobTableSqrt, imgCacheDim);
					}
				}
			}
		}
	} else if (tSpace->XZ == tSpace->dir) { // iterate XZ plane
		if (idy >= tSpace->minZ && idy <= tSpace->maxZ) { // map z -> y
			if (idx >= tSpace->minX && idx <= tSpace->maxX) {
				if (useFast) {
					float hitY =getY(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
					int y = (int)(hitY + 0.5f); // rounding
					processVoxel(tempVolumeGPU, tempWeightsGPU, idx, y, idy, xSize, ySize, FFT, tSpace);
				} else {
					float y1 = getY(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin); // lower plane
					float y2 = getY(idx, idy, tSpace->unitNormal, tSpace->topOrigin); // upper plane
					y1 = clamp(y1, 0, gpuC.cMaxVolumeIndexYZ);
					y2 = clamp(y2, 0, gpuC.cMaxVolumeIndexYZ);
					int lower = static_cast<int>(floorf(fminf(y1, y2)));
					int upper = static_cast<int>(ceilf(fmaxf(y1, y2)));
					for (int y = lower; y <= upper; y++) {
						processVoxelBlob<blobOrder, useFastKaiser>(tempVolumeGPU, tempWeightsGPU, idx, y, idy, xSize, ySize, FFT, tSpace, blobTableSqrt, imgCacheDim);
					}
				}
			}
		}
	} else { // iterate YZ plane
		if (idy >= tSpace->minZ && idy <= tSpace->maxZ) { // map z -> y
			if (idx >= tSpace->minY && idx <= tSpace->maxY) { // map y > x
				if (useFast) {
					float hitX = getX(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
					int x = (int)(hitX + 0.5f); // rounding
					processVoxel(tempVolumeGPU, tempWeightsGPU, x, idx, idy, xSize, ySize, FFT, tSpace);
				} else {
					float x1 = getX(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin); // lower plane
					float x2 = getX(idx, idy, tSpace->unitNormal, tSpace->topOrigin); // upper plane
					x1 = clamp(x1, 0, gpuC.cMaxVolumeIndexX);
					x2 = clamp(x2, 0, gpuC.cMaxVolumeIndexX);
					int lower = static_cast<int>(floorf(fminf(x1, x2)));
					int upper = static_cast<int>(ceilf(fmaxf(x1, x2)));
					for (int x = lower; x <= upper; x++) {
						processVoxelBlob<blobOrder, useFastKaiser>(tempVolumeGPU, tempWeightsGPU, x, idx, idy, xSize, ySize, FFT, tSpace, blobTableSqrt, imgCacheDim);
					}
				}
			}
		}
	}
}

/**
 * Method will load data from image at position tXindex, tYindex
 * and return them.
 * In case the data lies outside of the image boundaries, zeros (0,0)
 * are returned
 */
__device__
void getImgData(const Point3D<float> AABB[2],
                const int tXindex, const int tYindex,
                const float2* FFTs, const int fftSizeX, const int fftSizeY, const int imgIndex,
                float2& vComplex) {
	int imgXindex = tXindex + static_cast<int>(AABB[0].x);
	int imgYindex = tYindex + static_cast<int>(AABB[0].y);
	if ((imgXindex >= 0)
	    && (imgXindex < fftSizeX)
	    && (imgYindex >=0)
	    && (imgYindex < fftSizeY))	{
		int index = imgYindex * fftSizeX + imgXindex; // copy data from image
		vComplex = (FFTs + fftSizeX * fftSizeY * imgIndex)[index];
	} else {
		vComplex = {0.f, 0.f}; // out of image bound, so return zero
	}
}

/**
 * Method will copy imgIndex(th) data from buffer
 * to given destination (shared memory).
 * Only data within AABB will be copied.
 * Destination is expected to be continuous array of sufficient
 * size (imgCacheDim^2)
 */
__device__
void copyImgToCache(float2* dest, const Point3D<float> AABB[2],
                    const float2* FFTs, const int fftSizeX, const int fftSizeY, const int imgIndex,
                    const int imgCacheDim) {
	for (int y = threadIdx.y; y < imgCacheDim; y += blockDim.y) {
		for (int x = threadIdx.x; x < imgCacheDim; x += blockDim.x) {
			int memIndex = y * imgCacheDim + x;
			getImgData(AABB, x, y, FFTs, fftSizeX, fftSizeY, imgIndex, dest[memIndex]);
		}
	}
}

/**
 * Method will use data stored in the buffer and update temporal
 * storages appropriately.
 */
template<bool fastLateBlobbing, int blobOrder, bool useFastKaiser>
__global__
void processBufferKernel(
		float2* outVolumeBuffer, float *outWeightsBuffer,
		const int fftSizeX, const int fftSizeY,
		const int traverseSpaceCount, const RecFourierProjectionTraverseSpace* traverseSpaces,
		const float2* FFTs,
		const float* blobTableSqrt,
		int imgCacheDim) {

#if SHARED_BLOB_TABLE
	if ( ! fastLateBlobbing) {
		// copy blob table to shared memory
		volatile int id = threadIdx.y*blockDim.x + threadIdx.x;
		volatile int blockSize = blockDim.x * blockDim.y;
		for (int i = id; i < BLOB_TABLE_SIZE_SQRT; i+= blockSize)
			BLOB_TABLE[i] = blobTableSqrt[i];
		__syncthreads();
	}
#endif

	for (int i = blockIdx.z; i < traverseSpaceCount; i += gridDim.z) {
		const RecFourierProjectionTraverseSpace& space = traverseSpaces[i];

#if SHARED_IMG
		if ( ! fastLateBlobbing) {
			// make sure that all threads start at the same time
			// as they can come from previous iteration
			__syncthreads();
			if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
				// first thread calculates which part of the image should be shared
				calculateAABB(&space, SHARED_AABB);
			}
			__syncthreads();
			// check if the block will have to copy data from image
			if (isWithin(SHARED_AABB, fftSizeX, fftSizeY)) {
				// all threads copy image data to shared memory
				copyImgToCache(IMG, SHARED_AABB, FFTs, fftSizeX, fftSizeY, space.projectionIndex, imgCacheDim);
				__syncthreads();
			} else {
				continue; // whole block can exit, as it's not reading from image
			}
		}
#endif

		processProjection<fastLateBlobbing, blobOrder, useFastKaiser>(
				outVolumeBuffer, outWeightsBuffer,
				fftSizeX, fftSizeY,
				FFTs + fftSizeX * fftSizeY * space.projectionIndex,
				&space,
				blobTableSqrt,
				imgCacheDim);

		__syncthreads(); // sync threads to avoid write after read problems
	}
}

/**
 * Method will use data stored in the buffer and update temporal
 * storages appropriately.
 * Actual calculation is done asynchronously, but 'buffer' can be reused
 * once the method returns.
 */
template<int blobOrder, bool useFastKaiser>
void processBufferGPU(
		float2 *outVolumeBuffer, float *outWeightsBuffer,
		const int fftSizeX, const int fftSizeY,
		const int traverseSpaceCount, const RecFourierProjectionTraverseSpace *traverseSpaces,
		const float2 *inFFTs,
		const float *blobTableSqrt,
		const bool fastLateBlobbing,
		const float blobRadius, const int maxVolIndexYZ) {

	// enqueue kernel and return control
	const int imgCacheDim = static_cast<int>(ceil(sqrt(2.f) * sqrt(3.f) * (BLOCK_DIM + 2 * blobRadius)));
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);

	const int size2D = maxVolIndexYZ + 1;
	dim3 dimGrid(static_cast<unsigned int>(ceil(size2D / (float)dimBlock.x)),
	             static_cast<unsigned int>(ceil(size2D / (float)dimBlock.y)),
	             GRID_DIM_Z);

	// by using templates, we can save some registers, especially for 'fast' version
	if (fastLateBlobbing) {
		processBufferKernel<true, blobOrder,useFastKaiser><<<dimGrid, dimBlock, 0, starpu_cuda_get_local_stream()>>>(
				outVolumeBuffer, outWeightsBuffer,
				fftSizeX, fftSizeY,
				traverseSpaceCount, traverseSpaces,
				inFFTs,
				blobTableSqrt,
				imgCacheDim);
	} else {
		// if making copy of the image in shared memory, allocate enough space
		int sharedMemSize = SHARED_IMG ? (imgCacheDim*imgCacheDim*sizeof(float2)) : 0;
		processBufferKernel<false, blobOrder,useFastKaiser><<<dimGrid, dimBlock, sharedMemSize, starpu_cuda_get_local_stream()>>>(
				outVolumeBuffer, outWeightsBuffer,
				fftSizeX, fftSizeY,
				traverseSpaceCount, traverseSpaces,
				inFFTs,
				blobTableSqrt,
				imgCacheDim);
	}
	gpuErrchk(cudaPeekAtLastError());
}

void func_reconstruct_cuda(void* buffers[], void* cl_arg) {
	const ReconstructFftArgs& arg = *(ReconstructFftArgs*) cl_arg;
	const float2* inFFTs = (float2*)STARPU_VECTOR_GET_PTR(buffers[0]);
	const RecFourierProjectionTraverseSpace* inSpaces = (RecFourierProjectionTraverseSpace*)STARPU_MATRIX_GET_PTR(buffers[1]);
	const float* inBlobTableSqrt = (float*)(STARPU_VECTOR_GET_PTR(buffers[2]));
	float2* outVolumeBuffer = (float2*)(STARPU_VECTOR_GET_PTR(buffers[3])); // Actually std::complex<float>
	float* outWeightsBuffer = (float*)(STARPU_VECTOR_GET_PTR(buffers[4]));
	const uint32_t noOfImages = ((LoadedImagesBuffer*) STARPU_VARIABLE_GET_PTR(buffers[5]))->noOfImages;

	switch (arg.blobOrder) {
		case 0:
			if (arg.blobAlpha <= 15.0) {
				processBufferGPU<0, true>(outVolumeBuffer, outWeightsBuffer,
				                          arg.fftSizeX, arg.fftSizeY,
				                          arg.noOfSymmetries * noOfImages, inSpaces,
				                          inFFTs,
				                          inBlobTableSqrt,
				                          arg.fastLateBlobbing,
				                          arg.blobRadius, arg.maxVolIndexYZ);
			} else {
				processBufferGPU<0, false>(outVolumeBuffer, outWeightsBuffer,
				                           arg.fftSizeX, arg.fftSizeY,
				                           arg.noOfSymmetries * noOfImages, inSpaces,
				                           inFFTs,
				                           inBlobTableSqrt,
				                           arg.fastLateBlobbing,
				                           arg.blobRadius, arg.maxVolIndexYZ);
			}
			break;
		case 1:
			processBufferGPU<1, false>(outVolumeBuffer, outWeightsBuffer,
			                           arg.fftSizeX, arg.fftSizeY,
			                           arg.noOfSymmetries * noOfImages, inSpaces,
			                           inFFTs,
			                           inBlobTableSqrt,
			                           arg.fastLateBlobbing,
			                           arg.blobRadius, arg.maxVolIndexYZ);
			break;
		case 2:
			processBufferGPU<2, false>(outVolumeBuffer, outWeightsBuffer,
			                           arg.fftSizeX, arg.fftSizeY,
			                           arg.noOfSymmetries * noOfImages, inSpaces,
			                           inFFTs,
			                           inBlobTableSqrt,
			                           arg.fastLateBlobbing,
			                           arg.blobRadius, arg.maxVolIndexYZ);
			break;
		case 3:
			processBufferGPU<3, false>(outVolumeBuffer, outWeightsBuffer,
			                           arg.fftSizeX, arg.fftSizeY,
			                           arg.noOfSymmetries * noOfImages, inSpaces,
			                           inFFTs,
			                           inBlobTableSqrt,
			                           arg.fastLateBlobbing,
			                           arg.blobRadius, arg.maxVolIndexYZ);
			break;
		case 4:
			processBufferGPU<4, false>(outVolumeBuffer, outWeightsBuffer,
			                           arg.fftSizeX, arg.fftSizeY,
			                           arg.noOfSymmetries * noOfImages, inSpaces,
			                           inFFTs,
			                           inBlobTableSqrt,
			                           arg.fastLateBlobbing,
			                           arg.blobRadius, arg.maxVolIndexYZ);
			break;
		default:
			REPORT_ERROR(ERR_VALUE_INCORRECT, "m out of range [0..4] in kaiser_value()");
	}

	// gpuErrchk(cudaStreamSynchronize(starpu_cuda_get_local_stream())); disabled because codelet is async
}

//------------------------------------------------ CPU -----------------------------------------------------------------
// uses copies of the GPU functions, rewritten for single thread runtime

/** Atomically increments the value pointed at by ptr by value.
 * Uses relaxed memory model with no reordering guarantees. */
void atomicAddFloat(volatile float* ptr, float addedValue) {
	static_assert(sizeof(float) == sizeof(uint32_t), "atomicAddFloat requires floats to be 32bit");

	// This is probably fine, since the constructor/destructor should be trivial
	// (As of C++11, this is guaranteed only for integral type specializations, but it is probably reasonably safe to assume
	// that this will hold for floats as well. C++20 requies that by spec.)
	volatile std::atomic<float>& atomicPtr = *reinterpret_cast<volatile std::atomic<float>*>(ptr);
	float current = atomicPtr.load(std::memory_order::memory_order_relaxed);
	while (true) {
		const float newValue = current + addedValue;
		// Since x86 does not allow atomic add of floats (only integers), we have to implement it through CAS
		if (atomicPtr.compare_exchange_weak(current, newValue, std::memory_order::memory_order_relaxed)) {
			// Current was still current and was replaced with the newValue. Done.
			return;
		}
		// Comparison failed. current now contains the new value and we try again.
	}
}

void processVoxelCPU(
		float2* const tempVolumeGPU, float* const tempWeightsGPU,
		const int x, const int y, const int z,
		const int xSize, const int ySize,
		const float2* const __restrict__ FFT,
		const RecFourierProjectionTraverseSpace* const space)
{
	Point3D<float> imgPos;
	float wBlob = 1.f;

	float dataWeight = space->weight;

	// transform current point to center
	imgPos.x = x - cpuC.cMaxVolumeIndexX/2;
	imgPos.y = y - cpuC.cMaxVolumeIndexYZ/2;
	imgPos.z = z - cpuC.cMaxVolumeIndexYZ/2;
	if (imgPos.x*imgPos.x + imgPos.y*imgPos.y + imgPos.z*imgPos.z > space->maxDistanceSqr) {
		return; // discard iterations that would access pixel with too high frequency
	}
	// rotate around center
	multiply(space->transformInv, imgPos);
	if (imgPos.x < 0.f) return; // reading outside of the image boundary. Z is always correct and Y is checked by the condition above

	// transform back and round
	// just Y coordinate needs adjusting, since X now matches to picture and Z is irrelevant
	int imgX = clamp((int)(imgPos.x + 0.5f), 0, xSize - 1);
	int imgY = clamp((int)(imgPos.y + 0.5f + cpuC.cMaxVolumeIndexYZ / 2), 0, ySize - 1);

	int index3D = z * (cpuC.cMaxVolumeIndexYZ+1) * (cpuC.cMaxVolumeIndexX+1) + y * (cpuC.cMaxVolumeIndexX+1) + x;
	int index2D = imgY * xSize + imgX;

	float weight = wBlob * dataWeight;

	// use atomic as two blocks can write to same voxel
	atomicAddFloat(&tempVolumeGPU[index3D].x, FFT[index2D].x * weight);
	atomicAddFloat(&tempVolumeGPU[index3D].y, FFT[index2D].y * weight);
	atomicAddFloat(&tempWeightsGPU[index3D], weight);
	//tempVolumeGPU[index3D].x += FFT[index2D].x * weight;
	//tempVolumeGPU[index3D].y += FFT[index2D].y * weight;
	//tempWeightsGPU[index3D] += weight;
}

template<int blobOrder, bool useFastKaiser, bool usePrecomputedInterpolation>
void processVoxelBlobCPU(
		float2* const tempVolumeGPU, float* const tempWeightsGPU,
		const int x, const int y, const int z,
		const int xSize, const int ySize,
		const float2* const __restrict__ FFT,
		const RecFourierProjectionTraverseSpace* const space,
		const float* blobTableSqrt)
{
	Point3D<float> imgPos;
	// transform current point to center
	imgPos.x = x - cpuC.cMaxVolumeIndexX/2;
	imgPos.y = y - cpuC.cMaxVolumeIndexYZ/2;
	imgPos.z = z - cpuC.cMaxVolumeIndexYZ/2;
	if ((imgPos.x*imgPos.x + imgPos.y*imgPos.y + imgPos.z*imgPos.z) > space->maxDistanceSqr) {
		return; // discard iterations that would access pixel with too high frequency
	}
	// rotate around center
	multiply(space->transformInv, imgPos);
	if (imgPos.x < -cpuC.cBlobRadius) return; // reading outside of the image boundary. Z is always correct and Y is checked by the condition above
	// transform back just Y coordinate, since X now matches to picture and Z is irrelevant
	imgPos.y += cpuC.cMaxVolumeIndexYZ / 2;

	// check that we don't want to collect data from far far away ...
	float radiusSqr = cpuC.cBlobRadius * cpuC.cBlobRadius;
	float zSqr = imgPos.z * imgPos.z;
	if (zSqr > radiusSqr) return;

	// create blob bounding box
	int minX = ceilf(imgPos.x - cpuC.cBlobRadius);
	int maxX = floorf(imgPos.x + cpuC.cBlobRadius);
	int minY = ceilf(imgPos.y - cpuC.cBlobRadius);
	int maxY = floorf(imgPos.y + cpuC.cBlobRadius);
	minX = fmaxf(minX, 0);
	minY = fmaxf(minY, 0);
	maxX = fminf(maxX, xSize-1);
	maxY = fminf(maxY, ySize-1);

	int index3D = z * (cpuC.cMaxVolumeIndexYZ+1) * (cpuC.cMaxVolumeIndexX+1) + y * (cpuC.cMaxVolumeIndexX+1) + x;
	float2 vol;
	float w;
	vol.x = vol.y = w = 0.f;
	float dataWeight = space->weight;

	// check which pixel in the vicinity should contribute
	for (int i = minY; i <= maxY; i++) {
		float ySqr = (imgPos.y - i) * (imgPos.y - i);
		float yzSqr = ySqr + zSqr;
		if (yzSqr > radiusSqr) continue;
		for (int j = minX; j <= maxX; j++) {
			float xD = imgPos.x - j;
			float distanceSqr = xD*xD + yzSqr;
			if (distanceSqr > radiusSqr) continue;

			int index2D = i * xSize + j;

			float wBlob;
			if (usePrecomputedInterpolation) {
				int aux = (int) ((distanceSqr * cpuC.cIDeltaSqrt + 0.5f));
				wBlob = blobTableSqrt[aux];
			} else if (useFastKaiser) {
				wBlob = kaiserValueFast(distanceSqr);
			} else {
				wBlob = kaiserValue<blobOrder>(sqrtf(distanceSqr), cpuC.cBlobRadius) * cpuC.cIw0;
			}

			float weight = wBlob * dataWeight;
			w += weight;
			vol += FFT[index2D] * weight;
		}
	}

	atomicAddFloat(&tempVolumeGPU[index3D].x, vol.x);
	atomicAddFloat(&tempVolumeGPU[index3D].y, vol.y);
	atomicAddFloat(&tempWeightsGPU[index3D], w);
	//tempVolumeGPU[index3D].x += vol.x;
	//tempVolumeGPU[index3D].y += vol.y;
	//tempWeightsGPU[index3D] += w;
}

template<bool useFast, int blobOrder, bool useFastKaiser, bool usePrecomputedInterpolation>
void processProjectionCPU(
		float2* tempVolumeGPU, float *tempWeightsGPU,
		const int xSize, const int ySize,
		const float2* __restrict__ FFT,
		const RecFourierProjectionTraverseSpace* const tSpace,
		const float* blobTableSqrt) {

	if (tSpace->XY == tSpace->dir) { // iterate XY plane
		for (int idy = tSpace->minY; idy <= tSpace->maxY; idy++) {
			for (int idx = tSpace->minX; idx <= tSpace->maxX; idx++) {
				if (useFast) {
					float hitZ = getZ(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
					int z = (int)(hitZ + 0.5f); // rounding
					processVoxelCPU(tempVolumeGPU, tempWeightsGPU, idx, idy, z, xSize, ySize, FFT, tSpace);
				} else {
					float z1 = getZ(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin); // lower plane
					float z2 = getZ(idx, idy, tSpace->unitNormal, tSpace->topOrigin); // upper plane
					z1 = clamp(z1, 0, cpuC.cMaxVolumeIndexYZ);
					z2 = clamp(z2, 0, cpuC.cMaxVolumeIndexYZ);
					int lower = static_cast<int>(floorf(fminf(z1, z2)));
					int upper = static_cast<int>(ceilf(fmaxf(z1, z2)));
					for (int z = lower; z <= upper; z++) {
						processVoxelBlobCPU<blobOrder, useFastKaiser, usePrecomputedInterpolation>(tempVolumeGPU, tempWeightsGPU, idx, idy, z, xSize, ySize, FFT, tSpace, blobTableSqrt);
					}
				}
			}
		}
	} else if (tSpace->XZ == tSpace->dir) { // iterate XZ plane
		for (int idy = tSpace->minZ; idy <= tSpace->maxZ; idy++) { // map z -> y
			for (int idx = tSpace->minX; idx <= tSpace->maxX; idx++) {
				if (useFast) {
					float hitY =getY(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
					int y = (int)(hitY + 0.5f); // rounding
					processVoxelCPU(tempVolumeGPU, tempWeightsGPU, idx, y, idy, xSize, ySize, FFT, tSpace);
				} else {
					float y1 = getY(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin); // lower plane
					float y2 = getY(idx, idy, tSpace->unitNormal, tSpace->topOrigin); // upper plane
					y1 = clamp(y1, 0, cpuC.cMaxVolumeIndexYZ);
					y2 = clamp(y2, 0, cpuC.cMaxVolumeIndexYZ);
					int lower = static_cast<int>(floorf(fminf(y1, y2)));
					int upper = static_cast<int>(ceilf(fmaxf(y1, y2)));
					for (int y = lower; y <= upper; y++) {
						processVoxelBlobCPU<blobOrder, useFastKaiser, usePrecomputedInterpolation>(tempVolumeGPU, tempWeightsGPU, idx, y, idy, xSize, ySize, FFT, tSpace, blobTableSqrt);
					}
				}
			}
		}
	} else { // iterate YZ plane
		for (int idy = tSpace->minZ; idy <= tSpace->maxZ; idy++) { // map z -> y
			for (int idx = tSpace->minY; idx <= tSpace->maxY; idx++) { // map y > x
				if (useFast) {
					float hitX = getX(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
					int x = (int)(hitX + 0.5f); // rounding
					processVoxelCPU(tempVolumeGPU, tempWeightsGPU, x, idx, idy, xSize, ySize, FFT, tSpace);
				} else {
					float x1 = getX(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin); // lower plane
					float x2 = getX(idx, idy, tSpace->unitNormal, tSpace->topOrigin); // upper plane
					x1 = clamp(x1, 0, cpuC.cMaxVolumeIndexX);
					x2 = clamp(x2, 0, cpuC.cMaxVolumeIndexX);
					int lower = static_cast<int>(floorf(fminf(x1, x2)));
					int upper = static_cast<int>(ceilf(fmaxf(x1, x2)));
					for (int x = lower; x <= upper; x++) {
						processVoxelBlobCPU<blobOrder, useFastKaiser, usePrecomputedInterpolation>(tempVolumeGPU, tempWeightsGPU, x, idx, idy, xSize, ySize, FFT, tSpace, blobTableSqrt);
					}
				}
			}
		}
	}
}

template<int blobOrder, bool useFastKaiser, bool usePrecomputedInterpolation>
void processBufferCPU(
		float2 *outVolumeBuffer, float *outWeightsBuffer,
		const int fftSizeX, const int fftSizeY,
		const int traverseSpaceCount, const RecFourierProjectionTraverseSpace *traverseSpaces,
		const float2 *inFFTs,
		const float *blobTableSqrt,
		const bool fastLateBlobbing) {

	const int groupSize = starpu_combined_worker_get_size();
	const int groupRank = starpu_combined_worker_get_rank();

	for (int i = groupRank; i < traverseSpaceCount; i += groupSize) {
		const RecFourierProjectionTraverseSpace &space = traverseSpaces[i];

		const float2* spaceFFT = inFFTs + fftSizeX * fftSizeY * space.projectionIndex;

		// by using templates, we can save some registers, especially for 'fast' version
		if (fastLateBlobbing) {
			processProjectionCPU<true, blobOrder, useFastKaiser, usePrecomputedInterpolation>(
					outVolumeBuffer, outWeightsBuffer,
					fftSizeX, fftSizeY,
					spaceFFT,
					&space,
					blobTableSqrt);
		} else {
			processProjectionCPU<false, blobOrder, useFastKaiser, usePrecomputedInterpolation>(
					outVolumeBuffer, outWeightsBuffer,
					fftSizeX, fftSizeY,
					spaceFFT,
					&space,
					blobTableSqrt);
		}
	}
}

template<bool usePrecomputedInterpolation>
void func_reconstruct_cpu_template(void* buffers[], void* cl_arg) {
	const ReconstructFftArgs& arg = *(ReconstructFftArgs*) cl_arg;
	const float2* inFFTs = (float2*)STARPU_VECTOR_GET_PTR(buffers[0]);
	const RecFourierProjectionTraverseSpace* inSpaces = (RecFourierProjectionTraverseSpace*)STARPU_MATRIX_GET_PTR(buffers[1]);
	const float* inBlobTableSqrt = (float*)(STARPU_VECTOR_GET_PTR(buffers[2]));
	float2* outVolumeBuffer = (float2*)(STARPU_VECTOR_GET_PTR(buffers[3])); // Actually std::complex<float>
	float* outWeightsBuffer = (float*)(STARPU_VECTOR_GET_PTR(buffers[4]));
	const uint32_t noOfImages = ((LoadedImagesBuffer*) STARPU_VARIABLE_GET_PTR(buffers[5]))->noOfImages;

	switch (arg.blobOrder) {
		case 0:
			if (arg.blobAlpha <= 15.0) {
				processBufferCPU<0, true, usePrecomputedInterpolation>(outVolumeBuffer, outWeightsBuffer,
				                          arg.fftSizeX, arg.fftSizeY,
				                          arg.noOfSymmetries * noOfImages, inSpaces,
				                          inFFTs,
				                          inBlobTableSqrt,
				                          arg.fastLateBlobbing);
			} else {
				processBufferCPU<0, false, usePrecomputedInterpolation>(outVolumeBuffer, outWeightsBuffer,
				                           arg.fftSizeX, arg.fftSizeY,
				                           arg.noOfSymmetries * noOfImages, inSpaces,
				                           inFFTs,
				                           inBlobTableSqrt,
				                           arg.fastLateBlobbing);
			}
			break;
		case 1:
			processBufferCPU<1, false, usePrecomputedInterpolation>(outVolumeBuffer, outWeightsBuffer,
			                           arg.fftSizeX, arg.fftSizeY,
			                           arg.noOfSymmetries * noOfImages, inSpaces,
			                           inFFTs,
			                           inBlobTableSqrt,
			                           arg.fastLateBlobbing);
			break;
		case 2:
			processBufferCPU<2, false, usePrecomputedInterpolation>(outVolumeBuffer, outWeightsBuffer,
			                           arg.fftSizeX, arg.fftSizeY,
			                           arg.noOfSymmetries * noOfImages, inSpaces,
			                           inFFTs,
			                           inBlobTableSqrt,
			                           arg.fastLateBlobbing);
			break;
		case 3:
			processBufferCPU<3, false, usePrecomputedInterpolation>(outVolumeBuffer, outWeightsBuffer,
			                           arg.fftSizeX, arg.fftSizeY,
			                           arg.noOfSymmetries * noOfImages, inSpaces,
			                           inFFTs,
			                           inBlobTableSqrt,
			                           arg.fastLateBlobbing);
			break;
		case 4:
			processBufferCPU<4, false, usePrecomputedInterpolation>(outVolumeBuffer, outWeightsBuffer,
			                           arg.fftSizeX, arg.fftSizeY,
			                           arg.noOfSymmetries * noOfImages, inSpaces,
			                           inFFTs,
			                           inBlobTableSqrt,
			                           arg.fastLateBlobbing);
			break;
		default:
			REPORT_ERROR(ERR_VALUE_INCORRECT, "m out of range [0..4] in kaiser_value()");
	}
}

void func_reconstruct_cpu_lookup_interpolation(void* buffers[], void* cl_arg) {
	func_reconstruct_cpu_template<true>(buffers, cl_arg);
}

void func_reconstruct_cpu_dynamic_interpolation(void* buffers[], void* cl_arg) {
	func_reconstruct_cpu_template<false>(buffers, cl_arg);
}
