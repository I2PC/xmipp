/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
 *              (some code derived from other Xmipp programs by other authors)
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

#ifndef XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_STARPU_UTIL_H_
#define XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_STARPU_UTIL_H_

#include "core/numerical_recipes.h"
#include "data/point3D.h"
#include "data/blobs.h"
#include "reconstruction/reconstruct_fourier_projection_traverse_space.h"
#include "reconstruct_fourier_defines.h"


/** Method to allocate 3D array (not continuous) of given size */
template<typename T>
static T*** allocate(T***& where, int xSize, int ySize, int zSize) {
	where = new T**[zSize];
	for (int z = 0; z < zSize; z++) {
		where[z] = new T*[ySize];
		for (int y = 0; y < ySize; y++) {
			where[z][y] = new T[xSize];
			for (int x = 0; x < xSize; x++) {
				where[z][y][x] = (T) 0;
			}
		}
	}
	return where;
}

/** Method to release 3D array of given size */
template<typename T>
static void release(T***& array, int ySize, int zSize) {
	for(int z = 0; z < zSize; z++) {
		for(int y = 0; y < ySize; y++) {
			delete[] array[z][y];
		}
		delete[] array[z];
	}
	delete[] array;
	array = NULL;
}

/** Method will copy continuous 3D arrays (with each side of `size`) to non-continuous 3D arrays */
template<typename T>
void copyFlatTo3D(T*** target, const T* source, const int size) {
	for (int z = 0; z < size; z++) {
		for (int y = 0; y < size; y++) {
			int index = (z * size * size) + (y * size);
			memcpy(target[z][y], source + index, size * sizeof(T));
		}
	}
}


/** Function behaving like an identity, i.e returning passed value */
template<typename T>
static T identity(T val) { return val;}; // if used with some big type, use reference

/** Function returning conjugate of a complex number */
template<typename T>
static std::complex<T> conjugate(std::complex<T> f) { return conj(f);};


/** Do 3x3 x 1x3 matrix-vector multiplication */
static inline void multiply(const float transform[3][3], Point3D<float>& inOut) {
	float tmp0 = transform[0][0] * inOut.x + transform[0][1] * inOut.y + transform[0][2] * inOut.z;
	float tmp1 = transform[1][0] * inOut.x + transform[1][1] * inOut.y + transform[1][2] * inOut.z;
	float tmp2 = transform[2][0] * inOut.x + transform[2][1] * inOut.y + transform[2][2] * inOut.z;
	inOut.x = tmp0;
	inOut.y = tmp1;
	inOut.z = tmp2;
}

/**
*          6____5
*         2/___1/
*    +    | |  ||   y
* [0,0,0] |*|7 ||4
*    -    |/___|/  z  sizes are padded with blob-radius
*         3  x  0
* [0,0] is in the middle of the left side (point [2] and [3]), provided the blobSize is 0
* otherwise the values can go to negative values
* origin(point[0]) in in 'high' frequencies so that possible numerical instabilities are moved to high frequencies
*/
static void createProjectionCuboid(Point3D<float> (&cuboid)[8], float sizeX, float sizeY, float blobSize) {
	float halfY = sizeY / 2.0f;
	cuboid[3].x = cuboid[2].x = cuboid[7].x = cuboid[6].x = 0.f - blobSize;
	cuboid[0].x = cuboid[1].x = cuboid[4].x = cuboid[5].x = sizeX + blobSize;

	cuboid[3].y = cuboid[0].y = cuboid[7].y = cuboid[4].y = -(halfY + blobSize);
	cuboid[1].y = cuboid[2].y = cuboid[5].y = cuboid[6].y = halfY + blobSize;

	cuboid[3].z = cuboid[0].z = cuboid[1].z = cuboid[2].z = 0.f + blobSize;
	cuboid[7].z = cuboid[4].z = cuboid[5].z = cuboid[6].z = 0.f - blobSize;
}

/** Apply rotation transform to cuboid */
static inline void rotateCuboid(Point3D<float> (&cuboid)[8], const float transform[3][3]) {
	for (int i = 0; i < 8; i++) {
		multiply(transform, cuboid[i]);
	}
}

/** Add 'vector' to each element of 'cuboid' */
static inline void translateCuboid(Point3D<float> (&cuboid)[8], Point3D<float> vector) {
	for (int i = 0; i < 8; i++) {
		cuboid[i].x += vector.x;
		cuboid[i].y += vector.y;
		cuboid[i].z += vector.z;
	}
}

/**
 * Method will calculate Axis Aligned Bound Box of the cuboid and restrict
 * its maximum size
 */
static void computeAABB(Point3D<float> (&AABB)[2], Point3D<float> (&cuboid)[8],
                        float minX, float minY, float minZ,
                        float maxX, float maxY, float maxZ) {
	AABB[0].x = AABB[0].y = AABB[0].z = std::numeric_limits<float>::max();
	AABB[1].x = AABB[1].y = AABB[1].z = std::numeric_limits<float>::min();
	Point3D<float> tmp;
	for (int i = 0; i < 8; i++) {
		tmp = cuboid[i];
		if (AABB[0].x > tmp.x) AABB[0].x = tmp.x;
		if (AABB[0].y > tmp.y) AABB[0].y = tmp.y;
		if (AABB[0].z > tmp.z) AABB[0].z = tmp.z;
		if (AABB[1].x < tmp.x) AABB[1].x = tmp.x;
		if (AABB[1].y < tmp.y) AABB[1].y = tmp.y;
		if (AABB[1].z < tmp.z) AABB[1].z = tmp.z;
	}
	// limit to max size
	if (AABB[0].x < minX) AABB[0].x = minX;
	if (AABB[0].y < minY) AABB[0].y = minY;
	if (AABB[0].z < minZ) AABB[0].z = minZ;
	if (AABB[1].x > maxX) AABB[1].x = maxX;
	if (AABB[1].y > maxY) AABB[1].y = maxY;
	if (AABB[1].z > maxZ) AABB[1].z = maxZ;
}

/** DEBUG ONLY method, prints AABB to std::cout. Output can be used in e.g. GNUPLOT */
static void printAABB(Point3D<float> AABB[]) {
	std::cout
			// one base
			<< AABB[0].x << " " << AABB[0].y << " " << AABB[0].z << "\n"
			<< AABB[1].x << " " << AABB[0].y << " " << AABB[0].z << "\n"
			<< AABB[1].x << " " << AABB[1].y << " " << AABB[0].z << "\n"
			<< AABB[0].x << " " << AABB[1].y << " " << AABB[0].z << "\n"
			<< AABB[0].x << " " << AABB[0].y << " " << AABB[0].z << "\n"
			// other base with one connection
			<< AABB[0].x << " " << AABB[0].y << " " << AABB[1].z << "\n"
			<< AABB[1].x << " " << AABB[0].y << " " << AABB[1].z << "\n"
			<< AABB[1].x << " " << AABB[1].y << " " << AABB[1].z << "\n"
			<< AABB[0].x << " " << AABB[1].y << " " << AABB[1].z << "\n"
			<< AABB[0].x << " " << AABB[0].y << " " << AABB[1].z << "\n"
			// lines between bases
			<< AABB[1].x << " " << AABB[0].y << " " << AABB[1].z << "\n"
			<< AABB[1].x << " " << AABB[0].y << " " << AABB[0].z << "\n"
			<< AABB[1].x << " " << AABB[1].y << " " << AABB[0].z << "\n"
			<< AABB[1].x << " " << AABB[1].y << " " << AABB[1].z << "\n"
			<< AABB[0].x << " " << AABB[1].y << " " << AABB[1].z << "\n"
			<< AABB[0].x << " " << AABB[1].y << " " << AABB[0].z
			<< std::endl;
}

/** Method to convert temporal space to expected (original) format */
template<typename T, typename U>
static void convertToExpectedSpace(T*** input, int size, MultidimArray<U>& VoutFourier) {
	int halfSize = size / 2;
	for (int z = 0; z <= size; z++) {
		for (int y = 0; y <= size; y++) {
			// shift FFT from center to corners
			const size_t newY = (y < halfSize) ? VoutFourier.ydim - halfSize + y : y - halfSize;
			const size_t newZ = (z < halfSize) ? VoutFourier.zdim - halfSize + z : z - halfSize;

			for (int x = 0; x <= halfSize; x++) {
				// store to output array
				// += in necessary as VoutFourier might be used multiple times when used with MPI
				DIRECT_A3D_ELEM(VoutFourier, newZ, newY, x /* no need to move X */) += input[z][y][x];
			}
		}
	}
}

/**
 * Method calculates a traversal space information for specific projection
 * imgSizeX - X size of the projection
 * imgSizeY - Y size of the projection
 * transform - forward rotation that should be applied to the projection
 * transformInv - inverse transformation
 * space - which will be filled
 */
static void computeTraverseSpace(uint32_t imgSizeX, uint32_t imgSizeY,
                                 const float transform[3][3], const float transformInv[3][3], RecFourierProjectionTraverseSpace& space,
                                 uint32_t maxVolumeIndexX, uint32_t maxVolumeIndexYZ, bool useFast, float blobRadius) {
	Point3D<float> cuboid[8];
	Point3D<float> AABB[2];
	Point3D<float> origin = {maxVolumeIndexX/2.f, maxVolumeIndexYZ/2.f, maxVolumeIndexYZ/2.f};
	createProjectionCuboid(cuboid, imgSizeX, imgSizeY, useFast ? 0.f : blobRadius);
	rotateCuboid(cuboid, transform);
	translateCuboid(cuboid, origin);
	computeAABB(AABB, cuboid, 0, 0, 0, maxVolumeIndexX, maxVolumeIndexYZ, maxVolumeIndexYZ);

	// store data
	space.minZ = floor(AABB[0].z);
	space.minY = floor(AABB[0].y);
	space.minX = floor(AABB[0].x);
	space.maxZ = ceil(AABB[1].z);
	space.maxY = ceil(AABB[1].y);
	space.maxX = ceil(AABB[1].x);
	space.topOrigin = cuboid[4];
	space.bottomOrigin = cuboid[0];
	space.maxDistanceSqr = (imgSizeX + (useFast ? 0.f : blobRadius))
	                       * (imgSizeX + (useFast ? 0.f : blobRadius));
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			space.transformInv[i][j] = transformInv[i][j];
		}
	}

	// calculate best traverse direction
	space.unitNormal.x = space.unitNormal.y = 0.f;
	space.unitNormal.z = 1.f;
	multiply(transform, space.unitNormal);
	float nX = std::abs(space.unitNormal.x);
	float nY = std::abs(space.unitNormal.y);
	float nZ = std::abs(space.unitNormal.z);

	// biggest vector indicates ideal direction
	if (nX >= nY  && nX >= nZ) { // iterate YZ plane
		space.dir = RecFourierProjectionTraverseSpace::Direction::YZ;
	} else if (nY >= nX && nY >= nZ) { // iterate XZ plane
		space.dir = RecFourierProjectionTraverseSpace::Direction::XZ;
	} else if (nZ >= nX && nZ >= nY) { // iterate XY plane
		space.dir = RecFourierProjectionTraverseSpace::Direction::XY;
	}
}

/**
 * Method will take input array (of size
 * maxVolumeIndexYZ*maxVolumeIndexYZ*maxVolumeIndexYZ
 * and transfer data to newly created array of size
 * maxVolumeIndexX*maxVolumeIndexYZ*maxVolumeIndexYZ (i.e. X side is half).
 * so that data on the 'left side' are mirrored against the origin of the
 * array, and 'f' function is applied on them.
 * 'right hand side' is only transfered.
 * As a result, input will contain an array with X side half of the original
 * size, with all data transfered from 'missing' side to 'preserved' side
 */
template<typename T>
static void mirrorAndCrop(T***& input,T (*f)(T), uint32_t maxVolumeIndexX, uint32_t maxVolumeIndexYZ) {
	T*** output;
	// create new storage, notice that just 'right hand side - X axis' of the input will be preserved, left will be converted to its complex conjugate
	allocate(output, maxVolumeIndexX+1, maxVolumeIndexYZ+1, maxVolumeIndexYZ+1);
	// traverse old storage
	for (int z = 0; z <= maxVolumeIndexYZ; z++) {
		for (int y = 0; y <= maxVolumeIndexYZ; y++) {
			for (int x = 0; x <= maxVolumeIndexYZ; x++) {
				if (x < maxVolumeIndexX) {
					int newPos[3];
					// mirror against center of the volume, e.g. [0,0,0]->[size,size,size]. It will fit as the input space is one voxel bigger
					newPos[0] = maxVolumeIndexYZ - x;
					newPos[1] = maxVolumeIndexYZ - y;
					newPos[2] = maxVolumeIndexYZ - z;
					output[newPos[2]][newPos[1]][newPos[0]-maxVolumeIndexX] += f(input[z][y][x]);
				} else {
					// copy with X shifted by (-halfSize)
					output[z][y][x-maxVolumeIndexX] += input[z][y][x];
				}
			}
		}
	}
	// free original data
	release(input, maxVolumeIndexYZ+1, maxVolumeIndexYZ+1);
	// set new data
	input = output;
}

/**
 * Method will take temp spaces (containing complex conjugate values
 * in the 'right X side'), transfer them to 'left X side' and remove
 * the 'right X side'. As a result, the X dimension of the temp spaces
 * will be half of the original.
 */
static void mirrorAndCropTempSpaces(std::complex<float>***& tempVolume, float***& tempWeights, const uint32_t maxVolumeIndexX, const uint32_t maxVolumeIndexYZ) {
	mirrorAndCrop(tempWeights, &identity<float>, maxVolumeIndexX, maxVolumeIndexYZ);
	mirrorAndCrop(tempVolume, &conjugate, maxVolumeIndexX, maxVolumeIndexYZ);
}

/**
 * Method will enforce Hermitian symmetry, i.e will make sure
 * that the values in temporal space at X=0 are complex conjugate of
 * in respect to center of the space
 */
static void forceHermitianSymmetry(std::complex<float>***& tempVolume, float***& tempWeights, int maxVolumeIndexYZ) {
	const int x = 0;
	for (int z = 0; z <= maxVolumeIndexYZ; z++) {
		for (int y = 0; y <= maxVolumeIndexYZ/2; y++) {
			// mirror against center of the volume, e.g. [0,0,0]->[size,size,size]. It will fit as the input space is one voxel biger
			const int newX = x;
			const int newY = maxVolumeIndexYZ - y;
			const int newZ = maxVolumeIndexYZ - z;

			const std::complex<float> averageVol = 0.5f * (tempVolume[newZ][newY][newX] + conj(tempVolume[z][y][x]));
			const float averageWeight = 0.5f * (tempWeights[newZ][newY][newX] + tempWeights[z][y][x]);

			tempVolume[newZ][newY][newX] = averageVol;
			tempVolume[z][y][x] = conj(averageVol);
			tempWeights[newZ][newY][newX] = tempWeights[z][y][x] = averageWeight;
		}
	}
}

/**
 * Method will in effect do the point-wise division of
 * tempVolume and tempWeights
 * (i.e. correct Fourier coefficients by proper weight)
 */
static void processWeights(std::complex<float>***& tempVolume, float***& tempWeights, int maxVolumeIndexX, int maxVolumeIndexYZ, double paddingFactorProj, double paddingFactorVol, int imgSize) {
	// Get a first approximation of the reconstruction
	float corr2D_3D = static_cast<float>(pow(paddingFactorProj, 2.0) / (imgSize * pow(paddingFactorVol, 3.0)));
	for (int z = 0; z <= maxVolumeIndexYZ; z++) {
		for (int y = 0; y <= maxVolumeIndexYZ; y++) {
			for (int x = 0; x <= maxVolumeIndexX; x++) {
				const float weight = tempWeights[z][y][x];

				if (weight > ACCURACY)
					tempVolume[z][y][x] *= corr2D_3D / weight;
				else
					tempVolume[z][y][x] = 0;
			}
		}
	}
}


static float getBessiOrderAlpha(blobtype blob) {
	switch (blob.order) {
		case 0: return static_cast<float>(bessi0(blob.alpha));
		case 1: return static_cast<float>(bessi1(blob.alpha));
		case 2: return static_cast<float>(bessi2(blob.alpha));
		case 3: return static_cast<float>(bessi3(blob.alpha));
		case 4: return static_cast<float>(bessi4(blob.alpha));
		default:
			REPORT_ERROR(ERR_VALUE_INCORRECT,"Order must be in interval [0..4]");
	}
}

/**
 * Method will apply blob to input 3D array.
 * Original array will be released and new (blurred) will be returned
 */
template<typename T>
static T*** applyBlob(T***& input, float blobSize,
                      float* blobTableSqrt, float iDeltaSqrt,
                      const int maxVolumeIndexX, const int maxVolumeIndexYZ) {
	float blobSizeSqr = blobSize * blobSize;
	int blob = static_cast<int>(floor(blobSize)); // we are using integer coordinates, so we cannot hit anything further
	T*** output;
	// create new storage
	allocate(output, maxVolumeIndexX+1, maxVolumeIndexYZ+1, maxVolumeIndexYZ+1);

	// traverse new storage
	for (int i = 0; i <= maxVolumeIndexYZ; i++) {
		for (int j = 0; j <= maxVolumeIndexYZ; j++) {
			for (int k = 0; k <= maxVolumeIndexX; k++) {
				// traverse input storage
				T tmp = (T) 0;
				for (int z = std::max(0, i-blob); z <= std::min(maxVolumeIndexYZ, i+blob); z++) {
					float dZSqr = (i - z) * (i - z);
					for (int y = std::max(0, j-blob); y <= std::min(maxVolumeIndexYZ, j+blob); y++) {
						float dYSqr = (j - y) * (j - y);
						for (int x = std::max(0, k-blob); x <= std::min(maxVolumeIndexX, k+blob); x++) {
							float dXSqr = (k - x) * (k - x);
							float distanceSqr = dZSqr + dYSqr + dXSqr;
							if (distanceSqr > blobSizeSqr) {
								continue;
							}
							int aux = (int) ((distanceSqr * iDeltaSqrt + 0.5)); //Same as ROUND but avoid comparison
							float tmpWeight = blobTableSqrt[aux];
							tmp += tmpWeight * input[z][y][x];
						}
					}
				}
				output[i][j][k] = tmp;
			}
		}
	}
	// free original data
	release(input, maxVolumeIndexYZ+1, maxVolumeIndexYZ+1);
	return output;
}

#endif //XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_STARPU_UTIL_H_
