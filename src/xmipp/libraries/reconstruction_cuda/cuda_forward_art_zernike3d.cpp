// Xmipp includes
#include "cuda_forward_art_zernike3d.h"
#include <core/geometry.h>
#include "cuda_forward_art_zernike3d.cu"
#include "data/numerical_tools.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include "data/numerical_tools.h"

namespace cuda_forward_art_zernike3D {

// Cuda memory helper function
namespace {

	void processCudaError()
	{
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(err));
		}
	}

	// Copies data from CPU to the GPU
	template<typename T>
	void transportData(T **dest, const T *source, size_t n)
	{
		if (cudaMalloc(dest, sizeof(T) * n) != cudaSuccess) {
			processCudaError();
		}

		if (cudaMemcpy(*dest, source, sizeof(T) * n, cudaMemcpyHostToDevice) != cudaSuccess) {
			cudaFree(*dest);
			processCudaError();
		}
	}

	// Copies data from GPU to the CPU
	template<typename T>
	void transportDataFromGPU(T *dest, const T *source, size_t n)
	{
		if (cudaMemcpy(dest, source, sizeof(T) * n, cudaMemcpyDeviceToHost) != cudaSuccess) {
			processCudaError();
		}
	}

	template<typename T>
	T *transportMultidimArrayToGpu(const MultidimArray<T> &inputArray)
	{
		T *outputArrayData;
		transportData(&outputArrayData, inputArray.data, inputArray.xdim * inputArray.ydim * inputArray.zdim);
		return outputArrayData;
	}

	template<typename T>
	MultidimArrayCuda<T> *transportVectorOfMultidimArrayToGpu(const std::vector<MultidimArrayCuda<T>> &inputVector)
	{
		MultidimArrayCuda<T> *outputVectorData;
		transportData(&outputVectorData, inputVector.data(), inputVector.size());
		return outputVectorData;
	}

	template<typename T>
	T *transportMatrix1DToGpu(const Matrix1D<T> &inputVector)
	{
		T *outputVector;
		transportData(&outputVector, inputVector.vdata, inputVector.vdim);
		return outputVector;
	}

	template<typename T>
	T *transportStdVectorToGpu(const std::vector<T> &inputVector)
	{
		T *outputVector;
		transportData(&outputVector, inputVector.data(), inputVector.size());
		return outputVector;
	}

	template<typename T>
	T *transportMatrix2DToGpu(const Matrix2D<T> &inputMatrix)
	{
		T *outputMatrixData;
		transportData(&outputMatrixData, inputMatrix.mdata, inputMatrix.mdim);
		return outputMatrixData;
	}

	template<typename T>
	MultidimArrayCuda<T> initializeMultidimArrayCuda(const MultidimArray<T> &multidimArray)
	{
		struct MultidimArrayCuda<T> cudaArray = {
			.xdim = static_cast<unsigned>(multidimArray.xdim), .ydim = static_cast<unsigned>(multidimArray.ydim),
			.yxdim = static_cast<unsigned>(multidimArray.yxdim), .xinit = multidimArray.xinit,
			.yinit = multidimArray.yinit, .zinit = multidimArray.zinit,
			.data = transportMultidimArrayToGpu(multidimArray)
		};

		return cudaArray;
	}

	template<typename T>
	void updateMultidimArrayWithGPUData(MultidimArray<T> &multidimArray, const MultidimArrayCuda<T> &multidimArrayCuda)
	{
		transportDataFromGPU(
			multidimArray.data, multidimArrayCuda.data, multidimArray.xdim * multidimArray.ydim * multidimArray.zdim);
	}

	template<typename T>
	void updateVectorOfMultidimArrayWithGPUData(std::vector<Image<T>> &image,
												const std::vector<MultidimArrayCuda<T>> vectorMultidimArray)
	{
		assert(image.size() == vectorMultidimArray.size());
		for (int m = 0; m < image.size(); m++) {
			updateMultidimArrayWithGPUData(image[m](), vectorMultidimArray[m]);
		}
	}

	template<typename T>
	std::pair<MultidimArrayCuda<T> *, std::vector<MultidimArrayCuda<T>>> convertToMultidimArrayCuda(
		std::vector<Image<T>> &image)
	{
		std::vector<MultidimArrayCuda<T>> output;
		for (int m = 0; m < image.size(); m++) {
			output.push_back(initializeMultidimArrayCuda(image[m]()));
		}
		return std::make_pair(transportVectorOfMultidimArrayToGpu(output), output);
	}

	template<typename T>
	void freeCommonArgumentsKernel(struct Program<T>::CommonKernelParameters &commonParameters)
	{
		cudaFree(commonParameters.cudaClnm);
		//cudaFree(commonParameters.cudaR);
	}

	template<typename T>
	void freeVectorOfMultidimArray(std::vector<MultidimArrayCuda<T>> vector)
	{
		for (int m = 0; m < vector.size(); m++) {
			cudaFree(vector[m].data);
		}
	}

	template<typename T>
	Matrix2D<T> createRotationMatrix(const struct Program<T>::AngleParameters angles)
	{
		auto rot = angles.rot;
		auto tilt = angles.tilt;
		auto psi = angles.psi;
		constexpr size_t matrixSize = 3;
		auto tmp = Matrix2D<T>();
		tmp.initIdentity(matrixSize);
		Euler_angles2matrix(rot, tilt, psi, tmp, false);
		return tmp;
	}

	template<typename T>
	struct Program<T>::CommonKernelParameters getCommonArgumentsKernel(
		const struct Program<T>::DynamicParameters &parameters,
		const bool usesZernike,
		const int RmaxDef) {
		auto clnm = parameters.clnm;
		auto angles = parameters.angles;

		const size_t idxY0 = usesZernike ? (clnm.size() / 3) : 0;
		const size_t idxZ0 = usesZernike ? (2 * idxY0) : 0;
		const T RmaxF = usesZernike ? RmaxDef : 0;
		const T iRmaxF = usesZernike ? (1.0f / RmaxF) : 0;

		const Matrix2D<T> R = createRotationMatrix<T>(angles);

		struct Program<T>::CommonKernelParameters output = {
			.idxY0 = idxY0, .idxZ0 = idxZ0, .iRmaxF = iRmaxF, .cudaClnm = transportStdVectorToGpu(clnm), .R = R,
		};

		return output;

	}

	/*struct BlockSizes
	blockSizeArchitecture()

	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		struct BlockSizes output;
		switch (prop.major * 10 + prop.minor) {
			// 3.0 - 3.7 Kepler
			case 30 ... 37:
				output = {16, 4, 2};
				break;

			// 6.0 - 6.2 Pascal
			case 60 ... 62:
				output = {16, 8, 1};
				break;

			// 7.5 Turing
			case 75:
				output = {16, 8, 1};
				break;

			// 8.0 - 8.7 Ampere
			case 80 ... 87:
				output = {32, 1, 4};
				break;

			default:
				output = {8, 4, 4};
				break;
		}
		return output;
	}*/

	template<typename T>
	cudaTextureObject_t initTextureMultidimArray(MultidimArrayCuda<T> &array, size_t zdim)

	{
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = array.data;
		resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		resDesc.res.linear.desc.x = 32;
		resDesc.res.linear.sizeInBytes = zdim * array.yxdim * sizeof(T);
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;
		cudaTextureObject_t tex = 0;
		cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
		return tex;
	}

	template<typename T>
	bool checkStep(MultidimArray<T> &mask, int step, size_t position)

	{
		if (position % mask.xdim % step != 0) {
			return false;
		}
		if (position / mask.xdim % mask.ydim % step != 0) {
			return false;
		}
		if (position / mask.yxdim % step != 0) {
			return false;
		}
		return mask[position] != 0;
	}

	template<typename T>
	std::tuple<unsigned *, size_t> filterMaskTransportCoordinates(MultidimArray<T> &mask, int step)

	{
		std::vector<unsigned> coordinates;
		for (unsigned i = 0; i < static_cast<unsigned>(mask.yxdim * mask.zdim); i++) {
			if (checkStep(mask, step, static_cast<size_t>(i))) {
				coordinates.push_back(i);
			}
		}
		unsigned *coordinatesCuda = transportStdVectorToGpu(coordinates);
		return std::make_tuple(coordinatesCuda, coordinates.size());
	}

	template<typename T>
	std::tuple<unsigned *, size_t, int *> filterMaskTransportCoordinates(MultidimArray<T> &mask,
																		 int step,
																		 bool transportValues)

	{
		std::vector<unsigned> coordinates;
		std::vector<T> values;
		for (unsigned i = 0; i < static_cast<unsigned>(mask.yxdim * mask.zdim); i++) {
			if (checkStep(mask, step, static_cast<size_t>(i))) {
				coordinates.push_back(i);
				if (transportValues) {
					values.push_back(mask[i]);
				}
			}
		}
		unsigned *coordinatesCuda = transportStdVectorToGpu(coordinates);
		int *valuesCuda = transportStdVectorToGpu(values);
		return std::make_tuple(coordinatesCuda, coordinates.size(), valuesCuda);
	}

	template<typename T>
	std::tuple<unsigned *, size_t> filterMaskTransportCoordinatesSubsquares(MultidimArray<T> &mask,
																			int step,
																			int perimeter)

	{
		unsigned startX = (perimeter % 2) * mask.xdim / 2;
		unsigned endX = mask.xdim - ((perimeter + 1) % 2) * ceil((float)mask.xdim / 2);
		unsigned startY = (perimeter / 2 % 2) * mask.ydim / 2;
		unsigned endY = mask.ydim - ((perimeter + 2) / 2 % 2) * ceil((float)mask.ydim / 2);
		unsigned startZ = (perimeter / 4 % 2) * mask.zdim / 2;
		unsigned endZ = mask.zdim - ((perimeter + 4) / 4 % 2) * ceil((float)mask.zdim / 2);
		std::vector<unsigned> coordinates;
		for (unsigned x = startX; x < endX; x++) {
			for (unsigned y = startY; y < endY; y++) {
				for (unsigned z = startZ; z < endZ; z++) {
					unsigned xyz = x + y * mask.ydim + z * mask.yxdim;
					if (checkStep(mask, step, static_cast<size_t>(xyz))) {
						coordinates.push_back(xyz);
					}
				}
			}
		}
		unsigned *coordinatesCuda = transportStdVectorToGpu(coordinates);
		return std::make_tuple(coordinatesCuda, coordinates.size());
	}

	template<typename T>
	std::tuple<unsigned *, size_t, int *> filterMaskTransportCoordinatesSubsquares(MultidimArray<T> &mask,
																				   int step,
																				   bool transportValues,
																				   int perimeter)

	{
		unsigned startX = (perimeter % 2) * mask.xdim / 2;
		unsigned endX = mask.xdim - ((perimeter + 1) % 2) * ceil((float)mask.xdim / 2);
		unsigned startY = (perimeter / 2 % 2) * mask.ydim / 2;
		unsigned endY = mask.ydim - ((perimeter + 2) / 2 % 2) * ceil((float)mask.ydim / 2);
		unsigned startZ = (perimeter / 4 % 2) * mask.zdim / 2;
		unsigned endZ = mask.zdim - ((perimeter + 4) / 4 % 2) * ceil((float)mask.zdim / 2);
		std::vector<unsigned> coordinates;
		std::vector<T> values;
		for (unsigned x = startX; x < endX; x++) {
			for (unsigned y = startY; y < endY; y++) {
				for (unsigned z = startZ; z < endZ; z++) {
					unsigned xyz = x + y * mask.ydim + z * mask.yxdim;
					if (checkStep(mask, step, static_cast<size_t>(xyz))) {
						coordinates.push_back(xyz);
						if (transportValues) {
							values.push_back(mask[xyz]);
						}
					}
				}
			}
		}
		unsigned *coordinatesCuda = transportStdVectorToGpu(coordinates);
		int *valuesCuda = transportStdVectorToGpu(values);
		return std::make_tuple(coordinatesCuda, coordinates.size(), valuesCuda);
	}

	std::tuple<unsigned *, size_t, size_t, size_t, unsigned *, size_t, int *, size_t, size_t> fillBlockGridSubsquares(
		MultidimArray<int> &VRecMaskB,
		MultidimArray<int> &VRecMaskF,
		const int loopStep,
		int perimeter)
	{
		unsigned *cudaCoordinatesBX;
		size_t sizeBX;
		unsigned *cudaCoordinatesFX;
		size_t sizeFX;
		int *VRecMaskFX;
		std::tie(cudaCoordinatesBX, sizeBX) = filterMaskTransportCoordinatesSubsquares(VRecMaskB, 1, perimeter);
		auto optimalizedSize = ceil(sizeBX / BLOCK_SIZE) * BLOCK_SIZE;
		size_t blockXX = std::__gcd(BLOCK_SIZE, static_cast<int>(optimalizedSize));
		size_t gridXX = optimalizedSize / blockXX;
		std::tie(cudaCoordinatesFX, sizeFX, VRecMaskFX) =
			filterMaskTransportCoordinatesSubsquares(VRecMaskF, loopStep, true, perimeter);
		optimalizedSize = ceil(sizeFX / BLOCK_SIZE) * BLOCK_SIZE;
		size_t blockXXStep = std::__gcd(BLOCK_SIZE, static_cast<int>(optimalizedSize));
		size_t gridXXStep = optimalizedSize / blockXXStep;
		return std::make_tuple(
			cudaCoordinatesBX, sizeBX, blockXX, gridXX, cudaCoordinatesFX, sizeFX, VRecMaskFX, blockXXStep, gridXXStep);
	}

}  // namespace

template<typename PrecisionType>
Program<PrecisionType>::Program(const Program<PrecisionType>::ConstantParameters parameters)
	: cudaMV(initializeMultidimArrayCuda(parameters.Vrefined())),
	  sigma(parameters.sigma),
	  cudaSigma(transportStdVectorToGpu(parameters.sigma)),
	  RmaxDef(parameters.RmaxDef),
	  lastX(FINISHINGX(parameters.Vrefined())),
	  lastY(FINISHINGY(parameters.Vrefined())),
	  lastZ(FINISHINGZ(parameters.Vrefined())),
	  loopStep(parameters.loopStep),
	  cudaVL1(transportMatrix1DToGpu(parameters.vL1)),
	  cudaVL2(transportMatrix1DToGpu(parameters.vL2)),
	  cudaVN(transportMatrix1DToGpu(parameters.vN)),
	  cudaVM(transportMatrix1DToGpu(parameters.vM)),
	  xdimB(static_cast<unsigned>(parameters.VRecMaskB.xdim)),
	  ydimB(static_cast<unsigned>(parameters.VRecMaskB.ydim)),
	  xdimF(parameters.VRecMaskF.xdim),
	  ydimF(parameters.VRecMaskF.ydim)
{
	std::tie(
		cudaCoordinatesB0, sizeB0, blockX0, gridX0, cudaCoordinatesF0, sizeF0, VRecMaskF0, blockXStep0, gridXStep0) =
		fillBlockGridSubsquares(parameters.VRecMaskB, parameters.VRecMaskF, parameters.loopStep, 0);
	std::tie(
		cudaCoordinatesB1, sizeB1, blockX1, gridX1, cudaCoordinatesF1, sizeF1, VRecMaskF1, blockXStep1, gridXStep1) =
		fillBlockGridSubsquares(parameters.VRecMaskB, parameters.VRecMaskF, parameters.loopStep, 1);
	std::tie(
		cudaCoordinatesB2, sizeB2, blockX2, gridX2, cudaCoordinatesF2, sizeF2, VRecMaskF2, blockXStep2, gridXStep2) =
		fillBlockGridSubsquares(parameters.VRecMaskB, parameters.VRecMaskF, parameters.loopStep, 2);
	std::tie(
		cudaCoordinatesB3, sizeB3, blockX3, gridX3, cudaCoordinatesF3, sizeF3, VRecMaskF3, blockXStep3, gridXStep3) =
		fillBlockGridSubsquares(parameters.VRecMaskB, parameters.VRecMaskF, parameters.loopStep, 3);
	std::tie(
		cudaCoordinatesB4, sizeB4, blockX4, gridX4, cudaCoordinatesF4, sizeF4, VRecMaskF4, blockXStep4, gridXStep4) =
		fillBlockGridSubsquares(parameters.VRecMaskB, parameters.VRecMaskF, parameters.loopStep, 4);
	std::tie(
		cudaCoordinatesB5, sizeB5, blockX5, gridX5, cudaCoordinatesF5, sizeF5, VRecMaskF5, blockXStep5, gridXStep5) =
		fillBlockGridSubsquares(parameters.VRecMaskB, parameters.VRecMaskF, parameters.loopStep, 5);
	std::tie(
		cudaCoordinatesB6, sizeB6, blockX6, gridX6, cudaCoordinatesF6, sizeF6, VRecMaskF6, blockXStep6, gridXStep6) =
		fillBlockGridSubsquares(parameters.VRecMaskB, parameters.VRecMaskF, parameters.loopStep, 6);
	std::tie(
		cudaCoordinatesB7, sizeB7, blockX7, gridX7, cudaCoordinatesF7, sizeF7, VRecMaskF7, blockXStep7, gridXStep7) =
		fillBlockGridSubsquares(parameters.VRecMaskB, parameters.VRecMaskF, parameters.loopStep, 7);
}

template<typename PrecisionType>
Program<PrecisionType>::~Program()
{
	cudaFree(VRecMaskF0);
	cudaFree(VRecMaskF1);
	cudaFree(VRecMaskF2);
	cudaFree(VRecMaskF3);
	cudaFree(VRecMaskF4);
	cudaFree(VRecMaskF5);
	cudaFree(VRecMaskF6);
	cudaFree(VRecMaskF7);
	cudaFree(cudaMV.data);
	cudaFree(cudaCoordinatesB0);
	cudaFree(cudaCoordinatesB1);
	cudaFree(cudaCoordinatesB2);
	cudaFree(cudaCoordinatesB3);
	cudaFree(cudaCoordinatesB4);
	cudaFree(cudaCoordinatesB5);
	cudaFree(cudaCoordinatesB6);
	cudaFree(cudaCoordinatesB7);
	cudaFree(cudaCoordinatesF0);
	cudaFree(cudaCoordinatesF1);
	cudaFree(cudaCoordinatesF2);
	cudaFree(cudaCoordinatesF3);
	cudaFree(cudaCoordinatesF4);
	cudaFree(cudaCoordinatesF5);
	cudaFree(cudaCoordinatesF6);
	cudaFree(cudaCoordinatesF7);
	cudaFree(const_cast<PrecisionType *>(cudaSigma));

	cudaFree(const_cast<int *>(cudaVL1));
	cudaFree(const_cast<int *>(cudaVL2));
	cudaFree(const_cast<int *>(cudaVN));
	cudaFree(const_cast<int *>(cudaVM));
}

template<typename PrecisionType>
template<bool usesZernike>
void Program<PrecisionType>::runForwardKernel(struct DynamicParameters &parameters)

{
	// Unique parameters
	MultidimArrayCuda<PrecisionType> *cudaP, *cudaW;
	std::vector<MultidimArrayCuda<PrecisionType>> pVector, wVector;
	std::tie(cudaP, pVector) = convertToMultidimArrayCuda(parameters.P);
	std::tie(cudaW, wVector) = convertToMultidimArrayCuda(parameters.W);
	unsigned sigma_size = static_cast<unsigned>(sigma.size());

	// Common parameters
	auto commonParameters = getCommonArgumentsKernel<PrecisionType>(parameters, usesZernike, RmaxDef);

	forwardKernel<PrecisionType, usesZernike>
		<<<gridXStep0, blockXStep0>>>(cudaMV,
									  VRecMaskF0,
									  cudaCoordinatesF0,
									  xdimF,
									  ydimF,
									  static_cast<unsigned>(sizeF0),
									  cudaP,
									  cudaW,
									  sigma_size,
									  cudaSigma,
									  commonParameters.iRmaxF,
									  static_cast<unsigned>(commonParameters.idxY0),
									  static_cast<unsigned>(commonParameters.idxZ0),
									  cudaVL1,
									  cudaVN,
									  cudaVL2,
									  cudaVM,
									  commonParameters.cudaClnm,
									  commonParameters.R.mdata[0],
									  commonParameters.R.mdata[1],
									  commonParameters.R.mdata[2],
									  commonParameters.R.mdata[3],
									  commonParameters.R.mdata[4],
									  commonParameters.R.mdata[5]);

	forwardKernel<PrecisionType, usesZernike>
		<<<gridXStep1, blockXStep1>>>(cudaMV,
									  VRecMaskF1,
									  cudaCoordinatesF1,
									  xdimF,
									  ydimF,
									  static_cast<unsigned>(sizeF1),
									  cudaP,
									  cudaW,
									  sigma_size,
									  cudaSigma,
									  commonParameters.iRmaxF,
									  static_cast<unsigned>(commonParameters.idxY0),
									  static_cast<unsigned>(commonParameters.idxZ0),
									  cudaVL1,
									  cudaVN,
									  cudaVL2,
									  cudaVM,
									  commonParameters.cudaClnm,
									  commonParameters.R.mdata[0],
									  commonParameters.R.mdata[1],
									  commonParameters.R.mdata[2],
									  commonParameters.R.mdata[3],
									  commonParameters.R.mdata[4],
									  commonParameters.R.mdata[5]);

	forwardKernel<PrecisionType, usesZernike>
		<<<gridXStep2, blockXStep2>>>(cudaMV,
									  VRecMaskF2,
									  cudaCoordinatesF2,
									  xdimF,
									  ydimF,
									  static_cast<unsigned>(sizeF2),
									  cudaP,
									  cudaW,
									  sigma_size,
									  cudaSigma,
									  commonParameters.iRmaxF,
									  static_cast<unsigned>(commonParameters.idxY0),
									  static_cast<unsigned>(commonParameters.idxZ0),
									  cudaVL1,
									  cudaVN,
									  cudaVL2,
									  cudaVM,
									  commonParameters.cudaClnm,
									  commonParameters.R.mdata[0],
									  commonParameters.R.mdata[1],
									  commonParameters.R.mdata[2],
									  commonParameters.R.mdata[3],
									  commonParameters.R.mdata[4],
									  commonParameters.R.mdata[5]);

	forwardKernel<PrecisionType, usesZernike>
		<<<gridXStep3, blockXStep3>>>(cudaMV,
									  VRecMaskF3,
									  cudaCoordinatesF3,
									  xdimF,
									  ydimF,
									  static_cast<unsigned>(sizeF3),
									  cudaP,
									  cudaW,
									  sigma_size,
									  cudaSigma,
									  commonParameters.iRmaxF,
									  static_cast<unsigned>(commonParameters.idxY0),
									  static_cast<unsigned>(commonParameters.idxZ0),
									  cudaVL1,
									  cudaVN,
									  cudaVL2,
									  cudaVM,
									  commonParameters.cudaClnm,
									  commonParameters.R.mdata[0],
									  commonParameters.R.mdata[1],
									  commonParameters.R.mdata[2],
									  commonParameters.R.mdata[3],
									  commonParameters.R.mdata[4],
									  commonParameters.R.mdata[5]);

	forwardKernel<PrecisionType, usesZernike>
		<<<gridXStep4, blockXStep4>>>(cudaMV,
									  VRecMaskF4,
									  cudaCoordinatesF4,
									  xdimF,
									  ydimF,
									  static_cast<unsigned>(sizeF4),
									  cudaP,
									  cudaW,
									  sigma_size,
									  cudaSigma,
									  commonParameters.iRmaxF,
									  static_cast<unsigned>(commonParameters.idxY0),
									  static_cast<unsigned>(commonParameters.idxZ0),
									  cudaVL1,
									  cudaVN,
									  cudaVL2,
									  cudaVM,
									  commonParameters.cudaClnm,
									  commonParameters.R.mdata[0],
									  commonParameters.R.mdata[1],
									  commonParameters.R.mdata[2],
									  commonParameters.R.mdata[3],
									  commonParameters.R.mdata[4],
									  commonParameters.R.mdata[5]);

	forwardKernel<PrecisionType, usesZernike>
		<<<gridXStep5, blockXStep5>>>(cudaMV,
									  VRecMaskF5,
									  cudaCoordinatesF5,
									  xdimF,
									  ydimF,
									  static_cast<unsigned>(sizeF5),
									  cudaP,
									  cudaW,
									  sigma_size,
									  cudaSigma,
									  commonParameters.iRmaxF,
									  static_cast<unsigned>(commonParameters.idxY0),
									  static_cast<unsigned>(commonParameters.idxZ0),
									  cudaVL1,
									  cudaVN,
									  cudaVL2,
									  cudaVM,
									  commonParameters.cudaClnm,
									  commonParameters.R.mdata[0],
									  commonParameters.R.mdata[1],
									  commonParameters.R.mdata[2],
									  commonParameters.R.mdata[3],
									  commonParameters.R.mdata[4],
									  commonParameters.R.mdata[5]);

	forwardKernel<PrecisionType, usesZernike>
		<<<gridXStep6, blockXStep6>>>(cudaMV,
									  VRecMaskF6,
									  cudaCoordinatesF6,
									  xdimF,
									  ydimF,
									  static_cast<unsigned>(sizeF6),
									  cudaP,
									  cudaW,
									  sigma_size,
									  cudaSigma,
									  commonParameters.iRmaxF,
									  static_cast<unsigned>(commonParameters.idxY0),
									  static_cast<unsigned>(commonParameters.idxZ0),
									  cudaVL1,
									  cudaVN,
									  cudaVL2,
									  cudaVM,
									  commonParameters.cudaClnm,
									  commonParameters.R.mdata[0],
									  commonParameters.R.mdata[1],
									  commonParameters.R.mdata[2],
									  commonParameters.R.mdata[3],
									  commonParameters.R.mdata[4],
									  commonParameters.R.mdata[5]);

	forwardKernel<PrecisionType, usesZernike>
		<<<gridXStep7, blockXStep7>>>(cudaMV,
									  VRecMaskF7,
									  cudaCoordinatesF7,
									  xdimF,
									  ydimF,
									  static_cast<unsigned>(sizeF7),
									  cudaP,
									  cudaW,
									  sigma_size,
									  cudaSigma,
									  commonParameters.iRmaxF,
									  static_cast<unsigned>(commonParameters.idxY0),
									  static_cast<unsigned>(commonParameters.idxZ0),
									  cudaVL1,
									  cudaVN,
									  cudaVL2,
									  cudaVM,
									  commonParameters.cudaClnm,
									  commonParameters.R.mdata[0],
									  commonParameters.R.mdata[1],
									  commonParameters.R.mdata[2],
									  commonParameters.R.mdata[3],
									  commonParameters.R.mdata[4],
									  commonParameters.R.mdata[5]);
	cudaDeviceSynchronize();

	updateVectorOfMultidimArrayWithGPUData(parameters.P, pVector);
	updateVectorOfMultidimArrayWithGPUData(parameters.W, wVector);

	freeVectorOfMultidimArray(pVector);
	freeVectorOfMultidimArray(wVector);
	cudaFree(cudaP);
	cudaFree(cudaW);
	freeCommonArgumentsKernel<PrecisionType>(commonParameters);
}

template<typename PrecisionType>
template<bool usesZernike>
void Program<PrecisionType>::runBackwardKernel(struct DynamicParameters &parameters)
{
	// Unique parameters
	auto &mId = parameters.Idiff();
	auto cudaMId = initializeMultidimArrayCuda(mId);

	// Texture
	cudaTextureObject_t mIdTexture = initTextureMultidimArray<PrecisionType>(cudaMId, mId.zdim);

	// Common parameters
	auto commonParameters = getCommonArgumentsKernel<PrecisionType>(parameters, usesZernike, RmaxDef);

	backwardKernel<PrecisionType, usesZernike><<<gridX0, blockX0>>>(cudaMV,
																	cudaCoordinatesB0,
																	xdimB,
																	ydimB,
																	static_cast<unsigned>(sizeB0),
																	commonParameters.iRmaxF,
																	static_cast<unsigned>(commonParameters.idxY0),
																	static_cast<unsigned>(commonParameters.idxZ0),
																	cudaVL1,
																	cudaVN,
																	cudaVL2,
																	cudaVM,
																	commonParameters.cudaClnm,
																	commonParameters.R.mdata[0],
																	commonParameters.R.mdata[1],
																	commonParameters.R.mdata[2],
																	commonParameters.R.mdata[3],
																	commonParameters.R.mdata[4],
																	commonParameters.R.mdata[5],
																	mIdTexture,
																	mId.xinit,
																	mId.yinit,
																	static_cast<int>(mId.xdim),
																	static_cast<int>(mId.ydim));


	backwardKernel<PrecisionType, usesZernike><<<gridX1, blockX1>>>(cudaMV,
																	cudaCoordinatesB1,
																	xdimB,
																	ydimB,
																	static_cast<unsigned>(sizeB1),
																	commonParameters.iRmaxF,
																	static_cast<unsigned>(commonParameters.idxY0),
																	static_cast<unsigned>(commonParameters.idxZ0),
																	cudaVL1,
																	cudaVN,
																	cudaVL2,
																	cudaVM,
																	commonParameters.cudaClnm,
																	commonParameters.R.mdata[0],
																	commonParameters.R.mdata[1],
																	commonParameters.R.mdata[2],
																	commonParameters.R.mdata[3],
																	commonParameters.R.mdata[4],
																	commonParameters.R.mdata[5],
																	mIdTexture,
																	mId.xinit,
																	mId.yinit,
																	static_cast<int>(mId.xdim),
																	static_cast<int>(mId.ydim));

	backwardKernel<PrecisionType, usesZernike><<<gridX2, blockX2>>>(cudaMV,
																	cudaCoordinatesB2,
																	xdimB,
																	ydimB,
																	static_cast<unsigned>(sizeB2),
																	commonParameters.iRmaxF,
																	static_cast<unsigned>(commonParameters.idxY0),
																	static_cast<unsigned>(commonParameters.idxZ0),
																	cudaVL1,
																	cudaVN,
																	cudaVL2,
																	cudaVM,
																	commonParameters.cudaClnm,
																	commonParameters.R.mdata[0],
																	commonParameters.R.mdata[1],
																	commonParameters.R.mdata[2],
																	commonParameters.R.mdata[3],
																	commonParameters.R.mdata[4],
																	commonParameters.R.mdata[5],
																	mIdTexture,
																	mId.xinit,
																	mId.yinit,
																	static_cast<int>(mId.xdim),
																	static_cast<int>(mId.ydim));

	backwardKernel<PrecisionType, usesZernike><<<gridX3, blockX3>>>(cudaMV,
																	cudaCoordinatesB3,
																	xdimB,
																	ydimB,
																	static_cast<unsigned>(sizeB3),
																	commonParameters.iRmaxF,
																	static_cast<unsigned>(commonParameters.idxY0),
																	static_cast<unsigned>(commonParameters.idxZ0),
																	cudaVL1,
																	cudaVN,
																	cudaVL2,
																	cudaVM,
																	commonParameters.cudaClnm,
																	commonParameters.R.mdata[0],
																	commonParameters.R.mdata[1],
																	commonParameters.R.mdata[2],
																	commonParameters.R.mdata[3],
																	commonParameters.R.mdata[4],
																	commonParameters.R.mdata[5],
																	mIdTexture,
																	mId.xinit,
																	mId.yinit,
																	static_cast<int>(mId.xdim),
																	static_cast<int>(mId.ydim));

	backwardKernel<PrecisionType, usesZernike><<<gridX4, blockX4>>>(cudaMV,
																	cudaCoordinatesB4,
																	xdimB,
																	ydimB,
																	static_cast<unsigned>(sizeB4),
																	commonParameters.iRmaxF,
																	static_cast<unsigned>(commonParameters.idxY0),
																	static_cast<unsigned>(commonParameters.idxZ0),
																	cudaVL1,
																	cudaVN,
																	cudaVL2,
																	cudaVM,
																	commonParameters.cudaClnm,
																	commonParameters.R.mdata[0],
																	commonParameters.R.mdata[1],
																	commonParameters.R.mdata[2],
																	commonParameters.R.mdata[3],
																	commonParameters.R.mdata[4],
																	commonParameters.R.mdata[5],
																	mIdTexture,
																	mId.xinit,
																	mId.yinit,
																	static_cast<int>(mId.xdim),
																	static_cast<int>(mId.ydim));

	backwardKernel<PrecisionType, usesZernike><<<gridX5, blockX5>>>(cudaMV,
																	cudaCoordinatesB5,
																	xdimB,
																	ydimB,
																	static_cast<unsigned>(sizeB5),
																	commonParameters.iRmaxF,
																	static_cast<unsigned>(commonParameters.idxY0),
																	static_cast<unsigned>(commonParameters.idxZ0),
																	cudaVL1,
																	cudaVN,
																	cudaVL2,
																	cudaVM,
																	commonParameters.cudaClnm,
																	commonParameters.R.mdata[0],
																	commonParameters.R.mdata[1],
																	commonParameters.R.mdata[2],
																	commonParameters.R.mdata[3],
																	commonParameters.R.mdata[4],
																	commonParameters.R.mdata[5],
																	mIdTexture,
																	mId.xinit,
																	mId.yinit,
																	static_cast<int>(mId.xdim),
																	static_cast<int>(mId.ydim));

	backwardKernel<PrecisionType, usesZernike><<<gridX6, blockX6>>>(cudaMV,
																	cudaCoordinatesB6,
																	xdimB,
																	ydimB,
																	static_cast<unsigned>(sizeB6),
																	commonParameters.iRmaxF,
																	static_cast<unsigned>(commonParameters.idxY0),
																	static_cast<unsigned>(commonParameters.idxZ0),
																	cudaVL1,
																	cudaVN,
																	cudaVL2,
																	cudaVM,
																	commonParameters.cudaClnm,
																	commonParameters.R.mdata[0],
																	commonParameters.R.mdata[1],
																	commonParameters.R.mdata[2],
																	commonParameters.R.mdata[3],
																	commonParameters.R.mdata[4],
																	commonParameters.R.mdata[5],
																	mIdTexture,
																	mId.xinit,
																	mId.yinit,
																	static_cast<int>(mId.xdim),
																	static_cast<int>(mId.ydim));

	backwardKernel<PrecisionType, usesZernike><<<gridX7, blockX7>>>(cudaMV,
																	cudaCoordinatesB7,
																	xdimB,
																	ydimB,
																	static_cast<unsigned>(sizeB7),
																	commonParameters.iRmaxF,
																	static_cast<unsigned>(commonParameters.idxY0),
																	static_cast<unsigned>(commonParameters.idxZ0),
																	cudaVL1,
																	cudaVN,
																	cudaVL2,
																	cudaVM,
																	commonParameters.cudaClnm,
																	commonParameters.R.mdata[0],
																	commonParameters.R.mdata[1],
																	commonParameters.R.mdata[2],
																	commonParameters.R.mdata[3],
																	commonParameters.R.mdata[4],
																	commonParameters.R.mdata[5],
																	mIdTexture,
																	mId.xinit,
																	mId.yinit,
																	static_cast<int>(mId.xdim),
																	static_cast<int>(mId.ydim));

	cudaDeviceSynchronize();

	cudaFree(cudaMId.data);
	freeCommonArgumentsKernel<PrecisionType>(commonParameters);
}

template<typename PrecisionType>
void Program<PrecisionType>::recoverVolumeFromGPU(Image<PrecisionType> &Vrefined)
{
	updateMultidimArrayWithGPUData(Vrefined(), cudaMV);
}

// explicit template instantiation
template class Program<float>;
template void Program<float>::runForwardKernel<true>(struct DynamicParameters &);
template void Program<float>::runForwardKernel<false>(struct DynamicParameters &);
template void Program<float>::runBackwardKernel<true>(struct DynamicParameters &);
template void Program<float>::runBackwardKernel<false>(struct DynamicParameters &);
}  // namespace cuda_forward_art_zernike3D
