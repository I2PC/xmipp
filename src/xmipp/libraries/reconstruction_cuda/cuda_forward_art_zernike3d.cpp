// Xmipp includes
#include "cuda_forward_art_zernike3d.h"
#include <core/geometry.h>
#include "cuda_forward_art_zernike3d.cu"
#include "data/numerical_tools.h"

#include <algorithm>
#include <cassert>
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
			.xdim = multidimArray.xdim, .ydim = multidimArray.ydim, .yxdim = multidimArray.yxdim,
			.xinit = multidimArray.xinit, .yinit = multidimArray.yinit, .zinit = multidimArray.zinit,
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
		cudaFree(commonParameters.cudaR);
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
			.idxY0 = idxY0, .idxZ0 = idxZ0, .iRmaxF = iRmaxF, .cudaClnm = transportStdVectorToGpu(clnm),
			.cudaR = transportMatrix2DToGpu(R),
		};

		return output;

	}

	struct BlockSizes
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
	}

}  // namespace

template<typename PrecisionType>
Program<PrecisionType>::Program(const Program<PrecisionType>::ConstantParameters parameters)
	: cudaMV(initializeMultidimArrayCuda(parameters.Vrefined())),
	  VRecMaskF(initializeMultidimArrayCuda(parameters.VRecMaskF)),
	  VRecMaskB(initializeMultidimArrayCuda(parameters.VRecMaskB)),
	  sigma(parameters.sigma),
	  RmaxDef(parameters.RmaxDef),
	  lastX(FINISHINGX(parameters.Vrefined())),
	  lastY(FINISHINGY(parameters.Vrefined())),
	  lastZ(FINISHINGZ(parameters.Vrefined())),
	  loopStep(parameters.loopStep),
	  cudaVL1(transportMatrix1DToGpu(parameters.vL1)),
	  cudaVL2(transportMatrix1DToGpu(parameters.vL2)),
	  cudaVN(transportMatrix1DToGpu(parameters.vN)),
	  cudaVM(transportMatrix1DToGpu(parameters.vM)),
	  blockX(std::__gcd(blockSizeArchitecture().x, parameters.Vrefined().xdim)),
	  blockY(std::__gcd(blockSizeArchitecture().y, parameters.Vrefined().ydim)),
	  blockZ(std::__gcd(blockSizeArchitecture().z, parameters.Vrefined().zdim)),
	  gridX(parameters.Vrefined().xdim / blockX),
	  gridY(parameters.Vrefined().ydim / blockY),
	  gridZ(parameters.Vrefined().zdim / blockZ),
	  blockXStep(std::__gcd(blockSizeArchitecture().x, parameters.Vrefined().xdim / loopStep)),
	  blockYStep(std::__gcd(blockSizeArchitecture().y, parameters.Vrefined().ydim / loopStep)),
	  blockZStep(std::__gcd(blockSizeArchitecture().z, parameters.Vrefined().zdim / loopStep)),
	  gridXStep(parameters.Vrefined().xdim / loopStep / blockXStep),
	  gridYStep(parameters.Vrefined().ydim / loopStep / blockYStep),
	  gridZStep(parameters.Vrefined().zdim / loopStep / blockZStep)
{}

template<typename PrecisionType>
Program<PrecisionType>::~Program()
{
	cudaFree(VRecMaskF.data);
	cudaFree(VRecMaskB.data);
	cudaFree(cudaMV.data);

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
	auto sigma_size = sigma.size();
	auto cudaSigma = transportStdVectorToGpu(sigma);
	const int step = loopStep;

	// Common parameters
	auto commonParameters = getCommonArgumentsKernel<PrecisionType>(parameters, usesZernike, RmaxDef);

	forwardKernel<PrecisionType, usesZernike>
		<<<dim3(gridXStep, gridYStep, gridZStep), dim3(blockXStep, blockYStep, blockZStep)>>>(cudaMV,
																							  VRecMaskF,
																							  cudaP,
																							  cudaW,
																							  lastZ,
																							  lastY,
																							  lastX,
																							  step,
																							  sigma_size,
																							  cudaSigma,
																							  commonParameters.iRmaxF,
																							  commonParameters.idxY0,
																							  commonParameters.idxZ0,
																							  cudaVL1,
																							  cudaVN,
																							  cudaVL2,
																							  cudaVM,
																							  commonParameters.cudaClnm,
																							  commonParameters.cudaR);

	cudaDeviceSynchronize();

	updateVectorOfMultidimArrayWithGPUData(parameters.P, pVector);
	updateVectorOfMultidimArrayWithGPUData(parameters.W, wVector);

	freeVectorOfMultidimArray(pVector);
	freeVectorOfMultidimArray(wVector);
	cudaFree(cudaP);
	cudaFree(cudaW);
	cudaFree(cudaSigma);
	freeCommonArgumentsKernel<PrecisionType>(commonParameters);
}

template<typename PrecisionType>
template<bool usesZernike>
void Program<PrecisionType>::runBackwardKernel(struct DynamicParameters &parameters)
{
	// Unique parameters
	auto &mId = parameters.Idiff();

	//auto cudaMId = initializeMultidimArrayCuda(mId);

	struct MultidimArrayCuda<PrecisionType> cudaMId = {
		.xdim = mId.xdim, .ydim = mId.ydim, .yxdim = mId.yxdim, .xinit = mId.xinit, .yinit = mId.yinit,
		.zinit = mId.zinit
	};

	texture<PrecisionType, 2, cudaReadModeElementType> tex;
	size_t pitch, tex_ofs;
	PrecisionType *mIdTexture = 0;

	cudaMallocPitch((void **)&mIdTexture, &pitch, mId.xdim * sizeof(*mIdTexture), mId.ydim);
	cudaMemcpy2D(mIdTexture,
				 pitch,
				 mId.data,
				 mId.xdim * sizeof(PrecisionType),
				 mId.xdim * sizeof(PrecisionType),
				 mId.ydim,
				 cudaMemcpyHostToDevice);

	tex.normalized = false;
	tex.filterMode = cudaFilterModeLinear;

	cudaBindTexture2D(&tex_ofs, &tex, mIdTexture, &tex.channelDesc, mId.xdim, mId.ydim, pitch);

	const int step = 1;

	// Common parameters
	auto commonParameters = getCommonArgumentsKernel<PrecisionType>(parameters, usesZernike, RmaxDef);

	backwardKernel<PrecisionType, usesZernike>
		<<<dim3(gridX, gridY, gridZ), dim3(blockX, blockY, blockZ)>>>(cudaMV,
																	  cudaMId,
																	  VRecMaskB,
																	  lastZ,
																	  lastY,
																	  lastX,
																	  step,
																	  commonParameters.iRmaxF,
																	  commonParameters.idxY0,
																	  commonParameters.idxZ0,
																	  cudaVL1,
																	  cudaVN,
																	  cudaVL2,
																	  cudaVM,
																	  commonParameters.cudaClnm,
																	  commonParameters.cudaR,
																	  tex);

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
template class Program<double>;
template void Program<float>::runForwardKernel<true>(struct DynamicParameters &);
template void Program<float>::runForwardKernel<false>(struct DynamicParameters &);
template void Program<double>::runForwardKernel<true>(struct DynamicParameters &);
template void Program<double>::runForwardKernel<false>(struct DynamicParameters &);
template void Program<float>::runBackwardKernel<true>(struct DynamicParameters &);
template void Program<float>::runBackwardKernel<false>(struct DynamicParameters &);
template void Program<double>::runBackwardKernel<true>(struct DynamicParameters &);
template void Program<double>::runBackwardKernel<false>(struct DynamicParameters &);
}  // namespace cuda_forward_art_zernike3D
