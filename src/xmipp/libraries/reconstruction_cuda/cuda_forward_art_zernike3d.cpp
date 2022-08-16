// Xmipp includes
#include "cuda_forward_art_zernike3d.h"
#include <core/geometry.h>
#include "cuda_forward_art_zernike3d.cu"
#include "data/numerical_tools.h"

#include <cassert>
#include <stdexcept>
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
	T *tranportMultidimArrayToGpu(const MultidimArray<T> &inputArray)
	{
		T *outputArrayData;
		transportData(&outputArrayData, inputArray.data, inputArray.xdim * inputArray.ydim * inputArray.zdim);
		return outputArrayData;
	}

	template<typename T>
	MultidimArrayCuda<T> *tranportVectorOfMultidimArrayToGpu(const std::vector<MultidimArrayCuda<T>> &inputVector)
	{
		MultidimArrayCuda<T> *outputVectorData;
		transportData(&outputVectorData, inputVector.data(), inputVector.size());
		return outputVectorData;
	}

	template<typename T>
	T *tranportMatrix1DToGpu(const Matrix1D<T> &inputVector)
	{
		T *outputVector;
		transportData(&outputVector, inputVector.vdata, inputVector.vdim);
		return outputVector;
	}

	template<typename T>
	T *tranportStdVectorToGpu(const std::vector<T> &inputVector)
	{
		T *outputVector;
		transportData(&outputVector, inputVector.data(), inputVector.size());
		return outputVector;
	}

	template<typename T>
	T *tranportMatrix2DToGpu(const Matrix2D<T> &inputMatrix)
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
			.data = tranportMultidimArrayToGpu(multidimArray)
		};

		return cudaArray;
	}

	template<typename T>
	void updateMultidimArrayWithGPUData(MultidimArray<T> &multidimArray, const MultidimArrayCuda<T> &multidimArrayCuda)
	{
		assert(multidimArray.xdim * multidimArray.ydim * multidimArray.zdim
			   == multidimArrayCuda.xdim * multidimArrayCuda.ydim * multidimArrayCuda.zdim);
		transportDataFromGPU(
			multidimArray.data, multidimArrayCuda.data, multidimArray.xdim * multidimArray.ydim * multidimArray.zdim);
	}

	template<typename T>
	MultidimArrayCuda<T> *convertToMultidimArrayCuda(std::vector<Image<T>> &image)
	{
		std::vector<MultidimArrayCuda<T>> output;
		for (int m = 0; m < image.size(); m++) {
			output.push_back(initializeMultidimArrayCuda(image[m]()));
		}
		return tranportVectorOfMultidimArrayToGpu(output);
	}

}  // namespace

template<typename PrecisionType>
Program<PrecisionType>::Program(const Program<PrecisionType>::ConstantParameters parameters)
	: VRecMaskF(initializeMultidimArrayCuda(parameters.VRecMaskF)),
	  VRecMaskB(initializeMultidimArrayCuda(parameters.VRecMaskB)),
	  sigma(parameters.sigma),
	  RmaxDef(parameters.RmaxDef),
	  loopStep(parameters.loopStep),
	  cudaVL1(tranportMatrix1DToGpu(parameters.vL1)),
	  cudaVL2(tranportMatrix1DToGpu(parameters.vL2)),
	  cudaVN(tranportMatrix1DToGpu(parameters.vN)),
	  cudaVM(tranportMatrix1DToGpu(parameters.vM))
{}

template<typename PrecisionType>
Program<PrecisionType>::~Program()
{
	cudaFree(VRecMaskF.data);
	cudaFree(VRecMaskB.data);

	cudaFree(const_cast<int *>(cudaVL1));
	cudaFree(const_cast<int *>(cudaVL2));
	cudaFree(const_cast<int *>(cudaVN));
	cudaFree(const_cast<int *>(cudaVM));
}

template<typename PrecisionType>
template<bool usesZernike>
struct Program<PrecisionType>::CommonKernelParameters Program<PrecisionType>::setCommonArgumentsKernel(
	struct DynamicParameters &parameters) {
	auto clnm = parameters.clnm;
	auto angles = parameters.angles;

	// We can't set idxY0 to 0 because the compiler
	// would give irrelevant warnings.
	assert(usesZernike || clnm.size() == 0);
	const size_t idxY0 = clnm.size() / 3;
	const size_t idxZ0 = usesZernike ? (2 * idxY0) : 0;
	const PrecisionType RmaxF = usesZernike ? RmaxDef : 0;
	const PrecisionType iRmaxF = usesZernike ? (1.0f / RmaxF) : 0;

	// Rotation Matrix (has to pass the whole Matrix2D so it is not automatically deallocated)
	const Matrix2D<PrecisionType> R = createRotationMatrix(angles);

	CommonKernelParameters output = {
		.idxY0 = idxY0,
		.idxZ0 = idxZ0,
		.iRmaxF = iRmaxF,
		.cudaMV = initializeMultidimArrayCuda(parameters.Vrefined()),
		.cudaClnm = tranportStdVectorToGpu(clnm),
		.cudaR = tranportMatrix2DToGpu(R),
		.lastX = FINISHINGX(parameters.Vrefined()),
		.lastY = FINISHINGY(parameters.Vrefined()),
		.lastZ = FINISHINGZ(parameters.Vrefined()),
	};

	return output;

}

template<typename PrecisionType>
template<bool usesZernike>
void Program<PrecisionType>::runForwardKernel(struct DynamicParameters &parameters)

{
	// Unique parameters
	auto cudaP = convertToMultidimArrayCuda(parameters.P);
	auto cudaW = convertToMultidimArrayCuda(parameters.W);
	auto sigma_size = sigma.size();
	auto cudaSigma = tranportStdVectorToGpu(sigma);
	const int step = loopStep;

	// Common parameters
	auto commonParameters = setCommonArgumentsKernel<usesZernike>(parameters);
	auto idxY0 = commonParameters.idxY0;
	auto idxZ0 = commonParameters.idxZ0;
	auto iRmaxF = commonParameters.iRmaxF;
	auto cudaMV = commonParameters.cudaMV;
	auto cudaR = commonParameters.cudaR;
	auto cudaClnm = commonParameters.cudaClnm;
	auto lastZ = commonParameters.lastZ;
	auto lastY = commonParameters.lastY;
	auto lastX = commonParameters.lastX;

	forwardKernel<PrecisionType, usesZernike><<<1, 1>>>(cudaMV,
														VRecMaskF,
														cudaP,
														cudaW,
														lastZ,
														lastY,
														lastX,
														step,
														sigma_size,
														cudaSigma,
														iRmaxF,
														idxY0,
														idxZ0,
														cudaVL1,
														cudaVN,
														cudaVL2,
														cudaVM,
														cudaClnm,
														cudaR);
}

template<typename PrecisionType>
template<bool usesZernike>
void Program<PrecisionType>::runBackwardKernel(struct DynamicParameters &parameters)
{
	// Unique parameters
	auto &mId = parameters.Idiff();
	auto cudaMId = initializeMultidimArrayCuda(mId);
	const int step = 1;

	// Common parameters
	auto commonParameters = setCommonArgumentsKernel<usesZernike>(parameters);
	auto idxY0 = commonParameters.idxY0;
	auto idxZ0 = commonParameters.idxZ0;
	auto iRmaxF = commonParameters.iRmaxF;
	auto cudaMV = commonParameters.cudaMV;
	auto cudaR = commonParameters.cudaR;
	auto cudaClnm = commonParameters.cudaClnm;
	auto lastZ = commonParameters.lastZ;
	auto lastY = commonParameters.lastY;
	auto lastX = commonParameters.lastX;

	backwardKernel<PrecisionType, usesZernike><<<1, 1>>>(cudaMV,
														 cudaMId,
														 VRecMaskB,
														 lastZ,
														 lastY,
														 lastX,
														 step,
														 iRmaxF,
														 idxY0,
														 idxZ0,
														 cudaVL1,
														 cudaVN,
														 cudaVL2,
														 cudaVM,
														 cudaClnm,
														 cudaR);
}

template<typename PrecisionType>
Matrix2D<PrecisionType> Program<PrecisionType>::createRotationMatrix(struct AngleParameters angles) const
{
	auto rot = angles.rot;
	auto tilt = angles.tilt;
	auto psi = angles.psi;
	constexpr size_t matrixSize = 3;
	auto tmp = Matrix2D<PrecisionType>();
	tmp.initIdentity(matrixSize);
	Euler_angles2matrix(rot, tilt, psi, tmp, false);
	return tmp;
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
