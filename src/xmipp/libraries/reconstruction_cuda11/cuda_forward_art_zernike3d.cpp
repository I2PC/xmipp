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
#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}

	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
	{
		if (code != cudaSuccess) {
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort)
				exit(code);
		}
	}

	void processCudaError()
	{
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(err));
		}
	}

	// Copies data from CPU to the GPU
	template<typename T>
	void transferData(T **dest, const T *source, size_t n)
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
	void transferDataFromGPU(T *dest, const T *source, size_t n)
	{
		if (cudaMemcpy(dest, source, sizeof(T) * n, cudaMemcpyDeviceToHost) != cudaSuccess) {
			processCudaError();
		}
	}

	template<typename T>
	T *transferMultidimArrayToGpu(const MultidimArray<T> &inputArray)
	{
		T *outputArrayData;
		transferData(&outputArrayData, inputArray.data, inputArray.xdim * inputArray.ydim * inputArray.zdim);
		return outputArrayData;
	}

	template<typename T>
	MultidimArrayCuda<T> *transferVectorOfMultidimArrayToGpu(const std::vector<MultidimArrayCuda<T>> &inputVector)
	{
		MultidimArrayCuda<T> *outputVectorData;
		transferData(&outputVectorData, inputVector.data(), inputVector.size());
		return outputVectorData;
	}

	template<typename T>
	T *transferMatrix1DToGpu(const Matrix1D<T> &inputVector)
	{
		T *outputVector;
		transferData(&outputVector, inputVector.vdata, inputVector.vdim);
		return outputVector;
	}

	template<typename T>
	T *transferStdVectorToGpu(const std::vector<T> &inputVector)
	{
		T *outputVector;
		transferData(&outputVector, inputVector.data(), inputVector.size());
		return outputVector;
	}

	template<typename T>
	T *transferMatrix2DToGpu(const Matrix2D<T> &inputMatrix)
	{
		T *outputMatrixData;
		transferData(&outputMatrixData, inputMatrix.mdata, inputMatrix.mdim);
		return outputMatrixData;
	}

	template<typename T>
	MultidimArrayCuda<T> initializeMultidimArrayCuda(const MultidimArray<T> &multidimArray)
	{
		struct MultidimArrayCuda<T> cudaArray = {
			.xdim = static_cast<unsigned>(multidimArray.xdim), .ydim = static_cast<unsigned>(multidimArray.ydim),
			.zdim = static_cast<unsigned>(multidimArray.zdim),
			.yxdim = static_cast<unsigned>(multidimArray.yxdim), .xinit = multidimArray.xinit,
			.yinit = multidimArray.yinit, .zinit = multidimArray.zinit,
			.data = transferMultidimArrayToGpu(multidimArray)
		};

		return cudaArray;
	}

	template<typename T>
	void updateMultidimArrayWithGPUData(MultidimArray<T> &multidimArray, const MultidimArrayCuda<T> &multidimArrayCuda)
	{
		transferDataFromGPU(
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
		return std::make_pair(transferVectorOfMultidimArrayToGpu(output), output);
	}

	template<typename T>
	void freeCommonArgumentsKernel(struct Program<T>::CommonKernelParameters &commonParameters)
	{
		cudaFree(commonParameters.cudaClnm);
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
			.idxY0 = idxY0, .idxZ0 = idxZ0, .iRmaxF = iRmaxF, .cudaClnm = transferStdVectorToGpu(clnm), .R = R,
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
	std::tuple<unsigned *, size_t, int *> filterMasktransferCoordinates(MultidimArray<T> &mask,
																		int step,
																		bool transferValues)

	{
		std::vector<unsigned> coordinates;
		std::vector<T> values;
		for (unsigned i = 0; i < static_cast<unsigned>(mask.yxdim * mask.zdim); i++) {
			if (checkStep(mask, step, static_cast<size_t>(i))) {
				coordinates.push_back(i);
				if (transferValues) {
					values.push_back(mask[i]);
				}
			}
		}
		unsigned *coordinatesCuda = transferStdVectorToGpu(coordinates);
		int *valuesCuda = transferStdVectorToGpu(values);
		return std::make_tuple(coordinatesCuda, coordinates.size(), valuesCuda);
	}

}  // namespace

template<typename PrecisionType>
Program<PrecisionType>::Program(const Program<PrecisionType>::ConstantParameters parameters)
	: cudaMV(initializeMultidimArrayCuda(parameters.Vrefined())),
	  cudaDl1(initializeMultidimArrayCuda(parameters.VZero())),
	  cudaDx(initializeMultidimArrayCuda(parameters.VZero())),
	  cudaDy(initializeMultidimArrayCuda(parameters.VZero())),
	  cudaDz(initializeMultidimArrayCuda(parameters.VZero())),
	  cudaReg(initializeMultidimArrayCuda(parameters.VZero())),
	  VRecMaskB(initializeMultidimArrayCuda(parameters.VRecMaskB)),
	  sigma(parameters.sigma),
	  cudaSigma(transferStdVectorToGpu(parameters.sigma)),
	  RmaxDef(parameters.RmaxDef),
	  lastX(FINISHINGX(parameters.Vrefined())),
	  lastY(FINISHINGY(parameters.Vrefined())),
	  lastZ(FINISHINGZ(parameters.Vrefined())),
	  cudaVL1(transferMatrix1DToGpu(parameters.vL1)),
	  cudaVL2(transferMatrix1DToGpu(parameters.vL2)),
	  cudaVN(transferMatrix1DToGpu(parameters.vN)),
	  cudaVM(transferMatrix1DToGpu(parameters.vM)),
	  xdimF(parameters.VRecMaskF.xdim),
	  ydimF(parameters.VRecMaskF.ydim),
	  blockXB(std::__gcd(blockSizeArchitecture().x, parameters.Vrefined().xdim)),
	  blockYB(std::__gcd(blockSizeArchitecture().y, parameters.Vrefined().ydim)),
	  blockZB(std::__gcd(blockSizeArchitecture().z, parameters.Vrefined().zdim)),
	  gridXB(parameters.Vrefined().xdim / blockXB),
	  gridYB(parameters.Vrefined().ydim / blockYB),
	  gridZB(parameters.Vrefined().zdim / blockZB),
	  filterMR(parameters.filterMR)
{
	std::tie(cudaCoordinatesF, sizeF, VRecMaskF) =
		filterMasktransferCoordinates(parameters.VRecMaskF, parameters.loopStep, parameters.sigma.size() > 1);
	auto optimalizedSize = ceil(static_cast<double>(sizeF) / static_cast<double>(BLOCK_SIZE)) * BLOCK_SIZE;
	gridX = optimalizedSize / BLOCK_SIZE;
}

template<typename PrecisionType>
Program<PrecisionType>::~Program()
{
	cudaFree(VRecMaskF);
	cudaFree(VRecMaskB.data);
	cudaFree(cudaMV.data);
	cudaFree(cudaDx.data);
	cudaFree(cudaDy.data);
	cudaFree(cudaDz.data);
	cudaFree(cudaDl1.data);
	cudaFree(cudaReg.data);
	cudaFree(cudaCoordinatesF);
	cudaFree(const_cast<PrecisionType *>(cudaSigma));

	cudaFree(const_cast<int *>(cudaVL1));
	cudaFree(const_cast<int *>(cudaVL2));
	cudaFree(const_cast<int *>(cudaVN));
	cudaFree(const_cast<int *>(cudaVM));

	cudaFree(elems);
	cudaFree(avg);
	cudaFree(sumSqrNorm);
	cudaFree(stddev);

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

	forwardKernel<PrecisionType, usesZernike><<<gridX, BLOCK_SIZE>>>(cudaMV,
																	 VRecMaskF,
																	 cudaCoordinatesF,
																	 xdimF,
																	 ydimF,
																	 static_cast<unsigned>(sizeF),
																	 cudaP,
																	 cudaW,
																	 sigma_size,
																	 parameters.loopStep,
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
	gpuErrchk(cudaPeekAtLastError());

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
	auto &mIws = parameters.Iws();
	auto size = parameters.dSize;
	MultidimArray<PrecisionType> mId_small, mIws_small;
	if (size != 0)
	{
		resize2DArray(mId, mId_small, size);
		resize2DArray(mIws, mIws_small, size);
	}	
	else if (parameters.lmr != 0.0)
	{
		filter2DArray(mId, mId_small);
		filter2DArray(mIws, mIws_small);
	}
	else
	{
		mId_small.initZeros(size, size);
		mIws_small.initZeros(size, size);
		mIws_small.setXmippOrigin();
		mId_small.setXmippOrigin();
	}		
	
	auto cudaMId = initializeMultidimArrayCuda(mId);
	auto cudaMIws = initializeMultidimArrayCuda(mIws);
	auto cudaMId_small = initializeMultidimArrayCuda(mId_small);
	auto cudaMIws_small = initializeMultidimArrayCuda(mIws_small);

	const int step = 1;
	size_t n = 1;
	PrecisionType h_elems = 0.0;
	PrecisionType h_avg = 0.0;
	PrecisionType h_sumSqrNorm = 0.0;
	PrecisionType h_stddev = 0.0;
	transferData(&elems, &h_elems, n);
	transferData(&avg, &h_avg, n);
	transferData(&sumSqrNorm, &h_sumSqrNorm, n);
	transferData(&stddev, &h_stddev, n);

	// Common parameters
	auto commonParameters = getCommonArgumentsKernel<PrecisionType>(parameters, usesZernike, RmaxDef);

	auto cudaR = transferMatrix2DToGpu(createRotationMatrix<PrecisionType>(parameters.angles));


	computeTV<PrecisionType><<<dim3(gridXB, gridYB, gridZB), dim3(blockXB, blockYB, blockZB)>>>(cudaMV, cudaDx, cudaDy, cudaDz, cudaDl1, VRecMaskB, 
																								parameters.lambda, parameters.ltv, parameters.ltk, parameters.ll1, parameters.lst);
	computeDTV<PrecisionType><<<dim3(gridXB, gridYB, gridZB), dim3(blockXB, blockYB, blockZB)>>>(cudaReg, cudaDx, cudaDy, cudaDz, cudaDl1, VRecMaskB, 
																								 parameters.lambda, parameters.ltv, parameters.ltk, parameters.ll1, parameters.lst);

	backwardKernel<PrecisionType, usesZernike>
		<<<dim3(gridXB, gridYB, gridZB), dim3(blockXB, blockYB, blockZB)>>>(cudaMV,
																			cudaMId,
																			cudaMIws,
																			cudaMId_small,
																			cudaMIws_small,
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
																			cudaR,
																			cudaReg, 
																			parameters.lmr);

	// if (parameters.lst > 0.0)
	// {	
	// 	computeStdDevParams<PrecisionType>
	// 		<<<dim3(gridXB, gridYB, gridZB), dim3(blockXB, blockYB, blockZB)>>>(cudaMV, elems, avg,
	// 			sumSqrNorm, VRecMaskB);
	// 	computeStdDev<PrecisionType><<<1, 1>>>(elems, avg, sumSqrNorm, stddev);
	// 	softThreshold<PrecisionType>
	// 		<<<dim3(gridXB, gridYB, gridZB), dim3(blockXB, blockYB, blockZB)>>>(cudaMV, stddev, parameters.lst, VRecMaskB);
	// }	

	cudaDeviceSynchronize();

	cudaFree(cudaR);
	cudaFree(cudaMId.data);
	cudaFree(cudaMIws.data);
	cudaFree(cudaMId_small.data);
	cudaFree(cudaMIws_small.data);
	cudaFree(elems);
	cudaFree(avg);
	cudaFree(sumSqrNorm);
	cudaFree(stddev);
	freeCommonArgumentsKernel<PrecisionType>(commonParameters);
}

template<typename PrecisionType>
void Program<PrecisionType>::recoverVolumeFromGPU(Image<PrecisionType> &Vrefined)
{
	updateMultidimArrayWithGPUData(Vrefined(), cudaMV);
}

// Fourier 2D resizing
template<typename PrecisionType>
void Program<PrecisionType>::resize2DArray(const MultidimArray<PrecisionType> &mI, MultidimArray<PrecisionType> &mOut, int size)
{
	MultidimArray<double> mI_aux;
	typeCast(mI, mI_aux);
	mI_aux.setXmippOrigin();
	selfScaleToSizeFourier(1, size, size, mI_aux, 1);
	typeCast(mI_aux, mOut);
	mOut.setXmippOrigin();
}

// Fourier 2D filter
template<typename PrecisionType>
void Program<PrecisionType>::filter2DArray(const MultidimArray<PrecisionType> &mI, MultidimArray<PrecisionType> &mOut)
{
	FourierFilter filter;
	MultidimArray<double> mI_aux;
	typeCast(mI, mI_aux);
	mI_aux.setXmippOrigin();
	filter.FilterBand = LOWPASS;
	filter.FilterShape = REALGAUSSIAN;
	filter.w1 = 1.0;
	filter.do_generate_3dmask = false;
	filter.maskFourierd = filterMR;
	filter.applyMaskSpace(mI_aux);
	typeCast(mI_aux, mOut);
	mOut.setXmippOrigin();
}


// explicit template instantiation
template class Program<float>;
template void Program<float>::runForwardKernel<true>(struct DynamicParameters &);
template void Program<float>::runForwardKernel<false>(struct DynamicParameters &);
template void Program<float>::runBackwardKernel<true>(struct DynamicParameters &);
template void Program<float>::runBackwardKernel<false>(struct DynamicParameters &);
}  // namespace cuda_forward_art_zernike3D
