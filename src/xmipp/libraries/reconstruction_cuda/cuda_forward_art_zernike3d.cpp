// Xmipp includes
#include "cuda_forward_art_zernike3d.h"
#include <core/geometry.h>
#include "cuda_forward_art_zernike3d.cu"
#include "data/numerical_tools.h"

#include <cassert>
#include <stdexcept>
#include "data/numerical_tools.h"

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
CUDAForwardArtZernike3D<PrecisionType>::CUDAForwardArtZernike3D(
	const CUDAForwardArtZernike3D<PrecisionType>::ConstantParameters parameters)
	: V(initializeMultidimArrayCuda(parameters.Vrefined())),
	  VRecMask(initializeMultidimArrayCuda(parameters.VRecMask)),
	  sphMask(initializeMultidimArrayCuda(parameters.sphMask)),
	  sigma(parameters.sigma),
	  RmaxDef(parameters.RmaxDef),
	  lastZ(FINISHINGZ(parameters.Vrefined())),
	  lastY(FINISHINGY(parameters.Vrefined())),
	  lastX(FINISHINGX(parameters.Vrefined())),
	  loopStep(parameters.loopStep),
	  cudaVL1(tranportMatrix1DToGpu(parameters.vL1)),
	  cudaVL2(tranportMatrix1DToGpu(parameters.vL2)),
	  cudaVN(tranportMatrix1DToGpu(parameters.vN)),
	  cudaVM(tranportMatrix1DToGpu(parameters.vM))
{}

template<typename PrecisionType>
CUDAForwardArtZernike3D<PrecisionType>::~CUDAForwardArtZernike3D()
{
	cudaFree(V.data);
	cudaFree(VRecMask.data);
	cudaFree(sphMask.data);

	cudaFree(const_cast<int *>(cudaVL1));
	cudaFree(const_cast<int *>(cudaVL2));
	cudaFree(const_cast<int *>(cudaVN));
	cudaFree(const_cast<int *>(cudaVM));
}

template<typename PrecisionType>
template<bool usesZernike>
struct CUDAForwardArtZernike3D<PrecisionType>::CommonKernelParameters
CUDAForwardArtZernike3D<PrecisionType>::setCommonArgumentsKernel(struct DynamicParameters &parameters) {
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
		.cudaClnm = tranportStdVectorToGpu(clnm),
		.cudaR = tranportMatrix2DToGpu(R),
	};

	return output;

}

template<typename PrecisionType>
template<bool usesZernike>
void CUDAForwardArtZernike3D<PrecisionType>::runForwardKernel(struct DynamicParameters &parameters)

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
	auto cudaR = commonParameters.cudaR;
	auto cudaClnm = commonParameters.cudaClnm;

	forwardKernel<PrecisionType, usesZernike><<<1, 1>>>(V,
														VRecMask,
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
void CUDAForwardArtZernike3D<PrecisionType>::runBackwardKernel(struct DynamicParameters &parameters)
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
	auto cudaR = commonParameters.cudaR;
	auto cudaClnm = commonParameters.cudaClnm;

	for (int k = STARTINGZ(V); k <= lastZ; k += step) {
		for (int i = STARTINGY(V); i <= lastY; i += step) {
			for (int j = STARTINGX(V); j <= lastX; j += step) {
				PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;
				if (A3D_ELEM(sphMask, k, i, j) != 0) {
					if (usesZernike) {
						auto k2 = k * k;
						auto kr = k * iRmaxF;
						auto k2i2 = k2 + i * i;
						auto ir = i * iRmaxF;
						auto r2 = k2i2 + j * j;
						auto jr = j * iRmaxF;
						auto rr = SQRT(r2) * iRmaxF;
						for (size_t idx = 0; idx < idxY0; idx++) {
							auto l1 = cudaVL1[idx];
							auto n = cudaVN[idx];
							auto l2 = cudaVL2[idx];
							auto m = cudaVM[idx];
							if (rr > 0 || l2 == 0) {
								PrecisionType zsph = ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
								gx += cudaClnm[idx] * (zsph);
								gy += cudaClnm[idx + idxY0] * (zsph);
								gz += cudaClnm[idx + idxZ0] * (zsph);
							}
						}
					}

					auto r_x = j + gx;
					auto r_y = i + gy;
					auto r_z = k + gz;

					auto pos_x = cudaR[0] * r_x + cudaR[1] * r_y + cudaR[2] * r_z;
					auto pos_y = cudaR[3] * r_x + cudaR[4] * r_y + cudaR[5] * r_z;
					PrecisionType voxel = interpolatedElement2DCuda(pos_x, pos_y, cudaMId);
					A3D_ELEM(V, k, i, j) += voxel;
				}
			}
		}
	}
}

template<typename PrecisionType>
Matrix2D<PrecisionType> CUDAForwardArtZernike3D<PrecisionType>::createRotationMatrix(
	struct AngleParameters angles) const
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

template<typename PrecisionType>
PrecisionType CUDAForwardArtZernike3D<PrecisionType>::interpolatedElement2DCuda(
	PrecisionType x,
	PrecisionType y,
	MultidimArrayCuda<PrecisionType> &diffImage) const
{
	int x0 = floor(x);
	PrecisionType fx = x - x0;
	int x1 = x0 + 1;
	int y0 = floor(y);
	PrecisionType fy = y - y0;
	int y1 = y0 + 1;

	int i0 = STARTINGY(diffImage);
	int j0 = STARTINGX(diffImage);
	int iF = FINISHINGY(diffImage);
	int jF = FINISHINGX(diffImage);

#define ASSIGNVAL2DCUDA(d, i, j)                      \
	if ((j) < j0 || (j) > jF || (i) < i0 || (i) > iF) \
		d = (PrecisionType)0;                         \
	else                                              \
		d = A2D_ELEM(diffImage, i, j);

	PrecisionType d00, d10, d11, d01;
	ASSIGNVAL2DCUDA(d00, y0, x0);
	ASSIGNVAL2DCUDA(d01, y0, x1);
	ASSIGNVAL2DCUDA(d10, y1, x0);
	ASSIGNVAL2DCUDA(d11, y1, x1);

	PrecisionType d0 = LIN_INTERP(fx, d00, d01);
	PrecisionType d1 = LIN_INTERP(fx, d10, d11);
	return LIN_INTERP(fy, d0, d1);
}

// explicit template instantiation
template class CUDAForwardArtZernike3D<float>;
template class CUDAForwardArtZernike3D<double>;
template void CUDAForwardArtZernike3D<float>::runForwardKernel<true>(struct DynamicParameters &);
template void CUDAForwardArtZernike3D<float>::runForwardKernel<false>(struct DynamicParameters &);
template void CUDAForwardArtZernike3D<double>::runForwardKernel<true>(struct DynamicParameters &);
template void CUDAForwardArtZernike3D<double>::runForwardKernel<false>(struct DynamicParameters &);
template void CUDAForwardArtZernike3D<float>::runBackwardKernel<true>(struct DynamicParameters &);
template void CUDAForwardArtZernike3D<float>::runBackwardKernel<false>(struct DynamicParameters &);
template void CUDAForwardArtZernike3D<double>::runBackwardKernel<true>(struct DynamicParameters &);
template void CUDAForwardArtZernike3D<double>::runBackwardKernel<false>(struct DynamicParameters &);
