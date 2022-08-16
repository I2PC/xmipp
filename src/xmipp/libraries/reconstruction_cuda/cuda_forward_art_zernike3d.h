#ifndef CUDA_FORWARD_ART_ZERNIKE3D_H
#define CUDA_FORWARD_ART_ZERNIKE3D_H

// Xmipp includes
#include <core/matrix1d.h>
#include <core/matrix2d.h>
#include <core/multidim_array.h>
#include <core/xmipp_image.h>
// Standard includes
#include <vector>

struct float3;
struct double3;

template<typename T>
struct MultidimArrayCuda {
	size_t xdim;
	size_t ydim;
	size_t yxdim;
	int xinit;
	int yinit;
	int zinit;
	T *data;
};

template<typename PrecisionType = float>
class CUDAForwardArtZernike3D {
	static_assert(std::is_floating_point<PrecisionType>::value, "Floating point type is required.");

	using PrecisionType3 = std::conditional<std::is_same<PrecisionType, float>::value, float3, double3>;

   public:
	/// Constant parameters for the computation
	struct ConstantParameters {
		Image<PrecisionType> &Vrefined;
		MultidimArray<int> &VRecMaskF, &VRecMaskB;
		Matrix1D<int> &vL1, &vN, &vL2, &vM;
		std::vector<PrecisionType> &sigma;
		int RmaxDef;
		int loopStep;
	};

	struct AngleParameters {
		PrecisionType rot, tilt, psi;
	};

	struct DynamicParameters {
		const std::vector<PrecisionType> &clnm;
		std::vector<Image<PrecisionType>> &P;
		std::vector<Image<PrecisionType>> &W;
		const Image<PrecisionType> &Idiff;
		struct AngleParameters angles;
	};

	struct CommonKernelParameters {
		size_t idxY0, idxZ0;
		PrecisionType iRmaxF;
		PrecisionType *cudaClnm, *cudaR;
	};

   public:
	template<bool usesZernike>
	void runForwardKernel(struct DynamicParameters &parameters);

	template<bool usesZernike>
	void runBackwardKernel(struct DynamicParameters &parameters);

	explicit CUDAForwardArtZernike3D(const ConstantParameters parameters);
	~CUDAForwardArtZernike3D();

   private:
	const MultidimArrayCuda<PrecisionType> V;

	const MultidimArrayCuda<int> VRecMaskF, VRecMaskB;

	const int RmaxDef;

	const int loopStep;

	const int lastX, lastY, lastZ;

	const int *cudaVL1, *cudaVN, *cudaVL2, *cudaVM;

	const std::vector<PrecisionType> sigma;

   private:
	template<bool usesZernike>
	struct CommonKernelParameters setCommonArgumentsKernel(struct DynamicParameters &parameters);

	Matrix2D<PrecisionType> createRotationMatrix(struct AngleParameters angles) const;
};

#endif	// CUDA_FORWARD_ART_ZERNIKE3D_H
