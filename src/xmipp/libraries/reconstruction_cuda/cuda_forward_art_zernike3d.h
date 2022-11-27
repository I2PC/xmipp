#ifndef CUDA_FORWARD_ART_ZERNIKE3D_H
#define CUDA_FORWARD_ART_ZERNIKE3D_H

#if defined(__CUDACC__)	 // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__)	 // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)	 // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


// Xmipp includes
#include <core/matrix1d.h>
#include <core/matrix2d.h>
#include <core/multidim_array.h>
#include <core/xmipp_image.h>
// Standard includes
#include <vector>

namespace cuda_forward_art_zernike3D {

struct float3;
struct double3;

template<typename T>
struct MY_ALIGN(16) MultidimArrayCuda {
	unsigned xdim;
	unsigned ydim;
	unsigned yxdim;
	int xinit;
	int yinit;
	int zinit;
	T *data;
};

struct BlockSizes {
	size_t x, y, z;
};

template<typename PrecisionType = float>
class Program {
	static_assert(std::is_floating_point<PrecisionType>::value, "Floating point type is required.");

	using PrecisionType3 = std::conditional<std::is_same<PrecisionType, float>::value, float3, double3>;

   public:
	/// Constant parameters for the computation
	struct ConstantParameters {
		MultidimArray<int> &VRecMaskF, &VRecMaskB;
		Image<PrecisionType> &Vrefined;
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
		PrecisionType *cudaClnm;
		Matrix2D<PrecisionType> R;
	};

   public:
	template<bool usesZernike>
	void runForwardKernel(struct DynamicParameters &parameters);

	template<bool usesZernike>
	void runBackwardKernel(struct DynamicParameters &parameters);

	/// Moves Volume from GPU and writes it to Vrefined
	/// IMPORTANT: Memory heavy operation.
	void recoverVolumeFromGPU(Image<PrecisionType> &Vrefined);

	explicit Program(const ConstantParameters parameters);
	~Program();

   private:
	const MultidimArrayCuda<PrecisionType> cudaMV;

	const int lastX, lastY, lastZ;

	const int RmaxDef;

	const int loopStep;

	const int *cudaVL1, *cudaVN, *cudaVL2, *cudaVM;

	const std::vector<PrecisionType> sigma;

	const PrecisionType *cudaSigma;

	size_t blockXX, blockY, blockZ, gridXX, gridY, gridZ;

	size_t blockX0, blockX1, blockX2, blockX3, blockX4, blockX5, blockX6, blockX7;

	size_t gridX0, gridX1, gridX2, gridX3, gridX4, gridX5, gridX6, gridX7;

	size_t blockXStep0, blockXStep1, blockXStep2, blockXStep3, blockXStep4, blockXStep5, blockXStep6, blockXStep7;

	size_t gridXStep0, gridXStep1, gridXStep2, gridXStep3, gridXStep4, gridXStep5, gridXStep6, gridXStep7;

	size_t blockXXStep, gridXXStep;

	int *VRecMaskF0, *VRecMaskF1, *VRecMaskF2, *VRecMaskF3, *VRecMaskF4, *VRecMaskF5, *VRecMaskF6, *VRecMaskF7,
		*VRecMaskF8;

	unsigned *cudaCoordinatesB, *cudaCoordinatesF;

	unsigned *cudaCoordinatesB0, *cudaCoordinatesB1, *cudaCoordinatesB2, *cudaCoordinatesB3, *cudaCoordinatesB4,
		*cudaCoordinatesB5, *cudaCoordinatesB6, *cudaCoordinatesB7;

	unsigned *cudaCoordinatesF0, *cudaCoordinatesF1, *cudaCoordinatesF2, *cudaCoordinatesF3, *cudaCoordinatesF4,
		*cudaCoordinatesF5, *cudaCoordinatesF6, *cudaCoordinatesF7;

	const unsigned xdimB, ydimB;

	size_t sizeB0, sizeB1, sizeB2, sizeB3, sizeB4, sizeB5, sizeB6, sizeB7;

	const int xdimF, ydimF;

	size_t sizeF0, sizeF1, sizeF2, sizeF3, sizeF4, sizeF5, sizeF6, sizeF7;
};

}  // namespace cuda_forward_art_zernike3D
#endif	// CUDA_FORWARD_ART_ZERNIKE3D_H
