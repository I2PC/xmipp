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
#include <data/fourier_filter.h>
#include <data/fourier_projection.h>
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
		Image<PrecisionType> &VZero;
		Matrix1D<int> &vL1, &vN, &vL2, &vM;
		std::vector<PrecisionType> &sigma;
		int RmaxDef;
		int loopStep;
		MultidimArray<double> &filterMR;
	};

	struct AngleParameters {
		PrecisionType rot, tilt, psi;
	};

	struct DynamicParameters {
		const std::vector<PrecisionType> &clnm;
		std::vector<Image<PrecisionType>> &P;
		std::vector<Image<PrecisionType>> &W;
		const Image<PrecisionType> &Idiff;
		const Image<PrecisionType> &Iws;
		struct AngleParameters angles;
		double ltv;
		double ltk;
		double ll1;
		double lst;
		double lmr;
		int dSize;
		PrecisionType loopStep;
		double lambda;
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

	void resize2DArray(const MultidimArray<PrecisionType> &mI, MultidimArray<PrecisionType> &mOut, int size);

	void filter2DArray(const MultidimArray<PrecisionType> &mI, MultidimArray<PrecisionType> &mOut);

	explicit Program(const ConstantParameters parameters);
	~Program();

   private:
	const MultidimArrayCuda<PrecisionType> cudaMV, cudaDl1, cudaDx, cudaDy, cudaDz, cudaReg;

	const int lastX, lastY, lastZ;

	const int RmaxDef;

	const int *cudaVL1, *cudaVN, *cudaVL2, *cudaVM;

	PrecisionType *elems, *avg, *sumSqrNorm, *stddev, *d_tv, *d_l1;

	const std::vector<PrecisionType> sigma;

	const PrecisionType *cudaSigma;

	const size_t blockXB, blockYB, blockZB, gridXB, gridYB, gridZB;

	size_t gridX;

	int *VRecMaskF;

	const MultidimArrayCuda<int> VRecMaskB;

	unsigned *cudaCoordinatesF;

	const int xdimF, ydimF;

	size_t sizeF;

	MultidimArray<double> filterMR;
};

}  // namespace cuda_forward_art_zernike3D
#endif	// CUDA_FORWARD_ART_ZERNIKE3D_H
