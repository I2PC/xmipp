#ifndef CUDA_CDF
#define CUDA_CDF

namespace Gpu {

/** Gpu version of Cumulative density function.
 * This function computes a table with the cumulative density function*/
template< typename T >
struct CDF {
	static constexpr size_t type_size = sizeof(T);

	T* d_V;
	T* d_x;
	T* d_probXLessThanx;

	size_t volume_size;
	T probStep;
	T multConst;
	T Nsteps;


	CDF(size_t volume_size, T multConst = 1.0, T probStep = 0.005);
	~CDF();

	void calculateCDF(const T*  d_filtered1, const T* d_filtered2);
	void calculateCDF(const T* d_S);

		// Functions must be public because they use device lambda
	void _calculateDifference(const T* __restrict__ d_filtered1, const T* __restrict__ d_filtered2);
	void _calculateDifference(const T* __restrict__ d_S);
	void _updateProbabilities();

private:
	void sort();

};

} // namespace Gpu

#endif // CUDA_CDF