/***************************************************************************
 *
 * Authors:    Martin Horacek (horacek1martin@gmail.com)
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
#ifndef CUDA_VOLUME_HALVES_RESTORATOR
#define CUDA_VOLUME_HALVES_RESTORATOR

#include <iostream>

#include <core/multidim_array.h>

#include "reconstruction_cuda/cuda_fft.h"
#include "cuda_cdf.h"

/*
* Computation for ProgVolumeHalvesRestorationGpu
* This class does not check validity of parameters, it assumes
* that they were already checked
*/
template< typename T >
class VolumeHalvesRestorator {
	static_assert(std::is_floating_point<T>::value, "Only float and double are allowed as template parameters");

	using Complex = std::complex<T>;
    static constexpr size_t type_size = sizeof(T);
    static constexpr size_t complex_size = sizeof(Complex);

	const int verbosity;

	/*
	 * Parameters for denoising
	*/
	const unsigned denoisingIters;
	/*
	 * Parameters for deconvolution
	*/
	const unsigned deconvolutionIters;
	const T sigma;
	const T lambda;

	/*
	 * Parameters for difference
	*/
	const unsigned differenceIters;
	const T Kdiff;

	/*
	 * Parameters for filter bank
	*/
	const T bankStep;
	const T bankOverlap;
	const unsigned weightFun;
	const T weightPower;

	/*
	* Sizes of allocated arrays on gpu
	*/
	size_t xdim;
	size_t ydim;
	size_t zdim;
	size_t volume_size;
	size_t fourier_size; // size of complex array used in FFT
	size_t memsize;
	size_t fourier_memsize;

	/*
	* Plans for CuFFT
	*/
	cufftHandle* planForward;
	cufftHandle* planBackward;
	GPU gpu;

	/*
	* Device memory that is reused during all steps
	* In each step these pointers are renamed
	*/
	T* d_buf1;
	T* d_buf2;
	Complex* d_cbuf1;
	Complex* d_cbuf2;

	/*
	* Filter
	*/
	T* d_R2;

	/*
	* Results of computation
	*/
	MultidimArray<T> reconstructedVolume1;
	MultidimArray<T> reconstructedVolume2;
	MultidimArray<T> filterBankVolume;
	MultidimArray<T> deconvolvedS;
	MultidimArray<T> convolvedS;
	MultidimArray<T> averageDifference;

public:
	class Builder;
	friend std::ostream& operator<< (std::ostream& out, const VolumeHalvesRestorator& r) {
		out << "VolumeHalvesRestoration parameters:" << std::endl
	    << "    Denoising Iterations:" << r.denoisingIters << std::endl
	    << "    Deconvolution Iterations: " << r.deconvolutionIters << std::endl
	    << "    Sigma0:   " << r.sigma << std::endl
	    << "    Lambda:   " << r.lambda << std::endl
	    << "    Bank step:" << r.bankStep << std::endl
	    << "    Bank overlap:" << r.bankOverlap << std::endl
	    << "    Weight fun:" << r.weightFun << std::endl
	    << "    Weight power:" << r.weightPower << std::endl
	    << "    Difference Iterations: " << r.differenceIters << std::endl
	    << "    Kdiff: " << r.Kdiff << std::endl
	    ;
		return out;
	}

	VolumeHalvesRestorator(int verbosity, unsigned denoisingIters, unsigned deconvolutionIters, T sigma, T lambda,
						unsigned differenceIters, T Kdiff, T bankStep, T bankOverlap, unsigned weightFun, T weightPower)
	: verbosity(verbosity)
	, denoisingIters(denoisingIters)
	, deconvolutionIters(deconvolutionIters)
	, sigma(sigma)
	, lambda(lambda)
	, differenceIters(differenceIters)
	, Kdiff(Kdiff)
	, bankStep(bankStep)
	, bankOverlap(bankOverlap)
	, weightFun(weightFun)
	, weightPower(weightPower) {}

	/*
	 * Runs the volume halves restoration algorithm
	 * `volume1`, `volume2`, `mask` must have same dimensions
	*/
	void apply(const MultidimArray<T>& volume1, const MultidimArray<T>& volume2, const int* mask);

	/*
	 * Following methods are getters for reconstructed volumes and other results
	 */
	const MultidimArray<T>& getReconstructedVolume1() const { return reconstructedVolume1; }
	const MultidimArray<T>& getReconstructedVolume2() const { return reconstructedVolume2; }
	const MultidimArray<T>& getFilterBankVolume() const { return filterBankVolume; }
	const MultidimArray<T>& getDeconvolvedS() const { return deconvolvedS; }
	const MultidimArray<T>& getConvolvedS() const { return convolvedS; }
	const MultidimArray<T>& getAverageDifference() const { return averageDifference; }

	/*
	 * Powell optimizer uses error function, this struct provides
	 * data for computing the error using restorationSigmaCost static function
	*/
	struct {
        T* d_R2;
        Complex* d_fVol;
        Complex* d_fV1;
        Complex* d_fV2;
        size_t fourier_size;
    } restorationPointers;

private:

	void createFFTPlans();
	void initializeFilter();

	/*
	 * Filter bank functions
	*/
	void filterBank(T* d_volume1, T* d_volume2);
    void filterBand(const Complex* d_fV, T* d_filtered, Complex* d_buffer, T w, size_t fourier_size);

	/*
	 * Denoising functions
	*/
    void denoise(T* d_volume1, T* d_volume2, const int* d_mask);
    void estimateS(const T* d_volume1, const T* d_volume2, const int* d_mask, T* d_S, Complex* d_fVol, size_t size, size_t fourier_size);
    void significanceRealSpace(T* d_volume, const T* d_S, Gpu::CDF<T>& cdfS, Gpu::CDF<T>& cdfN, size_t size);

    /*
	 * Deconvolution functions
    */
    void deconvolution(T* d_volume1, T* d_volume2);
    void updateRestorationPointers(Complex* d_fVol, Complex* d_fV1, Complex* d_fV2);
    std::pair<T, T> optimizeSigma(T sigmaConv1, T sigmaConv2);
    void deconvolveS(Complex* d_fVol, Complex* d_fV1, Complex* d_fV2, T sigmaConv1, T sigmaConv2, size_t fourier_size);
    void convolveS(Complex* d_fVol, T sigmaConv1, T sigmaConv2, size_t fourier_size);

    /*
	 * Difference functions
    */
    void difference(T* d_volume1, T* d_volume2, const int* d_mask);

    void setSizes(const MultidimArray<T>& volume) {
	   	xdim = XSIZE(volume);
	    ydim = YSIZE(volume);
	    zdim = ZSIZE(volume);
	    volume_size = xdim * ydim * zdim;
	    fourier_size = xdim * ydim * (zdim / 2 + 1);
	    memsize = volume_size * type_size;
	    fourier_memsize = fourier_size * complex_size;
    }

    /*
	 * Used as cost function for powell optimizer
    */
    static double restorationSigmaCost(double *x, void *_prm);

};


/*
 * The builder separates the construction of the VolumeHalvesRestorator into
 * parts, each part represents one step in computation
*/
template< typename T >
struct VolumeHalvesRestorator<T>::Builder {
	int verbosity;
	unsigned denoisingIters;
	unsigned deconvolutionIters;
	T sigma;
	T lambda;
	unsigned differenceIters;
	T Kdiff;
	T bankStep;
	T bankOverlap;
	unsigned weightFun;
	T weightPower;

	Builder& setVerbosity(int verbosity) {
		this->verbosity = verbosity;
		return *this;
	}

	Builder& setFilterBank(T bankStep, T bankOverlap, unsigned weightFun, T weightPower) {
		this->bankStep = bankStep;
		this->bankOverlap = bankOverlap;
		this->weightFun = weightFun;
		this->weightPower = weightPower;
		return *this;
	}

	Builder& setDenoising(unsigned denoisingIters) {
		this->denoisingIters = denoisingIters;
		return *this;
	}

	Builder& setDeconvolution(unsigned deconvolutionIters, T sigma, T lambda) {
		this->deconvolutionIters = deconvolutionIters;
		this->sigma = sigma;
		this->lambda = lambda;
		return *this;
	}

	Builder& setDifference(unsigned differenceIters, T Kdiff) {
		this->differenceIters = differenceIters;
		this->Kdiff = Kdiff;
		return *this;
	}

	VolumeHalvesRestorator<T> build() {
		return { verbosity, denoisingIters, deconvolutionIters, sigma, lambda, differenceIters, Kdiff, bankStep,
				bankOverlap, weightFun, weightPower };
	}
};

#endif // CUDA_VOLUME_HALVES_RESTORATOR