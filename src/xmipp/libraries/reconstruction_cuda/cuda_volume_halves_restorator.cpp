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
#include "cuda_volume_halves_restorator.h"

#include "core/xmipp_funcs.h"
#include "data/numerical_tools.h"
#include "cuda_asserts.h"
#include "cuda_volume_restoration_kernels.h"

template< typename T >
void VolumeHalvesRestorator<T>::denoise(T* d_volume1, T* d_volume2, const int* d_mask) {
    if (denoisingIters == 0) {
        return;
    }

    size_t mask_size = 0;
    if (d_mask) {
    	mask_size = Gpu::VolumeRestorationKernels<T>::computeMaskSize(d_mask, volume_size);
    }

    T* d_S = d_buf1;
    T* d_aux = nullptr;
    Complex* d_fVol = d_cbuf1;
    if (d_mask) {
    	gpuErrchk( cudaMalloc(&d_aux, mask_size * type_size) );
    }

    const size_t S_size = d_mask ? mask_size : volume_size;

    Gpu::CDF<T> cdfS(S_size);
    Gpu::CDF<T> cdfN(volume_size);

    for (int i = 0; i < denoisingIters; ++i) {
        estimateS(d_volume1, d_volume2, d_mask, d_S, d_fVol, volume_size, fourier_size);
        Gpu::VolumeRestorationKernels<T>::normalizeForFFT(d_S, volume_size);
        if (d_mask) {
            Gpu::VolumeRestorationKernels<T>::maskForCDF(d_aux, d_S, d_mask, volume_size);
            cdfS.calculateCDF(d_aux);
        } else {
            cdfS.calculateCDF(d_S);
        }
        significanceRealSpace(d_volume1, d_S, cdfS, cdfN, volume_size);
        significanceRealSpace(d_volume2, d_S, cdfS, cdfN, volume_size);
    }

    gpuErrchk( cudaFree(d_aux) );
}

template< typename T >
void VolumeHalvesRestorator<T>::estimateS(const T* d_volume1, const T* d_volume2, const int* d_mask, T* d_S, Complex* d_fVol, size_t size, size_t fourier_size) {
    if (d_mask) {
        Gpu::VolumeRestorationKernels<T>::computeAveragePositivity(d_volume1, d_volume2, d_S, d_mask, size);
    } else {
        Gpu::VolumeRestorationKernels<T>::computeAveragePositivity(d_volume1, d_volume2, d_S, size);
    }

    CudaFFT<T>::fft(*planForward, d_S, d_fVol);
    Gpu::VolumeRestorationKernels<T>::filterS(d_R2, d_fVol, fourier_size);
    CudaFFT<T>::ifft(*planBackward, d_fVol, d_S);
}

template< typename T >
void VolumeHalvesRestorator<T>::significanceRealSpace(T* d_volume, const T* d_S, Gpu::CDF<T>& cdfS, Gpu::CDF<T>& cdfN, size_t size) {
	cdfN.calculateCDF(d_volume, d_S);
    Gpu::VolumeRestorationKernels<T>::maskWithNoiseProbability(d_volume, cdfS, cdfN, size);
}

// FFT_IDX2DIGFREQ macro from xmipp_fftw.h
double fft_idx2digfreq(int idx, size_t size) {
    return (size<=1)? 0:(( (((int)idx) <= (((int)(size)) >> 1)) ? ((int)(idx)) : -((int)(size)) + ((int)(idx))) / (double)(size));
}

template< typename T >
void VolumeHalvesRestorator<T>::initializeFilter() {
	MultidimArray<T> R2;
    R2.resizeNoCopy(zdim, ydim, xdim / 2 + 1);
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(R2)
    {
        double fz, fy, fx;
        fz = fft_idx2digfreq(k, zdim);
        fy = fft_idx2digfreq(i, ydim);
        fx = fft_idx2digfreq(j, xdim);
        A3D_ELEM(R2, k, i, j) = fx*fx + fy*fy + fz*fz;
    }

    gpuErrchk( cudaMalloc((void**)&d_R2, fourier_memsize) );
    gpuErrchk( cudaMemcpy(d_R2, R2.data, fourier_size * type_size, cudaMemcpyHostToDevice) );
}

template< typename T >
void VolumeHalvesRestorator<T>::createFFTPlans() {
    gpu.set();
    auto forward_settings = FFTSettingsNew<T>{ xdim, ydim, zdim, 1, 1, false, true };
    planForward = CudaFFT<T>::createPlan(gpu, forward_settings);
    planBackward = CudaFFT<T>::createPlan(gpu, forward_settings.createInverse());
}

template< typename T >
void VolumeHalvesRestorator<T>::deconvolution(T* d_volume1, T* d_volume2) {
	if (deconvolutionIters == 0) {
        return;
    }

    T* d_S = d_buf1;
    Complex* d_fVol = d_cbuf1;
    Complex* d_fV1;
    Complex* d_fV2;

    gpuErrchk( cudaMalloc(&d_fV1, fourier_memsize) );
    gpuErrchk( cudaMalloc(&d_fV2, fourier_memsize) );

    updateRestorationPointers(d_fVol, d_fV1, d_fV2);

    T sigmaConv1 = sigma;
    T sigmaConv2 = sigma;

    for (int i = 0; i < deconvolutionIters; ++i) {
        estimateS(d_volume1, d_volume2, nullptr, d_S, d_fVol,  volume_size, fourier_size);
        Gpu::VolumeRestorationKernels<T>::normalizeForFFT(d_S, volume_size);

        CudaFFT<T>::fft(*planForward, d_S, d_fVol);
        CudaFFT<T>::fft(*planForward, d_volume1, d_fV1);
        CudaFFT<T>::fft(*planForward, d_volume2, d_fV2);

        std::tie(sigmaConv1, sigmaConv2) = optimizeSigma(sigmaConv1, sigmaConv2);

        deconvolveS(d_fVol, d_fV1, d_fV2, sigmaConv1, sigmaConv2, fourier_size);

        CudaFFT<T>::ifft(*planBackward, d_fV1, d_volume1);
        CudaFFT<T>::ifft(*planBackward, d_fV2, d_volume2);

        Gpu::VolumeRestorationKernels<T>::normalizeForFFT(d_volume1, d_volume2, volume_size);

    }

    deconvolvedS.resize(zdim, ydim, xdim);
    gpuErrchk( cudaMemcpy(deconvolvedS.data, d_S, memsize, cudaMemcpyDeviceToHost) );

    convolveS(d_fVol, sigmaConv1, sigmaConv2, fourier_size);
    CudaFFT<T>::ifft(*planBackward, d_fVol, d_S);
    Gpu::VolumeRestorationKernels<T>::normalizeForFFT(d_S, volume_size);

    convolvedS.resize(zdim, ydim, xdim);
    gpuErrchk( cudaMemcpy(convolvedS.data, d_S, memsize, cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(d_fV1) );
    gpuErrchk( cudaFree(d_fV2) );
}

template< typename T >
std::pair<T, T> VolumeHalvesRestorator<T>::optimizeSigma(T sigmaConv1, T sigmaConv2) {
	Matrix1D<double> p(2), steps(2);
    p(0) = sigmaConv1;
    p(1) = sigmaConv2;
    steps.initConstant(1);
    double cost;
    int iter;
    powellOptimizer(p, 1, 2, &restorationSigmaCost, this, 0.01, cost, iter, steps, verbosity >= 2);

    return { p(0), p(1) };
}

template< typename T >
void VolumeHalvesRestorator<T>::updateRestorationPointers(Complex* d_fVol, Complex* d_fV1, Complex* d_fV2) {
    restorationPointers.d_R2 = d_R2;
    restorationPointers.d_fVol = d_fVol;
    restorationPointers.d_fV1 = d_fV1;
    restorationPointers.d_fV2 = d_fV2;
    restorationPointers.fourier_size = fourier_size;
}

template< typename T >
double VolumeHalvesRestorator<T>::restorationSigmaCost(double *x, void *_prm) {
	VolumeHalvesRestorator<T> *prm = (VolumeHalvesRestorator<T> *) _prm;
    const double sigma1 = x[1];
    const double sigma2 = x[2];
    if (sigma1 < 0 || sigma2 < 0 || sigma1 > 2 || sigma2 > 2)
        return 1e38;
    const T K1 = -0.5 / (sigma1 * sigma1);
    const T K2 = -0.5 / (sigma2 * sigma2);
    T error = 0;

    auto pointers = prm->restorationPointers;

    Gpu::VolumeRestorationKernels<T>::restorationSigmaCostError(error, pointers.d_fVol, pointers.d_fV1,
                                pointers.d_fV2, pointers.d_R2, K1, K2, pointers.fourier_size);

    return static_cast<T>(error);
}

template< typename T >
void VolumeHalvesRestorator<T>::deconvolveS(Complex* d_fVol, Complex* d_fV1, Complex* d_fV2, T sigmaConv1, T sigmaConv2, size_t fourier_size) {
    if (verbosity > 0) {
        std::cout << "   Deconvolving with sigma=" << sigmaConv1  << " " << sigmaConv2 << std::endl;
    }

    const T K1 = -0.5 / (sigmaConv1 * sigmaConv1);
    const T K2 = -0.5 / (sigmaConv2 * sigmaConv2);

    Gpu::VolumeRestorationKernels<T>::deconvolveRestored(d_fVol, d_fV1, d_fV2, d_R2,
                        K1, K2, lambda, volume_size, fourier_size);
}

template< typename T >
void VolumeHalvesRestorator<T>::convolveS(Complex* d_fVol, T sigmaConv1, T sigmaConv2, size_t fourier_size) {
    const T sigmaConv = (sigmaConv1 + sigmaConv2) / 2;
    const T K = -0.5 / (sigmaConv * sigmaConv);

    Gpu::VolumeRestorationKernels<T>::convolveFourierVolume(d_fVol, d_R2, K, fourier_size);
}

template< typename T >
void VolumeHalvesRestorator<T>::filterBank(T* d_volume1, T* d_volume2) {
	if (bankStep == 0) {
		return;
	}

	Gpu::VolumeRestorationKernels<T>::normalizeForFFT(d_volume1, d_volume2, volume_size);

    Complex* d_fV1 = d_cbuf1;
    Complex* d_fV2 = d_cbuf2;

    CudaFFT<T>::fft(*planForward, d_volume1, d_fV1);
    CudaFFT<T>::fft(*planForward, d_volume2, d_fV2);

    const T filterStep = bankStep * (1 - bankOverlap);
    int i = 0;
    const int imax = ceil(0.5 / filterStep);
    if (verbosity > 0) {
    	std::cerr << "Calculating filter bank ..." << std::endl;
    	std::cout << "Iterations: " << 0.5 / filterStep << std::endl;
    	init_progress_bar(imax);
    }

    Gpu::CDF<T> cdf_mN(volume_size, 0.5);

    T* d_S = d_buf1;
    T* d_Vfiltered1 = d_buf2;
    T* d_Vfiltered2;

    gpuErrchk( cudaMalloc(&d_Vfiltered2, memsize) );

    Complex* d_fVout;
    cudaMalloc(&d_fVout, fourier_memsize);

    gpuErrchk( cudaMemset(d_volume1, 0, memsize) );
    gpuErrchk( cudaMemset(d_volume2, 0, memsize) );
    gpuErrchk( cudaMemset(d_S, 0, memsize) );

    // This loop can be parallelized, but it requires lot of memory, maybe divide to 2 parallel parts
    for (T w = 0; w < 0.5; w += filterStep) {
        filterBand(d_fV1, d_Vfiltered1, d_fVout, w, fourier_size);
        filterBand(d_fV2, d_Vfiltered2, d_fVout, w, fourier_size);

        cdf_mN.calculateCDF(d_Vfiltered1, d_Vfiltered2);

        Gpu::VolumeRestorationKernels<T>::computeWeights(d_Vfiltered1, d_Vfiltered2, d_volume1, d_volume2, d_S, volume_size, cdf_mN, weightPower, weightFun);

        if (verbosity > 0) {
        	progress_bar(++i);
        }

    }
    if (verbosity > 0) {
	    progress_bar(imax);
    }

    const T notOverlap = 1 - bankOverlap;
    Gpu::VolumeRestorationKernels<T>::multiplyByConstant(d_S, notOverlap, volume_size);
    Gpu::VolumeRestorationKernels<T>::multiplyByConstant(d_volume1, notOverlap, volume_size);
    Gpu::VolumeRestorationKernels<T>::multiplyByConstant(d_volume2, notOverlap, volume_size);

    filterBankVolume.resize(zdim, ydim, xdim);
    gpuErrchk( cudaMemcpy(filterBankVolume.data, d_S, memsize, cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(d_Vfiltered2) );
    gpuErrchk( cudaFree(d_fVout) );
}

template< typename T >
void VolumeHalvesRestorator<T>::filterBand(const Complex* d_fV, T* d_filtered, Complex* d_buffer, T w, size_t fourier_size) {
    const T w2 = w * w;
    const T w2Step = (w + bankStep) * (w + bankStep);

    Gpu::VolumeRestorationKernels<T>::filterFourierVolume(d_R2, d_fV, d_buffer, fourier_size, w2, w2Step);
    CudaFFT<T>::ifft(*planBackward, d_buffer, d_filtered);
}

template< typename T >
void VolumeHalvesRestorator<T>::difference(T* d_volume1, T* d_volume2, const int* d_mask) {
	if (differenceIters == 0) {
		return;
	}

	T* d_S = d_buf1;
	T* d_N = d_buf2;

	size_t mask_size = 0;
	if (d_mask) {
		mask_size = Gpu::VolumeRestorationKernels<T>::computeMaskSize(d_mask, volume_size);
	}

    for (int i = 0; i < differenceIters; ++i) {
        Gpu::VolumeRestorationKernels<T>::computeDiffAndAverage(d_volume1, d_volume2, d_S, d_N, volume_size);
        T avg, std;
        if (d_mask) {
            std::tie(avg, std) = Gpu::VolumeRestorationKernels<T>::computeAvgStdWithMask(d_N, d_mask, mask_size, volume_size);
        } else {
            std::tie(avg, std) = Gpu::VolumeRestorationKernels<T>::computeAvgStd(d_N, volume_size);
        }
        std *= Kdiff;

        Gpu::VolumeRestorationKernels<T>::computeDifference(d_volume1, d_volume2, d_S, d_N, -static_cast<T>(0.5) / (std * std), volume_size);
    }

    Gpu::VolumeRestorationKernels<T>::computeDiffAndAverage(d_volume1, d_volume2, d_S, d_N, volume_size);

    averageDifference.resize(zdim, ydim, xdim);
    gpuErrchk( cudaMemcpy(averageDifference.data, d_S, memsize, cudaMemcpyDeviceToHost) );
}

template< typename T >
void VolumeHalvesRestorator<T>::apply(const MultidimArray<T>& volume1, const MultidimArray<T>& volume2, const int* mask) {
	setSizes(volume1);
	createFFTPlans();
	initializeFilter();

	T* d_volume1;
	T* d_volume2;
	int* d_mask = nullptr;

	gpuErrchk( cudaMalloc(&d_volume1, memsize) );
	gpuErrchk( cudaMalloc(&d_volume2, memsize) );
	if (mask) {
		gpuErrchk( cudaMalloc(&d_mask, volume_size * sizeof(int)) );
	}

	gpuErrchk( cudaMemcpy(d_volume1, volume1.data, memsize, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_volume2, volume2.data, memsize, cudaMemcpyHostToDevice) );
	if (mask) {
		gpuErrchk( cudaMemcpy(d_mask, mask, volume_size * sizeof(int), cudaMemcpyHostToDevice) );
	}

    gpuErrchk( cudaMalloc(&d_buf1, memsize) );
    gpuErrchk( cudaMalloc(&d_buf2, memsize) );
    gpuErrchk( cudaMalloc(&d_cbuf1, fourier_memsize) );
    gpuErrchk( cudaMalloc(&d_cbuf2, fourier_memsize) );

	denoise(d_volume1, d_volume2, d_mask);
    deconvolution(d_volume1, d_volume2);
    filterBank(d_volume1, d_volume2);
    difference(d_volume1, d_volume2, d_mask);

    reconstructedVolume1.resize(zdim, ydim, xdim);
    gpuErrchk( cudaMemcpy(reconstructedVolume1.data, d_volume1, memsize, cudaMemcpyDeviceToHost) );
    reconstructedVolume2.resize(zdim, ydim, xdim);
    gpuErrchk( cudaMemcpy(reconstructedVolume2.data, d_volume2, memsize, cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(d_R2) );
    gpuErrchk( cudaFree(d_mask) );
    gpuErrchk( cudaFree(d_buf1) );
    gpuErrchk( cudaFree(d_buf2) );
    gpuErrchk( cudaFree(d_cbuf1) );
    gpuErrchk( cudaFree(d_cbuf2) );
    CudaFFT<T>::release(planForward);
    CudaFFT<T>::release(planBackward);
}

template class VolumeHalvesRestorator<double>;
template class VolumeHalvesRestorator<float>;
