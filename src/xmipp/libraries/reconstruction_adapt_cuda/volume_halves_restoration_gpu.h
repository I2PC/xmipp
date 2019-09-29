#ifndef _PROG_VOLUME_HALVES_RESTORATION_GPU
#define _PROG_VOLUME_HALVES_RESTORATION_GPU

#include <chrono>
#include <iostream>
// #include <cuda_runtime_api.h>

#include <core/xmipp_program.h>
#include <core/xmipp_fftw.h>

#include <data/mask.h>
#include <data/numerical_tools.h>

#include "reconstruction_cuda/cuda_cdf.h"
#include "reconstruction_cuda/cuda_volume_restoration_kernels.h"
#include "reconstruction_cuda/cuda_fft.h"
#include "reconstruction_cuda/cuda_xmipp_utils.h"

#include "reconstruction_cuda/cuda_volume_restoration_denoise.h"

//comment this define to hide time measurements
// #define TIME_MSR

template< typename T >
class ProgVolumeHalvesRestorationGpu : public XmippProgram {
    static_assert(std::is_floating_point<T>::value, "Only float and double are allowed as template parameters");

    using Complex = std::complex<T>;
    static constexpr size_t type_size = sizeof(T);
    static constexpr size_t complex_size = sizeof(Complex);

    /** Filename of the two halves and the output root */
    FileName fnV1, fnV2, fnRoot;

    T bankStep, bankOverlap;
    /** Weight function */
    int weightFun;
    /** Weight power */
    T weightPower;

    T sigma0, lambda;

    Image<T> V1r, V2r, S, N;

    cufftHandle *planForward, *planBackward;

    int denoiseIterations, deconvolutionIterations, differenceIterations;

    T Kdiff;

    size_t xdim, ydim, zdim, volume_size, fourier_size;
    size_t memsize, fourier_memsize;

    T* d_R2;

    Mask mask;
    int* d_mask = nullptr;
    size_t pMaskSize = 0;

    GPU gpu;

public:
    // contains pointers to device memory used in `restorationSigmaCost` function
    struct {
        T* d_R2;
        Complex* d_fVol;
        Complex* d_fV1;
        Complex* d_fV2;
    } restorationPointers;

    size_t get_volume_size() const {
        return MULTIDIM_SIZE(V1r());
    }

    size_t get_fourier_size() const {
        return XSIZE(V1r()) * YSIZE(V1r()) * (ZSIZE(V1r()) / 2 + 1);
    }

    /// Read argument from command line
    void readParams() override;

    /// Show
    void show();

    /// Define parameters
    void defineParams() override;

    /// Run
    void run() override;

private:

    void setSizes();
    void readData();
    void checkInputDimensions();

    void createFFTPlans();

    void initializeFilter();

    void prepareData();

    void filterBank();
    void filterBand(const Complex* d_fV, T* d_filtered, Complex* d_buffer, T w, size_t fourier_size);

    void saveFilterBank();
    void saveResults();

    void denoise();

    void estimateS(const T* d_V1, const T* d_V2, T* d_S, Complex* d_fVol, size_t size, size_t fourier_size);

    void significanceRealSpace(T* d_V, const T* d_S, Gpu::CDF<T>& cdfS, Gpu::CDF<T>& cdfN, size_t size);

    void deconvolution();

    void updateRestorationPointers(Complex* d_fVol, Complex* d_fV1, Complex* d_fV2);

    void deconvolveS(Complex* d_fVol, Complex* d_fV1, Complex* d_fV2, T sigmaConv1, T sigmaConv2, size_t fourier_size);

    void convolveS(Complex* d_fVol, T sigmaConv1, T sigmaConv2, size_t fourier_size);

    std::pair<T, T> optimizeSigma(T sigmaConv1, T sigmaConv2);

    void difference();


    /*
    * Base case
    */
    void allocateDeviceMemory(size_t) {}

    /*
    * Allocates `size` bytes of device memory for each pointer
    */
    template< typename T1, typename... Args >
    void allocateDeviceMemory(size_t size, T1* &d_array, Args& ... args) {
        gpuMalloc((void**)&d_array, size);
        allocateDeviceMemory(size, args...);
    }

    /*
    * Base case
    */
    void freeDeviceMemory() {}

    /*
    * Deallocates device memory for each pointer
    */
    template< typename T1, typename... Args >
    void freeDeviceMemory(T1* d_array, Args... args) {
        gpuFree(d_array);
        freeDeviceMemory(args...);
    }

    static double restorationSigmaCost(double *x, void *_prm);
};

#endif
