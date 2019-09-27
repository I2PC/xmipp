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
    void readParams() override {
        fnV1 = getParam("--i1");
        fnV2 = getParam("--i2");
        fnRoot = getParam("--oroot");
        bankStep = getDoubleParam("--filterBank", 0);
        bankOverlap = getDoubleParam("--filterBank", 1);
        weightFun = getIntParam("--filterBank", 2);
        weightPower = getDoubleParam("--filterBank", 3);
        denoiseIterations = getIntParam("--denoising");
        deconvolutionIterations = getIntParam("--deconvolution");
        sigma0 = getDoubleParam("--deconvolution", 1);
        lambda = getDoubleParam("--deconvolution", 2);
        differenceIterations = getIntParam("--difference");
        Kdiff = getDoubleParam("--difference", 1);
        if (checkParam("--mask")) {
            mask.fn_mask = getParam("--mask");
            mask.mask_type = "binary_file";
            mask.type = READ_BINARY_MASK;

            if (checkParam("--center")) {
                mask.x0 = getDoubleParam("--center", 0);
                mask.y0 = getDoubleParam("--center", 1);
                mask.z0 = getDoubleParam("--center", 2);
            }
        }
    }

    /// Show
    void show() {
        if (!verbose)
            return;
        std::cout
        << "Volume1:  " << fnV1 << std::endl
        << "Volume2:  " << fnV2 << std::endl
        << "Rootname: " << fnRoot << std::endl
        << "Denoising Iterations:" << denoiseIterations << std::endl
        << "Deconvolution Iterations: " << deconvolutionIterations << std::endl
        << "Sigma0:   " << sigma0 << std::endl
        << "Lambda:   " << lambda << std::endl
        << "Bank step:" << bankStep << std::endl
        << "Bank overlap:" << bankOverlap << std::endl
        << "Weight fun:" << weightFun << std::endl
        << "Weight power:" << weightPower << std::endl
        << "Difference Iterations: " << differenceIterations << std::endl
        << "Kdiff: " << Kdiff << std::endl
        ;
        mask.show();
    }

    /// Define parameters
    void defineParams() override {
        addUsageLine("Given two halves of a volume (and an optional mask), produce a better estimate of the volume underneath");
        addParamsLine("   --i1 <volume1>              : First half");
        addParamsLine("   --i2 <volume2>              : Second half");
        addParamsLine("  [--oroot <root=\"volumeRestored\">] : Output rootname");
        addParamsLine("  [--denoising <N=0>]          : Number of iterations of denoising in real space");
        addParamsLine("  [--deconvolution <N=0> <sigma0=0.2> <lambda=0.001>]   : Number of iterations of deconvolution in Fourier space, initial sigma and lambda");
        addParamsLine("  [--filterBank <step=0> <overlap=0.5> <weightFun=1> <weightPower=3>] : Frequency step for the filter bank (typically, 0.01; between 0 and 0.5)");
        addParamsLine("                                        : filter overlap is between 0 (no overlap) and 1 (full overlap)");
        addParamsLine("                                : Weight function (0=mean, 1=min, 2=mean*diff");
        addParamsLine("  [--difference <N=0> <K=1.5>]  : Number of iterations of difference evaluation in real space");
        addParamsLine("  [--mask <binary_file>]        : Read from file and cast to binary");
        addParamsLine("  [--center <x0=0> <y0=0> <z0=0>]           : Mask center");
    }

    /// Run
    void run() override {
        std::cout << "Compile time:" << __TIME__ << std::endl;

        show();

            auto start_time = std::chrono::high_resolution_clock::now();
        prepareData();
            auto end_time = std::chrono::high_resolution_clock::now();
  #ifdef TIME_MSR
            std::cout << "Prepare data: " << std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count() << " ms" << std::endl;
  #endif

            start_time = std::chrono::high_resolution_clock::now();
        denoise();
            end_time = std::chrono::high_resolution_clock::now();
  #ifdef TIME_MSR
            std::cout << "Denoise data: " << std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count() << " ms" << std::endl;
  #endif
            start_time = std::chrono::high_resolution_clock::now();
        deconvolution();
            end_time = std::chrono::high_resolution_clock::now();
  #ifdef TIME_MSR
            std::cout << "Deconvolution: " << std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count() << " ms" << std::endl;
  #endif

        if (bankStep > 0) {
            filterBank();
        }

            start_time = std::chrono::high_resolution_clock::now();
        difference();
            end_time = std::chrono::high_resolution_clock::now();
  #ifdef TIME_MSR
            std::cout << "Difference: " << std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count() << " ms" << std::endl;
  #endif
        saveResults();

        freeDeviceMemory(d_R2, d_mask);
        CudaFFT<T>::release(planForward);
        CudaFFT<T>::release(planBackward);
    }

private:

    void setSizes() {
        xdim = XSIZE(V1r());
        ydim = YSIZE(V1r());
        zdim = ZSIZE(V1r());
        volume_size = xdim * ydim * zdim;
        fourier_size = xdim * ydim * (zdim / 2 + 1);
        memsize = volume_size * type_size;
        fourier_memsize = fourier_size * complex_size;
    }

    void readData() {
        V1r.read(fnV1);
        V2r.read(fnV2);
        V1r().setXmippOrigin();
        V2r().setXmippOrigin();

        checkInputDimensions();

        setSizes();

        if (mask.fn_mask != "") {
            mask.generate_mask();
            auto pMask = &mask.get_binary_mask();
            std::cout << "pMask size:" << MULTIDIM_SIZE(*pMask) << std::endl;
            std::cout << "pMask" << pMask << std::endl;
            const size_t mask_memsize = sizeof(int) * volume_size;
            gpuMalloc((void**)&d_mask, mask_memsize);
            gpuCopyFromCPUToGPU(pMask->data, d_mask, mask_memsize);
            pMaskSize = pMask->sum();
        }
    }

    void checkInputDimensions() {
        if (XSIZE(V1r()) != XSIZE(V2r()) || YSIZE(V1r()) != YSIZE(V2r())
            || ZSIZE(V1r()) != ZSIZE(V2r())) {
            throw std::runtime_error("Input volumes have different dimensions");
        }
    }

    void createFFTPlans() {
        gpu.set();
        auto forward_settings = FFTSettingsNew<T>{ xdim, ydim, zdim, 1, 1, false, true };
        planForward = CudaFFT<T>::createPlan(gpu, forward_settings);
        planBackward = CudaFFT<T>::createPlan(gpu, forward_settings.createInverse());
    }

    void initializeFilter() {
        MultidimArray<T> R2;
        R2.resizeNoCopy(zdim, ydim, xdim / 2 + 1);
        FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(R2)
        {
            double fz, fy, fx;
            FFT_IDX2DIGFREQ(k, zdim, fz);
            FFT_IDX2DIGFREQ(i, ydim, fy);
            FFT_IDX2DIGFREQ(j, xdim, fx);
            A3D_ELEM(R2, k, i, j) = fx*fx + fy*fy + fz*fz;
        }

        gpuMalloc((void**)&d_R2, fourier_memsize);
        gpuCopyFromCPUToGPU(R2.data, d_R2, fourier_size * type_size);
    }

    void prepareData() {
        readData();

        S().resizeNoCopy(V1r());

        createFFTPlans();

        initializeFilter();
    }

    void filterBank() {

            auto start_time = std::chrono::high_resolution_clock::now();

        V1r() *= 1.0 / volume_size;
        V2r() *= 1.0 / volume_size;

        T* d_V;
        Complex* d_fV1;
        Complex* d_fV2;

        gpuMalloc((void**)&d_V, memsize);
        allocateDeviceMemory(fourier_memsize, d_fV1, d_fV2);
        gpuCopyFromCPUToGPU(V1r().data, d_V, memsize);

        CudaFFT<T>::fft(*planForward, d_V, d_fV1);

        gpuCopyFromCPUToGPU(V2r().data, d_V, memsize);

        CudaFFT<T>::fft(*planForward, d_V, d_fV2);

        gpuFree(d_V);

            auto end_time = std::chrono::high_resolution_clock::now();

        const T filterStep = bankStep * (1 - bankOverlap);
        S().resizeNoCopy(V1r());
        int i = 0;
        const int imax = ceil(0.5 / filterStep);
        std::cerr << "Calculating filter bank ..." << std::endl;
        std::cout << "Iterations: " << 0.5 / filterStep << std::endl;
        init_progress_bar(imax);

        Gpu::CDF<T> cdf_mN(volume_size, 0.5);

        T *d_V1r, *d_V2r, *d_S, *d_Vfiltered1, *d_Vfiltered2;

        allocateDeviceMemory(memsize, d_V1r, d_V2r, d_S, d_Vfiltered1, d_Vfiltered2);

        Complex* d_fVout;
        gpuMalloc((void**)&d_fVout, fourier_memsize);

        gpuMemset(d_V1r, 0, memsize);
        gpuMemset(d_V2r, 0, memsize);
        gpuMemset(d_S, 0, memsize);

        // This loop can be parallelized, but it requires lot of memory, maybe divide to 2 parallel parts
        for (T w = 0; w < 0.5; w += filterStep)
        {
                auto start_time_loop = std::chrono::high_resolution_clock::now();
                start_time = std::chrono::high_resolution_clock::now();

            filterBand(d_fV1, d_Vfiltered1, d_fVout, w, fourier_size);
            filterBand(d_fV2, d_Vfiltered2, d_fVout, w, fourier_size);

                end_time = std::chrono::high_resolution_clock::now();
                #ifdef TIME_MSR
                std::cout << "Filter bands: " << std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count() << " ms" << std::endl;
                #endif

                auto start_time = std::chrono::high_resolution_clock::now();
            cdf_mN.calculateCDF(d_Vfiltered1, d_Vfiltered2);
                auto end_time = std::chrono::high_resolution_clock::now();
                #ifdef TIME_MSR
                std::cout << "CDF calculation: " << std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count() << " ms" << std::endl;
                #endif

                start_time = std::chrono::high_resolution_clock::now();



            Gpu::computeWeights(d_Vfiltered1, d_Vfiltered2, d_V1r, d_V2r, d_S, volume_size, cdf_mN, weightPower, weightFun);

            progress_bar(++i);

                end_time = std::chrono::high_resolution_clock::now();
                #ifdef TIME_MSR
                std::cout << "Compute weights: " << std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count() << " ms" << std::endl;
                #endif

                auto end_time_loop = std::chrono::high_resolution_clock::now();
                #ifdef TIME_MSR
                std::cout << "####filterStepLoop: " << std::chrono::duration_cast<std::chrono::milliseconds>( end_time_loop - start_time_loop ).count() << " ms" << std::endl;
                #endif
        }
        progress_bar(imax);

        gpuCopyFromGPUToCPU(d_S, S().data, memsize);
        gpuCopyFromGPUToCPU(d_V1r, V1r().data, memsize);
        gpuCopyFromGPUToCPU(d_V2r, V2r().data, memsize);

        // this to gpu
        S() *= 1-bankOverlap;
        V1r() *= 1-bankOverlap;
        V2r() *= 1-bankOverlap;

        saveFilterBank();

        freeDeviceMemory(d_fVout, d_fV1, d_fV2, d_V1r, d_V2r, d_S, d_Vfiltered1, d_Vfiltered2);
    }

    void filterBand(const Complex* d_fV, T* d_filtered, Complex* d_buffer, T w, size_t fourier_size) {
        const T w2 = w * w;
        T w2Step = (w + bankStep) * (w + bankStep);

        Gpu::filterFourierVolume(d_R2, d_fV, d_buffer, fourier_size, w2, w2Step);
        CudaFFT<T>::ifft(*planBackward, d_buffer, d_filtered);
    }

    void saveFilterBank() {
        S.write(fnRoot+"_filterBank_gpu.vol");
    }

    void saveResults() {
        V1r.write(fnRoot+"_restored1_gpu.vol");
        V2r.write(fnRoot+"_restored2_gpu.vol");
    }

    void denoise() {
        if (denoiseIterations <= 0) {
            return;
        }

        // allocate memory for denoising
        T* d_V1, *d_V2, *d_S, *d_aux;
        Complex* d_fVol;
        allocateDeviceMemory(memsize, d_V1, d_V2, d_S);
        allocateDeviceMemory(pMaskSize * type_size, d_aux);
        allocateDeviceMemory(fourier_memsize, d_fVol);

        gpuCopyFromCPUToGPU(V1r().data, d_V1, memsize);
        gpuCopyFromCPUToGPU(V2r().data, d_V2, memsize);

        const size_t S_size = d_mask ? pMaskSize : volume_size;

        Gpu::CDF<T> cdfS(S_size);
        Gpu::CDF<T> cdfN(volume_size);

        for (int i = 0; i < denoiseIterations; ++i) {
            estimateS(d_V1, d_V2, d_S, d_fVol, volume_size, fourier_size);
            Gpu::normalizeForFFT(d_S, volume_size);
            if (d_mask) {
                Gpu::maskForCDF(d_aux, d_S, d_mask, volume_size);
                cdfS.calculateCDF(d_aux);
            } else {
                cdfS.calculateCDF(d_S);
            }
            significanceRealSpace(d_V1, d_S, cdfS, cdfN, volume_size);
            significanceRealSpace(d_V2, d_S, cdfS, cdfN, volume_size);
        }

        gpuCopyFromGPUToCPU(d_V1, V1r().data, memsize);
        gpuCopyFromGPUToCPU(d_V2, V2r().data, memsize);

        freeDeviceMemory(d_V1, d_V2, d_S, d_fVol, d_aux);
    }

    void estimateS(const T* d_V1, const T* d_V2, T* d_S, Complex* d_fVol, size_t size, size_t fourier_size) {
        if (d_mask) {
            Gpu::computeAveragePositivity(d_V1, d_V2, d_S, d_mask, size);
        } else {
            Gpu::computeAveragePositivity(d_V1, d_V2, d_S, size);
        }

        CudaFFT<T>::fft(*planForward, d_S, d_fVol);
        Gpu::filterS(d_R2, d_fVol, fourier_size);
        CudaFFT<T>::ifft(*planBackward, d_fVol, d_S);
    }

    void significanceRealSpace(T* d_V, const T* d_S, Gpu::CDF<T>& cdfS, Gpu::CDF<T>& cdfN, size_t size) {
        cdfN.calculateCDF(d_V, d_S);
        Gpu::maskWithNoiseProbability(d_V, cdfS, cdfN, size);
    }

    void deconvolution() {
        if (deconvolutionIterations <= 0) {
            return;
        }

        T sigmaConv1 = sigma0;
        T sigmaConv2 = sigma0;

        // allocate memory for denoising
        T* d_V1, *d_V2, *d_S;
        Complex* d_fVol, *d_fV1, *d_fV2;

        allocateDeviceMemory(memsize, d_V1, d_V2, d_S);
        allocateDeviceMemory(fourier_memsize, d_fVol, d_fV1, d_fV2);

        updateRestorationPointers(d_fVol, d_fV1, d_fV2);

        gpuCopyFromCPUToGPU(V1r().data, d_V1, memsize);
        gpuCopyFromCPUToGPU(V2r().data, d_V2, memsize);

        for (int i = 0; i < deconvolutionIterations; ++i) {
            estimateS(d_V1, d_V2, d_S, d_fVol,  volume_size, fourier_size);
            Gpu::normalizeForFFT(d_S, volume_size);

            CudaFFT<T>::fft(*planForward, d_S, d_fVol);
            CudaFFT<T>::fft(*planForward, d_V1, d_fV1);
            CudaFFT<T>::fft(*planForward, d_V2, d_fV2);

            std::tie(sigmaConv1, sigmaConv2) = optimizeSigma(sigmaConv1, sigmaConv2);

            deconvolveS(d_fVol, d_fV1, d_fV2, sigmaConv1, sigmaConv2, fourier_size);

            CudaFFT<T>::ifft(*planBackward, d_fV1, d_V1);
            CudaFFT<T>::ifft(*planBackward, d_fV2, d_V2);

            Gpu::normalizeForFFT(d_V1, d_V2, volume_size);

        }
        gpuCopyFromGPUToCPU(d_S, S().data, memsize);
        S.write(fnRoot+"_deconvolved.vol");

        convolveS(d_fVol, sigmaConv1, sigmaConv2, fourier_size);

        CudaFFT<T>::ifft(*planBackward, d_fVol, d_S);

        Gpu::normalizeForFFT(d_S, volume_size);

        gpuCopyFromGPUToCPU(d_S, S().data, memsize);
        S.write(fnRoot+"_convolved.vol");

        gpuCopyFromGPUToCPU(d_V1, V1r().data, memsize);
        gpuCopyFromGPUToCPU(d_V2, V2r().data, memsize);

        V1r.write(fnRoot+"_deconvolvedV1.vol");
        V2r.write(fnRoot+"_deconvolvedV2.vol");

        freeDeviceMemory(d_V1, d_V2, d_S, d_fVol, d_fV1, d_fV2);
    }

    void updateRestorationPointers(Complex* d_fVol, Complex* d_fV1, Complex* d_fV2) {
        restorationPointers.d_R2 = d_R2;
        restorationPointers.d_fVol = d_fVol;
        restorationPointers.d_fV1 = d_fV1;
        restorationPointers.d_fV2 = d_fV2;
    }

    void deconvolveS(Complex* d_fVol, Complex* d_fV1, Complex* d_fV2, T sigmaConv1, T sigmaConv2, size_t fourier_size) {
        if (verbose > 0) {
            std::cout << "   Deconvolving with sigma=" << sigmaConv1  << " " << sigmaConv2 << std::endl;
        }

        const T K1 = -0.5 / (sigmaConv1 * sigmaConv1);
        const T K2 = -0.5 / (sigmaConv2 * sigmaConv2);

        Gpu::deconvolveRestored(d_fVol, d_fV1, d_fV2, d_R2,
                            K1, K2, lambda, volume_size, fourier_size);
    }

    void convolveS(Complex* d_fVol, T sigmaConv1, T sigmaConv2, size_t fourier_size) {
        const T sigmaConv = (sigmaConv1 + sigmaConv2) / 2;
        const T K = -0.5 / (sigmaConv * sigmaConv);

        Gpu::convolveFourierVolume(d_fVol, d_R2, K, fourier_size);
    }

    std::pair<T, T> optimizeSigma(T sigmaConv1, T sigmaConv2) {
        Matrix1D<double> p(2), steps(2);
        p(0) = sigmaConv1;
        p(1) = sigmaConv2;
        steps.initConstant(1);
        double cost;
        int iter;
        powellOptimizer(p, 1, 2, &restorationSigmaCost, this, 0.01, cost, iter, steps, verbose >= 2);

        return { p(0), p(1) };
    }

    void difference() {
        T *d_V1, *d_V2, *d_S, *d_N;
        allocateDeviceMemory(memsize, d_V1, d_V2, d_S, d_N);

        gpuCopyFromCPUToGPU(V1r().data, d_V1, memsize);
        gpuCopyFromCPUToGPU(V2r().data, d_V2, memsize);

        for (int i = 0; i < differenceIterations; ++i) {
            Gpu::computeDiffAndAverage(d_V1, d_V2, d_S, d_N, volume_size);
            T avg, std;
            if (d_mask) {
                std::tie(avg, std) = Gpu::computeAvgStdWithMask(d_N, d_mask, pMaskSize, volume_size);
            } else {
                std::tie(avg, std) = Gpu::computeAvgStd(d_N, volume_size);
            }
            std *= Kdiff;

            Gpu::computeDifference(d_V1, d_V2, d_S, d_N, -static_cast<T>(0.5) / (std * std), volume_size);
        }

        Gpu::computeDiffAndAverage(d_V1, d_V2, d_S, d_N, volume_size);

        gpuCopyFromGPUToCPU(d_V1, V1r().data, memsize);
        gpuCopyFromGPUToCPU(d_V2, V2r().data, memsize);
        gpuCopyFromGPUToCPU(d_S, S().data, memsize);

        S.write(fnRoot + "_avgDiff_gpu.vol");

        freeDeviceMemory(d_V1, d_V2, d_S, d_N);
    }


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

    static double restorationSigmaCost(double *x, void *_prm) {
        ProgVolumeHalvesRestorationGpu<T> *prm = (ProgVolumeHalvesRestorationGpu<T> *) _prm;
        const double sigma1 = x[1];
        const double sigma2 = x[2];
        if (sigma1 < 0 || sigma2 < 0 || sigma1 > 2 || sigma2 > 2)
            return 1e38;
        const T K1 = -0.5 / (sigma1 * sigma1);
        const T K2 = -0.5 / (sigma2 * sigma2);
        T error = 0;

        auto pointers = prm->restorationPointers;

        Gpu::restorationSigmaCostError(error, pointers.d_fVol, pointers.d_fV1,
                                    pointers.d_fV2, pointers.d_R2, K1, K2, prm->get_fourier_size());

        return static_cast<T>(error);
    }
};

#endif
