
#include "reconstruction_cuda/cuda_flexalign_scale.h"
#include "reconstruction_cuda/cuda_fft.h"
#include "reconstruction_adapt_cuda/basic_mem_manager.h"
#include "reconstruction_cuda/cuda_asserts.h"
#include "reconstruction_cuda/cuda_scaleFFT.h"

template <typename T>
CUDAFlexAlignScale<T>::~CUDAFlexAlignScale()
{
    CudaFFT<T>::release(reinterpret_cast<cufftHandle *>(mFT));
    CudaFFT<T>::release(reinterpret_cast<cufftHandle *>(mIT));
    BasicMemManager::instance().give(mAux1);
    BasicMemManager::instance().give(mAux2);
}

template <typename T>
size_t CUDAFlexAlignScale<T>::estimateBytesAlloc(bool alloc)
{
    if (mParams.doBinning)
    {
        auto sRaw = getRawSettings();
        auto bFT = CudaFFT<T>().estimatePlanBytes(sRaw);
        auto sMov = getMovieSettings();
        auto bIT = CudaFFT<T>().estimatePlanBytes(sMov);
        auto sOut = getOutputSettings();
        auto bAux1 = std::max(std::max(sRaw.sBytesBatch(), sMov.fBytesBatch()), sOut.fBytesBatch());
        auto bAux2 = std::max(sRaw.fBytesBatch(), sMov.sBytesBatch());

        if (alloc)
        {
            mFT = CudaFFT<T>().createPlan(mGpu, sRaw);
            mIT = CudaFFT<T>().createPlan(mGpu, sMov);
            mAux1 = BasicMemManager::instance().get(bAux1, MemType::CUDA);
            mAux2 = BasicMemManager::instance().get(bAux2, MemType::CUDA);
        }

        return bIT + bFT + bAux1 + bAux2;
    }
    else
    {
        const auto sMov = getMovieSettings();
        const auto bPlan = CudaFFT<T>().estimatePlanBytes(sMov);
        const auto sOut = getOutputSettings();
        const auto bAux1 = std::max(sMov.sBytesBatch(), sOut.fBytesSingle() * sMov.batch());
        const auto bAux2 = sMov.fBytesBatch();
        if (alloc)
        {
            mFT = CudaFFT<T>::createPlan(mGpu, sMov);
            mAux1 = BasicMemManager::instance().get(bAux1, MemType::CUDA);
            mAux2 = BasicMemManager::instance().get(bAux2, MemType::CUDA);
        }
        return bPlan + bAux1 + bAux2;
    }
}

template <typename T>
void CUDAFlexAlignScale<T>::runFFTScale(T *h_in, const FFTSettings<T> &in, std::complex<T> *h_out, T *d_filter)
{
    const auto stream = *(cudaStream_t *)mGpu.stream();
    // perform FFT of the movie
    gpuErrchk(cudaMemcpyAsync(mAux1, h_in, in.sBytesBatch(), cudaMemcpyHostToDevice, stream));
    cudaMemsetAsync(mAux2, 0, in.fBytesBatch(), stream); // clean it to ensure that FFT is properly computed
    CudaFFT<T>::fft(*reinterpret_cast<cufftHandle *>(mFT), asT(mAux1), asCT(mAux2));
    // scale frames in FD
    const auto out = getOutputSettings();
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid(ceil(out.fDim().x() / (float)dimBlock.x), ceil(out.fDim().y() / (float)dimBlock.y));
    scaleFFT2D(dimGrid, dimBlock,
               asCT(mAux2), asCT(mAux1),
               in.batch(), in.fDim().x(), in.fDim().y(),
               out.fDim().x(), out.fDim().y(),
               d_filter, 1.f / in.sDim().xy(), false, true, mGpu);
    // copy data out
    gpuErrchk(cudaMemcpyAsync(h_out, mAux1, out.fBytesSingle() * in.batch(), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaPeekAtLastError());
}

template <typename T>
void CUDAFlexAlignScale<T>::runScaleIFT(T *h_outBin)
{
    // assuming that data is already in FD, directly after the FD (non-normalized, not-centered), in mAux2
    // scale frames in FD to perform binning
    auto in = getRawSettings();
    auto movie = getMovieSettings();
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid(ceil(movie.fDim().x() / (float)dimBlock.x), ceil(movie.fDim().y() / (float)dimBlock.y));
    scaleFFT2D(dimGrid, dimBlock,
        asCT(mAux2), asCT(mAux1), // data will be in mAux1
        in.batch(), in.fDim().x(), in.fDim().y(),
        movie.fDim().x(), movie.fDim().y(),
        reinterpret_cast<T*>(NULL), 1.f/in.sDim().xy(), false, false, mGpu);
    // perform IFT and send data back
    auto stream = *(cudaStream_t *)mGpu.stream();
    cudaMemsetAsync(mAux2, 0, movie.sBytesBatch(), stream);                           // clean it to ensure that IFT is properly computed
    CudaFFT<T>::ifft(*reinterpret_cast<cufftHandle *>(mIT), asCT(mAux1), asT(mAux2)); // data in spacial domain is in mAux2
    gpuErrchk(cudaMemcpyAsync(h_outBin, mAux2, movie.sBytesBatch(), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaPeekAtLastError());
}

template class CUDAFlexAlignScale<float>;