/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
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

#include <cuda_runtime_api.h>
#include "cuda_gpu_geo_shift_transformer.h"
#include "cuda_xmipp_utils.cpp"
#include "cuda_gpu_geo_shift_transformer.cu"

template class GeoShiftTransformer<float> ;

template<typename T>
void GeoShiftTransformer<T>::release() {
    delete imgs;
    imgs = NULL;
    delete ffts;
    ffts = NULL;

    fftHandle.clear();
    ifftHandle.clear();

    // do not release stream, it has been passed by pointer, so we don't own it
    stream = NULL;
    device = -1;

    isReady = false;
}

template<typename T>
void GeoShiftTransformer<T>::init(size_t x, size_t y, size_t n, int device,
        myStreamHandle* stream) {
    release();

    this->device = device;
    this->stream = stream;

    setDevice(device);
    this->imgs = new GpuMultidimArrayAtGpu<T>(x, y, 1, n);
    this->ffts = new GpuMultidimArrayAtGpu<std::complex<T> >(x / 2 + 1, y, 1, n);

    fftHandle.ptr = new cufftHandle;
    ifftHandle.ptr = new cufftHandle;
    if (this->stream) {
        createPlanFFTStream(x, y, n, 1, true, (cufftHandle*)fftHandle.ptr, *stream);
        createPlanFFTStream(x, y, n, 1, false, (cufftHandle*)ifftHandle.ptr, *stream);
    } else {
        createPlanFFT(x, y, n, 1, true, (cufftHandle*)fftHandle.ptr);
        createPlanFFT(x, y, n, 1, false, (cufftHandle*)ifftHandle.ptr);
    }

    isReady = true;
}

template<typename T>
void GeoShiftTransformer<T>::initLazy(size_t x, size_t y, size_t n, int device,
        myStreamHandle* stream) {
    if (!isReady) {
        init(x, y, n, device, stream);
    }
}

template<typename T>
template<typename T_IN>
void GeoShiftTransformer<T>::applyShift(MultidimArray<T> &output,
        const MultidimArray<T_IN> &input, T shiftX, T shiftY) {
    checkRestrictions(output, input);
    if (output.xdim == 0) {
        output.resizeNoCopy(input.ndim, input.zdim, input.ydim, input.xdim);
    }
    if (shiftX == 0 && shiftY == 0) {
        typeCast(input, output);
        return;
    }

    GpuMultidimArrayAtGpu<std::complex<float> > mask;

    MultidimArray<T> tmp;
    typeCast(input, tmp);
    imgs->copyToGpu(tmp.data);
    if(stream) {
        imgs->fftStream(*ffts, fftHandle, *stream, false, mask);
    } else {
        imgs->fft(*ffts, fftHandle);
    }

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(ceil(ffts->Xdim / (T) dimBlock.x), ceil(ffts->Ydim / (T) dimBlock.y));

    if (stream) {
        shiftFFT2D<true><<<dimGrid, dimBlock, 0, *(cudaStream_t*)stream->ptr>>>(
            (float2*)ffts->d_data, ffts->Ndim, ffts->Xdim, imgs->Xdim,
            imgs->Ydim, shiftX, shiftY);
    } else {
        shiftFFT2D<true><<<dimGrid, dimBlock>>>((float2*)ffts->d_data,
            ffts->Ndim, ffts->Xdim, imgs->Xdim, imgs->Ydim, shiftX, shiftY);
    }
    gpuErrchk(cudaPeekAtLastError());

    if (stream) {
        ffts->ifftStream(*imgs, ifftHandle, *stream, false, mask);
    } else {
        ffts->ifft(*imgs, ifftHandle);
    }
    imgs->copyToCpu(output.data);
}

template<typename T>
void GeoShiftTransformer<T>::test() {
    int offsetX = 5;
    int offsetY = -7;
    size_t xSize = 2048;
    size_t ySize = 3072;
    MultidimArray<float> resGpu(ySize, xSize);
    MultidimArray<float> expected(ySize, xSize);
    MultidimArray<float> input(ySize, xSize);
    for (int y = 10; y < 15; ++y) {
        for (int x = 10; x < 15; ++x) {
            int indexIn = (y * input.xdim) + x;
            int indexExp = ((y + offsetY) * input.xdim) + (x + offsetX);
            input.data[indexIn] = 10;
            expected.data[indexExp] = 10;
        }
    }

    GeoShiftTransformer<float> tr;
    tr.initLazy(input.xdim, input.ydim, 1, 0);
    tr.applyShift(resGpu, input, offsetX, offsetY);

    bool error = false;
    for (int y = 0; y < expected.ydim; ++y) {
        for (int x = 0; x < expected.xdim; ++x) {
            int index = (y * input.xdim) + x;
            float gpu = resGpu.data[index];
            float cpu = expected.data[index];
            float threshold = std::max(gpu, cpu) / 1000.f;
            float diff = std::abs(cpu - gpu);
            if (diff > threshold && diff > 0.001) {
                error = true;
                printf("%d gpu %.4f cpu %.4f (%f > %f)\n", index, gpu, cpu, diff, threshold);
            }
        }
    }
#ifdef DEBUG
    Image<float> img(expected.xdim, expected.ydim);
    img.data = expected;
    img.write("expected.vol");
    img.data = resGpu;
    img.write("resGpu.vol");
#endif
    printf("\n SHIFT %s\n", error ? "FAILED" : "OK");
}

template<typename T>
template<typename T_IN>
void GeoShiftTransformer<T>::checkRestrictions(MultidimArray<T> &output,
        const MultidimArray<T_IN> &input) {
    if (!isReady)
        throw std::logic_error("Shift transformer: Not initialized");

    if (input.nzyxdim == 0)
        throw std::invalid_argument("Shift transformer: Input is empty");

    if ((imgs->Xdim != input.xdim) || (imgs->Ydim != input.ydim)
            || (imgs->Zdim != input.zdim) || (imgs->Ndim != input.ndim))
        throw std::logic_error(
            "Shift transformer: Initialized for different sizes");

    if (((output.xdim != input.xdim) || (output.ydim != input.ydim)
            || (output.zdim != input.zdim) || (output.ndim != input.ndim))
        && (output.nzyxdim != 0))
        throw std::logic_error(
            "Shift transformer: Input/output dimensions do not match");

    if (&input == (MultidimArray<T_IN>*) &output)
        throw std::invalid_argument(
            "Shift transformer: The input array cannot be the same as the output array");
}
