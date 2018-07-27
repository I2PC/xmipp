#include "cuda_gpu_geo_transformer.h"
#include "cuda_utils.h"
#include <cuda_runtime_api.h>
#include "cuda_all.cpp"

template class GeoTransformer<float> ;

template<typename T>
void GeoTransformer<T>::release() {
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_trInv);
    d_in = d_out = d_trInv = NULL;
    isReady = false;
}

template<typename T>
void GeoTransformer<T>::init(size_t x, size_t y, size_t z) {
    release();

    this->X = x;
    this->Y = y;
    this->Z = z;

    size_t matSize = (0 == z) ? 9 : 16;
    gpuErrchk(cudaMalloc((void** ) &d_trInv, matSize * sizeof(T)));
    gpuErrchk(cudaMalloc((void** ) &d_in, x * y * z * sizeof(T)));
    gpuErrchk(cudaMalloc((void** ) &d_out, x * y * z * sizeof(T)));

    isReady = true;
}

template<typename T>
void GeoTransformer<T>::initLazy(size_t x, size_t y, size_t z) {
    if (!isReady) {
        init(x, y, z);
    }
}

template<typename T>
template<typename T_IN>
void GeoTransformer<T>::applyShift(MultidimArray<T> &output,
        const MultidimArray<T_IN> &input, T shiftX, T shiftY) {
    if (output.xdim == 0) {
        output.resize(Z, Y, X);
    }
    if (shiftX == 0 && shiftY == 0) {
        typeCast(input, output);
        return;
    }

    static MultidimArray<T> tmp; // FIXME shouldnt be static
    typeCast(input, tmp);
    static GpuMultidimArrayAtGpu<T> img(input.xdim, input.ydim); // FIXME shouldnt be static
    img.copyToGpu(tmp.data);
    static GpuMultidimArrayAtGpu<std::complex<float> > fft; // FIXME shouldnt be static
    mycufftHandle handle;
    img.fft(fft, handle);
    handle.clear();

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(ceil(fft.Xdim / (T) dimBlock.x), ceil(fft.Ydim / (T) dimBlock.y));
    // perform row-wise pass
    shiftFFT2D<true><<<dimGrid, dimBlock>>>(
            (float2*)fft.d_data, 1, fft.Xdim, img.Xdim, img.Ydim, shiftX, shiftY);
    gpuErrchk(cudaPeekAtLastError());
//
//    Image<T> nevim(fft.Xdim, fft.Ydim);
//    std::complex<T>* data = new std::complex<T>[fft.yxdim];
//    fft.copyToCpu(data);
//    for (size_t i = 0; i < fft.yxdim; i++) {
//        T r = data[i].real();
//        if (std::abs(r) < 10)
//        nevim.data.data[i] = r;
//    }nevim.write("FFT.vol");

    fft.ifft(img, handle);
    img.copyToCpu(output.data);
}

void testShift() {
    int offsetX = 5;
    int offsetY = 7;
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

    GeoTransformer<float> tr;
    tr.initLazy(input.xdim, input.ydim);
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

    Image<float> img(expected.xdim, expected.ydim);
    img.data = expected;
    img.write("expected.vol");
    img.data = resGpu;
    img.write("resGpu.vol");

    printf("\n SHIFT %s\n", error ? "FAILED" : "OK");
}

template<typename T>
void GeoTransformer<T>::test() {
    testShift();
//    testCoeffsRow();
//    testTranspose();
//    testCoeffs();
//
//    Matrix1D<T> shift(2);
//    shift.vdata[0] = 0.45;
//    shift.vdata[1] = 0.62;
//    Matrix2D<T> transform;
//    translation2DMatrix(shift, transform, true);
//    test(transform);
}

template<typename T>
void GeoTransformer<T>::testCoeffsRow() {
    MultidimArray<T> resGpu(32, 32);
    MultidimArray<double> resCpu(32, 32);
    MultidimArray<double> inputDouble(32, 32); // size must be square, multiple of 32
    MultidimArray<T> inputFloat(32, 32); // size must be square, multiple of 32
    for (int i = 0; i < inputFloat.ydim; ++i) {
        for (int j = 0; j < inputFloat.xdim; ++j) {
            inputDouble.data[i * inputFloat.xdim + j] = i * 100 + j;
            inputFloat.data[i * inputFloat.xdim + j] = i * 100 + j;
        }
    }

    T* d_output;
    gpuErrchk(cudaMalloc(&d_output, resGpu.yxdim * sizeof(T)));
    T* d_input;
    gpuErrchk(cudaMalloc(&d_input, inputFloat.yxdim * sizeof(T)));
    gpuErrchk(
            cudaMemcpy(d_input, inputFloat.data, inputFloat.yxdim * sizeof(T),
                    cudaMemcpyHostToDevice));

    dim3 dimBlockConv(1, BLOCK_DIM_X);
    dim3 dimGridConv(1, ceil(inputFloat.ydim / (T) dimBlockConv.y));
    // perform row-wise pass
    iirConvolve2D_Cardinal_Bspline_3_MirrorOffBound<<<dimGridConv, dimBlockConv>>>(
        d_input, d_output,
        resGpu.xdim, resGpu.ydim);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(
            cudaMemcpy(resGpu.data, d_output, resGpu.yxdim * sizeof(T),
                    cudaMemcpyDeviceToHost));
    double pole = sqrt(3) - 2.0;

    for (int i = 0; i < inputFloat.ydim; i++) {
        ::IirConvolvePoles(inputDouble.data + (i * inputDouble.xdim),
                resCpu.data + (i * inputDouble.xdim), inputDouble.xdim, &pole,
                1, MirrorOffBounds, 0.0001);
    }

    bool failed = false;
    for (int i = 0; i < inputFloat.ydim; ++i) {
        for (int j = 0; j < inputFloat.xdim; ++j) {
            int index = i * inputFloat.xdim + j;
            T gpu = resGpu[index];
            T cpu = resCpu[index];
            if (std::abs(cpu - gpu) > 0.001) {
                failed = true;
                fprintf(stderr, "error testCoeffsRow [%d]: GPU %.4f CPU %.4f\n",
                        index, gpu, cpu);
            }
        }
    }

    fprintf(stderr, "testCoeffsRow result: %s\n", failed ? "FAIL" : "OK");
}

template<typename T>
void GeoTransformer<T>::testTranspose() {
    MultidimArray<T> resGpu(32, 32);
    MultidimArray<T> expected(32, 32);
    MultidimArray<T> input(32, 32); // size must be square, multiple of 32
    for (int i = 0; i < input.ydim; ++i) {
        for (int j = 0; j < input.xdim; ++j) {
            input.data[i * input.xdim + j] = i * 100 + j;
            expected.data[j * input.xdim + i] = i * 100 + j;
        }
    }

    T* d_output;
    gpuErrchk(cudaMalloc(&d_output, resGpu.yxdim * sizeof(T)));
    T* d_input;
    gpuErrchk(cudaMalloc(&d_input, input.yxdim * sizeof(T)));
    gpuErrchk(
            cudaMemcpy(d_input, input.data, input.yxdim * sizeof(T),
                    cudaMemcpyHostToDevice));

    dim3 dimGrid(input.xdim / 32, input.ydim / 32, 1); // FIXME this will work only for multiples of 32
    dim3 dimBlock(32, 8, 1);
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_output, d_input);
        gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(
            cudaMemcpy(resGpu.data, d_output, resGpu.yxdim * sizeof(T),
                    cudaMemcpyDeviceToHost));

    bool failed = false;
    for (int i = 0; i < input.ydim; ++i) {
        for (int j = 0; j < input.xdim; ++j) {
            int index = i * input.xdim + j;
            T gpu = resGpu[index];
            T cpu = expected[index];
            if (std::abs(cpu - gpu) > 0.001) {
                failed = true;
                fprintf(stderr, "error transpose [%d]: GPU %.4f CPU %.4f\n",
                        index, gpu, cpu);
            }
        }
    }

    fprintf(stderr, "test Transpose result: %s\n", failed ? "FAIL" : "OK");

}

template<typename T>
void GeoTransformer<T>::testCoeffs() {
    srand(42);
    MultidimArray<T> input(4096, 4096); // size must be square, multiple of 32
    MultidimArray<T> resGpu(input.ydim, input.xdim);
    MultidimArray<double> resCpu;
    for (int i = 0; i < input.ydim; ++i) {
        for (int j = 0; j < input.xdim; ++j) {
            double value = rand() / (RAND_MAX / 2000.);
            input.data[i * input.xdim + j] = value;
        }
    }

    this->init(input.xdim, input.ydim, input.zdim);
    this->produceAndLoadCoeffs(3, input);
    gpuErrchk(
            cudaMemcpy(resGpu.data, d_in, resGpu.yxdim * sizeof(T),
                    cudaMemcpyDeviceToHost));
    ::produceSplineCoefficients(3, resCpu, input);

    bool failed = false;
    for (int i = 0; i < input.ydim; ++i) {
        for (int j = 0; j < input.xdim; ++j) {
            int index = i * input.xdim + j;
            T gpu = resGpu[index];
            T cpu = resCpu[index];
            T threshold = std::abs(std::max(gpu, cpu)) / 1000.f;
            T diff = std::abs(cpu - gpu);
            if (diff > threshold && diff > 0.01) {
                failed = true;
                fprintf(stderr,
                        "error Coeffs [%d]: GPU %.4f CPU %.4f (%f > %f)\n",
                        index, gpu, cpu, diff, threshold);
            }
        }
    }

    fprintf(stderr, "test Coeffs result: %s\n", failed ? "FAIL" : "OK");
    this->release();
}

template<typename T>
void GeoTransformer<T>::test(const Matrix2D<T> &transform) {
    MultidimArray<T> resGpu, resCpu;
    MultidimArray<T> input(32, 32);
    for (int i = 0; i < input.ydim; ++i) {
        for (int j = 0; j < input.xdim; ++j) {
            input.data[i * input.xdim + j] = i * 10 + j;
        }
    }

    this->init(input.xdim, input.ydim, input.zdim);
    this->applyGeometry(3, resGpu, input, transform, false, true);
    ::applyGeometry(3, resCpu, input, transform, false, true);

    bool failed = false;
    for (int i = 0; i < input.ydim; ++i) {
        for (int j = 0; j < input.xdim; ++j) {
            int index = i * input.xdim + j;
            T gpu = resGpu[index];
            T cpu = resCpu[index];
            if (std::abs(cpu - gpu) > 0.001) {
                failed = true;
                fprintf(stderr, "error[%d]: GPU %.4f CPU %.4f\n", index, gpu,
                        cpu);
            }
        }
    }

    fprintf(stderr, "test transform result: %s\n", failed ? "FAIL" : "OK");
    this->release();
}

template<typename T>
template<typename T_IN, typename T_MAT>
void GeoTransformer<T>::applyGeometry(int splineDegree,
        MultidimArray<T> &output, const MultidimArray<T_IN> &input,
        const Matrix2D<T_MAT> &transform, bool isInv, bool wrap, T outside,
        const MultidimArray<T> *bCoeffsPtr) {
    checkRestrictions(splineDegree, output, input, transform);
    if (transform.isIdentity()) {
        typeCast(input, output);
        return;
    }

    loadTransform(transform, isInv);
    loadOutput(output, outside);

    if (splineDegree > 1) {
        if (NULL != bCoeffsPtr) {
            loadInput(*bCoeffsPtr);
        } else {
            produceAndLoadCoeffs(splineDegree, input);
        }
    } else {
        loadInput(input);
    }

    if (input.getDim() == 2) {
        if (wrap) {
            applyGeometry_2D_wrap(splineDegree);
        } else {
            throw std::logic_error("Not implemented yet");
        }
    } else {
        throw std::logic_error("Not implemented yet");
    }

    gpuErrchk(
            cudaMemcpy(output.data, d_out, output.zyxdim * sizeof(T),
                    cudaMemcpyDeviceToHost));

}

template<typename T>
template<typename T_MAT>
void GeoTransformer<T>::loadTransform(const Matrix2D<T_MAT> &transform,
        bool isInv) {
    Matrix2D<T_MAT> trInv = isInv ? transform : transform.inv();
    Matrix2D<T> tmp;
    typeCast(trInv, tmp);
    gpuErrchk(
            cudaMemcpy(d_trInv, tmp.mdata, tmp.mdim * sizeof(T),
                    cudaMemcpyHostToDevice));
}

template<typename T>
template<typename T_IN>
void GeoTransformer<T>::produceAndLoadCoeffs(int splineDegree,
        const MultidimArray<T_IN> &input) {
    MultidimArray<T> tmpIn;
    typeCast(input, tmpIn);
    // copy data to gpu
    gpuErrchk(
            cudaMemcpy(d_in, tmpIn.data, input.yxdim * sizeof(T),
                    cudaMemcpyHostToDevice));

    // apply line-wise filter
    dim3 dimBlockConv(1, BLOCK_DIM_X);
    dim3 dimGridConv(1, ceil(input.ydim / (T) dimBlockConv.y));
    // perform row-wise pass
    iirConvolve2D_Cardinal_Bspline_3_MirrorOffBound<<<dimGridConv, dimBlockConv>>>(
            d_in, d_out,
            input.xdim, input.ydim);
    gpuErrchk(cudaPeekAtLastError());

    // transpose data
    dim3 dimGrid(input.xdim / 32, input.ydim / 32, 1); // FIXME this will work only for multiples of 32
    dim3 dimBlock(32, 8, 1);
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_in, d_out);
    gpuErrchk(cudaPeekAtLastError());

    // perform column-wise pass (notice input/output is swapped)
    iirConvolve2D_Cardinal_Bspline_3_MirrorOffBound<<<dimGridConv, dimBlockConv>>>(
        d_in, d_out,
        input.xdim, input.ydim);
    gpuErrchk(cudaPeekAtLastError());

    // transpose data to row-wise again
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_in, d_out);
        gpuErrchk(cudaPeekAtLastError());
}

template<typename T>
void GeoTransformer<T>::applyGeometry_2D_wrap(int splineDegree) {
    T minxp = 0;
    T minyp = 0;
    T minxpp = minxp - XMIPP_EQUAL_ACCURACY;
    T minypp = minyp - XMIPP_EQUAL_ACCURACY;
    T maxxp = X - 1;
    T maxyp = Y - 1;
    T maxxpp = maxxp + XMIPP_EQUAL_ACCURACY;
    T maxypp = maxyp + XMIPP_EQUAL_ACCURACY;

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(ceil(X / (T) dimBlock.x), ceil(Y / (T) dimBlock.y));

    switch (splineDegree) {
    case 3:
        applyGeometryKernel_2D_wrap<T, 3,true><<<dimGrid, dimBlock>>>(d_trInv,
            minxpp, maxxpp, minypp, maxypp,
            minxp, maxxp, minyp, maxyp,
            d_out, (int)X, (int)Y, d_in, (int)X, (int)Y);
        gpuErrchk(cudaPeekAtLastError());
        break;
    default:
        throw std::logic_error("not implemented");
    }
}

template<typename T>
template<typename T_IN>
void GeoTransformer<T>::loadInput(const MultidimArray<T_IN> &input) {
    MultidimArray<T> tmp;
    typeCast(input, tmp);
    gpuErrchk(
            cudaMemcpy(d_in, tmp.data, tmp.zyxdim * sizeof(T),
                    cudaMemcpyHostToDevice));
}

template<typename T>
void GeoTransformer<T>::loadOutput(MultidimArray<T> &output, T outside) {
    if (output.xdim == 0) {
        output.resize(Z, Y, X);
    }
    if (outside != (T) 0) {
        // Initialize output matrix with value=outside
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(output)
        {
            DIRECT_MULTIDIM_ELEM(output, n) = outside;
        }
        gpuErrchk(
                cudaMemcpy(d_out, output.data, output.zyxdim * sizeof(T),
                        cudaMemcpyHostToDevice));
    } else {
        gpuErrchk(cudaMemset(d_out, 0, output.zyxdim * sizeof(T)));
    }
}

template<typename T>
template<typename T_IN, typename T_MAT>
void GeoTransformer<T>::checkRestrictions(int splineDegree,
        MultidimArray<T> &output, const MultidimArray<T_IN> &input,
        const Matrix2D<T_MAT> &transform) {
    if (!isReady)
        throw std::logic_error("Transformer is not ready yet.");
    if (!input.xdim)
        throw std::invalid_argument("Input is empty");
    if ((X != input.xdim) || (Y != input.ydim) || (Z != input.zdim))
        throw std::logic_error(
                "Transformer has been initialized for a different size of the input");
    if (&input == (MultidimArray<T_IN>*) &output)
        throw std::invalid_argument(
                "The input array cannot be the same as the output array");
    if ((input.getDim() == 2)
            && ((transform.Xdim() != 3) || (transform.Ydim() != 3)))
        throw std::invalid_argument("2D transformation matrix is not 3x3");
    if ((input.getDim() == 3)
            && ((transform.Xdim() != 4) || (transform.Ydim() != 4)))
        throw std::invalid_argument("3D transformation matrix is not 4x4");
}
