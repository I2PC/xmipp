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

#include "cuda_gpu_geo_transformer.h"
#include "cuda_utils.h"
#include <cuda_runtime_api.h>
#include "cuda_all.cpp"

template<typename T>
void GeoTransformer<T>::release() {
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_trInv);
    cudaFree(d_coeffsX);
    cudaFree(d_coeffsY);
    setDefaultValues();
}

template<typename T>
void GeoTransformer<T>::setDefaultValues() {
    isReadyForBspline = isReadyForMatrix = false;
    d_trInv = d_in = d_out = d_coeffsX = d_coeffsY = nullptr;
    inX = inY = inZ = splineX = splineY = splineN;
}

template<typename T>
void GeoTransformer<T>::initForMatrix(size_t x, size_t y, size_t z) {
    release();

    inX = x;
    inY = y;
    inZ = z;
    size_t matSize = (0 == z) ? 9 : 16;
    gpuErrchk(cudaMalloc((void** ) &d_trInv, matSize * sizeof(T)));
    gpuErrchk(cudaMalloc((void** ) &d_in, x * y * z * sizeof(T)));
    gpuErrchk(cudaMalloc((void** ) &d_out, x * y * z * sizeof(T)));

    isReadyForMatrix = true;
}

template<typename T>
void GeoTransformer<T>::initLazyForMatrix(size_t x, size_t y, size_t z) {
    if (!isReadyForMatrix) {
        initForMatrix(x, y, z);
    }
}

template<typename T>
void GeoTransformer<T>::initForBSpline(size_t inX, size_t inY, size_t inN,
        size_t splineX, size_t splineY, size_t splineN) {
    release();

    this->inX = inX;
    this->inY = inY;
    this->inZ = 1;
    this->inN = inN;
    this->splineX = splineX;
    this->splineY = splineY;
    this->splineN = splineN;
    // take into account end control points
    size_t coeffsSize = splineX * splineY * splineN;
    size_t inOutSize = inX * inY;
    gpuErrchk(cudaMalloc((void** ) &d_coeffsX, coeffsSize * sizeof(T)));
    gpuErrchk(cudaMalloc((void** ) &d_coeffsY, coeffsSize * sizeof(T)));
    gpuErrchk(cudaMalloc((void** ) &d_in, inOutSize * sizeof(T)));
    gpuErrchk(cudaMalloc((void** ) &d_out, inOutSize * sizeof(T)));

    isReadyForBspline = true;
}

template<typename T>
void GeoTransformer<T>::initLazyForBSpline(size_t inX, size_t inY, size_t inZ,
        size_t splineX, size_t splineY, size_t splineN) {
    if (!isReadyForBspline) {
        initForBSpline(inX, inY, inZ, splineX, splineY, splineN);
    }
}

template<typename T>
void GeoTransformer<T>::test() {
    testCoeffsRow();
    testCoeffsRowNew();
    testTranspose();
    testCoeffs();

    Matrix1D<T> shift(2);
    shift.vdata[0] = 0.45;
    shift.vdata[1] = 0.62;
    Matrix2D<T> transform;
    translation2DMatrix(shift, transform, true);
    test(transform);
}

template<typename T>
void GeoTransformer<T>::testCoeffsRowNew() {
    MultidimArray<double> inputDouble(53, 62);
    MultidimArray<T> resGpu(inputDouble.ydim, inputDouble.xdim);
    MultidimArray<double> resCpu(inputDouble.ydim, inputDouble.xdim);
    MultidimArray<T> inputFloat(inputDouble.ydim, inputDouble.xdim);
    for (int i = 0; i < inputFloat.ydim; ++i) {
        for (int j = 0; j < inputFloat.xdim; ++j) {
            inputDouble.data[i * inputFloat.xdim + j] = i * inputFloat.xdim + j;
            inputFloat.data[i * inputFloat.xdim + j] = i * inputFloat.xdim + j;
        }
    }

    T* d_output;
    gpuErrchk(cudaMalloc(&d_output, resGpu.yxdim * sizeof(T)));
    T* d_input;
    gpuErrchk(cudaMalloc(&d_input, inputFloat.yxdim * sizeof(T)));
    gpuErrchk(
            cudaMemcpy(d_input, inputFloat.data, inputFloat.yxdim * sizeof(T),
                    cudaMemcpyHostToDevice));

    //
    dim3 dimGrid(std::ceil(inputFloat.xdim / transposeTileDim), std::ceil(inputFloat.ydim / transposeTileDim), 1);
    dim3 dimBlock(transposeTileDim, transposeBlockRow, 1);
    transposeNoBankConflicts32x8<<<dimGrid, dimBlock>>>(d_output, d_input, inputFloat.xdim, inputFloat.ydim);
    gpuErrchk(cudaPeekAtLastError());


    dim3 dimBlockConv(BLOCK_DIM_X, 1);
    dim3 dimGridConv(ceil(inputFloat.xdim / (T) dimBlockConv.x));
    // perform row-wise pass
    iirConvolve2D_Cardinal_Bspline_3_MirrorOffBoundNew<<<dimGridConv, dimBlockConv>>>(
        d_output, d_input,
        (int)resGpu.xdim, (int)resGpu.ydim);
    gpuErrchk(cudaPeekAtLastError());

    // transform back
    transposeNoBankConflicts32x8<<<dimGrid, dimBlock>>>(d_output, d_input, inputFloat.ydim, inputFloat.xdim);
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

    fprintf(stderr, "testCoeffsRowNew result: %s\n", failed ? "FAIL" : "OK");
}


template<typename T>
void GeoTransformer<T>::testCoeffsRow() {
    MultidimArray<double> inputDouble(53, 62);
    MultidimArray<T> resGpu(inputDouble.ydim, inputDouble.xdim);
    MultidimArray<double> resCpu(inputDouble.ydim, inputDouble.xdim);
    MultidimArray<T> inputFloat(inputDouble.ydim, inputDouble.xdim);
    for (int i = 0; i < inputFloat.ydim; ++i) {
        for (int j = 0; j < inputFloat.xdim; ++j) {
            inputDouble.data[i * inputFloat.xdim + j] = i * inputFloat.xdim + j;
            inputFloat.data[i * inputFloat.xdim + j] = i * inputFloat.xdim + j;
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
    MultidimArray<T> input(33, 52);
    MultidimArray<T> resGpu(input.xdim, input.ydim);
    MultidimArray<T> expected(input.xdim, input.ydim);
    for (int i = 0; i < input.ydim; ++i) {
        for (int j = 0; j < input.xdim; ++j) {
            input.data[i * input.xdim + j] = i * input.xdim + j;
            expected.data[j * input.ydim + i] = i * input.xdim + j;
        }
    }

    T* d_output;
    gpuErrchk(cudaMalloc(&d_output, resGpu.yxdim * sizeof(T)));
    T* d_input;
    gpuErrchk(cudaMalloc(&d_input, input.yxdim * sizeof(T)));
    gpuErrchk(
            cudaMemcpy(d_input, input.data, input.yxdim * sizeof(T),
                    cudaMemcpyHostToDevice));

    dim3 dimGrid(std::ceil(input.xdim / transposeTileDim), std::ceil(input.ydim / transposeTileDim), 1);
    dim3 dimBlock(transposeTileDim, transposeBlockRow, 1);
    transposeNoBankConflicts32x8<<<dimGrid, dimBlock>>>(d_output, d_input, input.xdim, input.ydim);
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
    MultidimArray<T> input(52, 130);
    MultidimArray<T> resGpu(input.ydim, input.xdim);
    MultidimArray<double> resCpu;
    for (int i = 0; i < input.ydim; ++i) {
        for (int j = 0; j < input.xdim; ++j) {
            double value = rand() / (RAND_MAX / 2000.);
            input.data[i * input.xdim + j] = value;
        }
    }

    this->initForMatrix(input.xdim, input.ydim, input.zdim);
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

    this->initForMatrix(input.xdim, input.ydim, input.zdim);
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
void GeoTransformer<T>::applyBSplineTransform(
        int splineDegree,
        MultidimArray<T> &output, const MultidimArray<T> &input,
        const std::pair<Matrix1D<T>, Matrix1D<T>> &coeffs, size_t imageIdx, T outside) {
    checkRestrictions(3, output, input, coeffs, imageIdx);

    loadOutput(output, outside);
    produceAndLoadCoeffs(splineDegree, input);

    loadCoefficients(coeffs.first, coeffs.second);

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(ceil(inX / (T) dimBlock.x), ceil(inY / (T) dimBlock.y));

    switch (splineDegree) {
    case 3:
        applyLocalShiftGeometryKernel<T, 3><<<dimGrid, dimBlock>>>(d_coeffsX, d_coeffsY,
                d_out, (int)inX, (int)inY, (int)inN,
                d_in, imageIdx, (int)splineX, (int)splineY, (int)splineN);
            gpuErrchk(cudaPeekAtLastError());
        break;
    default:
        throw std::logic_error("not implemented");
    }

    gpuErrchk(
            cudaMemcpy(output.data, d_out, output.zyxdim * sizeof(T),
                    cudaMemcpyDeviceToHost));
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
void GeoTransformer<T>::loadCoefficients(const Matrix1D<T> &X,
        const Matrix1D<T> &Y) {
     gpuErrchk(
                 cudaMemcpy(d_coeffsX, X.vdata, X.vdim * sizeof(T),
                         cudaMemcpyHostToDevice));
     gpuErrchk(
                 cudaMemcpy(d_coeffsY, Y.vdata, Y.vdim * sizeof(T),
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

    // transpose data, because the kernel is faster that way
    dim3 dimGridTransF(std::ceil(tmpIn.xdim / transposeTileDim), std::ceil(tmpIn.ydim / transposeTileDim), 1);
    dim3 dimBlockTransF(transposeTileDim, transposeBlockRow, 1);
    transposeNoBankConflicts32x8<<<dimGridTransF, dimBlockTransF>>>
            (d_out, d_in, tmpIn.xdim, tmpIn.ydim);
    gpuErrchk(cudaPeekAtLastError());

    // apply line-wise filter
    dim3 dimBlockConvL(BLOCK_DIM_X, 1);
    dim3 dimGridConvL(ceil(input.ydim / (T) dimBlockConvL.x), 1);
    // perform row-wise pass
    iirConvolve2D_Cardinal_Bspline_3_MirrorOffBoundNew<<<dimGridConvL, dimBlockConvL>>>(
            d_out, d_in,
            tmpIn.xdim, tmpIn.ydim);
    gpuErrchk(cudaPeekAtLastError());

    // transpose data
    dim3 dimGridTransI(dimGridTransF.y, dimGridTransF.x);
    dim3 dimBlockTransI(transposeTileDim, transposeBlockRow, 1);
    transposeNoBankConflicts32x8<<<dimGridTransI, dimBlockTransI>>>
            (d_out, d_in, tmpIn.ydim, tmpIn.xdim);
    gpuErrchk(cudaPeekAtLastError());

    // perform column-wise pass
    dim3 dimBlockConvC(BLOCK_DIM_X, 1);
    dim3 dimGridConvC(ceil(input.xdim / (T) dimBlockConvL.x), 1);
    iirConvolve2D_Cardinal_Bspline_3_MirrorOffBoundNew<<<dimGridConvC, dimBlockConvC>>>(
        d_out, d_in,
        input.ydim, input.xdim);
    gpuErrchk(cudaPeekAtLastError());
}

template<typename T>
void GeoTransformer<T>::applyGeometry_2D_wrap(int splineDegree) {
    T minxp = 0;
    T minyp = 0;
    T minxpp = minxp - XMIPP_EQUAL_ACCURACY;
    T minypp = minyp - XMIPP_EQUAL_ACCURACY;
    T maxxp = inX - 1;
    T maxyp = inY - 1;
    T maxxpp = maxxp + XMIPP_EQUAL_ACCURACY;
    T maxypp = maxyp + XMIPP_EQUAL_ACCURACY;

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(ceil(inX / (T) dimBlock.x), ceil(inY / (T) dimBlock.y));

    switch (splineDegree) {
    case 3:
        applyGeometryKernel_2D_wrap<T, 3,true><<<dimGrid, dimBlock>>>(d_trInv,
            minxpp, maxxpp, minypp, maxypp,
            minxp, maxxp, minyp, maxyp,
            d_out, (int)inX, (int)inY, d_in, (int)inX, (int)inY);
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
        output.resize(inZ, inY, inX);
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
    if (!isReadyForMatrix)
        throw std::logic_error("Transformer is not ready yet.");
    if (!input.xdim)
        throw std::invalid_argument("Input is empty");
    if ((inX != input.xdim) || (inY != input.ydim) || (inZ != input.zdim))
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


template<typename T>
void GeoTransformer<T>::checkRestrictions(int splineDegree,
        MultidimArray<T> &output, const MultidimArray<T> &input,
        const std::pair<Matrix1D<T>, Matrix1D<T>> &coeffs, size_t frameIdx) {
    if (!isReadyForBspline)
        throw std::logic_error("Transformer is not ready yet.");
    if (!input.xdim)
        throw std::invalid_argument("Input is empty");
    if ((inX != input.xdim) || (inY != input.ydim) || (inZ != input.zdim))
        throw std::logic_error(
                "Transformer has been initialized for a different size of the input");
    if (&input == &output)
        throw std::invalid_argument(
                "The input array cannot be the same as the output array");
    if (frameIdx > inN)
        throw std::invalid_argument("Frame index is out of bound");
    size_t coeffsElems = splineX * splineY * splineN;
    if ((coeffs.first.size() != coeffsElems) || (coeffs.second.size() != coeffsElems))
        throw std::invalid_argument("Number of coefficients does not fit. "
                "To init function, pass N control points.");
}

template class GeoTransformer<float>;
