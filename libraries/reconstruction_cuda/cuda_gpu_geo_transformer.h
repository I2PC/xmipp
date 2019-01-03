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

/**
 * This class is able to apply geometrical transformation on given image(s),
 * using GPU.
 * Internally, it processes each pixel of the resulting image.
 * Typical workflowis as follows:
 * 1. - create instance of this class
 * 2. - initialize it
 * 3. - (in loop) apply transformation
 * 4. - release resources
 */

#ifndef CUDA_GEO_TRANSFORMER
#define CUDA_GEO_TRANSFORMER

#include <assert.h>
#include <stdexcept>
#include "core/multidim_array.h"
#include "core/transformations.h"
#include "core/xmipp_image.h" // for tests only
#include "core/bilib/iirconvolve.h" // for tests only
#include "reconstruction/gpu_geo_transformer_defines.h"
#include "cuda_xmipp_utils.h"

template<typename T>
class GeoTransformer {

public:
    /** Constructor */
    GeoTransformer() :
            X(0), Y(0), Z(0), isReady(false), d_in(NULL), d_out(NULL), d_trInv(
                    NULL) {
    }
    ;

    ~GeoTransformer() {
        release();
    }

    /**
     * Release previously obtained resources and initialize the transformer
     * for processing images of given size. It also allocates all resources on
     * GPU.
     * @param x dim (inner-most) of the resulting image
     * @param y dim of the resulting image
     * @param z dim (outer-most) of the resulting image
     */
    void init(size_t x, size_t y, size_t z);

    /**
     * Similar as init(), except this method has no effect should the instance
     * be already initialized.
     * It is useful for example in a for loop, where first call will initialize
     * resources and following calls will be ignored
     */
    void initLazy(size_t x, size_t y = 1, size_t z = 1);

    /**
     * Release all resources hold by this instance
     */
    void release();

    /**
     * Apply geometry transformation
     * Currently supported transformations: FIXME
     *  * 2D shift using cubic interpolation and mirrorOffBound
     * @param splineDegree used for interpolation
     * @param output where resulting image will be stored
     * @param input to process
     * @param transform to apply
     * @param isInv if true, 'transform' is expected to be from the resulting
     * image to source image
     * @param wrap true to wrap after boundaris
     * @param outside value to be used when reading outside of the image and wrap == false
     * @param bCoeffsPtr spline coefficients to use
     */
    template<typename T_IN, typename T_MAT>
    void applyGeometry(int splineDegree, MultidimArray<T> &output,
            const MultidimArray<T_IN> &input, const Matrix2D<T_MAT> &transform,
            bool isInv, bool wrap, T outside = 0,
            const MultidimArray<T> *bCoeffsPtr = NULL);

    void applyLocalShift(
            MultidimArray<T> &output, const MultidimArray<T> &input,
            const std::pair<Matrix1D<T>, Matrix1D<T>> &coefs, size_t frameIdx);

    template<typename T_IN>
    void applyShift(MultidimArray<T> &output,
            const MultidimArray<T_IN> &input, T shiftX, T shiftY);

    void test();

private:
    /**
     * Make sure that there's no logical mistake in the transformation
     * @param splineDegree of the transform
     * @param output image
     * @param input image
     * @param transform to perform
     */
    template<typename T_IN, typename T_MAT>
    void checkRestrictions(int splineDegree, MultidimArray<T> &output,
            const MultidimArray<T_IN> &input, const Matrix2D<T_MAT> &transform);

    /**
     * Makes sure that output is big enough and sets default value
     * @param output where result will be stored
     * @param outside (background) value to be used
     */
    void loadOutput(MultidimArray<T> &output, T outside);

    /**
     * Loads input image to GPU
     * @param input to load
     */
    template<typename T_IN>
    void loadInput(const MultidimArray<T_IN> &input);

    /**
     * Applies geometry transformation, wrap case
     * @param splineDegree to use
     */
    void applyGeometry_2D_wrap(int SplineDegree);

    /**
     * Computes spline coefficients and load them to GPU
     * @param splineDegree to be used
     * @param input image used to generate the coefficients
     */
    template<typename T_IN>
    void produceAndLoadCoeffs(int splineDegree,
            const MultidimArray<T_IN> &input);

    /**
     * Load transform matrix to GPU
     * @param transform to load
     * @param isInv if true, transform is expected to be from resulting image to
     * input image
     */
    template<typename T_MAT>
    void loadTransform(const Matrix2D<T_MAT> &transform, bool isInv);

    void test(const Matrix2D<T> &transform);

    void testCoeffs();

    void testTranspose();

    void testCoeffsRow();

    void loadCoefficients(const Matrix1D<T> &X,
            const Matrix1D<T> &Y);

private:
    bool isReady;

    T* d_trInv; // memory on GPU with inverse transformation (dest->src)
    T* d_in; // memory on GPU with input data
    T* d_out; // memory in GPU with output data

    T *d_coefsX;
    T *d_coefsY;

    /** dimensions of the output data */
    size_t X, Y, Z;
};

#endif // CUDA_GEO_TRANSFORMER
