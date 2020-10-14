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
 * Typical workflow is as follows:
 * 1. - create instance of this class
 * 2. - initialize it
 * 3. - (in loop) apply transformation
 * 4. - release resources
 */

#ifndef CUDA_GEO_TRANSFORMER
#define CUDA_GEO_TRANSFORMER

#include <assert.h>
#include <stdexcept>
#include <memory>
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
    GeoTransformer() { setDefaultValues(); };

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
    void initForMatrix(size_t x, size_t y, size_t z);

    /**
     * Similar as other init() function, except this method has no effect should the instance
     * be already initialized.
     * It is useful for example in a for loop, where first call will initialize
     * resources and following calls will be ignored
     */
    void initLazyForMatrix(size_t x, size_t y = 1, size_t z = 1);

    /**
     * Release previously obtained resources and initialize the transformer
     * for processing images using BSpline coefficients. It also allocates all resources on
     * GPU.
     * @param sizes of the input images and number of images to be processed
     * @param number of BSpline control points, including end points
     */
    void initForBSpline(size_t inX, size_t inY, size_t inN,
            size_t splineX, size_t splineY, size_t splineN);

    /**
     * Similar as the other init() function, except this method has no effect should the instance
     * be already initialized.
     * It is useful for example in a for loop, where first call will initialize
     * resources and following calls will be ignored
     */
    void initLazyForBSpline(size_t inX, size_t inY, size_t inN,
            size_t splineX, size_t splineY, size_t splineN);

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
    template<typename T_MAT>
    void applyGeometry(int splineDegree, MultidimArray<T> &output,
            const MultidimArray<T> &input, const Matrix2D<T_MAT> &transform,
            bool isInv, bool wrap, T outside = 0,
            const MultidimArray<T> *bCoeffsPtr = NULL);

    /**
     * Apply local transformation defined by a BSpline
     * @param splineDegree used for interpolation
     * @param output where resulting image will be stored
     * @param input to process
     * @param coeffs for the X and Y dimension of the input
     * @param imageIdx index of the current image. This function assumes that
     * multiple calls will be done and that interpolation is done also over time
     * @param outside value of the output, where the interpolation does not store anything
     */
    void applyBSplineTransform(int splineDegree,
        MultidimArray<T> &output, const MultidimArray<T> &input,
        const std::pair<Matrix1D<T>, Matrix1D<T>> &coeffs, size_t imageIdx, T outside = 0);

    void test();

private:
    /**
     * Make sure that there's no logical mistake in the transformation
     * @param splineDegree of the transform
     * @param output image
     * @param input image
     * @param transform to perform
     */
    template<typename T_MAT>
    void checkRestrictions(int splineDegree, MultidimArray<T> &output,
            const MultidimArray<T> &input, const Matrix2D<T_MAT> &transform);

    /**
     *  Make sure that there's no logical mistake in the transformation
     * @param splineDegree of the transform
     * @param output image
     * @param input image
     * @param coefficients of the transformation
     * @param imageIdx index of the frame
     */
    void checkRestrictions(int splineDegree,
            MultidimArray<T> &output, const MultidimArray<T> &input,
            const std::pair<Matrix1D<T>, Matrix1D<T>> &coeffs, size_t frameIdx);


    /**
     *  Make sure that there's no logical mistake in the transformation
     * @param output image
     * @param input image
     */
    void checkRestrictions(const MultidimArray<T> &output,
                                        const MultidimArray<T> &input);

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
    void loadInput(const MultidimArray<T> &input);

    /**
     * Applies geometry transformation, wrap case
     * @param splineDegree to use
     */
    void applyGeometry_2D_wrap(int SplineDegree);


    /**
     * Load transform matrix to GPU
     * @param transform to load
     * @param isInv if true, transform is expected to be from resulting image to
     * input image
     */
    template<typename T_MAT>
    void loadTransform(const Matrix2D<T_MAT> &transform, bool isInv);

    /**
     * Set default values to all private fields
     */
    void setDefaultValues();


    void test(const Matrix2D<T> &transform);

    /**
     * Load BSpline interpolation coefficients to GPU
     */
    void loadCoefficients(const Matrix1D<T> &X,
            const Matrix1D<T> &Y);

    /**
     * Resizes output so it can be used for computations
    */
    void setOutputSize(MultidimArray<T> &output);


protected:
    /*
     * Reference computation used for the testing of a faster kernel
    */
    void applyBSplineTransformRef(int splineDegree,
            MultidimArray<T> &output, const MultidimArray<T> &input,
            const std::pair<Matrix1D<T>, Matrix1D<T>> &coeffs, size_t imageIdx, T outside = 0);

        /**
     * Computes spline coefficients of the image and load them to GPU
     * @param splineDegree to be used
     * @param input image used to generate the coefficients
     */
    void produceAndLoadCoeffs(const MultidimArray<T> &input);

    /*
    * Creates a copy of device input memory
    * Used in tests
    */
    std::unique_ptr<T[]> copy_out_d_in( size_t size ) const;

private:
    bool isReadyForMatrix;
    bool isReadyForBspline;

    T *d_trInv; // memory on GPU with inverse transformation (dest->src)
    T *d_in; // memory on GPU with input data
    T *d_out; // memory in GPU with output data

    T *d_coeffsX; // coefficients of the BSpline transformation, X direction
    T *d_coeffsY; // coefficients of the BSpline transformation, Y direction

    // dimensions of the input/output data
    size_t inX;
    size_t inY;
    size_t inZ;
    size_t inN;

    // dimension of the coefficients control points
    size_t splineX;
    size_t splineY;
    size_t splineN;

    constexpr static const T transposeTileDim = (T)32;
    constexpr static const T transposeBlockRow = (T)8;
    constexpr static const int pixelsPerThread = 2;
};

#endif // CUDA_GEO_TRANSFORMER
