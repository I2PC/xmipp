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
 * This class is able to apply shift transformation on given image(s),
 * using GPU (by multiplication in Fourier space).
 * Internally, it processes each pixel of the resulting image.
 * Typical workflow is as follows:
 * 1. - create instance of this class
 * 2. - initialize it
 * 3. - (in loop) apply shift
 * 4. - release resources
 */

#ifndef CUDA_GEO_SHIFT_TRANSFORMER
#define CUDA_GEO_SHIFT_TRANSFORMER

#include <assert.h>
#include <exception>
#include "core/multidim_array.h"
#include "reconstruction/gpu_geo_transformer_defines.h"
#include "cuda_xmipp_utils.h"
#include "gpu.h"

template<typename T>
class GeoShiftTransformer {

public:
    /** Constructor */
    GeoShiftTransformer() :
        device(-1), imgs(NULL), ffts(NULL), stream(NULL), isReady(false) {
    };

    ~GeoShiftTransformer() {
        release();
    }

    /**
     * Release previously obtained resources and initialize the transformer
     * for processing images of given size. It also allocates all resources on
     * GPU.
     * @param gpu to use
     * @param x dim (inner-most) of the resulting image
     * @param y dim (outer-most) of the resulting image
     * @param n no. of images to process in a single batch
     * @param device to be used
     * @param stream to be used. NULL for default
     */
    void init(const GPU &gpu, size_t x, size_t y, size_t n, int device, myStreamHandle *stream);

    /**
     * Similar as init(), except this method has no effect should the instance
     * be already initialized.
     * It is useful for example in a for loop, where first call will initialize
     * resources and following calls will be ignored
     * Do NOT use it for reinitialization.
     */
    void initLazy(const GPU &gpu, size_t x, size_t y, size_t n, int device,
            myStreamHandle *stream = NULL);

    /**
     * Release all resources hold by this instance
     */
    void release();

    /**
     * Apply 2D shift. Image will be repeated at the border.
     * @param output where resulting images will be stored. Does not have to be initialized
     * @param input to process
     * @param shiftX to apply
     * @param shiftY to apply
     */
    template<typename T_IN>
    void applyShift(MultidimArray<T> &output, const MultidimArray<T_IN> &input,
            T shiftX, T shiftY);

    void test();

private:
    /**
     * Make sure that there's no logical mistake
     * @param output images
     * @param input images
     */
    template<typename T_IN>
    void checkRestrictions(MultidimArray<T> &output,
            const MultidimArray<T_IN> &input);

private:
    bool isReady;

    int device;
    myStreamHandle* stream;

    const GPU *m_gpu;

    GpuMultidimArrayAtGpu<T> *imgs; // object for FFT
    mycufftHandle fftHandle; // plan of the FFT
    GpuMultidimArrayAtGpu<std::complex<T> > *ffts; // object for FFT
    mycufftHandle ifftHandle; // plan of the FFT
};

#endif // CUDA_GEO_SHIFT_TRANSFORMER
