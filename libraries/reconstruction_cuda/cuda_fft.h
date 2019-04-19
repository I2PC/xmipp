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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_CUDA_FFT_H_
#define LIBRARIES_RECONSTRUCTION_CUDA_CUDA_FFT_H_


#include <array>
#include <typeinfo>

#include "data/aft.h"
#include "core/xmipp_error.h"
#include "core/utils/memory_utils.h"
#include "core/optional.h"
#include "gpu.h"

// XXX HACK to avoid including cufft.h in this header
// https://docs.nvidia.com/cuda/cufft/index.html#cuffthandle says that type is
// unsigned, but the header (v.8) uses int
typedef int cufftHandle;

template<typename T>
class CudaFFT : public AFT<T> {
public:
    CudaFFT() {
        setDefault();
    };
    ~CudaFFT() {
        release();
    }
    void init(const HW &gpu, const FFTSettingsNew<T> &settings, bool reuse=true);
    void release();
    std::complex<T>* fft(T *h_inOut);
    std::complex<T>* fft(const T *h_in, std::complex<T> *h_out);

    T* ifft(std::complex<T> *h_inOut);
    T* ifft(const std::complex<T> *h_in, T *h_out);


    size_t estimatePlanBytes(const FFTSettingsNew<T> &settings);
    static std::complex<T>* fft(cufftHandle plan, T *d_inOut);
    static std::complex<T>* fft(cufftHandle plan,
            const T *d_in, std::complex<T> *d_out);
    static T* ifft(cufftHandle plan, std::complex<T> *d_inOut);
    static T* ifft(cufftHandle plan,
            const std::complex<T> *d_in, T *d_out);
    static cufftHandle* createPlan(const GPU &gpu,
            const FFTSettingsNew<T> &settings);
    static core::optional<FFTSettingsNew<T>> findOptimal(GPU &gpu,
            const FFTSettingsNew<T> &settings,
            size_t reserveBytes, bool squareOnly, int sigPercChange,
            bool crop, bool verbose);
    static FFTSettingsNew<T> findMaxBatch(const FFTSettingsNew<T> &settings,
            size_t maxBytes);
    static FFTSettingsNew<T> findOptimalSizeOrMaxBatch(GPU &gpu,
            const FFTSettingsNew<T> &settings,
            size_t reserveBytes, bool squareOnly, int sigPercChange,
            bool crop, bool verbose);
    static void release(cufftHandle *plan);
private:
    cufftHandle *m_plan;
    const FFTSettingsNew<T> *m_settings;
    T *m_d_SD;
    std::complex<T> *m_d_FD;

    const GPU *m_gpu;

    bool m_isInit;

    void setDefault();
    template<typename F>
    static void manyHelper(const FFTSettingsNew<T> &settings, F function);
    void check();
};

#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_FFT_H_ */
