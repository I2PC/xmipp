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
#include <type_traits>
#include "core/xmipp_error.h"
#include "data/fft_settings_new.h"

// XXX HACK to avoid including cufft.h in this header
// https://docs.nvidia.com/cuda/cufft/index.html#cuffthandle says that type is
// unsigned, but the header (v.8) uses int
typedef int cufftHandle;

template<typename T>
class CudaFFT {
public:
    CudaFFT(): m_settings(0) {};
    ~CudaFFT() {
        release();
    }
    void init(const FFTSettingsNew<T> &settings);
    void release();
    std::complex<T>* fft(const T *h_in, std::complex<T> *h_out);

    static std::complex<T>* fft(cufftHandle plan,
            const T *d_in, std::complex<T> *d_out);
    static std::complex<T>* fft(cufftHandle plan, T *d_inOut);
    static cufftHandle createPlan(const FFTSettingsNew<T> &settings);
private:
    cufftHandle m_plan;
    FFTSettingsNew<T> m_settings;
    T *m_d_SD;
    std::complex<T> *m_d_FD;

    bool m_isInit;

    void setDefault();
};

#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_FFT_H_ */
