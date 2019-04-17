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

#ifndef LIBRARIES_RECONSTRUCTION_FFTWT_H_
#define LIBRARIES_RECONSTRUCTION_FFTWT_H_

#include <fftw3.h>
#include <array>
//#include <type_traits>
#include "core/xmipp_error.h"
#include "data/fft_settings_new.h"
#include "data/cpu.h"


namespace FFTwT_planType {
    template<class T>
    struct plan{ typedef T type; };
    template<>
    struct plan<float>{ typedef fftwf_plan type; };
    template<>
    struct plan<double>{ typedef fftw_plan type; };
}

class FFTwT_Startup {
public:
    FFTwT_Startup() { fftw_init_threads(); }
    ~FFTwT_Startup() { fftw_cleanup_threads(); }
};

FFTwT_Startup fftwt_startup;

template<typename T>
class FFTwT {
public:

    FFTwT() {
        setDefault();
    };
    ~FFTwT() {
        release();
    }
    void init(const CPU &cpu, const FFTSettingsNew<T> &settings, bool reuse=true);
    void release();
    std::complex<T>* fft(T *inOut);
    std::complex<T>* fft(const T *in, std::complex<T> *out);


    template<typename P>
    static std::complex<T>* fft(const P plan,
            const T *in, std::complex<T> *out);

    template<typename P>
    static std::complex<T>* fft(const P plan,
            T *inOut);

    static const fftw_plan createPlan(
            const CPU &cpu,
            const FFTSettingsNew<double> &settings);
    static const fftwf_plan createPlan(
            const CPU &cpu,
            const FFTSettingsNew<float> &settings);

    static void release(fftw_plan plan);
    static void release(fftwf_plan plan);
    static void release(void* plan);

private:
    static void *m_mockOut;

    void *m_plan;
    const FFTSettingsNew<T> *m_settings;
    T *m_SD;
    std::complex<T> *m_FD;

    const CPU *m_cpu;

    bool m_isInit;

    template<typename U, typename F>
    static U planHelper(const FFTSettingsNew<T> &settings, F function, int threads);

    void setDefault();
    void check();

    static typename FFTwT_planType::plan<T>::type cast(void *p) {
        return static_cast<typename FFTwT_planType::plan<T>::type>(p);
    }
};

#endif /* LIBRARIES_RECONSTRUCTION_FFTWT_H_ */
