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
#include <typeinfo>

#include "data/aft.h"
#include "core/xmipp_error.h"
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
    FFTwT_Startup() {
        fftw_init_threads();
        fftwf_init_threads();}
    ~FFTwT_Startup() {
        fftw_cleanup();
        fftwf_cleanup();
        fftw_cleanup_threads();
        fftwf_cleanup_threads();
    }
};

template<typename T>
class FFTwT : public AFT<T> {
public:

    FFTwT() {
        setDefault();
    };
    ~FFTwT() {
        release();
    }
    void init(const HW &cpu, const FFTSettingsNew<T> &settings, bool reuse=true);
    void release();
    size_t estimatePlanBytes(const FFTSettingsNew<T> &settings);
    std::complex<T>* fft(T *inOut);
    std::complex<T>* fft(const T *in, std::complex<T> *out);

    T* ifft(std::complex<T> *inOut);
    T* ifft(const std::complex<T> *in, T *out);

    template<typename P>
    static std::complex<T>* fft(const P plan,
            const T *in, std::complex<T> *out);
    template<typename P>
    static std::complex<T>* fft(const P plan,
            T *inOut);

    static std::complex<T>* fft(void *plan,
            const T *in, std::complex<T> *out) {
        return fft(cast(plan), in, out);
    }
    static std::complex<T>* fft(void *plan, T *inOut) {
        return fft(cast(plan), inOut);
    }

    template<typename P>
    static T* ifft(const P plan,
            std::complex<T> *in, T *out); // no const in as it can be overwritten!
    template<typename P>
    static T* ifft(const P plan,
            std::complex<T> *inOut);

    static T* ifft(void *plan,
            std::complex<T> *in, T *out) { // no const in as it can be overwritten!
        return ifft(cast(plan), in, out);
    }
    static T* ifft(void *plan,
            std::complex<T> *inOut) {
        return ifft(cast(plan), inOut);
    }

    static const fftw_plan createPlan(
            const CPU &cpu,
            const FFTSettingsNew<double> &settings,
            bool isDataAligned=false);
    static const fftwf_plan createPlan(
            const CPU &cpu,
            const FFTSettingsNew<float> &settings,
            bool isDataAligned=false);

    template<typename P>
    static void release(P plan);

    static void release(void *plan);

    template<typename D>
    static void release(D *alignedData);
    static void* allocateAligned(size_t bytes);

private:
    static void *m_mockOut;

    void *m_plan;
    const FFTSettingsNew<T> *m_settings;
    T *m_SD;
    std::complex<T> *m_FD;

    const CPU *m_cpu;

    bool m_isInit;

    template<typename U, typename F>
    static U planHelper(const FFTSettingsNew<T> &settings, F function,
            int threads, bool isDataAligned);

    void setDefault();
    void check();
    void allocate();

    void release(T *SD, std::complex<T> *FD);

    static typename FFTwT_planType::plan<T>::type cast(void *p) {
        return static_cast<typename FFTwT_planType::plan<T>::type>(p);
    }
};

#endif /* LIBRARIES_RECONSTRUCTION_FFTWT_H_ */
