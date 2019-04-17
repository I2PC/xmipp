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

#include "fftwT.h"

template<typename T>
void FFTwT<T>::init(const CPU &cpu, const FFTSettingsNew<T> &settings, bool reuse) {
    bool canReuse = m_isInit
            && reuse
            && (m_settings->sBytesBatch() <= settings.sBytesBatch())
            && (m_settings->fBytesBatch() <= settings.fBytesBatch());
    bool mustAllocate = !canReuse;
    if (mustAllocate) {
        release();
    }
    // previous plan has to be released, otherwise we will get GPU memory leak
    release(m_plan);

    m_settings = &settings;
    m_cpu = &cpu;

    check();

    m_plan = createPlan(*m_cpu, *m_settings);
    if (mustAllocate) {
        // allocate input data storage
        m_SD = (T*)fftw_malloc(m_settings->sBytesBatch());
        if (m_settings->isInPlace()) {
            // input data holds also the output
            m_FD = (std::complex<T>*)m_SD;
        } else {
            // allocate also the output buffer
            m_FD = (std::complex<T>*)fftw_malloc(m_settings->fBytesBatch());
        }
    }

    m_isInit = true;
}

template<typename T>
void FFTwT<T>::release(void *plan) {
    if (nullptr != plan) {
        FFTwT<T>::release(cast(plan));
    }
}

template<typename T>
void FFTwT<T>::check() {
    if (m_settings->sDim().x() < 1) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "X dim must be at least 1 (one)");
    }
    if ((m_settings->sDim().y() > 1)
            && (m_settings->sDim().x() < 2)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "X dim must be at least 2 (two) for 2D/3D transformations");
    }
    if ((m_settings->sDim().z() > 1)
            && (m_settings->sDim().y() < 2)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Y dim must be at least 2 (two) for 3D transformations");
    }
}

template<typename T>
void FFTwT<T>::setDefault() {
    m_isInit = false;
    m_settings = nullptr;
    m_SD = nullptr;
    m_FD = nullptr;
    m_plan = nullptr;
}

template<typename T>
void FFTwT<T>::release() {
    fftw_free(m_SD);
    if ((void*)m_FD != (void*)m_SD) {
        fftw_free(m_FD);
    }
    release(m_plan);
    setDefault();
}

template<typename T>
std::complex<T>* FFTwT<T>::fft(T *inOut) {
    return fft(inOut, (std::complex<T>*) inOut);
}

template<typename T>
std::complex<T>* FFTwT<T>::fft(const T *in,
        std::complex<T> *out) {
    auto isReady = m_isInit && m_settings->isForward();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to perform Fourier Transform. "
                "Call init() function first");
    }

    // process signals in batches
    for (size_t offset = 0; offset < m_settings->sDim().n(); offset += m_settings->batch()) {
        // how many signals to process
        size_t toProcess = std::min(m_settings->batch(), m_settings->sDim().n() - offset);

        // copy memory
        memcpy(m_SD,
                in + offset * m_settings->sDim().xyzPadded(),
                toProcess * m_settings->sBytesSingle());

        fft(cast(m_plan), m_SD, m_FD);

        // copy data back
        memcpy(out + offset * m_settings->fDim().xyzPadded(),
                m_FD,
                toProcess * m_settings->fBytesSingle());
    }
    return out;
}

template<>
const fftwf_plan FFTwT<float>::createPlan(const CPU &cpu, const FFTSettingsNew<float> &settings) {
    auto f = [&] (int rank, const int *n, int howmany,
            void *in, const int *inembed,
            int istride, int idist,
            void *out, const int *onembed,
            int ostride, int odist,
            unsigned flags) {
        if (settings.isForward()) {
            return fftwf_plan_many_dft_r2c(rank, n, howmany,
                (float *)in, nullptr,
                1, idist,
                (fftwf_complex *)out, nullptr,
                1, odist,
                flags);
        } else {
            return fftwf_plan_many_dft_c2r(rank, n, howmany,
                (fftwf_complex *)in, nullptr,
                1, idist,
                (float *)out, nullptr,
                1, odist,
                flags);
        }
    };
    return planHelper<const fftwf_plan>(settings, f, cpu.noOfParallUnits());
}

template<>
const fftw_plan FFTwT<double>::createPlan(const CPU &cpu, const FFTSettingsNew<double> &settings) {
    auto f = [&] (int rank, const int *n, int howmany,
            void *in, const int *inembed,
            int istride, int idist,
            void *out, const int *onembed,
            int ostride, int odist,
            unsigned flags) {
        if (settings.isForward()) {
            return fftw_plan_many_dft_r2c(rank, n, howmany,
                (double *)in, nullptr,
                1, idist,
                (fftw_complex *)out, nullptr,
                1, odist,
                flags);
        } else {
            return fftw_plan_many_dft_c2r(rank, n, howmany,
                (fftw_complex *)in, nullptr,
                1, idist,
                (double *)out, nullptr,
                1, odist,
                flags);
        }
    };
    auto result = planHelper<const fftw_plan>(settings, f, cpu.noOfParallUnits());
    return result;
}

template<typename T>
template<typename U, typename F>
U FFTwT<T>::planHelper(const FFTSettingsNew<T> &settings, F function, int threads) {
    auto n = std::array<int, 3>{(int)settings.sDim().z(), (int)settings.sDim().y(), (int)settings.sDim().x()};
    int rank = 3;
    if (settings.sDim().z() == 1) rank--;
    if ((2 == rank) && (settings.sDim().y() == 1)) rank--;
    int offset = 3 - rank;

    void *in = nullptr;
    void *out = settings.isInPlace() ? in : &m_mockOut;

    auto flags =  FFTW_ESTIMATE |  FFTW_PRESERVE_INPUT |  FFTW_UNALIGNED;

    int idist;
    int odist;
    if (settings.isForward()) {
        idist = settings.sDim().xyzPadded();
        odist = settings.fDim().xyzPadded();
    } else {
        idist = settings.fDim().xyzPadded();
        odist = settings.sDim().xyzPadded();
    }
    fftw_plan_with_nthreads(threads);
    auto tmp = function(rank, &n[offset], settings.batch(),
            in, nullptr,
            1, idist,
            out, nullptr,
            1, odist,
            flags);
    return tmp;
}

template<typename T>
void* FFTwT<T>::m_mockOut = {};

template<>
void FFTwT<float>::release(fftwf_plan plan) {
    fftwf_destroy_plan(plan);
    plan = nullptr;
}
template<>
void FFTwT<double>::release(fftw_plan plan) {
    fftw_destroy_plan(plan);
    plan = nullptr;
}

template<>
template<>
std::complex<float>* FFTwT<float>::fft(const fftwf_plan plan, const float *in, std::complex<float> *out) {
    // we can remove the const cast, as our plans do not touch input array
    fftwf_execute_dft_r2c(plan, (float*)in, (fftwf_complex*) out);
    return out;
}

template<>
template<>
std::complex<double>* FFTwT<double>::fft(const fftw_plan plan, const double *in, std::complex<double> *out) {
    // we can remove the const cast, as our plans do not touch input array
    fftw_execute_dft_r2c(plan, (double*)in, (fftw_complex*) out);
    return out;
}

template<typename T>
template<typename P>
std::complex<T>* FFTwT<T>::fft(const P plan, T *inOut) {
    return fft(plan, inOut, (std::complex<T>*)inOut);
}

// explicit instantiation
template class FFTwT<float>;
template class FFTwT<double>;
