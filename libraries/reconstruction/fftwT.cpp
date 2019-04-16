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

template<>
const fftwf_plan FFTwT<float>::createPlan(const FFTSettingsNew<float> &settings) {
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
    return planHelper<const fftwf_plan>(settings, f);
}

template<>
const fftw_plan FFTwT<double>::createPlan(const FFTSettingsNew<double> &settings) {
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
    auto result = planHelper<const fftw_plan>(settings, f);
//    printf("result: %p\n", result);
    return result;
}

//template<typename T>
//const fftw_plan FFTwT<T>::createPlan(const FFTSettingsNew<T> &settings) {
//
//}

template<typename T>
template<typename U, typename F>
U FFTwT<T>::planHelper(const FFTSettingsNew<T> &settings, F function) {
    auto n = std::array<int, 3>{(int)settings.sDim().z(), (int)settings.sDim().y(), (int)settings.sDim().x()};
    int rank = 3;
    if (settings.sDim().z() == 1) rank--;
    if ((2 == rank) && (settings.sDim().y() == 1)) rank--;
    int offset = 3 - rank;

    void *in = nullptr;//settings.isForward() ? malloc(settings.sBytesBatch()) : malloc(settings.fBytesBatch());
//    void *out;
//    if (settings.isInPlace()) {
//        out = in;
//    } else {
//        out = settings.isForward() ? malloc(settings.fBytesBatch()) : malloc(settings.sBytesBatch());
//    }
//    void *in = nullptr;
    void *out = settings.isInPlace() ? in : &m_mockOut;

    auto flags =  FFTW_ESTIMATE |  FFTW_PRESERVE_INPUT |  FFTW_UNALIGNED;

    int idist;
    int odist;
    if (settings.isForward()) {
        idist = settings.sDim().xyzPadded();
        odist = settings.fDim().xyzPadded();

//            return fftwf_plan_many_dft_r2c(rank, &n[offset], settings.batch(),
//                    (float *)in, nullptr,
//                    1, idist,
//                    (fftwf_complex *)out, nullptr,
//                    1, odist,
//                    flags);
//        } else if (std::is_same<T, double>::value) {
//            return fftw_plan_many_dft_r2c(rank, &n[offset], settings.batch(),
//                    (double *)in, nullptr,
//                    1, idist,
//                    (fftw_complex *)out, nullptr,
//                    1, odist,
//                    flags);        }
    } else {
        idist = settings.fDim().xyzPadded();
        odist = settings.sDim().xyzPadded();

//        if (std::is_same<T, float>::value) {
//            return fftwf_plan_many_dft_c2r(rank, &n[offset], settings.batch(),
//                    (fftwf_complex *)in, nullptr,
//                    1, idist,
//                    (float *)out, nullptr,
//                    1, odist,
//                    flags);
//        } else if (std::is_same<T, double>::value) {
//            return fftw_plan_many_dft_c2r(rank, &n[offset], settings.batch(),
//                    (fftw_complex *)in, nullptr,
//                    1, idist,
//                    (double *)out, nullptr,
//                    1, odist,
//                    flags);
//        }
    }
    auto tmp = function(rank, &n[offset], settings.batch(),
            in, nullptr,
            1, idist,
            out, nullptr,
            1, odist,
            flags);
//    free(in);
//    if ( ! settings.isInPlace()) free(out);
//    printf("rank %d: howMany %d in %p idist %d out %p odist %d \n",
//            rank, settings.batch(),
//            in, idist, out, odist);
    return tmp;
}

template<typename T>
void* FFTwT<T>::m_mockOut = {};

template<>
void FFTwT<float>::release(fftwf_plan plan) {
    fftwf_destroy_plan(plan);
}
template<>
void FFTwT<double>::release(fftw_plan plan) {
    fftw_destroy_plan(plan);
}

template<>
std::complex<float>* FFTwT<float>::fft(const fftwf_plan plan, const float *in, std::complex<float> *out) {
//    if (std::is_same<T, float>::value) {
        // we can get rid of the const, as the plans we create prohibit reuse of input array
        fftwf_execute_dft_r2c(plan, (float*)in, (fftwf_complex*) out);
        return out;
//    } else if (std::is_same<T, double>::value) {
//        fftw_execute_dft_r2c(plan, in, (fftw_complex*) out);
//    }
    // FIXME DS throw error unsupported type
}

template<>
std::complex<double>* FFTwT<double>::fft(const fftw_plan plan, const double *in, std::complex<double> *out) {
//    if (std::is_same<T, float>::value) {
        // we can get rid of the const, as the plans we create prohibit reuse of input array
//        printf("addresses: %p %p %p\n", plan, in, out);
        fftw_execute_dft_r2c(plan, (double*)in, (fftw_complex*) out);
        return out;
//    } else if (std::is_same<T, double>::value) {
//        fftw_execute_dft_r2c(plan, in, (fftw_complex*) out);
//    }
    // FIXME DS throw error unsupported type
}

// explicit instantiation
template class FFTwT<float>;
template class FFTwT<double>;
