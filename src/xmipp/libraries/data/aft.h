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

#ifndef LIBRARIES_DATA_AFT_H_
#define LIBRARIES_DATA_AFT_H_

#include <type_traits>
#include <complex>
#include "hw.h"
#include "data/fft_settings_new.h"

template<typename T>
class AFT {
public:
    virtual ~AFT() {}; // do nothing

    // utility functions
    virtual void init(const HW &hw, const FFTSettingsNew<T> &settings, bool reuse=true) = 0;
    virtual void release() = 0;
    virtual size_t estimatePlanBytes(const FFTSettingsNew<T> &settings) = 0;
    virtual size_t estimateTotalBytes(const FFTSettingsNew<T> &settings) {
        size_t planBytes = estimatePlanBytes(settings);
        size_t dataBytes = settings.sBytesBatch()
                + (settings.isInPlace() ? 0 : settings.fBytesBatch());
        return planBytes + dataBytes;
    }

    // Forward FT
    virtual std::complex<T>* fft(T *inOut) = 0;
    virtual std::complex<T>* fft(const T *in, std::complex<T> *out) = 0;

    // Inverse FT
    virtual T* ifft(std::complex<T> *inOut) = 0;
    virtual T* ifft(const std::complex<T> *in, T *out) = 0;
protected:
    virtual void setDefault() = 0;
};

#endif /* LIBRARIES_DATA_AFT_H_ */
