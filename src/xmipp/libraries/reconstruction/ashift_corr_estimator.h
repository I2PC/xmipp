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

#ifndef LIBRARIES_RECONSTRUCTION_ASHIFT_CORR_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_ASHIFT_CORR_ESTIMATOR_H_

#include "ashift_estimator.h"
#include "data/fft_settings_new.h"
#include "core/utils/memory_utils.h"
#include <complex>
#include <limits>

namespace Alignment {

template<typename T>
class AShiftCorrEstimator : public AShiftEstimator<T> {
public:
    AShiftCorrEstimator() {
        setDefault();
    }
    virtual ~AShiftCorrEstimator() {
        release();
    }

    virtual void init2D(const std::vector<HW*> &hw, AlignType type,
            const FFTSettingsNew<T> &dims, size_t maxShift,
            bool includingBatchFT, bool includingSingleFT,
            bool allowDataOverwrite) = 0;

    virtual void computeCorrelations2DOneToN(
            std::complex<T> *inOut, bool center) = 0;

    virtual void load2DReferenceOneToN(const std::complex<T> *ref) = 0;

    using AShiftEstimator<T>::load2DReferenceOneToN;

    virtual void computeCorrelations2DOneToN(
            const HW &hw,
            std::complex<T> *inOut,
            const std::complex<T> *ref,
            const Dimensions &dims,
            bool center) = 0;

    void release() override;

protected:
    FFTSettingsNew<T> *m_settingsInv;
    size_t m_centerSize;

    // flags
    bool m_includingBatchFT;
    bool m_includingSingleFT;
    bool m_is_ref_FD_loaded;
    bool m_allowDataOverwrite;

    void setDefault() override;
    virtual void init2D(AlignType type,
            const FFTSettingsNew<T> &dims, size_t maxShift,
            bool includingBatchFT, bool includingSingleFT,
            bool allowDataOverwrite);

    void check() override;
    virtual void init2DOneToN() {}; // nothing to do

    // parent init functions cannot be used, but cannot be hidden
    // in private block, to make compiler (NVCC) happy
    using AShiftEstimator<T>::init2D;
    void init2D(const std::vector<HW*> &hw, AlignType type,
                   const Dimensions &dims, size_t batch, size_t maxShift) {};
};

} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_ASHIFT_CORR_ESTIMATOR_H_ */
