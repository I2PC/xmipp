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

#ifndef LIBRARIES_RECONSTRUCTION_SHIFT_CORR_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_SHIFT_CORR_ESTIMATOR_H_

#include <typeinfo>
#include "ashift_corr_estimator.h"
#include "data/cpu.h"
#include "fftwT.h"

namespace Alignment {

template<typename T>
class ShiftCorrEstimator : public AShiftCorrEstimator<T> {
public:
    ShiftCorrEstimator() {
        setDefault();
    }

    virtual ~ShiftCorrEstimator() {
        release();
    }

    void release() override;

    void init2D(const HW &hw, AlignType type, const FFTSettingsNew<T> &dims, size_t maxShift,
            bool includingBatchFT=false, bool includingSingleFT=false) override;

    void load2DReferenceOneToN(const std::complex<T> *ref) override;

    void load2DReferenceOneToN(const T *ref) override;

    void computeShift2DOneToN(T *others) override;

    static std::vector<Point2D<float>> computeShifts2DOneToN(
        const CPU &cpu,
        std::complex<T> *othersF,
        T *othersS,
        std::complex<T> *ref,
        const FFTSettingsNew<T> &settings,
        void *plan,
        size_t maxShift);

    void computeCorrelations2DOneToN(std::complex<T> *inOut, bool center) override;

    void computeCorrelations2DOneToN(
        const HW &hw,
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        const Dimensions &dims,
        bool center) override;

    static void sComputeCorrelations2DOneToN(
        const HW &hw,
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        const Dimensions &dims,
        bool center);
private:
    const CPU *m_cpu;

    // host memory
    std::complex<T> *m_single_FD;
    std::complex<T> *m_batch_FD;
    T * m_batch_SD;

    // FT plans
    void *m_singleToFD;
    void *m_batchToFD;
    void *m_batchToSD;

    void init2DOneToN() override;
    void check() override;
    void setDefault() override;
};

}  /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_SHIFT_CORR_ESTIMATOR_H_ */
