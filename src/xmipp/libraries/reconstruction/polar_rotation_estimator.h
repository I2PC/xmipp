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

#ifndef LIBRARIES_RECONSTRUCTION_POLAR_ROTATION_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_POLAR_ROTATION_ESTIMATOR_H_

#include "arotation_estimator.cpp"
#include "data/cpu.h"
#include "data/polar.h"

namespace Alignment {

template<typename T>
class PolarRotationEstimator : public ARotationEstimator<T> {
public:
    PolarRotationEstimator() {
        setDefault();
    }

    virtual ~PolarRotationEstimator() {
        release();
    }

    PolarRotationEstimator(PolarRotationEstimator& o) = delete;
    PolarRotationEstimator& operator=(const PolarRotationEstimator& other) = delete;
    PolarRotationEstimator(PolarRotationEstimator &&o) {
        m_cpu = o.m_cpu;
        m_polarFourierI = o.m_polarFourierI;
        m_refPolarFourierI = o.m_refPolarFourierI;
        m_rotCorrAux = o.m_rotCorrAux;
        m_dataAux = o.m_dataAux;
        m_aux = o.m_aux;
        m_plans = o.m_plans;
        m_refPlans = o.m_refPlans;
        m_firstRing = o.m_firstRing;
        m_lastRing = o.m_lastRing;
        // remove data from other
        o.setDefault();
    }
    PolarRotationEstimator const & operator=(PolarRotationEstimator &&o) = delete;

private:
    CPU *m_cpu;
    Polar<std::complex<double>> m_polarFourierI; // FIXME DS add template
    Polar<std::complex<double>> m_refPolarFourierI; // FIXME DS add template
    MultidimArray<double> m_rotCorrAux;
    MultidimArray<double> m_dataAux;
    RotationalCorrelationAux m_aux;
    Polar_fftw_plans *m_plans; // fixme DS use unique_ptr
    Polar_fftw_plans *m_refPlans;
    int m_firstRing;
    int m_lastRing;

    void release();
    void setDefault();
    void check() override;

    MultidimArray<double> convert(T *data); // FIXME DS move to multidimarray.h

    void init2D() override;

    void load2DReferenceOneToN(const T *ref) override;

    void computeRotation2DOneToN(T *others) override;
    bool canBeReused2D(const RotationEstimationSetting &s) const override {
        return false; // FIXME DS implement
    }
};

} /* namespace Alignment */

#endif /* LIBRARIES_RECONSTRUCTION_POLAR_ROTATION_ESTIMATOR_H_ */
