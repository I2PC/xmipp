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

#ifndef LIBRARIES_RECONSTRUCTION_SINGLE_EXTREMA_FINDER_H_
#define LIBRARIES_RECONSTRUCTION_SINGLE_EXTREMA_FINDER_H_

#include "data/cpu.h"
#include <algorithm>
#include <functional>
#include "aextrema_finder.h"

namespace ExtremaFinder {

template<typename T>
class SingleExtremaFinder : public AExtremaFinder<T> {
public:
    SingleExtremaFinder() {
        setDefault();
    }

    virtual ~SingleExtremaFinder() {
        release();
    }

    SingleExtremaFinder(SingleExtremaFinder& o) = delete;
    SingleExtremaFinder& operator=(const SingleExtremaFinder& other) = delete;
    SingleExtremaFinder const & operator=(SingleExtremaFinder &&o) = delete;
    SingleExtremaFinder(SingleExtremaFinder &&o) {
        m_cpu = o.m_cpu;
        // clean original
        o.setDefault();
    }

    static void sFindMax(const CPU &cpu,
        const Dimensions &dims,
        const T *data,
        float *positions,
        T *values);

    static void sFindLowest(const CPU &cpu,
        const Dimensions &dims,
        const T *data,
        float *positions,
        T *values);

    template<typename C>
    static void sFindUniversal2DAroundCenter(
        const C &comp,
        T startVal,
        const CPU &cpu,
        const Dimensions &dims,
        const T *data,
        float *positions, // can be nullptr
        T * values, // can be nullptr
        size_t maxDist);

    static void sFindMax2DAroundCenter(const CPU &cpu,
        const Dimensions &dims,
        const T *data,
        float *positions, // can be nullptr
        T * values, // can be nullptr
        size_t maxDist);

    static void sFindLowest2DAroundCenter(const CPU &cpu,
        const Dimensions &dims,
        const T *data,
        float *positions, // can be nullptr
        T * values, // can be nullptr
        size_t maxDist);

private:
    CPU *m_cpu; // FIXME DS eventually use thread pool?

    void setDefault();
    void release();

    void check() const override;

    void initMax() override;
    void findMax(const T *data) override;
    bool canBeReusedMax(const ExtremaFinderSettings &s) const override;

    void initLowest() override;
    void findLowest(const T *data) override;
    bool canBeReusedLowest(const ExtremaFinderSettings &s) const override;

    void initMaxAroundCenter() override;
    void findMaxAroundCenter(const T *data) override;
    bool canBeReusedMaxAroundCenter(const ExtremaFinderSettings &s) const override;

    void initLowestAroundCenter() override;
    void findLowestAroundCenter(const T *data) override;
    bool canBeReusedLowestAroundCenter(const ExtremaFinderSettings &s) const override;

    static void sFindUniversalChecks(
            const Dimensions &dims,
            const T *__restrict__ data,
            float *__restrict__ positions,
            T *__restrict__ values);

    void initBasic();
    template<typename KERNEL>
    void findBasic(const T *data, const KERNEL &k);
};

} /* namespace ExtremaFinder */

#endif /* LIBRARIES_RECONSTRUCTION_SINGLE_EXTREMA_FINDER_H_ */
