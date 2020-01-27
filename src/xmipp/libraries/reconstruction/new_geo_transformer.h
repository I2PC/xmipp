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

#ifndef LIBRARIES_RECONSTRUCTION_NEW_GEO_TRANSFORMER_H_
#define LIBRARIES_RECONSTRUCTION_NEW_GEO_TRANSFORMER_H_

#include "reconstruction/ageo_transformer.h"
#include "data/filters.h"
#include "CTPL/ctpl_stl.h"
#include "data/cpu.h"

template<typename T>
class NewGeoTransformer : public AGeoTransformer<T> {
public:

    NewGeoTransformer() {
        setDefault();
    }

    virtual ~NewGeoTransformer() {
        release();
    }

    void setOriginal(const T *data) override {
        m_orig = data;
        this->setIsOrigLoaded(nullptr != data);
    }

    const T *getOriginal() const {
        return m_orig;
    }

    T *getCopy() const override {
        return m_copy.get();
    }

    void copyOriginalToCopy() override;

    T *interpolate(const std::vector<float> &matrices) override;

private:
    void init(bool doAllocation) override;
    void release();
    void setDefault() {};
    void check() override;

    void checkBSpline(const BSplineInterpolation<T> *i);

    bool canBeReused(const GeoTransformerSetting &s) const override;

    std::unique_ptr<T[]> m_copy;
    const T *m_orig;
    ctpl::thread_pool m_threadPool;
};


#endif /* LIBRARIES_RECONSTRUCTION_NEW_GEO_TRANSFORMER_H_ */
