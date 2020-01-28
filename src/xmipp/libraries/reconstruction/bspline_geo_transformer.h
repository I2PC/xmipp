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

#ifndef LIBRARIES_RECONSTRUCTION_BSPLINE_GEO_TRANSFORMER_H_
#define LIBRARIES_RECONSTRUCTION_BSPLINE_GEO_TRANSFORMER_H_

#include "reconstruction/ageo_transformer.h"
#include "CTPL/ctpl_stl.h"
#include "data/cpu.h"

template<typename T>
class BSplineTransformSettings : public GeoTransformerSettings<T> {
public:
    bool keepSrcCopy;

    void check() const override {
        GeoTransformerSettings<T>::check();
    }
};

template<typename T>
class BSplineGeoTransformer : public AGeoTransformer<BSplineTransformSettings<T>, T> {
public:

    BSplineGeoTransformer() {
        setDefault();
    }

    virtual ~BSplineGeoTransformer() {
        release();
    }

    virtual void setSrc(const T *data) override {
        m_src = data;
        this->setIsSrcSet(nullptr != data);
    }

    virtual const T *getSrc() const {
        return m_src;
    }

    virtual T *getDest() const override {
        return m_dest.get();
    }

    virtual std::unique_ptr<T[]> getDestOnCPU() const {// FIXME DS Remove (it's here because we need to compute correlation on cpu in the iterative aligner
        size_t elems = this->getSettings().dims.size();
        std::unique_ptr<T[]> p = std::unique_ptr<T[]>(new T[elems]);
        memcpy(p.get(),
            m_dest.get(),
            elems * sizeof(T));
        return p;
    }

    virtual void copySrcToDest() override;

    virtual T *interpolate(const std::vector<float> &matrices);

protected:
    virtual void initialize(bool doAllocation) override;
    virtual void release();
    virtual void setDefault();
    virtual void check() override;

    virtual bool canBeReused(const BSplineTransformSettings<T> &s) const override;
private:
    std::unique_ptr<T[]> m_dest;
    const T *m_src;
    ctpl::thread_pool m_threadPool;
};


#endif /* LIBRARIES_RECONSTRUCTION_BSPLINE_GEO_TRANSFORMER_H_ */
