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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_CUDA_BSPLINE_GEO_TRANSFORMER_H_
#define LIBRARIES_RECONSTRUCTION_CUDA_CUDA_BSPLINE_GEO_TRANSFORMER_H_

#include "gpu.h"
#include "reconstruction/bspline_geo_transformer.h"

template<typename T>
class CudaBSplineGeoTransformer : public BSplineGeoTransformer<T> {
public:
    CudaBSplineGeoTransformer() {
        setDefault();
    }

    void setSrc(const T *data) override;

    const T *getSrc() const override {
        return m_d_src;
    }

    T *getDest() const override {
        return m_d_dest;
    }

    void copySrcToDest() override;

    T *interpolate(const std::vector<float> &matrices) override; // each 3x3 values are a single matrix

    void sum(T *dest, size_t firstN) override;
private:
    T *m_d_src;
    T *m_d_dest;
    GPU *m_stream;
    float *m_d_matrices;

    void setDefault() override;
    void release() override;
    void initialize(bool doAllocation) override;
    void allocate();
    void check() override;
};


#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_BSPLINE_GEO_TRANSFORMER_H_ */
