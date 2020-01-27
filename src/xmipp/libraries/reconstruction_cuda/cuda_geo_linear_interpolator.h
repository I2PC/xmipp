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

//#ifndef LIBRARIES_RECONSTRUCTION_CUDA_CUDA_GEO_LINEAR_INTERPOLATOR_H_
//#define LIBRARIES_RECONSTRUCTION_CUDA_CUDA_GEO_LINEAR_INTERPOLATOR_H_

#include "gpu.h"
//#include "reconstruction/ageo_linear_interpolator.h"
//
////FIXME DS rework properly
//
//template<typename T>
//class CudaGeoLinearTransformer : public AGeoLinearTransformer<T> {
//public:
//    CudaGeoLinearTransformer() :
//        m_dims(0) {
//        setDefault();
//    }
//
//    void init(const Dimensions &d) override {
//        bool mustInit = (m_dims.size() < d.size());
//        if (mustInit) {
//            release();
//        }
//        m_dims = d;
//        if (mustInit) {
//            init();
//        }
//    }
//
//    virtual ~CudaGeoLinearTransformer() {
//        release();
//    }
//
//    void createCopyOnGPU(const T *h_data) override;
//
//    T *getCopy() override {
//        return m_d_dest;
//    }
//
//    T *interpolate(const std::vector<float> &matrices) override; // each 3x3 values are a single matrix
//private:
//    T *m_d_src;
//    T *m_d_dest;
//    float *m_d_matrices;
//
//    Dimensions m_dims;
//
//    void setDefault();
//    void release();
//    void init();
//};
//
//
//#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_GEO_LINEAR_INTERPOLATOR_H_ */
