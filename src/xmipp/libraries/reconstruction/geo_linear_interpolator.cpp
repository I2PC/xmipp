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

#include "geo_linear_interpolator.h"

template<typename T>
T *GeoLinearTransformer<T>::interpolate(const std::vector<float> &matrices) {
    const size_t n = dims.n();
    const size_t z = dims.z();
    const size_t y = dims.y();
    const size_t x = dims.x();

    auto futures = std::vector<std::future<void>>();

    auto workload = [&](int id, size_t signalId){
        size_t offset = signalId * dims.sizeSingle();
        auto in = MultidimArray<T>(1, z, y, x, const_cast<T*>(m_src + offset)); // removing const, but data should not be changed
        auto out = MultidimArray<T>(1, z, y, x, m_dest + offset);
        in.setXmippOrigin();
        out.setXmippOrigin();
        // compensate the movement
        Matrix2D<double> m;
        m.initIdentity(3);
        const float *f = matrices.data() + (9 * signalId);
        for (int i = 0; i < 9; ++i) {
            m.mdata[i] = f[i];
        }
        applyGeometry(LINEAR, out, in, m, false, DONT_WRAP);
    };

    for (size_t i = 0; i < n; ++i) {
        futures.emplace_back(m_threadPool.push(workload, i));
    }
    for (auto &f : futures) {
        f.get();
    }
}

// explicit instantiation
template class GeoLinearTransformer<float>;
template class GeoLinearTransformer<double>;
