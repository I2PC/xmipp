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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_CORRELATION_COMPUTER_H_
#define LIBRARIES_RECONSTRUCTION_CUDA_CORRELATION_COMPUTER_H_

#include "reconstruction/amerit_computer.h"
#include "gpu.h"
#include "core/utils/memory_utils.h"

template<typename T>
class CudaCorrelationComputer : public AMeritComputer<T> {
public:
    CudaCorrelationComputer() {
        setDefault();
    }

    ~CudaCorrelationComputer() {
        release();
    }

    void loadReference(const T *ref) override;
    void compute(T *others) override;

private:
    struct ResRaw { // non-normalize result
        T corr;
        T sum;
        T sumSqr;
    };

    struct ResRef {
        T sum;
        T sumSqr;
    };

    struct Stat {
        T avg;
        T stddev;
    };

    struct ResNorm { // normalized result
        T corr;
    };

    bool canBeReused(const MeritSettings &s) const override;
    void initialize(bool doAllocation) override;
    void release();
    void setDefault();
    void check() override;
    void allocate();

    void computeCorrStatOneToNNormalize();
    void computeAvgStddevForRef();
    template<bool NORMALIZE>
    void computeOneToN();
    template<bool NORMALIZE>
    void storeResultOneToN();

    template<typename U>
    Stat computeStat(U r, size_t norm);

    // GPU memory
    T *m_d_ref;
    T *m_d_others;
    T *m_d_corrRes;
    // CPU memory
    ResRef *m_h_ref_corrRes; // used only when normalization is requested
    void *m_h_corrRes; // actual type depends on the merit type
    // others
    GPU *m_stream;

};

#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CORRELATION_COMPUTER_H_ */
