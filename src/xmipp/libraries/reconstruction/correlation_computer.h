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

#ifndef LIBRARIES_RECONSTRUCTION_CORRELATION_COMPUTER_H_
#define LIBRARIES_RECONSTRUCTION_CORRELATION_COMPUTER_H_

#include "amerit_computer.h"
#include "CTPL/ctpl_stl.h"
#include "data/filters.h"
#include "data/cpu.h"

template<typename T>
class CorrelationComputer : public AMeritComputer<T> {
public:
    CorrelationComputer() {
        setDefault();
    }

    ~CorrelationComputer() {
        release();
    }

    void loadReference(const T *ref) override;
    void compute(T *others) override;

private:
    bool canBeReused(const MeritSettings &s) const override;
    void initialize(bool doAllocation) override;
    void release();
    void setDefault();
    template<bool NORMALIZE>
    void computeOneToN(T *others);
    void check() override;

    const T *m_ref;
    ctpl::thread_pool m_threadPool;
};

#endif /* LIBRARIES_RECONSTRUCTION_CORRELATION_COMPUTER_H_ */
