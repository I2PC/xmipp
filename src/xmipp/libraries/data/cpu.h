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

#ifndef LIBRARIES_DATA_CPU_H_
#define LIBRARIES_DATA_CPU_H_

#include <thread>
#include <unistd.h>
#include "hw.h"
#include "core/xmipp_error.h"

class CPU : public HW {
public:
    CPU(unsigned cores=1) : HW(cores) {}

    static unsigned findCores() {
        return std::max(std::thread::hardware_concurrency(), 1u);
    }

    void synch() const {}; // nothing to do
    void synchAll() const {}; // nothing to do

    void updateMemoryInfo();

    void lockMemory(const void *h_mem, size_t bytes) override {
        // FIXME DS implement
    }

    void unlockMemory(const void *h_mem) override {
        // FIXME DS implement
    }

    bool isMemoryLocked(const void *h_mem) override {
        // FIXME DS implement
        return false;
    }

protected:
    void obtainUUID();

private:
    void native_cpuid(unsigned int *eax, unsigned int *ebx,
            unsigned int *ecx, unsigned int *edx);
};


#endif /* LIBRARIES_DATA_CPU_H_ */
