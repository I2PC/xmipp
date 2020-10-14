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

#ifndef LIBRARIES_DATA_HW_H_
#define LIBRARIES_DATA_HW_H_

#include <string>
#include <cstddef>


class HW {
public:
    explicit HW(unsigned parallelUnits) :
        m_parallUnits(parallelUnits),
        m_lastFreeBytes(0),
        m_totalBytes(0) {}

    virtual ~HW(){};

    inline unsigned noOfParallUnits() const {
        return m_parallUnits;
    }

    virtual void synch() const = 0;
    virtual void synchAll() const = 0;
    virtual void set() {
        updateMemoryInfo();
        obtainUUID();
    }

    virtual void updateMemoryInfo() = 0;

    virtual inline size_t lastFreeBytes() const {
        return m_lastFreeBytes;
    }

    virtual inline size_t totalBytes() const {
        return m_totalBytes;
    }

    virtual inline size_t lastUsedBytes() const {
        return m_totalBytes - m_lastFreeBytes;
    }

    virtual std::string getUUID() const {
        return m_uuid;
    }

    virtual void lockMemory(const void *h_mem, size_t bytes) = 0;

    virtual void unlockMemory(const void *h_mem) = 0;

    virtual bool isMemoryLocked(const void *h_mem) = 0;
protected:
    unsigned m_parallUnits;
    size_t m_totalBytes;
    size_t m_lastFreeBytes;
    std::string m_uuid;

    virtual void obtainUUID() = 0;
};

#endif /* LIBRARIES_DATA_HW_H_ */
