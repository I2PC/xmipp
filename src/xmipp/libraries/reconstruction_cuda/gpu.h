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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_GPU_H_
#define LIBRARIES_RECONSTRUCTION_CUDA_GPU_H_

#include <assert.h>
#include "data/hw.h"
#include "core/xmipp_error.h"

class GPU : public HW {
public:
    explicit GPU(int device = 0, int stream = 0):
        HW(1),
        m_device(device),
        m_streamId(stream), m_stream(nullptr),
        m_isSet(false) {};

    ~GPU();

    inline int device() const {
        return m_device;
    }

    inline void* stream() const {
        check();
        return m_stream;
    }

    inline int streamId() const {
        return m_streamId;
    }

    inline size_t lastFreeBytes() const {
        check();
        return HW::lastFreeBytes();
    }

    inline size_t totalBytes() const {
        check();
        return HW::totalBytes();
    }

    inline size_t lastUsedBytes() const {
        check();
        return HW::lastUsedBytes();
    }

    void updateMemoryInfo();

    void peekLastError() const;

    static void pinMemory(const void *h_mem, size_t bytes, unsigned int flags=0); // must not be nullptr

    static void unpinMemory(const void *h_mem); // must not be nullptr

    static bool isMemoryPinned(const void *h_mem);

    void set();

    void synchAll() const;

    void synch() const;

    inline std::string getUUID() const {
        check();
        return HW::getUUID();
    }

    inline bool isSet() const {
        return m_isSet;
    }

    // FIXME DS do not use, it's for backward compatibility only
    static void setDevice(int device);

    static int getDeviceCount();

    void lockMemory(const void *h_mem, size_t bytes) override {
        GPU::pinMemory(h_mem, bytes, 0);
    }

    void unlockMemory(const void *h_mem) override {
        GPU::unpinMemory(h_mem);
    }

    bool isMemoryLocked(const void *h_mem) override {
        return GPU::isMemoryPinned(h_mem);
    }

    bool isGpuPointer(const void *);

private:
    int m_device;
    int m_streamId;
    void* m_stream;
    bool m_isSet;

    inline void check() const {
        if ( ! m_isSet) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "You have to set() this GPU before using it");
        }
    }

    void obtainUUID();
};

#endif /* LIBRARIES_RECONSTRUCTION_CUDA_GPU_H_ */
