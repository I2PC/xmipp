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

#include "gpu_new.h"
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <nvml.h>

void GPUNew::check() const {
    if ( ! m_isSet) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "You have to set() this GPU before using it");
    }
}

GPUNew::~GPUNew() {
    if (m_isSet) {
        auto s = (cudaStream_t*)m_stream;
        gpuErrchk(cudaStreamDestroy(*s));
        delete (cudaStream_t*)m_stream;
        m_stream = nullptr;
    }
}

void GPUNew::set() {
    if (m_isSet) {
        return;
    }
    // set device
    gpuErrchk(cudaSetDevice(m_device));
    gpuErrchk(cudaPeekAtLastError());
    // create stream
    m_stream = new cudaStream_t;
    gpuErrchk(cudaStreamCreate((cudaStream_t*)m_stream));
    // remember the state
    m_isSet = true;
    // get memory info
    updateMemoryInfo();

    peekLastError();
}

void GPUNew::obtainUUID() {
    std::stringstream ss;
    nvmlDevice_t device;
    // https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g84dca2d06974131ccec1651428596191
    if (NVML_SUCCESS == nvmlInit()) {
        if (NVML_SUCCESS == nvmlDeviceGetHandleByIndex(m_device, &device)) {
            char uuid[80];
            if (NVML_SUCCESS == nvmlDeviceGetUUID(device, uuid, 80)) {
                ss <<  uuid;
            }
        }
    } else {
        ss << m_device;
    }
    m_uuid = ss.str();
}

void GPUNew::updateMemoryInfo() {
    check();
    gpuErrchk(cudaMemGetInfo(&m_lastFreeBytes, &m_totalBytes));
}

void GPUNew::peekLastError() const {
    check();
    gpuErrchk(cudaPeekAtLastError());
}

void GPUNew::pinMemory(void *h_mem, size_t bytes,
        unsigned int flags) const {
    check();
    assert(0 == cudaHostRegisterDefault); // default value should be 0
    gpuErrchk(cudaHostRegister(h_mem, bytes, flags));
}

void GPUNew::unpinMemory(void *h_mem) const {
    check();
    gpuErrchk(cudaHostUnregister(h_mem));
}

int GPUNew::getDeviceCount() {
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

void GPUNew::synch() const {
    check();
    gpuErrchk(cudaDeviceSynchronize());
}

void GPUNew::synchStream() const {
    check();
    auto stream = (cudaStream_t*)m_stream;
    gpuErrchk(cudaStreamSynchronize(*stream));
}
