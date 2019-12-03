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

#include "gpu.h"
#include <sstream>
#include <cuda_runtime.h>
#include "cuda_asserts.h"
#include <nvml.h>

GPU::~GPU() {
    if (m_isSet) {
        synch();
        auto s = (cudaStream_t*)m_stream;
        gpuErrchk(cudaStreamDestroy(*s));
        delete (cudaStream_t*)m_stream;
        m_stream = nullptr;
        m_uuid = std::string();
    }
    m_isSet = false;
}

void GPU::set() {
    // set device (for current context / thread)
    setDevice(m_device);
    if ( ! m_isSet) {
        // create stream
        m_stream = new cudaStream_t;
        gpuErrchk(cudaStreamCreate((cudaStream_t*)m_stream));
        // remember the state
        m_isSet = true;
        // get additional info
        HW::set();
    }
    peekLastError();
}

void GPU::obtainUUID() {
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

void GPU::updateMemoryInfo() {
    check();
    gpuErrchk(cudaMemGetInfo(&m_lastFreeBytes, &m_totalBytes));
}

void GPU::peekLastError() const {
    check();
    gpuErrchk(cudaPeekAtLastError());
}

void GPU::pinMemory(const void *h_mem, size_t bytes,
        unsigned int flags) {
    if (isMemoryPinned(h_mem)
            && (isMemoryPinned((char*)h_mem + bytes - 1))) {
        return;
    }
    assert(0 == cudaHostRegisterDefault); // default value should be 0
    // check that it's aligned properly to the beginning of the page
    if (0 != ((size_t)h_mem % 4096)) {
        // otherwise the cuda-memcheck and cuda-gdb tends to randomly crash (confirmed on cuda 8 - cuda 10)
        REPORT_ERROR(ERR_PARAM_INCORRECT, "Only pointer aligned to the page size can be registered");
    }
    // we remove const, but we don't change the data
    gpuErrchk(cudaHostRegister(const_cast<void*>(h_mem), bytes, flags));
}

void GPU::unpinMemory(const void *h_mem) {
    // we remove const, but we don't change the data
    auto err = cudaHostUnregister(const_cast<void*>(h_mem));
    if (cudaErrorHostMemoryNotRegistered == err) {
        cudaGetLastError(); // clear out the previous API error
    } else {
        gpuErrchk(err);
    }
}

int GPU::getDeviceCount() {
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

void GPU::synchAll() const {
    check();
    gpuErrchk(cudaDeviceSynchronize());
}

void GPU::synch() const {
    check();
    auto stream = (cudaStream_t*)m_stream;
    gpuErrchk(cudaStreamSynchronize(*stream));
}

void GPU::setDevice(int device) {
    gpuErrchk(cudaSetDevice(device));
    gpuErrchk(cudaPeekAtLastError());
}

bool GPU::isMemoryPinned(const void *h_mem) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, h_mem) == cudaErrorInvalidValue) {
        cudaGetLastError(); // clear out the previous API error
        return false;
    }
    return true;
}

bool GPU::isGpuPointer(const void *p) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, p) == cudaErrorInvalidValue) {
        cudaGetLastError(); // clear out the previous API error
        return false;
    }
    return cudaMemoryTypeDevice == attr.memoryType;
}
