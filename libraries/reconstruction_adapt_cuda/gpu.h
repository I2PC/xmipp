/*
 * gpu.h
 *
 *  Created on: Dec 6, 2018
 *      Author: david
 */

#ifndef LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_H_
#define LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_H_

#include <string>
#include "reconstruction_cuda/cuda_xmipp_utils.h"

class GPU {
public:
    GPU(size_t device) :
            m_device(device), m_UUID(getUUID(device)), m_lastFreeMem(
                    getFreeMem(device)) {
    }
    ;

    size_t device() const { return m_device; };
    std::string UUID() const { return m_UUID; };
    size_t lastFreeMem() const { return m_lastFreeMem; };

private:
    const size_t m_device;
    const std::string m_UUID;
    const size_t m_lastFreeMem;
};



#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_H_ */
