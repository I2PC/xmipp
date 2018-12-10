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
            device(device), UUID(getUUID(device)), lastFreeMem(
                    getFreeMem(device)) {
    }
    ;

    const size_t device;
    const std::string UUID;
    size_t lastFreeMem;
};



#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_H_ */
