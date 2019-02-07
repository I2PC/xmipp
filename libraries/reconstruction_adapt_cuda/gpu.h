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

#ifndef LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_H_
#define LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_H_

#include <string>
#include "reconstruction_cuda/cuda_xmipp_utils.h"

class GPU {
public:
    explicit GPU(size_t device) :
        m_device(device), m_UUID(getUUID(device)),
        m_lastFreeMem(getFreeMem(device)) {};

    size_t device() const { return m_device; };
    std::string UUID() const { return m_UUID; };
    size_t lastFreeMem() const { return m_lastFreeMem; };

    /**
     * Method checks currently available free GPU memory
     * Obtained value is stored in this instance
     */
    size_t checkFreeMem() {
        m_lastFreeMem = getFreeMem(m_device);
        return m_lastFreeMem;
    }

private:
    const size_t m_device;
    const std::string m_UUID;
    size_t m_lastFreeMem;
};

#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_H_ */
