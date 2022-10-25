/***************************************************************************
 *
 * Authors:    Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#include "align_spectral_gpu.h"

#include <iostream>

namespace Alignment {

void ProgAlignSpectralGPU::defineParams() {
    ProgAlignSpectral::defineParams();
    this->addParamsLine("  [--dev <...>]                    : space-separated list of GPU device(s) to use. Single, 0th GPU used by default");
}

void ProgAlignSpectralGPU::show() const {
    ProgAlignSpectral::show();
    std::cout <<  "Device(s)                   :";
    for (auto d : m_devices) {
        std::cout << " " << d;
    }
    std::cout << std::endl;
}

void ProgAlignSpectralGPU::readParams() {
    ProgAlignSpectral::readParams();
    // read GPU
    StringVector devs;
    this->getListParam("--dev", devs);
    if (devs.empty()) {
        devs.emplace_back("0"); // by default, use one GPU, 0th
    }
    auto noOfAvailableDevices = GPU::getDeviceCount();
    for (auto &a : devs) {
        int d = std::stoi(a);
        if (0 > d) {
            REPORT_ERROR(ERR_ARG_INCORRECT, "Invalid GPU device '" + a + "' (must be non-negative number)");
        }
        // FIXME DS uncomment once we are decided if we want to run multiple executions on the same GPU
//        if (std::find(m_devices.begin(), m_devices.end(), d) != m_devices.end()) {
//            REPORT_ERROR(ERR_ARG_INCORRECT, "Invalid GPU device '" + a + "' (repeated index)");
//        }
        if (d >= noOfAvailableDevices) {
            REPORT_ERROR(ERR_ARG_INCORRECT, "Invalid GPU device '" + a + "' (index higher than number of available devices)");
        }
        m_devices.emplace_back(d);
    }
}

} // namespace Alignment