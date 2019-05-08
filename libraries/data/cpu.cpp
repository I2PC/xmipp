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

#include "cpu.h"
#include <sstream>

void CPU::native_cpuid(unsigned int *eax, unsigned int *ebx,
        unsigned int *ecx, unsigned int *edx)
{
    /* ecx is often an input as well as an output. */
    asm volatile("cpuid"
    : "=a" (*eax),
      "=b" (*ebx),
      "=c" (*ecx),
      "=d" (*edx)
    : "0" (*eax), "2" (*ecx));
}

void CPU::updateMemoryInfo() {
    size_t pages = sysconf(_SC_PHYS_PAGES);
    size_t page_size = sysconf(_SC_PAGE_SIZE);
    size_t total = pages * page_size;
    m_totalBytes = total;
    m_lastFreeBytes = total; // assume we own all memory
}

void CPU::obtainUUID() {
    // https://stackoverflow.com/a/6491964/5484355
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;

    std::stringstream ss;

    eax = 1; /* processor info and feature bits */
    native_cpuid(&eax, &ebx, &ecx, &edx);

    ss << (eax & 0xF); ss << " "; // stepping
    ss << ((eax >> 4) & 0xF); ss << " "; // model
    ss << ((eax >> 8) & 0xF); ss << " "; // family
    ss << ((eax >> 12) & 0x3); ss << " "; // processor type
    ss << ((eax >> 16) & 0xF); ss << " "; // extended model
    ss << ((eax >> 20) & 0xFF); ss << " "; // extended family

    m_uuid = ss.str();
}
