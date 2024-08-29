/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
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

#ifndef XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_UTIL_H_
#define XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_UTIL_H_

#include <cstdint>

/**
 * Default alignment of most StarPU buffers. Required mostly by FFTW, which mentions the need for 16 byte alignment,
 * but does not specify any upper bound. So, just to be sure, we require 32 byte alignment.
 * http://fftw.org/fftw3_doc/SIMD-alignment-and-fftw_005fmalloc.html
 */
const uint32_t ALIGNMENT = 32;

inline uint32_t alignmentOf(size_t ptr) {
	uintptr_t p = (uintptr_t) ptr;
	uint32_t alignment = 1;
	while (true) {
		uint32_t nextAlignment = alignment << 1;
		if (nextAlignment == 0)
			break;
		if ((p & nextAlignment) != 0)
			return nextAlignment;
		alignment = nextAlignment;
	}
	return alignment;
}

inline uint32_t alignmentOf(void * ptr) {
	return alignmentOf((size_t) ptr);
}

template<typename T>
inline T align(T number, uint32_t alignment) {
	T off = number % alignment;
	if (off == 0) {
		return number;
	} else {
		return number + alignment - off;
	}
}

/** Wrap calls to error-returning StarPU functions with this to error on non-zero values. */
#define CHECK_STARPU(operationWithReturnCode) do {\
	int check_result = (operationWithReturnCode);\
	STARPU_CHECK_RETURN_VALUE(check_result, #operationWithReturnCode);\
} while (0)

/** Wrap calls to error-returning MPI functions with this to error on non-zero values. */
#define CHECK_MPI(operationWithReturnCode) do {\
	int check_result = (operationWithReturnCode);\
	if (STARPU_UNLIKELY(check_result != 0)) {\
        fprintf(stderr, "Unexpected value: <%d> returned for %s\n", check_result, #operationWithReturnCode);\
        fprintf(stderr, "[abort][%s:%d]\n", __FILE__, __LINE__);\
        STARPU_DUMP_BACKTRACE(); _starpu_abort();\
	}}while(0)


#endif //XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_UTIL_H_
