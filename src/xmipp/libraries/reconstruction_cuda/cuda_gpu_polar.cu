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

#include "cuda_basic_math.h"
#include "core/xmipp_macros.h"

template<typename T, bool FULL_CIRCLE>
__global__
void polarFromCartesian(const T *__restrict__ in, int inX, int inY,
        T *__restrict__ out, int samples, int rings, int signals, int posOfFirstRing)
{
    // input is 2D signal - each row is a ring of samples
    // map thread to pixel in the polar coordinate
    int s = (blockIdx.x*blockDim.x + threadIdx.x) % samples; // sample position == column
    int n = (blockIdx.x*blockDim.x + threadIdx.x) / samples; // signal index
    int r = blockIdx.y*blockDim.y + threadIdx.y; // ring position == row

    if ((n >= signals)
        || (s >= samples)
        || (r >= rings)) return;

    T piConst = FULL_CIRCLE ? 2 * M_PI : M_PI;
    T dphi = piConst / (T)samples;
    T phi = s * dphi;

    // transform current polar position to cartesian
    // we don't wrap, as we expect that the biggest ring has some edge around, so we cannot read
    // data out of domain
    T cartX = sin(phi) * (T)(r + posOfFirstRing);
    T cartY = cos(phi) * (T)(r + posOfFirstRing);

//        printf("thread %d [%d, %d, %d]:"
//                "\tcart [%f, %f], "
//                "\tin: [%d, %d], "
//                "\tout: [%d, %d, %d],"
//                "\tmin max: [%d, %d] [%d, %d]\n",
//                (n * samples * rings) + (r * samples) + s,
//                s, r+firstRing, n,
//                cartX, cartY,
//                inX, inY,
//                samples, rings, n,
//                FIRST_XMIPP_INDEX(inX), LAST_XMIPP_INDEX(inX), FIRST_XMIPP_INDEX(inY), LAST_XMIPP_INDEX(inY));
    int offset = (n * samples * rings) + (r * samples) + s;
    T val = biLerp(in + (n * inX * inY),
            inX, inY,
            cartX - FIRST_XMIPP_INDEX(inX), cartY - FIRST_XMIPP_INDEX(inY));
//    printf("sample: %d [%d %d %d] reading from [%f %f] value %f (stored at %d)\n",
//            (n * samples * rings) + (r * samples) + s,
//            s, r+firstRing, n,
//            cartX - FIRST_XMIPP_INDEX(inX), cartY - FIRST_XMIPP_INDEX(inY),
//            val, offset);
    out[offset] = val;
}
