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
//#include "core/xmipp_macros.h" // for Nearest Neighbor interpolation
#include "cuda_compatibility.h"

template<typename T, bool FULL_CIRCLE>
__global__
void polarFromCartesian(const T *__restrict__ in, int inX, int inY,
        T *__restrict__ out, int samples, int rings, int signals, int posOfFirstRing)
{
    // input is 2D signal - each row is a ring of samples
    // map thread to sample in the polar coordinate
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (samples * signals)) return;

    int s = idx % samples; // sample position == column
    int n = idx / samples; // signal index

    T piConst = FULL_CIRCLE ? 2 * M_PI : M_PI;
    T dphi = piConst / (T)samples;
    T phi = s * dphi;

    T sinPhi = sin(phi);
    T cosPhi = cos(phi);

    T maxR = rings + posOfFirstRing - 1;
    // transform current polar position to cartesian
    // shift origin to center of the input image
    for (int r = 0; r < rings; ++r) {
        T cartX = sinPhi * (T)(r + posOfFirstRing) + (int)(inX / (T)2);
        T cartY = cosPhi * (T)(r + posOfFirstRing) + (int)(inY / (T)2);

        int offset = (n * samples * rings) + (r * samples) + s;
        // Bilinear interpolation
        // we don't wrap, as we expect that the biggest ring has some edge around, so we cannot read
        // data out of domain
        T val = biLerp(in + (n * inX * inY),
                inX, inY,
                cartX , cartY);
//        printf("sample: [%d %d+%d=%d %d] reading from [%f %f] value %f (stored at %d)\n",
//                s, r,firstRing, r + firstRing, n,
//                cartX, cartY,
//                val, offset);

        // Nearest neighbour interpolation
//        int cartXRound = (int)(cartX + (T)0.5) - FIRST_XMIPP_INDEX(inX);
//        int cartYRound = (int)(cartY + (T)0.5) - FIRST_XMIPP_INDEX(inY);
//        T val = in[(n * inX * inY) + (cartYRound * inX) + cartXRound];
//        printf("sample: [%d %d+%d=%d %d] reading from [%f %f] value %f (stored at %d)\n",
//                s, r,firstRing, r + firstRing, n,
//                cartXRound, cartYRound,
//                val, offset);
        T weight = (r + posOfFirstRing) / maxR;

        out[offset] = val * weight;
    }
}

template<typename T, bool FULL_CIRCLE>
__global__
void computeSumSumSqr(const T * __restrict__ in,
    int samples, int rings, int signals,
    T * __restrict__ outSum, // must be zero-initialized
    T * __restrict__ outSumSqr, // must be zero-initialized
    int posOfFirstRing) {

    // input is 2D signal - each row is a ring of samples
    // map thread to sample in the polar coordinate
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int maxThread = samples * signals;
    if (idx >= maxThread) return;

    int s = idx % samples; // sample position == column
    int n = idx / samples; // signal index

    bool isSameSignalInWarp = false;
    int warpId = idx / warpSize; // warp has 32 threads
    int idOfFirstThreadInWarp = warpId * warpSize;
//#if (CUDART_VERSION >= 9000) // FIXME DS correct and test
//    const unsigned mask = 0xffffffff;
//    if (mask = (__activemask & mask)) {
//        __match_all_sync(mask, n, &isSameSignalInWarp);
//    }
//#else
    int idOfLastThreadInWarp = idOfFirstThreadInWarp + warpSize - 1;
    int nForFirst = idOfFirstThreadInWarp / samples;
    int nForLast = idOfLastThreadInWarp / samples;
    if ((idOfLastThreadInWarp < maxThread) && (nForFirst == nForLast)) {
        isSameSignalInWarp = true;
    }
//#endif

    const T piConst = FULL_CIRCLE ? (2 * M_PI) : M_PI;

    // compute statistics
    T sum = 0;
    T sum2 = 0;
    // shift origin to center of the input image
    for (int r = 0; r < rings; ++r) {
        T w = (piConst * (r + posOfFirstRing)) / (T)samples;
        int offset = (n * samples * rings) + (r * samples) + s;
        T val = in[offset];
//        printf("val %f w %f ring %d sample %d sum %f sum2 %f (thread %d)\n",
//                val, w, r, s, w * val, w * val * val, idx);
        sum += w * val;
        sum2 += w * val * val;
    }
    if (isSameSignalInWarp) {
        // intrawarp sum
//#if (CUDART_VERSION >= 9000) // FIXME DS correct and test
//        __syncwarp();
//        for (int offset = 16; offset > 0; offset /= 2) {
//            sum += __shfl_down_sync(mask, sum, offset);
//            sum2 += __shfl_down_sync(mask, sum2, offset);
//        }
//#else
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
            sum2 += __shfl_down(sum2, offset);
        }
//#endif
        if (idx != idOfFirstThreadInWarp) {
            // only first thread in the warp will write to the output
            return;
        }
    }
    // update the result
    atomicAdd(&outSum[n], sum);
    atomicAdd(&outSumSqr[n], sum2);
}

template<typename T>
__global__
void normalize(T * __restrict__ inOut,
    int samples, int rings, int signals,
    T norm,
    const T * __restrict__ sums,
    const T * __restrict__ sumsSqr) {
    // input is 2D signal - each row is a ring of samples
    // map thread to sample in the polar coordinate
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int maxThread = samples * signals;
    if (idx >= maxThread) return;

    int s = idx % samples; // sample position == column
    int n = idx / samples; // signal index

    T avg = sums[n] / norm;
    T sumSqrNorm = sumsSqr[n] / norm;
    T stddev = sqrt(abs(sumSqrNorm - (avg * avg)));
    T istddev = 1 / stddev;
    for (int r = 0; r < rings; ++r) {
        int offset = (n * samples * rings) + (r * samples) + s;
        T val = inOut[offset];
        inOut[offset] = (val - avg) * istddev;
    }
}
