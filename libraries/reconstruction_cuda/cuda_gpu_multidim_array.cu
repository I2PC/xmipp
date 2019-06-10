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

#ifndef CUDA_GPU_MUTLIDIM_ARRAY_CU
#define CUDA_GPU_MUTLIDIM_ARRAY_CU

#define LOOKUP_TABLE_LEN 6

template<typename T>
__device__
T interpolatedElementBSpline2D_Degree3(T x, T y, int xdim, int ydim, T* data)
{
    bool firstTime=true; // Inner loop first time execution flag.
    T *ref;

    int l1 = (int)ceil(x - 2);
    int l2 = l1 + 3;
    int m1 = (int)ceil(y - 2);
    int m2 = m1 + 3;

    T columns = 0.0;
    T aux;

    int equivalent_l_Array[LOOKUP_TABLE_LEN];
    T aux_Array[LOOKUP_TABLE_LEN];

    for (int m = m1; m <= m2; m++)
    {
        int equivalent_m=m;
        if      (m<0)
            equivalent_m=-m-1;
        else if (m>=ydim)
            equivalent_m=2*ydim-m-1;
        T rows = 0.0;
        int	index=0;
        ref = data + (equivalent_m*xdim);
        for (int l = l1; l <= l2; l++)
        {
            int equivalent_l;
            // Check if it is first time executing inner loop.
            if (firstTime)
            {
                T xminusl = x - (T) l;
                equivalent_l=l;
                if (l<0)
                {
                    equivalent_l=-l-1;
                }
                else if (l>=xdim)
                {
                    equivalent_l=2*xdim-l-1;
                }

                equivalent_l_Array[index] = equivalent_l;
                aux = bspline03(xminusl);
                aux_Array[index] = aux;
                index++;
            }
            else
            {
                equivalent_l = equivalent_l_Array[index];
                aux = aux_Array[index];
                index++;
            }

            T Coeff = ref[equivalent_l];
            rows += Coeff * aux;
        }

        // Set first time inner flag is executed to false.
        firstTime = false;

        T yminusm = y - (T) m;
        aux = bspline03(yminusm);
        columns += rows * aux;
    }

    return columns;
}

#undef LOOKUP_TABLE_LEN

#define LOOKUP_TABLE_LEN 6

/*
* This version doesn't subtract shift from position in float and doesn't lose
* precision with higher values of `x` and `y`
*/

template<typename T>
__device__
T interpolatedElementBSpline2D_Degree3New(int x, int y, T x_shift, T y_shift, int xdim, int ydim, T* data)
{
    bool firstTime=true; // Inner loop first time execution flag.
    T *ref;

    int l1 = x + (int)ceil(-x_shift) - 2;
    int l2 = l1 + 3;
    int m1 = y + (int)ceil(-y_shift) - 2;
    int m2 = m1 + 3;

    T columns = 0.0;
    T aux;

    int equivalent_l_Array[LOOKUP_TABLE_LEN];
    T aux_Array[LOOKUP_TABLE_LEN];

    for (int m = m1; m <= m2; m++)
    {
        int equivalent_m=m;
        if      (m<0)
            equivalent_m=-m-1;
        else if (m>=ydim)
            equivalent_m=2*ydim-m-1;
        T rows = 0.0;
        int index=0;
        ref = data + (equivalent_m*xdim);
        for (int l = l1; l <= l2; l++)
        {
            int equivalent_l;
            // Check if it is first time executing inner loop.
            if (firstTime)
            {
                T xminusl = 2 - (l - l1) - x_shift - (int)ceil(-x_shift);
                equivalent_l=l;
                if (l<0)
                {
                    equivalent_l=-l-1;
                }
                else if (l>=xdim)
                {
                    equivalent_l=2*xdim-l-1;
                }

                equivalent_l_Array[index] = equivalent_l;
                aux = bspline03(xminusl);
                aux_Array[index] = aux;
                index++;
            }
            else
            {
                equivalent_l = equivalent_l_Array[index];
                aux = aux_Array[index];
                index++;
            }

            T Coeff = ref[equivalent_l];
            rows += Coeff * aux;
        }

        // Set first time inner flag is executed to false.
        firstTime = false;

        T yminusm = 2 - (m - m1) - y_shift - (int)ceil(-y_shift);
        aux = bspline03(yminusm);
        columns += rows * aux;
    }

    return columns;
}

#undef LOOKUP_TABLE_LEN

#define LOOKUP_TABLE_LEN 4

template< typename T >
__device__
T interpolatedElementBSpline2D_Degree3MorePixelsInner(int x, int y, T x_shift, T y_shift, int xdim, int ydim, const T* __restrict__ data)
{
    const T* __restrict__ ref;

    int x_shift_ceiled = static_cast<int>(ceil(-x_shift));
    int y_shift_ceiled = static_cast<int>(ceil(-y_shift));
    T x_diff = 2 - (x_shift + x_shift_ceiled);
    T y_diff = 2 - (y_shift + y_shift_ceiled);

    int l1 = x + x_shift_ceiled - 2;
    int m1 = y + y_shift_ceiled - 2;

    T columns = 0.0;
    T aux_Array[LOOKUP_TABLE_LEN];

    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        aux_Array[i] = bspline03( x_diff - i );
    }

    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        const int m = m1 + i;

        T rows = 0.0;
        ref = data + (m*xdim);
        #pragma unroll
        for ( int j = 0; j < 4; ++j ) {
            const int l = l1 + j;
            rows += ref[l] * aux_Array[j];
        }

        columns += rows * bspline03( y_diff - i );
    }

    return columns;
}

template< typename T >
__device__
T interpolatedElementBSpline2D_Degree3MorePixelsEdge(int x, int y, T x_shift, T y_shift, int xdim, int ydim, const T* __restrict__ data)
{
    const T* __restrict__ ref;

    int x_shift_ceiled = static_cast<int>(ceil(-x_shift));
    int y_shift_ceiled = static_cast<int>(ceil(-y_shift));
    T x_diff = 2 - (x_shift + x_shift_ceiled);
    T y_diff = 2 - (y_shift + y_shift_ceiled);

    int l1 = x + x_shift_ceiled - 2;
    int m1 = y + y_shift_ceiled - 2;

    T columns = 0.0;

    int equivalent_l_Array[LOOKUP_TABLE_LEN];
    T aux_Array[LOOKUP_TABLE_LEN];

    // precompute values that would have to be computed in the inner for loop
    // equivalent_l wraps around the edge if `l` is outside of the image
    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        const int l = l1 + i;
        aux_Array[i] = bspline03( x_diff - i );
        int equivalent_l = l;
        if ( l < 0 ) {
            equivalent_l = -l - 1;
        } else if ( l >= xdim ) {
            equivalent_l = (2 * xdim) - l - 1;
        }
        equivalent_l_Array[i] = equivalent_l;
    }

    // interpolates the value of (x, y) coordinate using the 4x4 square around the pixel
    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        const int m = m1 + i;

        int equivalent_m = m;
        if ( m < 0 ) {
            equivalent_m = -m - 1;
        } else if ( m >= ydim ) {
            equivalent_m = (2 * ydim) - m - 1;
        }

        T rows = 0.0;
        ref = data + (equivalent_m * xdim);
        #pragma unroll
        for ( int j = 0; j < 4; ++j ) {
            int equivalent_l = equivalent_l_Array[j];
            rows += ref[equivalent_l] * aux_Array[j];
        }

        columns += rows * bspline03( y_diff - i );
    }

    return columns;
}

#undef LOOKUP_TABLE_LEN

#endif //* CUDA_GPU_MUTLIDIM_ARRAY_CU */
