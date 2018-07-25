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

#endif //* CUDA_GPU_MUTLIDIM_ARRAY_CU */
