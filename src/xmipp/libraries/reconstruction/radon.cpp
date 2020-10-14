/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#include "radon.h"

#include <core/geometry.h>

void Radon_Transform(const MultidimArray<double> &vol, double rot, double tilt,
		MultidimArray<double> &RT)
{
    MultidimArray<double> Rvol;

    // Align the Radon direction with the Z axis
    if (rot != 0 || tilt != 0) Euler_rotate(vol, rot, tilt, 0.0F, Rvol);
    else Rvol = vol;

    // Project onto one line
    RT.initZeros(Rvol.zdim);
    STARTINGX(RT) = STARTINGZ(Rvol);

    for (int k = STARTINGZ(Rvol); k <= FINISHINGZ(Rvol); k++)
        for (int i = STARTINGY(Rvol); i <= FINISHINGY(Rvol); i++)
            for (int j = STARTINGX(Rvol); j <= FINISHINGX(Rvol); j++)
                A1D_ELEM(RT, k) += A3D_ELEM(Rvol, k, i, j);
}

// The number of voxels in each layer is a double to make easier some
// computations outside
void Local_Radon_Transform(const MultidimArray<double> &vol, double rot, double tilt,
                           int label, const MultidimArray<double> &vol_label,
                           MultidimArray<double> &RT,
                           MultidimArray<double> &RT_n)
{
	MultidimArray<double>   Rvol;
	MultidimArray<double>   Lvol;

    // Align the Radon direction with the Z axis
    if (rot != 0 || tilt != 0)
    {
        Euler_rotate(vol, rot, tilt, 0.0F, Rvol);
        Euler_rotate(vol_label, rot, tilt, 0.0F, Lvol);
    }
    else
    {
        Rvol = vol;
        Lvol = vol_label;
    }

    // Project onto one line
    RT.initZeros(Rvol.zdim);
    STARTINGX(RT) = STARTINGZ(Rvol);
    RT_n = RT;

    for (int k = STARTINGZ(Rvol); k <= FINISHINGZ(Rvol); k++)
        for (int i = STARTINGY(Rvol); i <= FINISHINGY(Rvol); i++)
            for (int j = STARTINGX(Rvol); j <= FINISHINGX(Rvol); j++)
                if (A3D_ELEM(Lvol, k, i, j) == label)
                {
                    A1D_ELEM(RT, k) += A3D_ELEM(Rvol, k, i, j);
                    A1D_ELEM(RT_n, k)++;
                }
}

/* Radon Transform of an image --------------------------------------------- */
void Radon_Transform(const MultidimArray<double> &I, double rot_step,
		MultidimArray<double> &RT)
{
	MultidimArray<double> rot_I;
    RT.initZeros(CEIL(360.0 / rot_step), XSIZE(I));
    STARTINGX(RT) = STARTINGX(I);
    int l = 0;
    for (double rot = 0; rot < 360; rot += rot_step, l++)
    {
        // Rotate image
    	rotate(LINEAR, rot_I, I, rot);
        // Sum by columns
        FOR_ALL_ELEMENTS_IN_ARRAY2D(rot_I) RT(l, j) += rot_I(i, j);
    }
}
