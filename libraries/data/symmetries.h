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
/* ------------------------------------------------------------------------- */
/* SYMMETRIES                                                                */
/* ------------------------------------------------------------------------- */
#ifndef _SYMMETRIES_HH
#define _SYMMETRIES_HH

#include <core/matrix1d.h>
#include <core/matrix2d.h>
#include <core/xmipp_funcs.h>
#include <core/args.h>
#include <core/symmetries.h>
#include <data/grids.h>

/** Applies to the crystal vectors de n-th symmetry  matrix, It also
   initializes the shift vector. The crystal vectors and the basis must be
   the same  except for a constant!!
   A note: Please realize that we are not repeating code here.
   The class SymList deals with symmetries when expressed in
   Cartesian space, that is the basis is orthonormal. Here
   we describe symmetries in the crystallographic way
   that is, the basis and the crystal vectors are the same.
   For same symmetries both representations are almost the same
   but in general they are rather different.
 */
void symmetrizeCrystalVectors(Matrix1D<double> &aint,
                                Matrix1D<double> &bint,
                                Matrix1D<double> &shift,
                                int space_group,
                                int sym_no,
                                const Matrix1D<double> &eprm_aint,
                                const Matrix1D<double> &eprm_bint);

/** Symmetrizes a crystal volume.
 */
void symmetrizeCrystalVolume(GridVolume &vol,
                               const Matrix1D<double> &eprm_aint,
                               const Matrix1D<double> &eprm_bint,
                               int eprm_space_group, const MultidimArray<int> &mask,
                               int grid_type);

/** Symmetrizes a simple grid with P2_122  symmetry
*/
void symmetry_P2_122(Image<double> &vol, const SimpleGrid &grid,
                     const Matrix1D<double> &eprm_aint,
                     const Matrix1D<double> &eprm_bint,
                     const MultidimArray<int> &mask, int volume_no,
                     int grid_type);

/** Symmetrizes a simple grid with P22_12  symmetry
*/
void symmetry_P22_12(Image<double> &vol, const SimpleGrid &grid,
                     const Matrix1D<double> &eprm_aint,
                     const Matrix1D<double> &eprm_bint,
                     const MultidimArray<int> &mask, int volume_no,
                     int grid_type);

/** Symmetrizes a simple grid with P4  symmetry
*/
void symmetry_P4(Image<double> &vol, const SimpleGrid &grid,
                 const Matrix1D<double> &eprm_aint,
                 const Matrix1D<double> &eprm_bint,
                 const MultidimArray<int> &mask, int volume_no,
                 int grid_type);

/** Symmetrizes a simple grid with P4212 symmetry
*/
void symmetry_P42_12(Image<double> &vol, const SimpleGrid &grid,
                     const Matrix1D<double> &eprm_aint,
                     const Matrix1D<double> &eprm_bint,
                     const MultidimArray<int> &mask, int volume_no,
                     int grid_type);

/** Symmetrizes a simple grid with P6 symmetry
*/
void symmetry_P6(Image<double> &vol, const SimpleGrid &grid,
                 const Matrix1D<double> &eprm_aint,
                 const Matrix1D<double> &eprm_bint,
                 const MultidimArray<int> &mask, int volume_no,
                 int grid_type);

/** Symmetrize with a helical symmetry. */
void symmetry_Helical(MultidimArray<double> &Vout, const MultidimArray<double> &Vin, double zHelical, double rotHelical,
		double rot0=0, MultidimArray<int> *mask=NULL, bool dihedral=false, double heightFraction=1.0);

/** Symmetrize with a helical symmetry Low resolution.
 * This function applies the helical symmetry in such a way that only the low resolution information is kept (i.e.,
 * the general shape of the helices). */
void symmetry_HelicalLowRes(MultidimArray<double> &Vout, const MultidimArray<double> &Vin, double zHelical, double rotHelical,
		double rot0=0, MultidimArray<int> *mask=NULL);

/** Find dihedral symmetry and apply it */
void symmetry_Dihedral(MultidimArray<double> &Vout, const MultidimArray<double> &Vin, double rotStep=1,
		double zmin=-3, double zmax=3, double zStep=0.5, MultidimArray<int> *mask=NULL);
//@}
#endif
