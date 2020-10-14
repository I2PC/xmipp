/***************************************************************************
 *
 * Authors:     Roberto Marabini (roberto@mipg.upenn.edu)
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
/*****************************************************************************/
/* INTERACTION WITH VIRUSES                                                  */
/*****************************************************************************/

#ifndef _XMIPP_VIRUS_HH
#define _XMIPP_VIRUS_HH

#include <core/xmipp_funcs.h>
#include <core/matrix2d.h>

#define Vir_Eq_Views 60
#define Vir_Com_Lin_Pairs 37

/**@defgroup VirusInterface Virus
   @ingroup InterfaceLibrary */
//@{
/** Virus Euler Files.
    This is a class to read the Euler matrices used by the icosahedral virus recon programs.
  The input is a stack of 60 3x3 matrices<double> matrices
*/
class VirusEulerMatrices
{
public:
    /** Virus Euler Filename */
    FileName fh_Euler;
    /** 60 Matrices with the Euler rotations */
    Matrix2D<double> E_Matrices[Vir_Eq_Views];
    /** Read an Euler matrix file with the symmetry relationships in a virus */
    void read(const FileName &fn);
};
//@}
#endif
