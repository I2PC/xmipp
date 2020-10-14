/***************************************************************************
 *
 * Authors:    Sjors Scheres           scheres@cnb.csic.es (2004)
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

#ifndef _DIRECTIONS_H
#define _DIRECTIONS_H

#include <core/xmipp_funcs.h>
#include <core/metadata.h>

#include <core/symmetries.h>

/**@defgroup Directions Operations for projection directions
   @ingroup ReconsLibrary */
//@{
/// Check whether projection directions are unique
bool directions_are_unique(double rot,  double tilt,
                           double rot2, double tilt2,
                           double rot_limit, double tilt_limit,
                           SymList &SL, bool include_mirrors,
                           Matrix2D<double> &Laux, Matrix2D<double> &Raux);

/// Calculate angular distance between two directions
double distance_directions(double rot1, double tilt1,
                           double rot2, double tilt2,
                           bool include_mirrors);

/// Make even distribution, taking symmetry into account
void make_even_distribution(std::vector<double> &rotList, std::vector<double> &tiltList,
							double sampling, SymList &SL, bool include_mirror);

/// Select a user-provided tilt range
void limit_tilt_range(MetaData &DF, double tilt_range0, double tilt_rangeF);

/// Determine which of the entries in DFlib is closest to [rot1,tilt1]
int find_nearest_direction(double rot1, double tilt1,
                           std::vector<double> &rotList, std::vector<double> &tiltList,
                           SymList &SL, Matrix2D<double> &Laux, Matrix2D<double> &Raux);
//@}
#endif
