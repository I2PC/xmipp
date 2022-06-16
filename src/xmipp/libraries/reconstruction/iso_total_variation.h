/***************************************************************************
 *
 * Authors:     Edgar Garduno Angeles (edgargar@ieee.org)
 * Created on: Sep 6, 2019
 *
 * Department of Computer Science, Institute for Applied Mathematics
 * and Systems Research (IIMAS), National Autonomous University of
 * Mexico (UNAM)
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
#ifndef ITV_3D_HH
#define ITV_3D_HH

#include <core/xmipp_program.h>

#include "second_criterion.h"

class itv:public secc
{
 private:
    double phi_2(const MultidimArray<double>& v);
    double phi_3(const MultidimArray<double>& v);
    MultidimArray<double> nav_2(const MultidimArray<double>& v);
    MultidimArray<double> nav_3(const MultidimArray<double>& v);
 public:
    double phi(const MultidimArray<double>& v);
    MultidimArray<double> nav(const MultidimArray<double>& v);
}; /* class for Total Variation functions */

#endif /* ITV_HH */
