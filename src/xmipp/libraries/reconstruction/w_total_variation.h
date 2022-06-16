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
#ifndef WTV_3D_HH
#define WTV_3D_HH

#include <core/xmipp_program.h>

#include "second_criterion.h"

class wtv:public secc
{
 private:
    MultidimArray<double> w;
    double eps;
    
    double phi_2(const MultidimArray<double>& u);
    double phi_3(const MultidimArray<double>& u);
    MultidimArray<double> nav_2(const MultidimArray<double>& u);
    MultidimArray<double> nav_3(const MultidimArray<double>& u);
 public:
    wtv();
    ~wtv();
    double phi(const MultidimArray<double>& u);
    MultidimArray<double> nav(const MultidimArray<double>& u);
    void init(MultidimArray<double>& u);
    void postupdate(MultidimArray<double>& u);
}; /* class for Weighted Total Variation functions */

#endif /* WTV_3D_HH */
