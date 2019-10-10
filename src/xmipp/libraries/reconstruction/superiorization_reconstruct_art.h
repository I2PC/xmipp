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
#ifndef REC_METHOD_ART_HH
#define REC_METHOD_ART_HH

#include "superiorization_reconstruct_method.h"

#include <core/xmipp_program.h>

class RecART:public RecMeth
{
 private:
    struct partial_inter{
           uint   p; //pixel ID
	   double w; // Weight --> amount of intersection between line and pixel
	  };
    double lambda;
    MultidimArray<double> S, // sinogram
                          X;
 private:
    bool GetIntersections(const double angle,const uint p,const uint length,std::vector<partial_inter> list);
    void ART(MultidimArray<double>& x,const MultidimArray<double>& S, const std::vector<double>& A,const int k);
 public:
    RecART();
    void B(MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& A,const int k);
    void setParam(const double l);
}; /* virtual class for ART method */

#endif /* REC_METHOD_ART_HH */

