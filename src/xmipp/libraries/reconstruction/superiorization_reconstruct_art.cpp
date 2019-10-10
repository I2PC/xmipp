/***************************************************************************
 *
 * Authors:     Edgar Garduno Angeles (edgargar@ieee.org)
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

#include "superiorization_reconstruct_art.h"

#include <core/alglib/ap.h>

#include <functional>
#include <cmath>

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Private Methods *****************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
 **
 **
 **
 **
 **
 **/
bool RecART::GetIntersections(const double angle,const uint p,const uint length,std::vector<partial_inter> list)
{
 return false;
}

/**
**
** Carries out the 2D reconstruction with ART
**
** x  --- The reconstructed "image".
** S  --- The 2D sinogram.
** A  --- List of angles.
** k  --- Iteration number.
**
*/
void RecART::ART(MultidimArray<double>& x,const MultidimArray<double>& S, const std::vector<double>& A,const int k)
{
 const double deg2rad = M_PI/180.0;
 double angle = 0.0, cosp, sinp;
 std::vector<partial_inter> I; // Intersections and their weights
 
 for(uint j=0;j<A.size();j++){ // go over the projection angles
     angle = deg2rad * (A[j]+90.0);
     cosp  = cos(angle);
     sinp  = sin(angle);
     for(uint l=0;l<S.xdim;l++){ // go over the integral lines in a projection
         if(GetIntersections(angle,l,S.xdim,I)){
            angle = angle;
           }
        }
    }
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Public Methods ******************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Computes the Second Criterion
**
*/
RecART::RecART()
{
 lambda = 0.0;
}

/**
**
** Computes the Second Criterion
**
*/
void RecART::setParam(const double l)
{
 lambda = l;
}

/**
**
** Computes the Second Criterion
**
*/
void RecART::B(MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& A,const int k)
{
 //
 // The input 3D array is going to be reconstructed slice by slice
 // (i.e., divided into several 2D problems) considering how the
 // data is acquired in Cryo-EM 3D Electron Tomography. For other acquisition
 // geometries, it might be necessary to use a different implementation of
 // ART.
 //
 
 //
 // A multi-dimensional array V in Xmipp is stored as follows:
 // V.data[l*V.xdim*V.ydim*V.zdim + k*V.xdim*V.ydim + j*V.xdim + i]
 //
 // where i is the counter for the 'x' direction.
 //       j is the counter for the 'y' direction.
 //       k is the counter for the 'z' direction.
 //       l is the counter for the 't' direction (e.g., 'time').
 //
 if(k == 0){ // First iteration, allocating the local arrays
    S.resize(P.zdim,P.xdim);
    X.resize(v.zdim,v.xdim);
   }
 
 // Moving along the different 2D planes
 for(uint jj=0;jj<P.ydim;jj++){
     // Beginning
     // Copying the appropriate sinogram plane and reconstruction plane
     for(uint kk=0;kk<std::max(P.zdim,v.zdim);kk++){
         if(kk < P.zdim)
            memcpy(&S.data[kk*S.xdim],&P.data[kk*P.xdim*P.ydim +
	                                      jj*P.xdim],P.xdim*sizeof(double));
         if(kk < X.zdim)
            memcpy(&X.data[kk*X.xdim],&v.data[kk*v.xdim*v.ydim +
	                                      jj*v.xdim],v.xdim*sizeof(double));
	}
     // End of copying section
     this->ART(X,S,A,k);
    }
}
#undef DEBUG
