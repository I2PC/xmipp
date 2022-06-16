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
#include <algorithm>
#include <fstream>

#define OUTPUT_PROGRESS
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Private Methods *****************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Method that computes the L2 squared proximity function
**
** || Ax - Y ||^2
**
** v  -- The current solution (reconstruction)
** P  -- Set of Measurements (i.e., stack of sinograms)
** LA -- List of projection angles
**
*/
double RecART::L2SQ_2(const MultidimArray<double>& x,const MultidimArray<double>& P, const std::vector<double>& LA, const uint slice)
{
 /*
 **
 ** The acquisition for cryo-Tomo the projections occur on a 2D plane X-Y and the Z axis
 ** represents the different orientations. Therefore, the 3D reconstruction process can be
 ** thought as a decomposition of 2D reconstructions along the Y axis, in other workds, a 2D sinogram
 ** is thought as line integrals occurring over the X-axis (i.e., vector of measurements) and angles
 ** occurring over the Z-axis. Thus, it is necessary to specify the dimensions along the Y-axis to
 ** produce a 2D reconstruction. For example, the following occurs with the measurements:
    P is a 3D array of dimensions [xdim x ydim x zdim]  ---> S is a 2D array of dimensions [xdim x zdim]
 **
 */
 //
 // A multi-dimensional array V in Xmipp is stored as follows:
 // V.data[l*V.xdim*V.ydim*V.zdim + k*V.xdim*V.ydim + j*V.xdim + i]
 //
 // where i is the counter for the 'x' direction.
 //       j is the counter for the 'y' direction.
 //       k is the counter for the 'z' direction.
 //       l is the counter for the 't' direction (e.g., 'time').
 //
 double diff,
        sum,
        sumsq = 0.0;
 std::vector<recu::reg_R> R; // Intersections and their weights
 
 for(ushort i_a = 0; i_a < LA.size(); i_a ++){ // go through the projection     angles (i.e., Z-dimension of sinogram)
     for(short l_p = 0; l_p < P.xdim; l_p ++){ // go through the line integr    als in a projection (i.e., X-dimension of sinogram)
         R = PR.pixray(i_a,l_p,LA);
         if(R.size() > 0){ // The l_p-th ray from the i_a-th projection inte    rsected the image
            sum = 0.0;
            for(int i_p = 0; i_p<R.size(); i_p++){
                // Pixel's Row                     --> (R.at(i_p)).r
                // Pixel's Column                  --> (R.at(i_p)).c
                // Weight (length of intersection) --> (R.at(i_p)).w
                sum += (R.at(i_p)).w * x.data[(R.at(i_p)).r*x.xdim + (R.at(i_p)).c];
               }
            diff = sum - P.data[i_a*P.xdim*P.ydim + slice*P.xdim + l_p];
            sumsq += diff*diff;
           }
         R.clear();
        }
    }
 return sumsq;
}

/**
**
** Method that computes the L2 squared proximity function
**
** || Ax - Y ||^2
**
** v  -- The current solution (reconstruction)
** P  -- Set of Measurements (i.e., stack of sinograms)
** LA -- List of projection angles
**
*/
double RecART::L2SQ_3(const MultidimArray<double>& x,const MultidimArray<double>& P, const std::vector<double>& LA)
{
 /*
 **
 ** The acquisition for cryo-Tomo the projections occur on a 2D plane X-Y and the Z axis
 ** represents the different orientations. Therefore, the 3D reconstruction process can be
 ** thought as a decomposition of 2D reconstructions along the Y axis, in other workds, a 2D sinogram
 ** is thought as line integrals occurring over the X-axis (i.e., vector of measurements) and angles
 ** occurring over the Z-axis. Thus, it is necessary to specify the dimensions along the Y-axis to
 ** produce a 2D reconstruction. For example, the following occurs with the measurements:
    P is a 3D array of dimensions [xdim x ydim x zdim]  ---> S is a 2D array of dimensions [xdim x zdim]
 **
 */
 //
 // A multi-dimensional array V in Xmipp is stored as follows:
 // V.data[l*V.xdim*V.ydim*V.zdim + k*V.xdim*V.ydim + j*V.xdim + i]
 //
 // where i is the counter for the 'x' direction.
 //       j is the counter for the 'y' direction.
 //       k is the counter for the 'z' direction.
 //       l is the counter for the 't' direction (e.g., 'time').
 //
 std::cout << "ART LSQR 3" << std::endl;
 double diff,
        sum,
        sumsq = 0.0;
 std::vector<recu::reg_R> R; // Intersections and their weights
 
 for(ushort i_a = 0; i_a < LA.size(); i_a ++){ // go through the projection angles (i.e., Z-dimension of sinogram)
     for(short l_p = 0; l_p < P.xdim; l_p ++){ // go through the line integrals in a projection (i.e., X-dimension of sinogram)
         R = PR.pixray(i_a,l_p,LA);
         //  
         // Computing the inner product
         //  
         if(R.size() > 0){ // The l_p-th ray from the i_a-th projection intersected the image
            for(uint plane=0; plane<P.ydim; plane++){ // Moving along the planes                sum = 0.0;
                for(int i_p=0; i_p<R.size(); i_p++){
                    // Pixel's Row                     --> (R.at(i_p)).r
                    // Pixel's Column                  --> (R.at(i_p)).c
                    // Weight (length of intersection) --> (R.at(i_p)).w
                    sum += x.data[(R.at(i_p)).r*P.ydim*P.xdim + plane*P.xdim + (R.at(i_p)).c] * (R.at(i_p)).w; 
                   }
                diff = sum - P.data[i_a*P.xdim*P.ydim + plane*P.xdim + l_p];
                sumsq += diff*diff;
               }    
            }      
         R.clear();
        }    
    }       
 
 return sumsq;
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
** Constructor for the ART class
**
*/
RecART::RecART()
{
 lambda = 0.0;
 PrType = proximityType::none;
}

/**
**
** Initializes the class parameters
**
*/
void RecART::init(const uint xdim, const uint ydim,const std::vector<double>& LA)
{
 PR.init(xdim,ydim,1.0/*Dv*/,1.0/*Dh*/,1.0/*Dp*/);
}

/**
**
** Method to initialize some required variables
**
*/
void RecART::setParam(const double l)
{
 lambda = l;
}

/**
**
** Method that performs the actual recosntruction
**
** v  -- The solution (reconstruction)
** P  -- Set of Measurements (i.e., stack of sinograms)
** LA -- List of projection angles
** slice -- Slice
** k  -- Iteration
**
*/
void RecART::B(MultidimArray<double>& x,const MultidimArray<double>& P, const std::vector<double>& LA,const uint slice,const int k)
{
 //
 // The input is a 2D array representing a slice of the 3D image that is meant
 // to be reconstructed in a plane-by-plane scheme consistent with how
 // data is acquired in Cryo-EM 3D Electron Tomography. For other acquisition
 // geometries, it might be necessary to use a different implementation of
 // ART.
 //
 
 /*
 **
 ** The acquisition for cryo-Tomo the projections occur on a 2D plane X-Y and the Z axis
 ** represents the different orientations. Therefore, the 3D reconstruction process can be
 ** thought as a decomposition of 2D reconstructions along the Y axis, in other workds, a 2D sinogram
 ** is thought as line integrals occurring over the X-axis (i.e., vector of measurements) and angles
 ** occurring over the Z-axis. Thus, it is necessary to specify the dimensions along the Y-axis to
 ** produce a 2D reconstruction. For example, the following occurs with the measurements:
    P is a 3D array of dimensions [xdim x ydim x zdim]  ---> S is a 2D array of dimensions [xdim x zdim]
 **
 */
 double sum, snorm, factor;
 std::vector<recu::reg_R> R; // Intersections and their weights
 
 //
 // A multi-dimensional array V in Xmipp is stored as follows:
 // V.data[l*V.xdim*V.ydim*V.zdim + k*V.xdim*V.ydim + j*V.xdim + i]
 //
 // where i is the counter for the 'x' direction.
 //       j is the counter for the 'y' direction.
 //       k is the counter for the 'z' direction.
 //       l is the counter for the 't' direction (e.g., 'time').
 //
 
 for(ushort i_a = 0; i_a < LA.size(); i_a ++){ // go through the projection angles (i.e., Z-dimension of sinogram)
#ifdef OUTPUT_PROGRESS
     fprintf(stdout,"Angle: %7.4f\r",LA[i_a]);
     std::cout << std::flush;
#endif
     for(short l_p = 0; l_p < P.xdim; l_p ++){ // go through the line integrals in a projection (i.e., X-dimension of sinogram)
         R = PR.pixray(i_a,l_p,LA);
         //
         // Computing the inner product
         //
         if(R.size() > 0){ // The l_p-th ray from the i_a-th projection intersected the image
            sum = 0.0;
            snorm = 0.0;
            
            for(uint i_p=0; i_p<R.size(); i_p++){
                // Pixel's Row			  --> (R.at(i_p)).r
                // Pixel's Column		  --> (R.at(i_p)).c
                // Weight (length of intersection) --> (R.at(i_p)).w
                sum += x.data[(R.at(i_p)).r*x.xdim + (R.at(i_p)).c] * (R.at(i_p)).w;
                snorm += ( (R.at(i_p)).w * (R.at(i_p)).w );
               }
            if(snorm > eps_zero){
               factor = (P.data[i_a*P.xdim*P.ydim + slice*P.xdim + l_p] - sum)/snorm;
               //
               // Backrpojecting
               //
               for(int i_p=0; i_p<R.size(); i_p++)
                   x.data[R.at(i_p).r*x.xdim + R.at(i_p).c] += (lambda*factor*R.at(i_p).w);
              }
           }
        }
    }
}

/**
**
** Method that computes the proximity of a solution
**
*/
double RecART::Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA)
{
 double val = 0.0;
 
 switch(PrType){
     case proximityType::L2SQ:
          val = this->L2SQ_3(v,P,LA);
          break;
    }
 
 return val;
}

/**
**
** Method that computes the proximity of a solution
**
*/
double RecART::Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA, const uint slice)
{
 double val = 0.0;
 
 switch(PrType){
     case proximityType::L2SQ:
          val = this->L2SQ_2(v,P,LA,slice);
          break;
    }
 
 return val;
}

/**
**
** Method that computes the proximity of a solution
**
*/
void RecART::setPr(const proximityType type)
{
 if(type == proximityType::L2SQ)
    PrType = proximityType::L2SQ;
}

/**
**
** Method that computes the proximity of a solution
**
*/
void RecART::setPr(const std::string strType)
{
 std::string str = strType;
 std::transform(str.begin(), str.end(),str.begin(), ::toupper);
 
 if(str == std::string("L2SQ"))
    PrType = proximityType::L2SQ;
}

/**
**
** Method that computes the proximity of a solution
**
*/
reconType RecART::getType(void)
{
 return reconType::ART;
}

/**
**
** Method that computes the proximity of a solution
**
*/
proximityType RecART::getPrType(void)
{
 return PrType;
}
