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

#include "superiorization_reconstruct_sart.h"

#include <core/alglib/ap.h>

#include <functional>
#include <cmath>
#include <algorithm>

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
double RecSART::L2SQ(const MultidimArray<double>& x,const MultidimArray<double>& P, const std::vector<double>& LA)
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
 
 for(ushort i_a = 0; i_a < LA.size(); i_a ++){ // go through the projection angles (i.e., Z-dimension of sinogram)
     for(short l_p = 0; l_p < P.xdim; l_p ++){ // go through the line integrals in a projection (i.e., X-dimension of sinogram)
         R = PR.pixray(i_a,l_p,LA);
         //
         // Computing the inner product
         //
         if(R.size() > 0){ // The l_p-th ray from the i_a-th projection intersected the image
            for(uint plane=0; plane<P.ydim; plane++){ // Moving along the planes
                sum = 0.0;
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
RecSART::RecSART()
{
 lambda = 0.0;
 PrType = proximityType::none;
}

/**
**
** Destructor for the ART class
**
*/
RecSART::~RecSART()
{
 lambda = 0.0;
 PrType = proximityType::none;
 RW.resize(0,0);
 RWS.resize(0,0);
 Q.resize(0,0);
}

/**
**
** Initializes the matrix A
**
*/
void RecSART::init(const uint xdim, const uint ydim,const std::vector<double>& LA)
{
 M = ydim; // Height of the reconstructed region
 N = xdim; // Width of the reconstructed region
 PR.init(N,M,1.0/*Dv*/,1.0/*Dh*/,1.0/*Dp*/);
 
 RW.resize(M,N);
 RWS.resize(M,N);
 Q.resize(M,N);
}

/**
**
** Method to initialize some required variables
**
*/
void RecSART::setParam(const double l)
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
** k  -- Iteration
**
*/
void RecSART::B(MultidimArray<double>& x,const MultidimArray<double>& P, const std::vector<double>& LA,const int k)
{
 //
 // The input 3D array is going to be reconstructed slice by slice
 // (i.e., divided into several 2D problems) considering how the
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
 double raySum, weightSum,
        factor;
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
 
 // Moving along the planes
 for(uint plane=0; plane<P.ydim; plane++){
     fprintf(stdout,"Plane: %6.2f\r",(100.0*plane)/P.ydim);
     std::cout << std::flush;
     
     memset(RW.data,0,RW.xdim*RW.ydim*sizeof(double));
     memset(RWS.data,0,RWS.xdim*RWS.ydim*sizeof(double));
     memset(Q.data,0,Q.xdim*Q.ydim*sizeof(double));
     
     //
     // Go through all the rays
     //
     for(ushort i_a = 0; i_a < LA.size(); i_a ++){ // go through the projection angles (i.e., Z-dimension of sinogram)
         for(short l_p = 0; l_p < P.xdim; l_p ++){ // go through the line integrals in a projection (i.e., X-dimension of sinogram)
             R = PR.pixray(i_a,l_p,LA);
             //
             // Computing the inner product
             //
             raySum = 0.0;
             weightSum = 0.0;
             
             for(int i_r=0; i_r<R.size(); i_r++){
                 raySum += x.data[(R.at(i_r)).r*x.xdim*x.ydim + plane*x.xdim + (R.at(i_r)).c] * (R.at(i_r)).w;
                 weightSum += (R.at(i_r)).w;
                 RW.data[R.at(i_r).r*RW.xdim + R.at(i_r).c] = R.at(i_r).w;
                 RWS.data[R.at(i_r).r*RWS.xdim + R.at(i_r).c] += R.at(i_r).w;
                }
             if(weightSum < eps_zero)
                continue;
             for(int i_r=0; i_r<R.size(); i_r++){
                 Q.data[R.at(i_r).r*Q.xdim + R.at(i_r).c] += RW[R.at(i_r).r*RW.xdim + R.at(i_r).c] * (P.data[i_a*P.xdim*P.ydim + plane*P.xdim + l_p] - raySum)/weightSum;
                 RW.data[R.at(i_r).r*Q.xdim + R.at(i_r).c] = 0.0;
                }
            }
        }
     //
     // Updating pixel values
     //
     for(int i_r=0;i_r<M;i_r++){
         for(int i_c=0;i_c<N;i_c++){
             if(RWS.data[i_r*RWS.xdim + i_c] < eps_zero)
                continue;
             x.data[i_r*x.xdim*x.ydim + plane*x.xdim + i_c] += lambda*Q.data[i_r*Q.xdim + i_c]/RWS.data[i_r*RWS.xdim + i_c];
             if(x.data[i_r*x.xdim*x.ydim + plane*x.xdim + i_c] < 0.0)
                x.data[i_r*x.xdim*x.ydim + plane*x.xdim + i_c] = 0.0;
            }
        }
    }
}

/**
**
** Method that computes the proximity of a solution
**
*/
double RecSART::Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA)
{
 switch(PrType){
     case proximityType::L2SQ:
          return this->L2SQ(v,P,LA);
          break;
    }
 
 return 0.0;
}

/**
**
** Method that computes the proximity of a solution
**
*/
void RecSART::setPr(const proximityType type)
{
 if(type == proximityType::L2SQ)
    PrType = proximityType::L2SQ;
}

/**
**
** Method that computes the proximity of a solution
**
*/
void RecSART::setPr(const std::string strType)
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
reconType RecSART::getType(void)
{
 return reconType::SART;
}

/**
**
** Method that computes the proximity of a solution
**
*/
proximityType RecSART::getPrType(void)
{
 return PrType;
}

#undef DEBUG
