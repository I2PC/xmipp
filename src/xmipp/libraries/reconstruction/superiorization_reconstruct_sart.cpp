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
 ** Method to find the pixel intersected by a line given by (p,r)
 ** p -- projection number (determined by the rotation angle)
 ** r -- ray number (position of the line in the projection)
 **
*/
// return type vector
std::vector<RecSART::reg_R> RecSART::pixray(const int np,const int nr,const std::vector<double>& LA)
{
#define d2r(angleDegrees) ((angleDegrees) * M_PI / 180.0)
 enum PixAccess{Bottom, Left};
 PixAccess Entrance;
 double theta, mv, mh;
 double sint, cost, tant, cott;
 double l,plh,plv;
 double fh,fv;
 int i,j,K;
 std::vector<double> q(2,0.0),
                     p(2, 0.0),
                     r(2, 0.0);
 reg_R P;
 std::vector<reg_R> R;
 bool Flip = false;

 theta = fmod(LA[np]+90.0,360.0);
 if(theta < 0.0)
    theta += 360.0;
 sint = sin(d2r(theta));
 cost = cos(d2r(theta));
 
 l = Dp*(nr - Pc); // Position of the current ray/line in the projection
 // q is the perpendicular vector to o
 q[0] = -1.0*sint;
 q[1] = cost;
 
 plh = l*q[0]/Dh; // horizontal position of the ray in the pixel grid (discretized)
 plv = l*q[1]/Dv; // vertical position of the ray in the pixel grid (discretized)
 
 if(fmod(theta,90.0) == 0.0){ // Vertical and Horizontal projections
    if(theta == 90.0 || theta == 270.0){ // Vertical projections
       i = (int)(plh + C[0]);
       if(i>=0 && i<N){
          for(j=0; j<M ; j++){
              P.w = Dv;
              P.r = j;
              P.c = i;
              R.push_back(P);
             }
         }
      }
    else{  // Horizontal projections
       j = (int)(plv + C[1]);
       if(j>=0 && j<M){
          for(i=0; i<N ; i++){
              P.w = Dh;
              P.r = j;
              P.c = i;
              R.push_back(P);
             }
         }
      }
   }
 else{ // Rest of orientations
    if(sint < 0.0){
       cost = -cost;
       sint = -sint;
      }
    
    if(cost*sint < 0.0){// Checking if the line is NOT in the positive (or negative) quadrant
       Flip = true;
       cost = -cost;
      }
    
    if(fabs(sint) <= eps_zero)
       if(sint < 0.0) 
          sint = -eps_zero;
       else   
          sint = eps_zero;
    if(fabs(cost) <= eps_zero)
       if(cost < 0.0)
          cost = -eps_zero;
       else
          cost = eps_zero;
    tant = sint/cost;
    cott = cost/sint;
    
    double b = l*cost + tant*l*sint; // b = y - mx
    // Checks whether the ray intersects the image
    // y = tan*x + b  <--->  x = (y - b)*cot
    //
    r[0] = cott*(Lim[3] - fabs(b));
    if(r[0] > Lim[0]){ // The integral line DOES intersect the image
       fh = Dh;
       fv = Dv;
       r[0] = Lim[0]; // Test whether the ray enters from the Left of the image
       r[1] = r[0]*tant + b;
       if(r[1] < Lim[2]){
          r[0] = 0.0;    // The ray enters from the bottom of the image
          r[1] = Lim[2];
          r[0] = (r[1] - b)*cott;
          Entrance = Bottom;
          
          i = (int)( r[0]/Dh + C[0] + 0.5);
          j = 0;
          fh = fabs(r[0]-((i+0.5)*Dh - C[0]));
         }
       else{
          Entrance = Left;
          
          i = 0;
          j = (int)( r[1]/Dv + C[1] + 0.5);
          fv = fabs(r[1] - ((j+0.5)*Dv - C[1]));
        }
       
       bool Flag = true;
       K = 0;
       while(Flag == true){
           K = K + 1;
           if(Flip == true){
              P.w = 0.0;
              P.r = j;
              P.c = i;
              R.push_back(P);
             }
           else{
              P.w = 0.0;
              P.r = M-(j+1);
              P.c = i;
              R.push_back(P);
             }
           if((fabs(Dh-fh) <= eps_zero) && (fabs(Dv-fv) <= eps_zero)){
              R.at(K-1).w = sqrt(Dh*Dh + Dv*Dv);
              i = i + 1;
              j = j + 1;
              fv = Dv;
              Entrance = Left;
              if(i >= N || j>=M)
                 Flag = false;
             }
           else{
              if(Entrance == Left){
                 fh = fv*cott;
                 if((fh<Dh) || (fabs(Dh-fh) <= eps_zero)){
                    j = j + 1;
                    R.at(K-1).w = fv/sint;
                    if(fabs(Dh-fh) < eps_zero){
                       fh = Dh;
                       i = i + 1;
                       if(i >= N)
                          Flag = false;
                      }
                    else
                       fh = Dh - fh;
                    Entrance = Bottom;
                    if(j >= M)
                       Flag = false;
                   }
                 else{
                    i = i + 1;
                    R.at(K-1).w = Dh/cost;
                    Entrance = Left;
                    fv = fv - Dh*tant;
                    if(i >= N)
                       Flag = false;
                   }
                }
              else{
                 if(Entrance == Bottom){
                    fv = fh*tant;
                    if((fv<Dv)  || (fabs(Dv-fv)<eps_zero)){
                       i = i + 1;
                       R.at(K-1).w = fh/cost;
                       if(fabs(Dv-fv) < eps_zero){
                          fv = Dv;
                          j = j + 1;
                          if(j >= M)
                             Flag = false;
                         }
                       else
                          fv = Dv - fv;
                       Entrance = Left;
                       if(i >= N)
                          Flag = false;
                      }
                    else{
                       j = j + 1;
                       R.at(K-1).w = Dv/sint;
                       Entrance = Bottom;
                       fh = fh - Dv*cott;
                       if(j >= M)
                          Flag = false;
                      }
                   }
                }
             }
          }
      }
   }
 
 return R;
#undef d2r
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
 std::vector<reg_R> R; // Intersections and their weights
 
 for(ushort i_a = 0; i_a < LA.size(); i_a ++){ // go through the projection angles (i.e., Z-dimension of sinogram)
     for(short l_p = 0; l_p < P.xdim; l_p ++){ // go through the line integrals in a projection (i.e., X-dimension of sinogram)
         R = pixray(i_a,l_p,LA);
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
 L = xdim; // Length of a projection array
 Dv = 1.0; // Sampling distance for the vertical direction
 Dh = 1.0; // Sampling distance for the horizontal direction
 Dp = 1.0; // Sampling distance among ray integrals
 
 C.push_back(0.0);
 C.push_back(0.0);
 C[0] = 0.5*Dh*(M - 1); // Center of the reconstructed region
 C[1] = 0.5*Dv*(N - 1);
 
 Pc = 0.5*(L - 1);
 Cp = Dp*Pc; // Center of the projection vector
 
 Lim[0] =  -0.5*Dh*N;
 Lim[1] =   0.5*Dh*N;
 Lim[2] =  -0.5*Dv*M;
 Lim[3] =   0.5*Dv*M;
 
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
 std::vector<reg_R> R; // Intersections and their weights
 
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
             R = pixray(i_a,l_p,LA);
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
