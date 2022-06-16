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

#include "recons_util.h"

#include <core/alglib/ap.h>

#include <functional>
#include <cmath>
#include <algorithm>

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Public Methods ******************************/
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
std::vector<recu::reg_R> recu::pixRay::pixray(const int np,const int nr,const std::vector<double>& LA)
{
#define d2r(angleDegrees) ((angleDegrees) * M_PI / 180.0)
 enum PixAccess{Bottom, Left};
 PixAccess Entrance;
 double theta, mv, mh;
 double sint, cost, tant, cott;
 double l,plh,plv;
 double fh,fv;
 int i,j;
 std::vector<double> q(2, 0.0),
                     p(2, 0.0),
                     r(2, 0.0);
 recu::reg_R P;
 std::vector<recu::reg_R> R;
 bool Flip = false;

 // theta = fmod(LA[np]+90.0,360.0); previous
 theta = fmod(LA[np] + 0.0, 360.0);
 if(theta < 0.0)
    theta = theta + 360.0;
 sint = sin(d2r(theta));
 cost = cos(d2r(theta));
 
 l = Dp*(nr - Pc); // Position of the current ray/line in the projection
                   // q is the perpendicular vector to o
 q[0] = sint;
 q[1] = -1.0*cost;
 
 plh = l*q[0]/Dh; // horizontal position of the ray in the pixel grid (discretized)
 plv = l*q[1]/Dv; // vertical position of the ray in the pixel grid (discretized)
 
 if(fmod(theta,90.0) == 0.0){                // Vertical and Horizontal projections
    if(theta == 90.0 || theta == 270.0){    // Vertical projections
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
    else{            // Horizontal projections
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
 else{               // Rest of orientations
    if(sint < 0.0){
       cost = -cost;
       sint = -sint;
      }
    
    if(cost*sint < 0.0){  // Checking if the line is NOT in the positive (or negative) quadrant
       Flip = true;       // the vector o (direction) is flipped to
       cost = -cost;      // o =(sint, cost)
      }
    
    if(fabs(sint) <= eps_zero){
       if(sint < 0.0)
          sint = -eps_zero;
       else   
          sint = eps_zero;
      }
    if(fabs(cost) <= eps_zero){
       if(cost < 0.0)
          cost = -eps_zero;
       else
          cost = eps_zero;
      }
    tant = sint/cost;
    cott = cost/sint;
    
    double b = l*cost + tant*l*sint; // b = y - mx
    // Checks whether the ray intersects the image
    // y = tan*x + b  <--->  x = (y - b)*cot
    //
    r[0] = cott*(Lim[3] - fabs(b));
    
    if(r[0] > Lim[0]){  // The integral line DOES intersect the image
       fh = Dh;
       fv = Dv;
       r[0] = Lim[0];       // Test whether the ray enters from the Left of the image
       r[1] = r[0]*tant + b;
       
       if(r[1] < Lim[2]){
          r[0] = 0.0;     // The ray enters from the bottom of the image
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
          fv = abs(r[1] - ((j+0.5)*Dv - C[1]));
         }
       
       bool Flag = true;
       int i_r = 0;
       int i_c = 0;
       while(Flag == true){
           if(Flip == true){
              P.w = 0.0;
              P.r = j;
              P.c = N-i-1;
              i_r = j;
              i_c = N-i-1;
             }
           else{
              P.w = 0.0;
              P.r = j;
              P.c = i;
              i_r = j;
              i_c = i;
             }
           if((fabs(Dh-fh) <= eps_zero) && (fabs(Dv-fv) <= eps_zero)){
              P.w = sqrt(Dh*Dh + Dv*Dv);
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
                 if((fh<Dh) || (abs(Dh-fh) <= eps_zero)){
                    j = j + 1;
                    P.w = fv/sint;
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
                    P.w = Dh/cost;
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
                       P.w = fh/cost;
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
                       P.w = Dv/sint;
                       Entrance = Bottom;
                       fh = fh - Dv*cott;
                       if(j >= M)
                          Flag = false;
                      }
                   }
                }
             }
           
           if(i_c<0 || i_r<0)
              continue;
           R.push_back(P);
          }
      }
   }
  
 return R;
#undef d2r
}


/**
 **
 ** Method to find the pixel intersected by a line given by (p,r)
 ** p -- projection number (determined by the rotation angle)
 ** r -- ray number (position of the line in the projection)
 **
*/
// return type vector
std::vector<recu::reg_R> recu::pixRay::pixray(const double angle)
{
#define d2r(angleDegrees) ((angleDegrees) * M_PI / 180.0)
 enum PixAccess{Bottom, Left};
 PixAccess Entrance;
 double theta, mv, mh;
 double sint, cost, tant, cott;
 double l,plh,plv;
 double fh,fv;
 int i,j,K;
 std::vector<double> q(2, 0.0),
                     p(2, 0.0),
                     r(2, 0.0);
 recu::reg_R P;
 std::vector<recu::reg_R> R;
 bool Flip = false;

 theta = angle;
 if(theta < 0.0)
    theta = theta + 360.0;
 sint = sin(d2r(theta));
 cost = cos(d2r(theta));
 
 l = 0;
 //l = Dp*(nr - Pc); // Position of the current ray/line in the projection
 // q is the perpendicular vector to o
 q[0] = sint;
 q[1] = -1.0*cost;
 
 plh = l*q[0]/Dh; // horizontal position of the ray in the pixel grid (discretized)
 plv = l*q[1]/Dv; // vertical position of the ray in the pixel grid (discretized)
 
 if(fmod(theta,90.0) == 0.0){                // Vertical and Horizontal projections
    if(theta == 90.0 || theta == 270.0){    // Vertical projections
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
    else{            // Horizontal projections
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
 else{               // Rest of orientations
    if(sint < 0.0){
       cost = -cost;
       sint = -sint;
      }
    
    if(cost*sint < 0.0){  // Checking if the line is NOT in the positive (or negative) quadrant
       Flip = true;       // the vector o (direction) is flipped to
       cost = -cost;      // o =(sint, cost)
      }
    
    if(fabs(sint) <= eps_zero){
       if(sint < 0.0)
          sint = -eps_zero;
       else   
          sint = eps_zero;
      }
    if(fabs(cost) <= eps_zero){
       if(cost < 0.0)
          cost = -eps_zero;
       else
          cost = eps_zero;
      }
    tant = sint/cost;
    cott = cost/sint;
    
    double b = l*cost + tant*l*sint; // b = y - mx
    // Checks whether the ray intersects the image
    // y = tan*x + b  <--->  x = (y - b)*cot
    //
    r[0] = cott*(Lim[3] - fabs(b));
    
    if(r[0] > Lim[0]){  // The integral line DOES intersect the image
       fh = Dh;
       fv = Dv;
       r[0] = Lim[0];       // Test whether the ray enters from the Left of the image
       r[1] = r[0]*tant + b;
       
       if(r[1] < Lim[2]){
          r[0] = 0.0;     // The ray enters from the bottom of the image
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
          fv = abs(r[1] - ((j+0.5)*Dv - C[1]));
         }
       
       bool Flag = true;
       int i_r = 0;
       int i_c = 0;
       while(Flag == true){
           if(Flip == true){
              P.w = 0.0;
              P.r = j;
              P.c = N-i-1;
              i_r = j;
              i_c = N-i-1;
             }
           else{
              P.w = 0.0;
              P.r = j;
              P.c = i;
              i_r = j;
              i_c = i;
             }
           if((fabs(Dh-fh) <= eps_zero) && (fabs(Dv-fv) <= eps_zero)){
              P.w = sqrt(Dh*Dh + Dv*Dv);
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
                 if((fh<Dh) || (abs(Dh-fh) <= eps_zero)){
                    j = j + 1;
                    P.w = fv/sint;
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
                    P.w = Dh/cost;
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
                       P.w = fh/cost;
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
                       P.w = Dv/sint;
                       Entrance = Bottom;
                       fh = fh - Dv*cott;
                       if(j >= M)
                          Flag = false;
                      }
                   }
                }
             }
           
           if(i_c<0 || i_r<0)
              continue;
           R.push_back(P);
          }
      }
   }
 
 return R;
#undef d2r
}

/**
**
** Initializes the class parameters
**
*/
void recu::pixRay::init(const uint xdim,const uint ydim,double sv,double sh,double sp)
{
 M = ydim; // Height of the reconstructed region
 N = xdim; // Width of the reconstructed region
 L = xdim; // Length of a projection array
 Dv = sv; // Sampling distance for the vertical direction
 Dh = sh; // Sampling distance for the horizontal direction
 Dp = sp; // Sampling distance among ray integrals
 
 if(C.size() != 2){
    C.clear();
    C.push_back(0.0);
    C.push_back(0.0);
   }
 C[0] = 0.5*Dh*(N - 1); // Center of the reconstructed region
 C[1] = 0.5*Dv*(M - 1);
 
 if(Q.size() != 2){
    Q.clear();
    Q.push_back(0.0);
    Q.push_back(0.0);
   }
 Q[0] = Dh*C[0];
 Q[1] = Dh*C[1];
 
 Pc = 0.5*(L - 1);
 
 Lim[0] =  -0.5*Dh*N;
 Lim[1] =   0.5*Dh*N;
 Lim[2] =  -0.5*Dv*M;
 Lim[3] =   0.5*Dv*M;
}

/**
**
** Initializes the class parameters
**
*/
recu::pixRay::pixRay(const uint xdim,const uint ydim,double sv,double sh,double sp)
{
 M = ydim; // Height of the reconstructed region
 N = xdim; // Width of the reconstructed region
 L = xdim; // Length of a projection array
 Dv = sv; // Sampling distance for the vertical direction
 Dh = sh; // Sampling distance for the horizontal direction
 Dp = sp; // Sampling distance among ray integrals
 
 C.push_back(0.0);
 C.push_back(0.0);
 C[0] = 0.5*Dh*(N - 1); // Center of the reconstructed region
 C[1] = 0.5*Dv*(M - 1);
 
 Q.push_back(0.0);
 Q.push_back(0.0);
 Q[0] = Dh*C[0];
 Q[1] = Dh*C[1];
 
 Pc = 0.5*(L - 1);
 
 Lim[0] =  -0.5*Dh*N;
 Lim[1] =   0.5*Dh*N;
 Lim[2] =  -0.5*Dv*M;
 Lim[3] =   0.5*Dv*M;
}

/**
**
** Initializes the class parameters
**
*/
recu::pixRay::pixRay()
{
 M = 1; // Height of the reconstructed region
 N = 1; // Width of the reconstructed region
 L = 1; // Length of a projection array
 Dv = 1.0; // Sampling distance for the vertical direction
 Dh = 1.0; // Sampling distance for the horizontal direction
 Dp = 1.0; // Sampling distance among ray integrals
 
 C.push_back(0.0);
 C.push_back(0.0);
 C[0] = 0.5*Dh*(N - 1); // Center of the reconstructed region
 C[1] = 0.5*Dv*(M - 1);
 
 Q.push_back(0.0);
 Q.push_back(0.0);
 Q[0] = Dh*C[0];
 Q[1] = Dh*C[1];
 
 Pc = 0.5*(L - 1);
 
 Lim[0] =  -0.5*Dh*N;
 Lim[1] =   0.5*Dh*N;
 Lim[2] =  -0.5*Dv*M;
 Lim[3] =   0.5*Dv*M;
}

#undef DEBUG
