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

#include "w_total_variation.h"
#include <core/alglib/ap.h>

#include <functional>
#include <cmath>

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Local Methods *******************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Computes the Weighted Total Variation
**
*/
double wtv::phi_2(const MultidimArray<double>& v)
{
#define P(i,j)((i) + (j)*v.xdim)
 double sum = 0.0;
 double dw,dh;
 
// std::cout<<v.xdim; // "physical" horizontal limit (x direction)
// std::cout<<v.ydim; // "physical" horizontal limit (y direction)
 
 for(uint j=0;j < v.ydim;j++){     // Height
     for(uint i=0;i < v.xdim;i++){ // Width
         dw = ((i+1) < v.xdim) ? (v.data[P(i,j)] - v.data[P(i+1,j)]) : 0.0;
         dh = ((j+1) < v.ydim) ? (v.data[P(i,j)] - v.data[P(i,j+1)]) : 0.0;
         sum = sum + w.data[P(i,j)]*sqrt(dw*dw + dh*dh);
 	}
    }
#undef P
 
 return sum;
}

/**
**
** Computes the Weighted Total Variation
**
*/
double wtv::phi_3(const MultidimArray<double>& v)
{
#define P(i,j,k)((i) + (j)*v.xdim + (k)*v.xdim*v.ydim)
 double sum = 0.0;
 double dw,dh,dd;
 
// std::cout<<v.xdim; // "physical" horizontal limit (x direction)
// std::cout<<v.ydim; // "physical" horizontal limit (y direction)
// std::cout<<v.zdim; // "physical" horizontal limit (z direction)
 
 for(uint k=0; k < v.zdim;k++){        // Depth
     for(uint j=0;j < v.ydim;j++){     // Height
         for(uint i=0;i < v.xdim;i++){ // Width
             dw = ((i+1) < v.xdim) ? (v.data[P(i,j,k)] - v.data[P(i+1,j,k)]) : 0.0;
             dh = ((j+1) < v.ydim) ? (v.data[P(i,j,k)] - v.data[P(i,j+1,k)]) : 0.0;
             dd = ((k+1) < v.zdim) ? (v.data[P(i,j,k)] - v.data[P(i,j,k+1)]) : 0.0;
             sum = sum + w.data[P(i,j,k)]*sqrt(dw*dw + dh*dh + dd*dd);
            }
        }
    }
#undef P
 
 return sum;
}

/**
**
** Computes the normalized non-ascending vector for the Weighted Total Variation
** TV(x) = SUM of the w(x(i,j,k))*sqrt( (x(i,j,k) - x(i+1,j,k))^2 + (x(i,j,k) - x(i,j+1,k))^2 +(x(i,j,k) - x(i,j,k+1))^2 ) =
**       = SUM w(x_i)*sqrt( (x_i - x_r)^2 + (x_i - x_u)^2 + (x_i - x_b)^2 )
** d/dx(i,j,k) TV / || d/dx(i,j,k) TV ||
**
*/
MultidimArray<double> wtv::nav_2(const MultidimArray<double>& u)
{
 MultidimArray<double> v(u);
#define P(i,j)((i) + (j)*v.xdim)
 const double ZERO=pow(10,-15);
 double denom = 0.0;
 double dw,dh;
 
 // std::cout<<u.xdim; // "physical" horizontal limit (x direction)
 // std::cout<<u.ydim; // "physical" horizontal limit (y direction)
 // Guaranteeing the array of weights exists and initializes it
 
 //
 // Computing the gradient of the total variation function
 //
 memset(v.data,0,v.xdim*v.ydim*sizeof(double));
 for(uint j=0;j < u.ydim;j++)	   // Height
     for(uint i=0;i < u.xdim;i++){ // Width
 	 //
 	 // First Case
 	 // (d/d x_i) of TV
 	 //
 	 if(i<(u.xdim-1) && j<(u.ydim-1)){
 	    dw = u.data[P(i,j)] - u.data[P(i+1,j)];
 	    dh = u.data[P(i,j)] - u.data[P(i,j+1)];
 	    //Computing the denominator
 	    denom = sqrt(dw*dw + dh*dh);
 	    if(denom > ZERO)
 	       v.data[P(i,j)] += w.data[P(i,j)]*(2*u.data[P(i,j)] - 
                                                   u.data[P(i+1,j)] -
                                                   u.data[P(i,j+1)])/denom;
 	   }
 	 //
 	 // Second Case
 	 // (d/d x_r) of TV (x_r is the base and not x_i)
 	 //
 	 if(i>0 && i<u.xdim && j<(u.ydim-1)){
 	    dw = u.data[P(i-1,j)] - u.data[P(i,j)];
 	    dh = u.data[P(i-1,j)] - u.data[P(i-1,j+1)];
 	    //Computing the denominator
 	    denom = sqrt(dw*dw + dh*dh);
 	    if(denom > ZERO)
 	       v.data[P(i,j)] += w.data[P(i-1,j)]*(u.data[P(i,j)] -
                                 u.data[P(i-1,j)])/denom;
 	   }
 	 //
 	 // Third Case
 	 // (d/d x_u) of TV (x_u is the base and not x_i)
 	 //
 	 if(i<(u.xdim-1) && j>0 && j<u.ydim){
 	    dw = u.data[P(i,j-1)] - u.data[P(i+1,j-1)];
 	    dh = u.data[P(i,j-1)] - u.data[P(i,j)];
 	    //Computing the denominator
 	    denom = sqrt(dw*dw + dh*dh);
 	    if(denom > ZERO)
 	       v.data[P(i,j)] += w.data[P(i,j-1)]*(u.data[P(i,j)] -
                                 u.data[P(i,j-1)])/denom;
 	   }
 	}
 
 //
 // Failsafe & Finding the norm of the gradient (vector)
 //
 denom = 0.0;
 for(uint j=0;j < v.ydim;j++)	   // Height
     for(uint i=0;i < v.xdim;i++){ // Width
 	 if(std::isnan(v.data[P(i,j)]) || fabs(v.data[P(i,j)])<=ZERO)
 	    v.data[P(i,j)] = 0.0;
 	 denom += v.data[P(i,j)]*v.data[P(i,j)];
 	}
 
 //
 // Normalizing the resulting vector
 //
 if(denom <= ZERO)
    memset(v.data,0,v.xdim*v.ydim*sizeof(double));
 else{
    denom = sqrt(denom);
    for(uint j=0;j < v.ydim;j++)      // Height
    	for(uint i=0;i < v.xdim;i++){ // Width
    	    v.data[P(i,j)] = -1.0 * v.data[P(i,j)]/denom;
    	    if(fabs(v.data[P(i,j)])<=ZERO)
    	       v.data[P(i,j)] = 0.0;
    	   }
   }
#undef P
 
 return v;
}

/**
**
** Computes the normalized non-ascending vector for the Weighted Total Variation
** TV(x) = SUM of the w(x(i,j,k))*sqrt( (x(i,j,k) - x(i+1,j,k))^2 + (x(i,j,k) - x(i,j+1,k))^2 +(x(i,j,k) - x(i,j,k+1))^2 ) =
**       = SUM w(x_i)*sqrt( (x_i - x_r)^2 + (x_i - x_u)^2 + (x_i - x_b)^2 )
** d/dx(i,j,k) TV / || d/dx(i,j,k) TV ||
**
*/
MultidimArray<double> wtv::nav_3(const MultidimArray<double>& u)
{
 MultidimArray<double> v(u);
#define P(i,j,k)((i) + (j)*v.xdim + (k)*v.xdim*v.ydim)
 const double ZERO=pow(10,-15);
 double denom = 0.0;
 double dw,dh,dd;
 
 // std::cout<<u.xdim; // "physical" horizontal limit (x direction)
 // std::cout<<u.ydim; // "physical" horizontal limit (y direction)
 // std::cout<<u.zdim; // "physical" horizontal limit (z direction)
 // Guaranteeing the array of weights exists and initializes it
 
 //
 // Computing the gradient of the total variation function
 //
 memset(v.data,0,v.xdim*v.ydim*v.zdim*sizeof(double));
 for(uint k=0; k < u.zdim;k++)         // Depth
     for(uint j=0;j < u.ydim;j++)      // Height
         for(uint i=0;i < u.xdim;i++){ // Width
             //
             // First Case
             // (d/d x_i) of TV
             //
             if(i<(u.xdim-1) && j<(u.ydim-1) && k<(u.zdim-1)){
                dw = u.data[P(i,j,k)] - u.data[P(i+1,j,k)];
                dh = u.data[P(i,j,k)] - u.data[P(i,j+1,k)];
                dd = u.data[P(i,j,k)] - u.data[P(i,j,k+1)];
                //Computing the denominator
                denom = sqrt(dw*dw + dh*dh + dd*dd);
                if(denom > ZERO)
                   v.data[P(i,j,k)] += w.data[P(i,j,k)]*(3*u.data[P(i,j,k)] -
                                        u.data[P(i+1,j,k)] -
                                        u.data[P(i,j+1,k)] -
                                        u.data[P(i,j,k+1)])/denom;
               }
             //
             // Second Case
             // (d/d x_r) of TV (x_r is the base and not x_i)
             //
             if(i>0 && i<u.xdim && j<(u.ydim-1) && k<(u.zdim-1)){
                dw = u.data[P(i-1,j,k)] - u.data[P(i,j,k)];
                dh = u.data[P(i-1,j,k)] - u.data[P(i-1,j+1,k)];
                dd = u.data[P(i-1,j,k)] - u.data[P(i-1,j,k+1)];
                //Computing the denominator
                denom = sqrt(dw*dw + dh*dh + dd*dd);
                if(denom > ZERO)
                   v.data[P(i,j,k)] += w.data[P(i-1,j,k)]*(u.data[P(i,j,k)] -
                                              u.data[P(i-1,j,k)])/denom;
               }
             //
             // Third Case
             // (d/d x_u) of TV (x_u is the base and not x_i)
             //
             if(i<(u.xdim-1) && j>0 && j<u.ydim && k<(u.zdim-1)){
                dw = u.data[P(i,j-1,k)] - u.data[P(i+1,j-1,k)];
                dh = u.data[P(i,j-1,k)] - u.data[P(i,j,k)];
                dd = u.data[P(i,j-1,k)] - u.data[P(i,j-1,k+1)];
                //Computing the denominator
                denom = sqrt(dw*dw + dh*dh + dd*dd);
                if(denom > ZERO)
                   v.data[P(i,j,k)] += w.data[P(i,j-1,k)]*(u.data[P(i,j,k)] -
                                              u.data[P(i,j-1,k)])/denom;
               }
             //
             // Fourth Case
             // (d/d x_b) of TV (x_b is the base and not x_i)
             //
             if(i<(u.xdim-1) && j<(u.ydim-1) && k>0 && k<u.zdim){
                dw = u.data[P(i,j,k-1)] - u.data[P(i+1,j,k-1)];
                dh = u.data[P(i,j,k-1)] - u.data[P(i,j+1,k-1)];
                dd = u.data[P(i,j,k-1)] - u.data[P(i,j,k)];
                //Computing the denominator
                denom = sqrt(dw*dw + dh*dh + dd*dd);
                if(denom > ZERO)
                   v.data[P(i,j,k)] += w.data[P(i,j,k-1)]*(u.data[P(i,j,k)] -
                                              u.data[P(i,j,k-1)])/denom;
               }
            }
 
 //
 // Failsafe & Finding the norm of the gradient (vector)
 //
 denom = 0.0;
 for(uint k=0; k < v.zdim;k++)         // Depth
     for(uint j=0;j < v.ydim;j++)      // Height
         for(uint i=0;i < v.xdim;i++){ // Width
             if(std::isnan(v.data[P(i,j,k)]) || fabs(v.data[P(i,j,k)])<=ZERO)
                v.data[P(i,j,k)] = 0.0;
             denom += v.data[P(i,j,k)]*v.data[P(i,j,k)];
            }
 
 //
 // Normalizing the resulting vector
 //
 if(denom <= ZERO)
    memset(v.data,0,v.xdim*v.ydim*v.zdim*sizeof(double));
 else{
    denom = sqrt(denom);
    for(uint k=0; k < v.zdim;k++)         // Depth
        for(uint j=0;j < v.ydim;j++)      // Height
            for(uint i=0;i < v.xdim;i++){ // Width
                v.data[P(i,j,k)] = -1.0 * v.data[P(i,j,k)]/denom;
                if(fabs(v.data[P(i,j,k)])<=ZERO)
                   v.data[P(i,j,k)] = 0.0;
               }
   }
#undef P
 
 return v;
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Public Methods ******************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/*
 * Default Constructor
 */
wtv::wtv()
{
 eps = 1.00;
}

/*
 * Desstructor
 */
wtv::~wtv()
{
 w.clear();
}

/**
**
** Computes the Weighted Total Variation
**
*/
double wtv::phi(const MultidimArray<double>& v)
{
 if(v.zdim == 0)
    return this->phi_2(v);
 if(v.zdim > 0)
    return this->phi_3(v);
}

/**
**
** Computes the normalized non-ascending vector for the Weighted Total Variation
** TV(x) = SUM of the w(x(i,j,k))*sqrt( (x(i,j,k) - x(i+1,j,k))^2 + (x(i,j,k) - x(i,j+1,k))^2 +(x(i,j,k) - x(i,j,k+1))^2 ) =
**       = SUM w(x_i)*sqrt( (x_i - x_r)^2 + (x_i - x_u)^2 + (x_i - x_b)^2 )
** d/dx(i,j,k) TV / || d/dx(i,j,k) TV ||
**
*/
MultidimArray<double> wtv::nav(const MultidimArray<double>& u)
{
 if(u.zdim == 1)
    return this->nav_2(u);
 if(u.zdim > 1)
    return this->nav_3(u);
}

/**
**
** Computes the weighting vector
**
*/
void wtv::init(MultidimArray<double>& v)
{
 // Guaranteeing the array of weights exists and initializes it
 if(w.getArrayPointer() == NULL){
    if(v.zdim > 0){
       w.resize(v.zdim,v.ydim,v.xdim);
       memset(w.data,0,w.xdim*w.ydim*w.zdim*sizeof(double));
      }
    else{
       w.resize(v.ydim,v.xdim);
       memset(w.data,0,w.xdim*w.ydim*sizeof(double));
      }
   }
}

/**
**
** Computes the weighting vector
**
*/
void wtv::postupdate(MultidimArray<double>& v)
{
 double dw,dh,dd;
 
 if(v.zdim > 0){
#define P(i,j,k)((i) + (j)*v.xdim + (k)*v.xdim*v.ydim)
    for(uint k=0; k < v.zdim;k++){	  // Depth
    	for(uint j=0;j < v.ydim;j++){	  // Height
    	    for(uint i=0;i < v.xdim;i++){ // Width
    		dw = ((i+1) < v.xdim) ? (v.data[P(i,j,k)] - v.data[P(i+1,j,k)]) : 0.0;
    		dh = ((j+1) < v.ydim) ? (v.data[P(i,j,k)] - v.data[P(i,j+1,k)]) : 0.0;
    		dd = ((k+1) < v.zdim) ? (v.data[P(i,j,k)] - v.data[P(i,j,k+1)]) : 0.0;
    		w.data[P(i,j,k)] = 1.0/(sqrt(dw*dw + dh*dh + dd*dd) + eps);
    	       }
    	   }
       }
#undef P
   }
 else{
#define P(i,j)((i) + (j)*v.xdim)
    for(uint j=0;j < v.ydim;j++){     // Height
    	for(uint i=0;i < v.xdim;i++){ // Width
    	    dw = ((i+1) < v.xdim) ? (v.data[P(i,j)] - v.data[P(i+1,j)]) : 0.0;
    	    dh = ((j+1) < v.ydim) ? (v.data[P(i,j)] - v.data[P(i,j+1)]) : 0.0;
    	    w.data[P(i,j)] = 1.0/(sqrt(dw*dw + dh*dh) + eps);
    	   }
       }
#undef P
   }
}
