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

#include "aniso_total_variation.h"
#include <core/alglib/ap.h>

#include <functional>
#include <cmath>

#include <stdexcept>
#include <limits>

#include <core/matrix2d.h>
#include <fstream>

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************* Definition of Local Methods ******************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

// Function to create Gaussian filter
void atv::GaussKernel(MultidimArray<double>& K, const double sigma)
{
 const double s = 2.0 * sigma * sigma;
 const double div = 1.0/(M_PI*s);
 const double ch = 0.5*(K.xdim-1);
 const double cv = 0.5*(K.ydim-1);
 
 double p;
 
 // sum is for normalization
 double sum = 0.0;
 
#define P(i,j)((i) + (j)*K.xdim)
 // generating SIZE x SIZE kernel
 for(int r = 0; r < K.ydim; r++){
     for(int c = 0; c < K.xdim; c++){
         p = ((double)c-ch)*((double)c-ch) + ((double)r-cv)*((double)r-cv);
         K[P(c,r)] = div*exp(-p / s);
         sum += K[P(c,r)];
        }
    }
 
 // normalizing the kernel
 for(int r = 0; r < K.ydim; r++)
     for(int c = 0; c < K.xdim; c++)
         K[P(c,r)] /= sum;
#undef P
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
atv::atv()
{
 eps = 1.00;
 maxA = std::numeric_limits<double>::min();
 minA = std::numeric_limits<double>::max();
 
 angle_delta = 1.0;
 Ku  = 1.0;
 Me  = 1.0;
}

/*
 * Desstructor
 */
atv::~atv()
{
//	w.clear();
}

/**
 **
 ** Computes the Weighted Total Variation
 **
 */
double atv::phi(const MultidimArray<double>& u)
{
#define P(i,j)((i) + (j)*u.xdim)
 double sum  = 0.0,
        suma;
 double sigma;
 double cosd,sind;
 
 gradUV(u,Gx,Gy);
 for(double alpha=0.0;alpha<=180.0;alpha+=angle_delta){
     suma = 0.0;
     cosd =  cos(alpha * M_PI / 180.0);
     sind = -sin(alpha * M_PI / 180.0);
     I = cosd*Gx + sind*Gy;
     // Computing the Gaussian kernel and filtering the gradient
     sigma = 1 + Me*exp( -(alpha - 90.0)*(alpha - 90.0)/(Sigma*Sigma) );
     GaussKernel(G, sigma);
     // Convolution between Gradient and Gaussian filter
     convolutionFFT(G,I,W);
     
     // Computing the final filter array
     for(uint r=0; r<u.ydim;r++){
         for(uint c=0; c<u.xdim; c++){
             // Computing the filter's factor
             W.data[P(c,r)] = 1.0/(1 + Ku*fabs(W.data[P(c,r)]));
            }
        }
     
     // Computing the Total Variation for this gradient
     // in Matlab it would be:
     // tv = sum(sum(sqrt(Gx(1:size(x,1)-1,:).^2 + Gy(:,1:size(x,2)-1).^2)));
     for(uint r=0; r<u.ydim-1; r++){
         for(uint c=0; c<u.xdim-1; c++){
             //cosd = W.data[P(c,r)];
             //sind = I.data[P(c,r)];
             suma += W.data[P(c,r)]*fabs(I.data[P(c,r)]);
            }
        }
     sum += suma;
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
MultidimArray<double> atv::nav(const MultidimArray<double>& u)
{
 const double ZERO=pow(10,-15);
 MultidimArray<double> v(u);
 const double epsi = 1e-20;
 double pval;
 double ct,st;
 int    alpha;
 double sigma,den,w;
 
#define P(i,j)((i) + (j)*v.xdim)
 
 memset(v.data,0,v.xdim*v.ydim*sizeof(double));
 
 //
 // Computing the integral over theta (an interval of angles)
 //
 Image<double> outf;
 char siter[6];
 std::string scnt,oname;
 
 gradUV(u,Gx,Gy);
 
 for(alpha=0;alpha<=180;alpha+=angle_delta){
     //std::cout << "Angle: " << alpha << std::endl;
     // Computing the gradient along an angle
     switch(alpha){
         case   0:
             ct = 1.0;
             st = 0.0;
             break;
         case  30:
             ct = 0.5*sqrt(3);
             st = -0.5;
             break;
         case  45:
             ct = 0.5*sqrt(2);
             st = -0.5*sqrt(2);
             break;
         case  60:
             ct = 0.5;
             st = -0.5*sqrt(3);
             break;
         case  90:
             ct = 0.0;
             st = -1.0;
             break;
         case 120:
             ct = -0.5;
             st = -0.5*sqrt(3);
             break;
         case 135:
             ct = -0.5*sqrt(2);
             st = -0.5*sqrt(2);
             break;
         case 150:
             ct = -0.5*sqrt(3);
             st = -0.5;
             break;
         case 180:
             ct = -1.0;
             st =  0.0;
             break;
         default:
             ct =  cos(alpha * M_PI/180.0);
             st = -sin(alpha * M_PI/180.0);
             break;
        }
     
     I = ct*Gx + st*Gy;
     // Computing the Gaussian kernel and filtering the gradient
     sigma = 1 + Me*exp( -(alpha - 90.0)*(alpha - 90.0)/(Sigma*Sigma) );
     GaussKernel(G, sigma);
     // Convolution between Gradient and Gaussian filter
     convolutionFFT(G,I,W);
     //
     // Computing the summatory over the valid positions in the image S
     //
     for(uint r=0;r<u.ydim;r++){
         for(uint c=0;c<u.xdim;c++){
             //
             // Case 1: d/(d f_k)
             // 
             // First Operand
             //
             w = (1 + Ku * fabs(W.data[P(c,r)]));
             den = w*fabs( I.data[P(c,r)] );
             pval = 0.0;
             if(fabs(den) > epsi)
                pval = I.data[P(c,r)]/den;
             // 
             // Second Operand
             //
	     den = w*w * fabs(W.data[P(c,r)]);
             if(fabs(den) > epsi)
                pval -= (Ku*G.data[P(c,r)]*fabs(I.data[P(c,r)])*
                            W.data[P(c,r)]/den);
             
             v.data[P(c,r)] += (ct+st)*pval;
             //
	     // Case 2: d/(d f_r(k))
             // 
             pval = 0.0;
             if(c > 0){
                // 
                // First Operand
                //
                w = (1 + Ku*fabs(W.data[P(c-1,r)]));
	     	den = w*w*fabs(W.data[P(c-1,r)]);
                if(fabs(den) > epsi)
                   pval = Ku*G.data[P(c-1,r)]*fabs(I.data[P(c-1,r)])*
                             W.data[P(c-1,r)]/den;
                
                den = w*fabs(I.data[P(c-1,r)]);
	     	if(fabs(den) > epsi)
                   pval -= I.data[P(c-1,r)]/den;
               }
             v.data[P(c,r)] += ct*pval;
             //
	     // Case 3: d/(d f_b(k))
             // 
             pval = 0.0;
             if(r > 0){
                // 
                // First Operand
                //
                w = (1 + Ku*fabs(W.data[P(c,r-1)]));
	     	den = w*w*fabs(W.data[P(c,r-1)]);
                if(fabs(den) > epsi)
                   pval = Ku*G.data[P(c,r-1)]*fabs(I.data[P(c,r-1)])*
                             W.data[P(c,r-1)]/den;
                
                den = w*fabs(I.data[P(c,r-1)]);
	     	if(fabs(den) > epsi)
                   pval -= I.data[P(c,r-1)]/den;
               }
             v.data[P(c,r)] += st*pval;
            }
        }
    }

 //
 // Failsafe & Finding the norm of the gradient (vector)
 //
 den = 0.0;
 for(uint j=0;j < v.ydim;j++)	   // Height
     for(uint i=0;i < v.xdim;i++){ // Width
 	 if(std::isnan(v.data[j*v.xdim + i]) ||
	    fabs(v.data[j*v.xdim + i])<=ZERO)
 	    v.data[j*v.xdim + i] = 0.0;
 	 den += v.data[j*v.xdim + i]*v.data[j*v.xdim + i];
 	}
 
 //
 // Normalizing the resulting vector
 //
 if(den <= ZERO)
    memset(v.data,0,v.xdim*v.ydim*sizeof(double));
 else{
    den = sqrt(den);
    for(uint j=0;j < v.ydim;j++)      // Height
      for(uint i=0;i < v.xdim;i++){ // Width
	  v.data[j*v.xdim + i] = -1.0 * v.data[j*v.xdim + i]/den;
	  if(fabs(v.data[j*v.xdim + i]) < ZERO)
	     v.data[j*v.xdim + i] = 0.0;
	 }
   }

#undef P
 
 return v;
}

/**
 **
 ** Computes the Gradient of input image u
 ** The arrays Gx, Gy and Gz must be initialized before calling this method.
 ** R signals whether to set the gradient for the region M (where there is
 ** no projection data) or P (where there is projection data).
 **
 */
void atv::gradUV(const MultidimArray<double>& u,
                  MultidimArray<double>& Gx,
                  MultidimArray<double>& Gy)
{
#define P(V,i,j)((i) + (j)*V.xdim)
 memset(Gx.data,0,Gx.xdim*Gx.ydim*sizeof(double));
 memset(Gy.data,0,Gy.xdim*Gy.ydim*sizeof(double));
 
 for(uint j=0;j < u.ydim;j++){     // Height
     for(uint i=0;i < u.xdim;i++){ // Width
         if(i < (u.xdim - 1))
           Gx.data[P(Gx,i,j)] = u.data[P(u,i,j)] - u.data[P(u,i+1,j)];
        if(j < (u.ydim - 1))
           Gy.data[P(Gy,i,j)] = u.data[P(u,i,j)] - u.data[P(u,i,j+1)];
 	}//end i index
    }//end j index
 
#undef P
}

/**
 **
 ** Method to carry out initial calculations and assignments
 **
 */
void atv::init(MultidimArray<double>& u, const double sigma, const double ku, const double me, const int delta, const double Amin, const double Amax)
{
 minA = Amin;
 maxA = Amax;
 std::cout<<formatString("\033[1;31mmin-max angles:\033[0m %.2f, %.2f\n",minA,maxA);
 
 // weights for anisotropic
 if(W.getArrayPointer() == NULL)
    W.resize(u.ydim, u.xdim);
 memset(W.data,0,W.xdim*W.ydim*sizeof(double));
 
 // Gaussian filter
 if(G.getArrayPointer() == NULL)
    G.resize(u.ydim, u.xdim);
 memset(G.data,0,G.xdim*G.ydim*sizeof(double));
 
 // Horizontal filter
 if(Gx.getArrayPointer() == NULL)
    Gx.resize(u.ydim, u.xdim);
 memset(Gx.data,0,Gx.xdim*Gx.ydim*sizeof(double));
 
 // Vertical filter
 if(Gy.getArrayPointer() == NULL)
    Gy.resize(u.ydim, u.xdim);
 memset(Gy.data,0,Gy.xdim*Gy.ydim*sizeof(double));
 
 Sigma = sigma;
 angle_delta = delta;
 Ku  = ku;
 Me  = me;
}
