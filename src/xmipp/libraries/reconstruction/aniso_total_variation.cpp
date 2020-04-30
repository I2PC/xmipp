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

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************* Definition of Local Methods ******************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

// Function to create Gaussian filter
void atv::GaussKernel(MultidimArray<double>& K, const double sigma, const unsigned short size)
{
	const double s = 2.0 * sigma * sigma;
	const double div = 1.0/(2*M_PI*sqrt(2*M_PI)*sigma*sigma*sigma);
	const double c = (0.5 * (size - 1));
	const int ix = (int)(0.5*(K.xdim - size));
	const int iy = (int)(0.5*(K.ydim - size));
	const int iz = (int)(0.5*(K.zdim - size));

	double r;

	// sum is for normalization
	// double sum = 0.0;

#define P(i,j,k)((i) + (j)*K.xdim + (k)*K.xdim*K.ydim)
	// generating SIZE x SIZE kernel
	for(int z = 0; z < size; z++){
		for(int y = 0; y < size; y++){
			for(int x = 0; x < size; x++){
				r = (x-c)*(x-c) + (y-c)*(y-c) + (z-c)*(z-c);
				K[P(x+ix,y+iy,z+iz)] = div * exp(-(r  / s));
				//             sum += K[P(x+ix,y+iy,z+iz)];
			}
		}
	}
	/*
 // normalizing the Kernel
 for(int k = 0; k < size; k++)
     for(int j = 0; j < size; j++)
         for(int i = 0; i < size; i++)
             K[P(i,j,k)] /= sum;
	 */
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
	kappaP = 1.0;
	kappaM = 1.0;
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
#define P(i,j,k)((i) + (j)*u.xdim + (k)*u.xdim*u.ydim)
	double sum = 0.0;
	double dw,dh,dd;

	// std::cout<<v.xdim; // "physical" horizontal limit (x direction)
	// std::cout<<v.ydim; // "physical" horizontal limit (y direction)
	// std::cout<<v.zdim; // "physical" horizontal limit (z direction)

	for(uint k=0; k < u.zdim;k++){        // Depth
		for(uint j=0;j < u.ydim;j++){     // Height
			for(uint i=0;i < u.xdim;i++){
				dw = ((i+1) < u.xdim) ? (u.data[P(i,j,k)] -
						u.data[P(i+1,j,k)]) : 0.0;
				dh = ((j+1) < u.ydim) ? (u.data[P(i,j,k)] -
						u.data[P(i,j+1,k)]) : 0.0;
				dd = ((k+1) < u.zdim) ? (u.data[P(i,j,k)] -
						u.data[P(i,j,k+1)]) : 0.0;
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
void atv::nav(const MultidimArray<double>& u, MultidimArray<double>& v)
{
#define P(i,j,k)((i) + (j)*v.xdim + (k)*v.xdim*v.ydim)
	const double ZERO=pow(10,-15);
	double denom = 0.0;
	double dw,dh,dd;

	double denom2 = 0.0;
	double dw2,dh2,dd2;
	double kappaPM = 0.0;



	// std::cout<<u.xdim; // "physical" horizontal limit (x direction)
	// std::cout<<u.ydim; // "physical" horizontal limit (y direction)
	// std::cout<<u.zdim; // "physical" horizontal limit (z direction)
	// Guaranteeing the array of weights exists and initializes it

	//
	// Computing the gradient of the total variation function
	//
	//std::cout<<formatString("\033[1;31mVolumeSize:\033[0m %dx%dx%d",u.xdim,u.ydim,u.zdim);

	MultidimArray<double> nabla_tau, tau;
	nabla_tau.resize(u.xdim,u.xdim,u.xdim);
	memset(nabla_tau.data,0,u.xdim*u.ydim*u.zdim*sizeof(double));
	tau.resize(u.xdim,u.xdim,u.xdim);
	memset(tau.data,0,u.xdim*u.ydim*u.zdim*sizeof(double));


	memset(v.data,0,v.xdim*v.ydim*v.zdim*sizeof(double));
	for(uint k=0; k < u.zdim;k++){        // Depth
		for(uint j=0;j < u.ydim;j++){     // Height
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
					tau.data[P(i,j,k)] = denom;
					// tau.data[P(i,j,k)] = w.data[P(i,j,k)] * denom; // o debería ser ???
					if(denom > ZERO){
						nabla_tau.data[P(i,j,k)] = (3*u.data[P(i,j,k)] -
								                 u.data[P(i+1,j,k)] -
								                 u.data[P(i,j+1,k)] -
								                 u.data[P(i,j,k+1)])/denom;
						v.data[P(i,j,k)] += w.data[P(i,j,k)] * nabla_tau.data[P(i,j,k)];
					}
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
					tau.data[P(i,j,k)] = denom;
					if(denom > ZERO){
						nabla_tau.data[P(i,j,k)] = (u.data[P(i,j,k)] -
								               u.data[P(i-1,j,k)])/denom;
						v.data[P(i,j,k)] += w.data[P(i,j,k)] * nabla_tau.data[P(i,j,k)];
					}
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
					tau.data[P(i,j,k)] = denom;
					if(denom > ZERO){
						nabla_tau.data[P(i,j,k)] = (u.data[P(i,j,k)] -
								               u.data[P(i,j-1,k)])/denom;
						v.data[P(i,j,k)] += w.data[P(i,j,k)] * nabla_tau.data[P(i,j,k)];
					}
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
					tau.data[P(i,j,k)] = denom;
					if(denom > ZERO){
						nabla_tau.data[P(i,j,k)] = (u.data[P(i,j,k)] -
								               u.data[P(i,j,k-1)])/denom;
						v.data[P(i,j,k)] += w.data[P(i,j,k)] * nabla_tau.data[P(i,j,k)];
					}
				}
			}//end i index
		}//end j index
	}//end k index

	//convolution of nabla_tau with kernels
	MultidimArray<double> nabla_tau_filtP, nabla_tau_filtM;
	convolutionFFT(nabla_tau, M, nabla_tau_filtM);
	convolutionFFT(nabla_tau, P, nabla_tau_filtP);

	for(uint k=0; k < u.zdim;k++){        // Depth
		for(uint j=0;j < u.ydim;j++){     // Height
			for(uint i=0;i < u.xdim;i++){ // Width

				if(dAij(MPregionsMask, k, i)) // region P
					v.data[P(i, j, k)] += tau.data[P(i, j, k)] *
							              (w.data[P(i, j, k)] * w.data[P(i, j, k)] * (-kappaP) *
						 			       nabla_tau_filtP.data[P(i, j, k)]);
				else// region M
					v.data[P(i, j, k)] += tau.data[P(i, j, k)]*
							              (w.data[P(i, j, k)] * w.data[P(i, j, k)] * (-kappaM) *
									      nabla_tau_filtM.data[P(i, j, k)]);
			}//end i index
		}//end j index
	}//end k index


	//
	// Failsafe & Finding the norm of the gradient (vector)
	//
	denom = 0.0;
	for(uint k=0; k < v.zdim;k++){         // Depth
		for(uint j=0;j < v.ydim;j++){      // Height
			for(uint i=0;i < v.xdim;i++){ // Width
				if(std::isnan(v.data[P(i,j,k)]) || fabs(v.data[P(i,j,k)])<=ZERO)
					v.data[P(i,j,k)] = 0.0;
				denom += v.data[P(i,j,k)]*v.data[P(i,j,k)];
			}
		}
	}

	//
	// Normalizing the resulting vector
	//
	if(denom <= ZERO)
		memset(v.data,0,v.xdim*v.ydim*v.zdim*sizeof(double));
	else{
		denom = sqrt(denom);
		for(uint k=0; k < v.zdim;k++){         // Depth
			for(uint j=0;j < v.ydim;j++){      // Height
				for(uint i=0;i < v.xdim;i++){ // Width
					v.data[P(i,j,k)] = -1.0 * v.data[P(i,j,k)]/denom;
					if(fabs(v.data[P(i,j,k)])<=ZERO)
						v.data[P(i,j,k)] = 0.0;
				}
			}
		}
	}

#undef P
}

void atv::createMask(const size_t xdim, const size_t ydim, const size_t zdim)
{
	MultidimArray<int> oriMask(zdim,xdim);
	oriMask.initConstant(1);
	double Cx = (double)xdim / 2.0;
	double Cy = (double)zdim / 2.0;
	double angle= 90. - maxA;
	angle = DEG2RAD(angle);
	double m = tan(angle);

	for(int y=0; y<zdim; ++y){
		for(int x=0; x<xdim; ++x){
			if(((x >= (int) (m * (double)(y - Cy) + Cx)) &&
					(x <= (int) (-m * (double)(y - Cy) + Cx))) ||
					((x <= (int) (m * (double)(y - Cy) + Cx)) &&
							(x >= (int) (-m * (double) (y - Cy) + Cx))))
				dAij(oriMask,y,x)=0;
		}
	}
	Image<int> save;
	MPregionsMask = oriMask;
	 /*comment
save()=MPregionsMask;
String rootTestFiles = String("/home/jeison/Escritorio/");
save.write(rootTestFiles+"testMask.xmp"); // */
}

/**
 **
 ** Computes the weighting vector
 **
 */
void atv::init(MultidimArray<double>& u,const double sigmaP, const unsigned short sizeP,const double kP, const double sigmaM, const unsigned short sizeM,const double kM, double Amin,double Amax)
{
	minA = Amin;
	maxA = Amax;
	std::cout<<formatString("\033[1;31mmin-max angles:\033[0m %.2f, %.2f\n",minA,maxA);
	createMask(u.xdim, u.ydim, u.zdim);
	std::cout<< formatString(
					"\033[1;31men ::Init() valores para P y M; size, sigma, kappa:\n\033[0m%d  %.2f  %.2f\n%d  %.2f  %.2f\n",
					sizeP, sigmaP, kP,
					sizeM, sigmaM, kM);

	// kappa values instead of values by constructor
	kappaP = kP;
	kappaM = kM;

	// weights for anisotropic
	if(w.getArrayPointer() == NULL)
		w.resize(u.xdim, u.ydim, u.zdim);
	memset(w.data,0,w.xdim*w.ydim*w.zdim*sizeof(double));

	P.resize(u.xdim,u.xdim,u.xdim);
	memset(P.data,0,P.xdim*P.ydim*P.zdim*sizeof(double));
	GaussKernel(P, sigmaP, sizeP);

	M.resize(u.xdim,u.xdim,u.xdim);
	memset(M.data,0,M.xdim*M.ydim*M.zdim*sizeof(double));
	GaussKernel(M, sigmaM, sizeM);
 /*
String rootTestFiles = String("/home/jeison/Escritorio/");
 Image<double> kernel;
 kernel() = P;
 kernel.write(rootTestFiles+"Filter_P.mrc");
 kernel() = M;
 kernel.write(rootTestFiles+"Filter_M.mrc");
	 // */

	/*comment
String rootTestFiles = String("/home/jeison/Escritorio/");
Image<double> save;
save() = G;
save.write(rootTestFiles+"testGkernel.xmp");
save() = tempG;
save.write(rootTestFiles+"testGkernel_resized.xmp");
save() = H;
save.write(rootTestFiles+"testHkernel.xmp");
save() = tempH;
save.write(rootTestFiles+"testHkernel_resized.xmp");

MultidimArray<double> magnitude;
FFT_magnitude(G_fourier,magnitude);
save() = magnitude;
save.write(rootTestFiles+"testMagnitudeG.xmp");
FFT_magnitude(H_fourier,magnitude);
save() = magnitude;
save.write(rootTestFiles+"testMagnitudeH.xmp");

// instruction to display testfiles: xmipp_showj -i ~/Escritorio/test*
// exit(1); // */
}


void atv::preupdate(MultidimArray<double>& u)
{
#define P(i,j,k)((i) + (j)*u.xdim + (k)*u.xdim*u.ydim)
	double dw,dh,dd;

	for(uint k=0; k < u.zdim;k++){        // Depth
		for(uint j=0;j < u.ydim;j++){     // Height
			for(uint i=0;i < u.xdim;i++){ // Width
				dw = ((i+1) < u.xdim) ? (u.data[P(i,j,k)] -
						u.data[P(i+1,j,k)]) : 0.0;
				dh = ((j+1) < u.ydim) ? (u.data[P(i,j,k)] -
						u.data[P(i,j+1,k)]) : 0.0;
				dd = ((k+1) < u.zdim) ? (u.data[P(i,j,k)] -
						u.data[P(i,j,k+1)]) : 0.0;
				w.data[P(i,j,k)] = sqrt(dw*dw + dh*dh + dd*dd);
			}
		}
	}



    MultidimArray<double> filt_M,filt_P;
	convolutionFFT(w,M,filt_M);
	convolutionFFT(w,P,filt_P);

/*
String rootTestFiles = String("/home/jeison/Escritorio/");
 Image<double> kernel;
 kernel() = filt_M;
 kernel.write(rootTestFiles+"FilteredMT.mrc");
 kernel() = filt_P;
 kernel.write(rootTestFiles+"FilteredPT.mrc");
 printf("\033[32mvalores kappaP y kappaM en preUpdate():\033[0m\n%.2f  %.2f\n",kappaP, kappaM);
// exit(0);
	 // */

	//double vmax = std::numeric_limits<double>::min();
	//double vmin = std::numeric_limits<double>::max();
	for(uint k=0; k < w.zdim;k++){        // Depth
		for(uint j=0;j < w.ydim;j++){     // Height
			for(uint i=0;i < w.xdim;i++){ // Width
				if(dAij(MPregionsMask,k,i)){ // region P
					w.data[P(i,j,k)] = 1.0/(eps + kappaP*filt_P.data[P(i,j,k)]);
				}
				else{ //region M
					w.data[P(i,j,k)] = 1.0/(eps + kappaM*filt_M.data[P(i,j,k)]);
				}
				/*
             if(w.data[P(i,j,k)] < vmin)
                vmin = w.data[P(i,j,k)];
             if(w.data[P(i,j,k)] > vmax)
                vmax = w.data[P(i,j,k)];
				 */
			}
		}
	}
	//fprintf(stdout,"min: %20.18f, max: %20.18f\n",vmin,vmax);
/*
Image<double> outImg;
outImg() = w;
outImg.write(rootTestFiles+"filteredW.mrc");
//exit(0);
// */

#undef P
}
#undef DEBUG


/**
 **
 ** Computes the weighting vector
 **
 */
//void atv::postupdate(MultidimArray<double>& u)
//{
//#define P(i,j,k)((i) + (j)*u.xdim + (k)*u.xdim*u.ydim)
//	double dw,dh,dd;
//
//	for(uint k=0; k < u.zdim;k++){        // Depth
//		for(uint j=0;j < u.ydim;j++){     // Height
//			for(uint i=0;i < u.xdim;i++){ // Width
//				dw = ((i+1) < u.xdim) ? (u.data[P(i,j,k)] -
//						u.data[P(i+1,j,k)]) : 0.0;
//				dh = ((j+1) < u.ydim) ? (u.data[P(i,j,k)] -
//						u.data[P(i,j+1,k)]) : 0.0;
//				dd = ((k+1) < u.zdim) ? (u.data[P(i,j,k)] -
//						u.data[P(i,j,k+1)]) : 0.0;
//				w.data[P(i,j,k)] = sqrt(dw*dw + dh*dh + dd*dd);
//			}
//		}
//	}
//
//	MultidimArray<double> filt_M,
//	filt_P;
//
//	convolutionFFT(w,M,filt_M);
//	convolutionFFT(w,P,filt_P);
//
///*
//String rootTestFiles = String("/home/jeison/Escritorio/");
// Image<double> kernel;
// kernel() = filt_M;
// kernel.write(rootTestFiles+"FilteredMT.mrc");
// kernel() = filt_P;
// kernel.write(rootTestFiles+"FilteredPT.mrc");
// printf("\033[32mvalores kappaP y kappaM en postUpdate():\033[0m\n%.2f  %.2f\n",kappaP, kappaM);
//// exit(0);
//	 // */
//
//	//double vmax = std::numeric_limits<double>::min();
//	//double vmin = std::numeric_limits<double>::max();
//	for(uint k=0; k < w.zdim;k++){        // Depth
//		for(uint j=0;j < w.ydim;j++){     // Height
//			for(uint i=0;i < w.xdim;i++){ // Width
//				if(dAij(MPregionsMask,k,i)){ // region P
//					w.data[P(i,j,k)] = 1.0/(eps + kappaP*filt_P.data[P(i,j,k)]);
//				}
//				else{ //region M
//					w.data[P(i,j,k)] = 1.0/(eps + kappaM*filt_M.data[P(i,j,k)]);
//				}
//				/*
//             if(w.data[P(i,j,k)] < vmin)
//                vmin = w.data[P(i,j,k)];
//             if(w.data[P(i,j,k)] > vmax)
//                vmax = w.data[P(i,j,k)];
//				 */
//			}
//		}
//	}
//	//fprintf(stdout,"min: %20.18f, max: %20.18f\n",vmin,vmax);
//// /*
//String rootTestFiles = String("/home/jeison/Escritorio/");
//Image<double> outImg;
//outImg() = w;
//outImg.write(rootTestFiles+"filteredW.mrc");
//exit(0);
//// */
//
//#undef P
//}
//#undef DEBUG
