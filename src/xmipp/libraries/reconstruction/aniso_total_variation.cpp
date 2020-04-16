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

std::vector<recu::reg_R> atv::pixray(const int np,const int nr,const std::vector<double>& LA)
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
	recu::reg_R P;
	std::vector<recu::reg_R> R;
	bool Flip = false;

	return R;
#undef d2r
}

// Function to create Gaussian filter
void atv::GaussKernel(MultidimArray<double>& K, const double sigma, const unsigned short size)
{
	const double s = 2.0 * sigma * sigma;
	const short a = 0.5 * size;
	double r;

	// sum is for normalization
	double sum = 0.0;

#define P(i,j,k)((i) + (j)*K.xdim + (k)*K.xdim*K.ydim)
	// generating SIZE x SIZE kernel
	for(int z = -a; z <= a; z++){
		for(int y = -a; y <= a; y++){
			for(int x = -a; x <= a; x++){
				r = sqrt(x*x + y*y + z*z);
				K[P(x+a,y+a,z+a)] = (exp(-(r * r) / s)) / (M_PI * s);
				sum += K[P(x+a,y+a,z+a)];
			}
		}
	}

	// normalising the Kernel
	for(int k = 0; k < size; k++)
		for(int j = 0; j < size; j++)
			for(int i = 0; i < size; i++)
				K[P(i,j,k)] /= sum;

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
double atv::phi(const MultidimArray<double>& v)
{
#define P(i,j,k)(i + j*v.xdim + k*v.xdim*v.ydim)
	double sum = 0.0;
	double dwP,dhP,ddP;
	double dwM,dhM,ddM;

	// std::cout<<v.xdim; // "physical" horizontal limit (x direction)
	// std::cout<<v.ydim; // "physical" horizontal limit (y direction)
	// std::cout<<v.zdim; // "physical" horizontal limit (z direction)

	for(uint k=0; k < v.zdim;k++){        // Depth
		for(uint j=0;j < v.ydim;j++){     // Height
			for(uint i=0;i < v.xdim;i++){ // Width
				if(dAij(MPregionsMask,k,i)){ //region P
					dwP = ((i+1) < v.xdim) ? (v.data[P(i,j,k)] - v.data[P(i+1,j,k)]) : 0.0;
					dhP = ((j+1) < v.ydim) ? (v.data[P(i,j,k)] - v.data[P(i,j+1,k)]) : 0.0;
					ddP = ((k+1) < v.zdim) ? (v.data[P(i,j,k)] - v.data[P(i,j,k+1)]) : 0.0;
					sum = sum + piP.data[P(i,j,k)]*sqrt(dwP*dwP + dhP*dhP + ddP*ddP);
				}
				else{ //region M
					dwM = ((i+1) < v.xdim) ? (v.data[P(i,j,k)] - v.data[P(i+1,j,k)]) : 0.0;
					dhM = ((j+1) < v.ydim) ? (v.data[P(i,j,k)] - v.data[P(i,j+1,k)]) : 0.0;
					ddM = ((k+1) < v.zdim) ? (v.data[P(i,j,k)] - v.data[P(i,j,k+1)]) : 0.0;
					sum = sum + muM.data[P(i,j,k)]*sqrt(dwM*dwM + dhM*dhM + ddM*ddM);
				}
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
#define P(i,j,k)(i + j*v.xdim + k*v.xdim*v.ydim)
	const double ZERO=pow(10,-15);
	double denom = 0.0;
	double dwP,dhP,ddP;
	double dwM,dhM,ddM;

	// std::cout<<u.xdim; // "physical" horizontal limit (x direction)
	// std::cout<<u.ydim; // "physical" horizontal limit (y direction)
	// std::cout<<u.zdim; // "physical" horizontal limit (z direction)
	// Guaranteeing the array of weights exists and initializes it

	//
	// Computing the gradient of the total variation function
	//
//std::cout<<formatString("\033[1;31mVolumeSize:\033[0m %dx%dx%d",u.xdim,u.ydim,u.zdim);

	memset(v.data,0,v.xdim*v.ydim*v.zdim*sizeof(double));
	for(uint k=0; k < u.zdim;k++){        // Depth
		for(uint j=0;j < u.ydim;j++){     // Height
			for(uint i=0;i < u.xdim;i++){ // Width

				if(dAij(MPregionsMask,k,i)){ // region P
//std::cout<< formatString("\033[1;33mregion P, valor: %d \033[0m\n %d, %d, %d",dAij(MPregionsMask, k, i), i, j, k);
					//
					// First Case
					// (d/d x_i) of TV
					//
					if(i<(u.xdim-1) && j<(u.ydim-1) && k<(u.zdim-1)){
						dwP = u.data[P(i,j,k)] - u.data[P(i+1,j,k)];
						dhP = u.data[P(i,j,k)] - u.data[P(i,j+1,k)];
						ddP = u.data[P(i,j,k)] - u.data[P(i,j,k+1)];
						//Computing the denominator
						denom = sqrt(dwP*dwP + dhP*dhP + ddP*ddP);
						if(denom > ZERO)
							v.data[P(i,j,k)] += piP.data[P(i,j,k)]*(3*u.data[P(i,j,k)] -
									u.data[P(i+1,j,k)] -
									u.data[P(i,j+1,k)] -
									u.data[P(i,j,k+1)])/denom;
					}
					//
					// Second Case
					// (d/d x_r) of TV (x_r is the base and not x_i)
					//
					if(i>0 && i<u.xdim && j<(u.ydim-1) && k<(u.zdim-1)){
						dwP = u.data[P(i-1,j,k)] - u.data[P(i,j,k)];
						dhP = u.data[P(i-1,j,k)] - u.data[P(i-1,j+1,k)];
						ddP = u.data[P(i-1,j,k)] - u.data[P(i-1,j,k+1)];
						//Computing the denominator
						denom = sqrt(dwP*dwP + dhP*dhP + ddP*ddP);
						if(denom > ZERO)
							v.data[P(i,j,k)] += piP.data[P(i,j,k)]*(u.data[P(i,j,k)] -
									u.data[P(i-1,j,k)])/denom;
					}
					//
					// Third Case
					// (d/d x_u) of TV (x_u is the base and not x_i)
					//
					if(i<(u.xdim-1) && j>0 && j<u.ydim && k<(u.zdim-1)){
						dwP = u.data[P(i,j-1,k)] - u.data[P(i+1,j-1,k)];
						dhP = u.data[P(i,j-1,k)] - u.data[P(i,j,k)];
						ddP = u.data[P(i,j-1,k)] - u.data[P(i,j-1,k+1)];
						//Computing the denominator
						denom = sqrt(dwP*dwP + dhP*dhP + ddP*ddP);
						if(denom > ZERO)
							v.data[P(i,j,k)] += piP.data[P(i,j,k)]*(u.data[P(i,j,k)] -
									u.data[P(i,j-1,k)])/denom;
					}
					//
					// Fourth Case
					// (d/d x_b) of TV (x_b is the base and not x_i)
					//
					if(i<(u.xdim-1) && j<(u.ydim-1) && k>0 && k<u.zdim){
						dwP = u.data[P(i,j,k-1)] - u.data[P(i+1,j,k-1)];
						dhP = u.data[P(i,j,k-1)] - u.data[P(i,j+1,k-1)];
						ddP = u.data[P(i,j,k-1)] - u.data[P(i,j,k)];
						//Computing the denominator
						denom = sqrt(dwP*dwP + dhP*dhP + ddP*ddP);
						if(denom > ZERO)
							v.data[P(i,j,k)] += piP.data[P(i,j,k)]*(u.data[P(i,j,k)] -
									u.data[P(i,j,k-1)])/denom;
					}
				}

				else{ //region M
//std::cout<< formatString("\033[1;32mregion M, valor: %d \033[0m\n %d, %d, %d",dAij(MPregionsMask, k, i), i, j, k);
					//
					// First Case
					// (d/d x_i) of TV
					//
					if(i<(u.xdim-1) && j<(u.ydim-1) && k<(u.zdim-1)){
						dwM = u.data[P(i,j,k)] - u.data[P(i+1,j,k)];
						dhM = u.data[P(i,j,k)] - u.data[P(i,j+1,k)];
						ddM = u.data[P(i,j,k)] - u.data[P(i,j,k+1)];
						//Computing the denominator
						denom = sqrt(dwM*dwM + dhM*dhM + ddM*ddM);
						if(denom > ZERO)
							v.data[P(i,j,k)] += muM.data[P(i,j,k)]*(3*u.data[P(i,j,k)] -
									u.data[P(i+1,j,k)] -
									u.data[P(i,j+1,k)] -
									u.data[P(i,j,k+1)])/denom;
					}
					//
					// Second Case
					// (d/d x_r) of TV (x_r is the base and not x_i)
					//
					if(i>0 && i<u.xdim && j<(u.ydim-1) && k<(u.zdim-1)){
						dwM = u.data[P(i-1,j,k)] - u.data[P(i,j,k)];
						dhM = u.data[P(i-1,j,k)] - u.data[P(i-1,j+1,k)];
						ddM = u.data[P(i-1,j,k)] - u.data[P(i-1,j,k+1)];
						//Computing the denominator
						denom = sqrt(dwM*dwM + dhM*dhM + ddM*ddM);
						if(denom > ZERO)
							v.data[P(i,j,k)] += muM.data[P(i,j,k)]*(u.data[P(i,j,k)] -
									u.data[P(i-1,j,k)])/denom;
					}
					//
					// Third Case
					// (d/d x_u) of TV (x_u is the base and not x_i)
					//
					if(i<(u.xdim-1) && j>0 && j<u.ydim && k<(u.zdim-1)){
						dwM = u.data[P(i,j-1,k)] - u.data[P(i+1,j-1,k)];
						dhM = u.data[P(i,j-1,k)] - u.data[P(i,j,k)];
						ddM = u.data[P(i,j-1,k)] - u.data[P(i,j-1,k+1)];
						//Computing the denominator
						denom = sqrt(dwM*dwM + dhM*dhM + ddM*ddM);
						if(denom > ZERO)
							v.data[P(i,j,k)] += muM.data[P(i,j,k)]*(u.data[P(i,j,k)] -
									u.data[P(i,j-1,k)])/denom;
					}
					//
					// Fourth Case
					// (d/d x_b) of TV (x_b is the base and not x_i)
					//
					if(i<(u.xdim-1) && j<(u.ydim-1) && k>0 && k<u.zdim){
						dwM = u.data[P(i,j,k-1)] - u.data[P(i+1,j,k-1)];
						dhM = u.data[P(i,j,k-1)] - u.data[P(i,j+1,k-1)];
						ddM = u.data[P(i,j,k-1)] - u.data[P(i,j,k)];
						//Computing the denominator
						denom = sqrt(dwM*dwM + dhM*dhM + ddM*ddM);
						if(denom > ZERO)
							v.data[P(i,j,k)] += muM.data[P(i,j,k)]*(u.data[P(i,j,k)] -
									u.data[P(i,j,k-1)])/denom;
					}

				} // end region M

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

void atv::createMask(const size_t xdim, const size_t ydim, const size_t zdim) {
	MultidimArray<int> oriMask(zdim,xdim);
	oriMask.initConstant(1);
	double Cx = (double)xdim / 2.0;
	double Cy = (double)zdim / 2.0;
	double angle= 90. - maxA;
	angle = DEG2RAD(angle);
	double m = tan(angle);
	for (int y=0; y<zdim; ++y){
		for (int x=0; x<xdim; ++x){
			if (((x >= (int) (m * (double)(y - Cy) + Cx))
					&& (x <= (int) (-m * (double)(y - Cy) + Cx)))
					|| ((x <= (int) (m * (double)(y - Cy) + Cx))
							&& (x >= (int) (-m * (double) (y - Cy) + Cx))))
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


/*prepare kernel with propper size for Fourier*/
MultidimArray<double> kernelResized(const MultidimArray<double>& orKernel, const MultidimArray<double>& v){
	MultidimArray<double> tempKernel;
	tempKernel.setXmippOrigin();
	tempKernel.initZeros(v);

	int Cx = (int)((double)v.xdim / 2.0);
	int Cy = (int)((double)v.ydim / 2.0);
	int Cz = (int)((double)v.zdim / 2.0);
	int a = 0.5 * orKernel.xdim;


	for (int z = -a; z <= a; ++z) {        // Depth
		for (int y = -a; y <= a; ++y) {     // Height
			for (int x = -a; x <= a; ++x) { // Width
				dAkij(tempKernel,Cz+z,Cy+y,Cx+x) = dAkij(orKernel,z+a,y+a,x+a);
			}
		}
	}

	return tempKernel;
}

/**
 **
 ** Computes the weighting vector
 **
 */
void atv::init(MultidimArray<double>& v,const double sigmaG, const unsigned short sizeG,const double sigmaH, const unsigned short sizeH,double Amin,double Amax)
{
	minA = Amin;
	maxA = Amax;
std::cout<<formatString("\033[1;31mmin-max angles:\033[0m %.2f, %.2f\n",minA,maxA);
	createMask(v.xdim, v.ydim, v.zdim);

//	// Guaranteeing the array of weights exists and initializes it
//	if(w.getArrayPointer() == NULL)
//		w.resize(v.zdim,v.ydim,v.xdim);
//	memset(w.data,0,w.xdim*w.ydim*w.zdim*sizeof(double));

	// weights for anisotropic
	if(muM.getArrayPointer() == NULL)
		muM.resize(v.zdim,v.ydim,v.xdim);
	memset(muM.data,0,muM.xdim*muM.ydim*muM.zdim*sizeof(double));

	if(piP.getArrayPointer() == NULL)
		piP.resize(v.zdim,v.ydim,v.xdim);
	memset(piP.data,0,piP.xdim*piP.ydim*piP.zdim*sizeof(double));

	// kernels for Fourier
	MultidimArray<double> tempG, tempH;

	if(G.getArrayPointer() == NULL)
		G.resize(sizeG,sizeG,sizeG);
	memset(G.data,0,sizeG*sizeG*sizeG*sizeof(double));
	GaussKernel(G, sigmaG, sizeG);
	tempG = kernelResized(G,v);

	if(H.getArrayPointer() == NULL)
		H.resize(sizeH,sizeH,sizeH);
	memset(H.data,0,sizeH*sizeH*sizeH*sizeof(double));
	GaussKernel(H, sigmaH, sizeH);
	tempH = kernelResized(H,v);

	// Fourier transform for Gaussian kernels
	transformer.FourierTransform(tempG, G_fourier, true);
	transformer.FourierTransform(tempH, H_fourier, true);

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

/**
 **
 ** Computes the weighting vector
 **
 */
void atv::update(MultidimArray<double>& v)
{
#define P(i,j,k)(i + j*v.xdim + k*v.xdim*v.ydim)
	double dwP,dhP,ddP;
	double dwM,dhM,ddM;

	// convolve v with gaussian kernels
	MultidimArray<double> filt_vP, filt_vM;
	MultidimArray<std::complex<double> > fourierV;
	transformer.FourierTransform(v, fourierV, true);

	MultidimArray<std::complex<double> > fourierFilt_P, fourierFilt_M;
	fourierFilt_P.resizeNoCopy(fourierV);
	fourierFilt_M.resizeNoCopy(fourierV);

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(fourierV){
		dAkij(fourierFilt_P, k, i, j) = dAkij(fourierV, k, i, j) * dAkij(G_fourier, k, i, j);
		dAkij(fourierFilt_M, k, i, j) = dAkij(fourierV, k, i, j) * dAkij(H_fourier, k, i, j);
	}

	InverseFourierTransform(fourierFilt_P, filt_vP);
	InverseFourierTransform(fourierFilt_M, filt_vM);

	for(uint k=0; k < v.zdim;k++){        // Depth
		for(uint j=0;j < v.ydim;j++){     // Height
			for(uint i=0;i < v.xdim;i++){ // Width

				if(dAij(MPregionsMask,k,i)){ // region P
					dwP = ((i+1) < v.xdim) ? (filt_vP.data[P(i,j,k)] - filt_vP.data[P(i+1,j,k)]) : 0.0;
					dhP = ((j+1) < v.ydim) ? (filt_vP.data[P(i,j,k)] - filt_vP.data[P(i,j+1,k)]) : 0.0;
					ddP = ((k+1) < v.zdim) ? (filt_vP.data[P(i,j,k)] - filt_vP.data[P(i,j,k+1)]) : 0.0;
					piP.data[P(i,j,k)] = 1.0/(sqrt(dwP*dwP + dhP*dhP + ddP*ddP) + eps);
				}
				else{ //region M
					dwM = ((i+1) < v.xdim) ? (filt_vM.data[P(i,j,k)] - filt_vM.data[P(i+1,j,k)]) : 0.0;
					dhM = ((j+1) < v.ydim) ? (filt_vM.data[P(i,j,k)] - filt_vM.data[P(i,j+1,k)]) : 0.0;
					ddM = ((k+1) < v.zdim) ? (filt_vM.data[P(i,j,k)] - filt_vM.data[P(i,j,k+1)]) : 0.0;
					muM.data[P(i,j,k)] = 1.0/(sqrt(dwM*dwM + dhM*dhM + ddM*ddM) + eps);
				}
			}
		}
	}
#undef P
}
#undef DEBUG
