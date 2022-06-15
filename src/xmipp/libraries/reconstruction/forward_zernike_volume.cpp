/***************************************************************************
 *
 * Authors:    Carlos Oscar             coss@cnb.csic.es
 * 			   David Herreros Calero    dherreros@cnb.csic.es
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

/***************************************************************************

MatLab code to compute symbolic expression (in C format) for any degree/order
of the bases (normalized Zernikes + Spherical Harmonics)

clear, clc;

N=15; L=15;

% R_vec = ZernPoly(L,N,1);
% vpa(poly2sym(flip(R_vec)), 4)

if rem(L, 2) == 0
    start = 0;
else
    start = 1;
end

syms r
for l=start:2:N
    R_vec = ZernPoly(l,N,1);
    R = ccode(vpa(poly2sym(flip(R_vec), r), 4))
end

for m = -L:1:L
    S = ccode(realSphericalHarmonic(L,m))
end




function plm = assocLegendre(l,m)
    % get symbolic associated legendre function P_lm(x) based on
    % legendre function P_l(x)
    
    syms costh; %% x represents the cos(theta)
    % get symbolic form of Legendre function P_l(x)
    leg = legendreP(l,costh);
    % differentiate it m times
    legDiff = diff(leg,costh,m);
    % calculate associated legendre function P_lm(x)
    plm = ((-1)^m)*((1 - costh^2)^(m/2))*legDiff;
end

function ylm = sphericalHarmonic(l,m)
    % get symbolic spherical harmonic Y_lm(x) based on
    % associated legendre function P_lm(x)
    
    a1 = (2*l+1)/(4*pi);
    a2 = factorial(l-abs(m))/factorial(l+abs(m));
    n = sqrt(a1*a2); %Normalization Lengendre Polynomials
    ylm = vpa(n * assocLegendre(l,m), 4);
end

function rylm = realSphericalHarmonic(l,m)
    % get symbolic real spherical harmonic Yr_lm(x) based on
    % associated legendre function P_lm(x)
    
    syms sinth %% y represents the sin(|m|*phi)
    syms cosph %% z represents the cos(m*phi)
    
    if m < 0
        a1 = (2*l+1)/(4*pi);
        a2 = factorial(l-abs(m))/factorial(l+abs(m));
        n = sqrt(a1*a2); %Normalization Lengendre Polynomials
        rylm = ((-1)^m)* sqrt(2) * n * assocLegendre(l,abs(m)) * sinth;
    elseif m == 0
        a1 = (2*l+1)/(4*pi);
        n = sqrt(a1);
        rylm = n * assocLegendre(l,m);
    else
        a1 = (2*l+1)/(4*pi);
        a2 = factorial(l-abs(m))/factorial(l+abs(m));
        n = sqrt(a1*a2); %Normalization Lengendre Polynomials
        rylm = ((-1)^m)* sqrt(2) * n * assocLegendre(l,m) * cosph;
    end
    
    rylm = vpa(rylm, 4);
      
end

 ***************************************************************************/

#include <fstream>
#include <iterator>
#include <numeric>
#include "forward_zernike_volume.h"
#include "data/fourier_filter.h"
#include "data/normalize.h"
#include "data/mask.h"

// Params definition =======================================================
// -i ---> V2 (paper) / -r --> V1 (paper)
void ProgForwardZernikeVol::defineParams() {
	addUsageLine("Compute the deformation that properly fits two volumes using spherical harmonics");
	addParamsLine("   -i <volume>                         : Volume to deform");
	addParamsLine("   -r <volume>                         : Reference volume");
	addParamsLine("  [-o <volume=\"\">]                   : Output volume which is the deformed input volume");
	addParamsLine("  [--maski <m=\"\">]            		  : Input volume mask");
	addParamsLine("  [--maskr <m=\"\">]            		  : Reference volume mask");
	addParamsLine("  [--oroot <rootname=\"Volumes\">]     : Root name for output files");
	addParamsLine("                                       : By default, the input file is rewritten");
	addParamsLine("  [--analyzeStrain]                    : Save the deformation of each voxel for local strain and rotation analysis");
	addParamsLine("  [--optimizeRadius]                   : Optimize the radius of each spherical harmonic");
	addParamsLine("  [--l1 <l1=3>]                        : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                        : Harmonical depth of the deformation=1,2,3,...");
	addParamsLine("  [--regularization <l=0.00025>]       : Regularization weight");
	addParamsLine("  [--Rmax <r=-1>]                      : Maximum radius for the transformation");
	addParamsLine("  [--blobr <b=4>]                      : Blob radius for forward mapping splatting");
	addParamsLine("  [--step <step=1>]                    : Voxel index step");
	addParamsLine("  [--clnm <metadata_file=\"\">]        : List of deformation coefficients");
	addExampleLine("xmipp_forward_zernike_volume -i vol1.vol -r vol2.vol -o vol1DeformedTo2.vol");
}

// Read arguments ==========================================================
void ProgForwardZernikeVol::readParams() {
    std::string aux;
	fnVolI = getParam("-i");
	fnVolR = getParam("-r");
	fnMaskR = getParam("--maskr");
	fnMaskI = getParam("--maski");
	L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	fnRoot = getParam("--oroot");

	VI.read(fnVolI);
	VR.read(fnVolR);
	VI().setXmippOrigin();
	VR().setXmippOrigin();

	VR2().initZeros(VR());
	VR2().setXmippOrigin();

	fnVolOut = getParam("-o");
	if (fnVolOut=="")
		fnVolOut=fnVolI;
	analyzeStrain=checkParam("--analyzeStrain");
	optimizeRadius=checkParam("--optimizeRadius");
	lambda = getDoubleParam("--regularization");
	Rmax = getDoubleParam("--Rmax");
	if (Rmax<0)
		Rmax=XSIZE(VI())/2;
	applyTransformation = false;

	// Read Reference mask if avalaible (otherwise sphere of radius RmaxDef is used)
	Mask mask;
	mask.type = BINARY_CIRCULAR_MASK;
	mask.mode = INNER_MASK;
	if (fnMaskI != "") {
		Image<double> auxi, auxr;
		auxi.read(fnMaskI);
		typeCast(auxi(), V_maski);
		V_maski.setXmippOrigin();
		double Rmax2 = Rmax*Rmax;
		for (int k=STARTINGZ(V_maski); k<=FINISHINGZ(V_maski); k++) {
			for (int i=STARTINGY(V_maski); i<=FINISHINGY(V_maski); i++) {
				for (int j=STARTINGX(V_maski); j<=FINISHINGX(V_maski); j++) {
					double r2 = k*k + i*i + j*j;
					if (r2>=Rmax2)
					{
						A3D_ELEM(V_maski,k,i,j) = 0;
					}
				}
			}
		}
	}
	else {
		mask.R1 = Rmax;
		mask.generate_mask(VR());
		V_maski = mask.get_binary_mask();
		V_maski.setXmippOrigin();
	}

	if (fnMaskR != "") {
		Image<double> auxr;
		auxr.read(fnMaskR);
		typeCast(auxr(), V_maskr);
		V_maskr.setXmippOrigin();
		double Rmax2 = Rmax*Rmax;
		for (int k=STARTINGZ(V_maskr); k<=FINISHINGZ(V_maskr); k++) {
			for (int i=STARTINGY(V_maskr); i<=FINISHINGY(V_maskr); i++) {
				for (int j=STARTINGX(V_maskr); j<=FINISHINGX(V_maskr); j++) {
					double r2 = k*k + i*i + j*j;
					if (r2>=Rmax2)
					{
						A3D_ELEM(V_maskr,k,i,j) = 0;
					}
				}
			}
		}
	}
	else {
		mask.R1 = Rmax;
		mask.generate_mask(VR());
		V_maskr = mask.get_binary_mask();
		V_maskr.setXmippOrigin();
	}

	Mask mask2;
	mask2.type = BINARY_CIRCULAR_MASK;
	mask2.mode = INNER_MASK;
	mask2.R1 = Rmax;
	mask2.generate_mask(VR());
	V_mask2 = mask2.get_binary_mask();
	V_mask2.setXmippOrigin();
	// Image<double> aux3;
	// aux3.read("mask_closed.mrc");
	// typeCast(aux3(), V_mask2);
	// V_mask2.setXmippOrigin();

	// For debugging purposes
	// Image<int> save;
	// save() = V_mask;
	// save.write("Mask.vol");

    loop_step = getIntParam("--step");

	// Blob
	blob_r = getDoubleParam("--blobr");
	blob.radius = blob_r;   // Blob radius in voxels
	blob.order  = 2;        // Order of the Bessel function
    blob.alpha  = 3.6;      // Smoothness parameter

	fn_sph = getParam("--clnm");
	if (fn_sph != "") {
		std::string line;
		line = readNthLine(0);
		std::vector<double> basisParams = string2vector(line);
		L1 = basisParams[0];
		L2 = basisParams[1];
		Rmax = basisParams[2];
		line = readNthLine(1);
		vec = string2vector(line);
	}

	sigma4 = 2 * blob_r;
	gaussianProjectionTable.resize(CEIL(sigma4 * sqrt(2) * 1000));
	FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
	gaussianProjectionTable(i) = gaussian1D(i / 1000.0, blob_r);
	gaussianProjectionTable *= gaussian1D(0, blob_r);
	gaussianProjectionTable2 = gaussianProjectionTable;
	gaussianProjectionTable2 *= gaussianProjectionTable;
}

// Show ====================================================================
void ProgForwardZernikeVol::show() {
	if (verbose==0)
		return;
	std::cout
	        << "Volume to deform:     " << fnVolI         << std::endl
			<< "Reference volume:     " << fnVolR         << std::endl
			<< "Output volume:        " << fnVolOut       << std::endl
			<< "Reference mask:       " << fnMaskR        << std::endl
			<< "Zernike Degree:       " << L1             << std::endl
			<< "SH Degree:            " << L2             << std::endl
			<< "Save deformation:     " << analyzeStrain  << std::endl
			<< "Regularization:       " << lambda         << std::endl
			<< "Step:                 " << loop_step      << std::endl
			<< "Blob radius:          " << blob_r         << std::endl
	;

}

template<bool SAVE_DEFORMATION>
void ProgForwardZernikeVol::deformVolume() {
	size_t idxY0=VEC_XSIZE(clnm)/3;
	size_t idxZ0=2*idxY0;
    const auto &mVI = VI();
	auto &mVO = VO();
	auto &mVO2 = VO2();
	mVO.initZeros(mVI);
	mVO2.initZeros(mVI);
    size_t vec_idx = 0;
	const double iRmax = 1.0 / Rmax;
	auto stepsMask = std::vector<size_t>();
	if (fn_sph == "") {
		for (size_t idx = 0; idx < idxY0; idx++)
		{
			if (1 == VEC_ELEM(steps_cp, idx))
			{
				stepsMask.emplace_back(idx);
			}
		}
	}
	else {
		for (size_t idx = 0; idx < idxY0; idx++)
		{
			stepsMask.emplace_back(idx);
		}
	}
	sumVD = 0.0;
	deformation = 0.0;
	double Ncount = 0.0;

	// int startz = STARTINGZ(mVI) + rand() % loop_step + 1;
	// int starty = STARTINGY(mVI) + rand() % loop_step + 1;
	// int startx = STARTINGX(mVI) + rand() % loop_step + 1;

	int startz = STARTINGZ(mVI);
	int starty = STARTINGY(mVI);
	int startx = STARTINGX(mVI);

    for (int k=startz; k<=FINISHINGZ(mVI); k+=loop_step) {
        for (int i=starty; i<=FINISHINGY(mVI); i+=loop_step) {
            for (int j=startx; j<=FINISHINGX(mVI); j+=loop_step) {
				if (A3D_ELEM(V_maski,k,i,j) == 1) {
					double gx=0.0, gy=0.0, gz=0.0;
					double k2=k*k;
					double kr=k*iRmax;
					double k2i2=k2+i*i;
					double ir=i*iRmax;
					double r2=k2i2+j*j;
					double jr=j*iRmax;
					double rr=sqrt(r2)*iRmax;
					for (auto idx : stepsMask) {
						auto l1 = VEC_ELEM(vL1,idx);
						auto n = VEC_ELEM(vN,idx);
						auto l2 = VEC_ELEM(vL2,idx);
						auto m = VEC_ELEM(vM,idx);
						auto zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
						if (rr>0 || l2==0) {
							gx += VEC_ELEM(clnm,idx)        *(zsph);
							gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
							gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
						}
					}
					// XX(p) += gx; YY(p) += gy; ZZ(p) += gz;
					// XX(pos) = 0.0; YY(pos) = 0.0; ZZ(pos) = 0.0;
					// for (size_t i = 0; i < R.mdimy; i++)
					// 	for (size_t j = 0; j < R.mdimx; j++)
					// 		VEC_ELEM(pos, i) += MAT_ELEM(R, i, j) * VEC_ELEM(p, j);

					auto pos = std::array<double, 3>{};
					pos[0] = j + gx;
					pos[1] = i + gy;
					pos[2] = k + gz;
					
					double voxel_mVI = A3D_ELEM(mVI,k,i,j);
					splattingAtPos(pos, voxel_mVI, mVO, mVO2);

					if (SAVE_DEFORMATION) {
						Gx(k,i,j) = gx;
						Gy(k,i,j) = gy;
						Gz(k,i,j) = gz;
					}

					// if (!mVI.outside(pos[2], pos[1], pos[0]))
						// sumVD += splatVal(pos, voxel_mVI, mVI);
					deformation += gx*gx+gy*gy+gz*gz;
					Ncount++;
				}
            }
        }
    }
	deformation = sqrt(deformation/Ncount);
}

void ProgForwardZernikeVol::splattingAtPos(std::array<double, 3> r, double weight, MultidimArray<double> &mVO1, MultidimArray<double> &mVO2) {
	// Find the part of the volume that must be updated
	double x_pos = r[0];
	double y_pos = r[1];
	double z_pos = r[2];
	int k0 = XMIPP_MAX(FLOOR(z_pos - blob_r), STARTINGZ(mVO1));
	int kF = XMIPP_MIN(CEIL(z_pos + blob_r), FINISHINGZ(mVO1));
	int i0 = XMIPP_MAX(FLOOR(y_pos - blob_r), STARTINGY(mVO1));
	int iF = XMIPP_MIN(CEIL(y_pos + blob_r), FINISHINGY(mVO1));
	int j0 = XMIPP_MAX(FLOOR(x_pos - blob_r), STARTINGX(mVO1));
	int jF = XMIPP_MIN(CEIL(x_pos + blob_r), FINISHINGX(mVO1));
	// int k0 = XMIPP_MAX(FLOOR(z_pos - sigma4), STARTINGZ(mVO1));
	// int kF = XMIPP_MIN(CEIL(z_pos + sigma4), FINISHINGZ(mVO1));
	// int i0 = XMIPP_MAX(FLOOR(y_pos - sigma4), STARTINGY(mVO1));
	// int iF = XMIPP_MIN(CEIL(y_pos + sigma4), FINISHINGY(mVO1));
	// int j0 = XMIPP_MAX(FLOOR(x_pos - sigma4), STARTINGX(mVO1));
	// int jF = XMIPP_MIN(CEIL(x_pos + sigma4), FINISHINGX(mVO1));
	int size = gaussianProjectionTable.size();
	for (int k = k0; k <= kF; k++)
	{
		double k2 = (z_pos - k) * (z_pos - k);
		for (int i = i0; i <= iF; i++)
		{
			double y2 = (y_pos - i) * (y_pos - i);
			for (int j = j0; j <= jF; j++)
			{
				double mod = sqrt((x_pos - j) * (x_pos - j) + y2 + k2);
				double didx = mod * 1000;
				int idx = ROUND(didx);
				// if (idx < size)
				// {
				// 	double gw = gaussianProjectionTable.vdata[idx];
				// 	A3D_ELEM(mVO1, k, i, j) += weight * gw;
				// 	A3D_ELEM(mVO2, k, i, j) += weight * gw;
				// }
				A3D_ELEM(mVO1,k, i, j) += weight * kaiser_value(mod, blob.radius, blob.alpha, blob.order);
				A3D_ELEM(mVO2,k, i, j) += weight * kaiser_value(mod, 2, blob.alpha, blob.order);
			}
		}
	}
}

double ProgForwardZernikeVol::splatVal(std::array<double, 3> r, double weight, const MultidimArray<double> &mV) {
	// Find the part of the volume that must be updated
	double x_pos = r[0];
	double y_pos = r[1];
	double z_pos = r[2];
	double val = 0.0;
	int k0 = XMIPP_MAX(FLOOR(z_pos - blob_r), STARTINGZ(mV));
	int kF = XMIPP_MIN(CEIL(z_pos + blob_r), FINISHINGZ(mV));
	int i0 = XMIPP_MAX(FLOOR(y_pos - blob_r), STARTINGY(mV));
	int iF = XMIPP_MIN(CEIL(y_pos + blob_r), FINISHINGY(mV));
	int j0 = XMIPP_MAX(FLOOR(x_pos - blob_r), STARTINGX(mV));
	int jF = XMIPP_MIN(CEIL(x_pos + blob_r), FINISHINGX(mV));
	for (int k = k0; k <= kF; k++)
	{
		double k2 = (z_pos - k) * (z_pos - k);
		for (int i = i0; i <= iF; i++)
		{
			double y2 = (y_pos - i) * (y_pos - i);
			for (int j = j0; j <= jF; j++)
			{
				double mod = sqrt((x_pos - j) * (x_pos - j) + y2 + k2);
				val += weight * kaiser_value(mod, blob.radius, blob.alpha, blob.order);
			}
		}
	}
	return val;
}

// Distance function =======================================================
// #define DEBUG
double ProgForwardZernikeVol::distance(double *pclnm)
{
	const MultidimArray<double> &mVR=VR();
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];

	if (saveDeformation)
    	deformVolume<true>();
	else
		deformVolume<false>();

	if (applyTransformation)
		VO.write(fnVolOut);
	
	// MultidimArray<int> bg_mask;
	// bg_mask.resizeNoCopy(VO().zdim, VO().ydim, VO().xdim);
    // bg_mask.setXmippOrigin();
	// normalize_Robust(VO(), bg_mask, true);

	// double val = 0.0;
	// rmsd(VO(), VR(), val);
	// return val;

	// double corr1 = -0.9 * correlationIndex(VO(), VR(), &V_mask2);
	// double corr2 = -0.1 * correlationIndex(VO2(), VR2(), &V_mask2);
	// return corr1+corr2+lambda*(deformation);

	double corr1 = -correlationIndex(VO(), VR(), &V_mask2);
	return corr1+lambda*(deformation);

	// volume2Mask(VO(), 0.01);

	// double massDiff=std::abs(sumVI-sumVD)/sumVI;
	// double corr2 = -0.25*correlationIndex(VO(), VR(), &V_mask2);
	// return corr1+corr2+lambda*(deformation);
}
#undef DEBUG

double continuousZernikeCostVol(double *p, void *vprm)
{
    ProgForwardZernikeVol *prm=(ProgForwardZernikeVol *) vprm;
	return prm->distance(p);
}

// Run =====================================================================
void ProgForwardZernikeVol::run() {
	// Matrix1D<int> nh;
	// nh.resize(depth+2);
	// nh.initConstant(0);
	// nh(1)=1;

	VO().initZeros(VR());
	VO().setXmippOrigin();
	VO2().initZeros(VR());
	VO2().setXmippOrigin();

	saveDeformation=false;
	sumVI = 0.0;
	// Numsph(nh);

		// This filtered version of the input volume is needed to interpolate more accurately VD
	// MultidimArray<int> bg_mask;
	// bg_mask.resizeNoCopy(VI().zdim, VI().ydim, VI().xdim);
    // bg_mask.setXmippOrigin();
	// normalize_Robust(VI(), bg_mask, true);
	// // auxI.write("input_filt.vol");
	// bg_mask *= 0;
	// normalize_Robust(VR(), bg_mask, true);
	// auxR.write("ref_filt.vol");

	// volume2Mask(VR(), 0.01);
	// volume2Mask(VI(), 0.01);
	MultidimArray<double> blobV, blobV2;
	volume2Blobs(blobV, blobV2, VR(), V_maskr);
	VR() = blobV;
	VR2() = blobV2;
	VR.write("reference_blob.vol");
	VR2.write("reference_blob2.vol");
	// volume2Mask(VR(), 0.01);

	// volume2Blobs(blobV, blobV2, VI(), V_maski);
	// VI() = blobV;
	// VI.write("input_blob.vol");
	// volume2Mask(VI(), 0.01);
	
	// Total Volume Mass (Inside Mask)
	for (int k = STARTINGZ(VI()); k <= FINISHINGZ(VI()); k += loop_step)
	{
		for (int i = STARTINGY(VI()); i <= FINISHINGY(VI()); i += loop_step)
		{
			for (int j = STARTINGX(VI()); j <= FINISHINGX(VI()); j += loop_step)
			{
				if (A3D_ELEM(V_maski, k, i, j) == 1)
				{
					auto pos = std::array<double, 3>{};
					pos[0] = j;
					pos[1] = i;
					pos[2] = k;
					sumVI += splatVal(pos, A3D_ELEM(VI(), k, i, j), VI());
				}
			}
		}
	}

    Matrix1D<double> steps, x;
	vecSize = 0;
	numCoefficients(L1,L2,vecSize);
	size_t totalSize = 3*vecSize;
	fillVectorTerms(L1,L2);
	clnm.resize(totalSize);
	x.initZeros(totalSize);

	int start=0;

	if (fn_sph != "") {
		for (int idx=0; idx<vec.size(); idx++){
			clnm[idx] = vec[idx];
			x[idx] = vec[idx];
		}
		start = L2;
	}

    for (int h=start;h<=L2;h++)
    {
    	// L = nh(h+1);
    	// prevL = nh(h);
    	// prevsteps=steps;
		steps.clear();
    	steps.initZeros(totalSize);
		minimizepos(L1,h,steps);
		steps_cp = steps;

    	std::cout<<std::endl;
    	std::cout<<"-------------------------- Basis Degrees: ("<<L1<<","<<h<<") --------------------------"<<std::endl;
        int iter;
        double fitness;
        powellOptimizer(x, 1, totalSize, &continuousZernikeCostVol, this,
		                0.01, fitness, iter, steps, true);

        std::cout<<std::endl;
        std::cout << "Deformation " << deformation << std::endl;
        std::ofstream deformFile;
        deformFile.open (fnRoot+"_deformation.txt");
        deformFile << deformation;
        deformFile.close();

// #define DEBUG
#ifdef DEBUG
	Image<double> save;
	save() = VI();
	save.write(fnRoot+"_PPPIdeformed.vol");
	save()-=VR();
	save.write(fnRoot+"_PPPdiff.vol");
	save()=VR();
	save.write(fnRoot+"_PPPR.vol");
	std::cout << "Error=" << deformation << std::endl;
	std::cout << "Press any key\n";
	char c; std::cin >> c;
#endif

    }
    applyTransformation=true;
	// x.write(fnRoot+"_clnm.txt");
	Matrix1D<double> degrees;
	degrees.initZeros(3);
	VEC_ELEM(degrees,0) = L1;
	VEC_ELEM(degrees,1) = L2;
	VEC_ELEM(degrees,2) = Rmax;
	writeVector(fnRoot+"_clnm.txt", degrees, false);
	writeVector(fnRoot+"_clnm.txt", x, true);
    if (analyzeStrain)
    {
    	saveDeformation=true;
    	Gx().initZeros(VR());
    	Gy().initZeros(VR());
    	Gz().initZeros(VR());
    	Gx().setXmippOrigin();
    	Gy().setXmippOrigin();
    	Gz().setXmippOrigin();
    }

    distance(x.adaptForNumericalRecipes()); // To save the output volume

#ifdef DEBUG
		Image<double> save;
		save() = Gx();
		save.write(fnRoot+"_PPPGx.vol");
		save() = Gy();
		save.write(fnRoot+"_PPPGy.vol");
		save() = Gz();
		save.write(fnRoot+"_PPPGz.vol");
#endif

    if (analyzeStrain)
    	computeStrain();
}

// // Copy Vectors ============================================================
// void ProgForwardZernikeVol::copyvectors(Matrix1D<double> &oldvect,Matrix1D<double> &newvect)
// {
// 	size_t groups = 4;
// 	size_t olditems = VEC_XSIZE(oldvect)/groups;
// 	size_t newitems = VEC_XSIZE(newvect)/groups;
// 	for (int g=0;g<groups;g++)
// 	{
// 		for (int i=0;i<olditems;i++)
// 			{
// 			    newvect(g*newitems+i) = oldvect(g*olditems+i);
// 			}
// 	}
// }

// // Minimize Positions ======================================================
// void ProgForwardZernikeVol::minimizepos(Matrix1D<double> &vectpos, Matrix1D<double> &prevpos)
// {
// 	size_t groups = 4;
// 	size_t olditems = VEC_XSIZE(prevpos)/groups;
// 	size_t newitems = VEC_XSIZE(vectpos)/groups;
// 	for (int i=0;i<olditems;i++)
// 	{
// 		vectpos(3*newitems+i) = 0;
// 	}
// 	if (!optimizeRadius)
// 	{
// 		for (int j=olditems;j<newitems;j++)
// 		{
// 			vectpos(3*newitems+j) = 0;
// 		}
// 	}
// }

// Minimize Positions ======================================================
// void ProgForwardZernikeVol::minimizepos(Matrix1D<double> &vectpos, int &current_l2)
// {
// 	size_t currentSize = std::floor((4+4*L1+std::pow(L1,2))/4)*std::pow(current_l2+1,2);
// 	for (int i=0;i<currentSize;i++)
// 	{
// 		VEC_ELEM(vectpos,i) = 1;
// 		VEC_ELEM(vectpos,i+vecSize) = 1;
// 		VEC_ELEM(vectpos,i+2*vecSize) = 1;
// 	}
// }

void ProgForwardZernikeVol::minimizepos(int L1, int l2, Matrix1D<double> &steps)
{
    int size = 0;
	numCoefficients(L1,l2,size);
    int totalSize = steps.size()/3;
    for (int idx=0; idx<size; idx++)
    {
        VEC_ELEM(steps,idx) = 1;
        VEC_ELEM(steps,idx+totalSize) = 1;
        VEC_ELEM(steps,idx+2*totalSize) = 1;
    }	
}

// // Number Spherical Harmonics ==============================================
// void ProgForwardZernikeVol::Numsph(Matrix1D<int> &sphD)
// {
// 	for (int d=1;d<(VEC_XSIZE(sphD)-1);d++)
// 	{
// 	    if (d%2==0)
// 	    {
// 	    	sphD(d+1) = sphD(d)+((d/2)+1)*(2*d+1);
// 	    }
// 	    else
// 	    {
// 	    	sphD(d+1) = sphD(d)+(((d-1)/2)+1)*(2*d+1);
// 	    }
// 	}
// }

// Length of coefficients vector
// void ProgForwardZernikeVol::numCoefficients(int l1, int l2, int &vecSize)
// {
// 	vecSize = std::floor((4+4*l1+std::pow(l1,2))/4)*std::pow(l2+1,2);
// }

void ProgForwardZernikeVol::numCoefficients(int l1, int l2, int &vecSize)
{
    for (int h=0; h<=l2; h++)
    {
        int numSPH = 2*h+1;
        int count=l1-h+1;
        int numEven=(count>>1)+(count&1 && !(h&1));
        if (h%2 == 0)
            vecSize += numSPH*numEven;
        else
        	vecSize += numSPH*(l1-h+1-numEven);
    }
}

void ProgForwardZernikeVol::fillVectorTerms(int l1, int l2)
{
    int idx = 0;
	vL1.initZeros(vecSize);
	vN.initZeros(vecSize);
	vL2.initZeros(vecSize);
	vM.initZeros(vecSize);
    for (int h=0; h<=l2; h++) {
        int totalSPH = 2*h+1;
        int aux = std::floor(totalSPH/2);
        for (int l=h; l<=l1; l+=2) {
            for (int m=0; m<totalSPH; m++) {
                VEC_ELEM(vL1,idx) = l;
                VEC_ELEM(vN,idx) = h;
                VEC_ELEM(vL2,idx) = h;
                VEC_ELEM(vM,idx) = m-aux;
                idx++;
            }
        }
    }
}

// void ProgForwardZernikeVol::fillVectorTerms(Matrix1D<int> &vL1, Matrix1D<int> &vN, Matrix1D<int> &vL2, Matrix1D<int> &vM)
// {
// 	vL1.initZeros(vecSize);
// 	vN.initZeros(vecSize);
// 	vL2.initZeros(vecSize);
// 	vM.initZeros(vecSize);
// 	for (int idx=0;idx<vecSize;idx++)
// 	{
// 		int l1,n,l2,m;
// 		spherical_index2lnm(idx,l1,n,l2,m,L1);
// 		VEC_ELEM(vL1,idx) = l1;
// 		VEC_ELEM(vN,idx)  = n;
// 		VEC_ELEM(vL2,idx) = l2;
// 		VEC_ELEM(vM,idx)  = m;
// 	}
// }

// Number Spherical Harmonics ==============================================
#define Dx(V) (A3D_ELEM(V,k,i,jm2)-8*A3D_ELEM(V,k,i,jm1)+8*A3D_ELEM(V,k,i,jp1)-A3D_ELEM(V,k,i,jp2))/12.0
#define Dy(V) (A3D_ELEM(V,k,im2,j)-8*A3D_ELEM(V,k,im1,j)+8*A3D_ELEM(V,k,ip1,j)-A3D_ELEM(V,k,ip2,j))/12.0
#define Dz(V) (A3D_ELEM(V,km2,i,j)-8*A3D_ELEM(V,km1,i,j)+8*A3D_ELEM(V,kp1,i,j)-A3D_ELEM(V,kp2,i,j))/12.0

void ProgForwardZernikeVol::computeStrain()
{
	Image<double> LS, LR;
	LS().initZeros(Gx());
	LR().initZeros(Gx());

	// Gaussian filter of the derivatives
    FourierFilter f;
    f.FilterBand=LOWPASS;
    f.FilterShape=REALGAUSSIAN;
    f.w1=2;
    f.applyMaskSpace(Gx());
    f.applyMaskSpace(Gy());
    f.applyMaskSpace(Gz());

	Gx.write(fnRoot+"_PPPGx.vol");
	Gy.write(fnRoot+"_PPPGy.vol");
	Gz.write(fnRoot+"_PPPGz.vol");

	MultidimArray<double> &mLS=LS();
	MultidimArray<double> &mLR=LR();
	MultidimArray<double> &mGx=Gx();
	MultidimArray<double> &mGy=Gy();
	MultidimArray<double> &mGz=Gz();
	Matrix2D<double> U(3,3), D(3,3), H(3,3);
	std::vector< std::complex<double> > eigs;
	H.initZeros();
	for (int k=STARTINGZ(mLS)+2; k<=FINISHINGZ(mLS)-2; ++k)
	{
		int km1=k-1;
		int kp1=k+1;
		int km2=k-2;
		int kp2=k+2;
		for (int i=STARTINGY(mLS)+2; i<=FINISHINGY(mLS)-2; ++i)
		{
			int im1=i-1;
			int ip1=i+1;
			int im2=i-2;
			int ip2=i+2;
			for (int j=STARTINGX(mLS)+2; j<=FINISHINGX(mLS)-2; ++j)
			{
				int jm1=j-1;
				int jp1=j+1;
				int jm2=j-2;
				int jp2=j+2;
				MAT_ELEM(U,0,0)=Dx(mGx); MAT_ELEM(U,0,1)=Dy(mGx); MAT_ELEM(U,0,2)=Dz(mGx);
				MAT_ELEM(U,1,0)=Dx(mGy); MAT_ELEM(U,1,1)=Dy(mGy); MAT_ELEM(U,1,2)=Dz(mGy);
				MAT_ELEM(U,2,0)=Dx(mGz); MAT_ELEM(U,2,1)=Dy(mGz); MAT_ELEM(U,2,2)=Dz(mGz);

				MAT_ELEM(D,0,0) = MAT_ELEM(U,0,0);
				MAT_ELEM(D,0,1) = MAT_ELEM(D,1,0) = 0.5*(MAT_ELEM(U,0,1)+MAT_ELEM(U,1,0));
				MAT_ELEM(D,0,2) = MAT_ELEM(D,2,0) = 0.5*(MAT_ELEM(U,0,2)+MAT_ELEM(U,2,0));
				MAT_ELEM(D,1,1) = MAT_ELEM(U,1,1);
				MAT_ELEM(D,1,2) = MAT_ELEM(D,2,1) = 0.5*(MAT_ELEM(U,1,2)+MAT_ELEM(U,2,1));
				MAT_ELEM(D,2,2) = MAT_ELEM(U,2,2);

				MAT_ELEM(H,0,1) = 0.5*(MAT_ELEM(U,0,1)-MAT_ELEM(U,1,0));
				MAT_ELEM(H,0,2) = 0.5*(MAT_ELEM(U,0,2)-MAT_ELEM(U,2,0));
				MAT_ELEM(H,1,2) = 0.5*(MAT_ELEM(U,1,2)-MAT_ELEM(U,2,1));
				MAT_ELEM(H,1,0) = -MAT_ELEM(H,0,1);
				MAT_ELEM(H,2,0) = -MAT_ELEM(H,0,2);
				MAT_ELEM(H,2,1) = -MAT_ELEM(H,1,2);

				A3D_ELEM(mLS,k,i,j)=fabs(D.det());
				allEigs(H,eigs);
				for (size_t n=0; n < eigs.size(); n++)
				{
					double imagabs=fabs(eigs[n].imag());
					if (imagabs>1e-6)
					{
						A3D_ELEM(mLR,k,i,j)=imagabs*180/PI;
						break;
					}
				}
			}
		}
		LS.write(fnVolOut.withoutExtension()+"_strain.mrc");
		LR.write(fnVolOut.withoutExtension()+"_rotation.mrc");
	}
}

void ProgForwardZernikeVol::writeVector(std::string outPath, Matrix1D<double> v, bool append)
{
    std::ofstream outFile;
    if (append == false)
        outFile.open(outPath);
    else
        outFile.open(outPath, std::ios_base::app);
    FOR_ALL_ELEMENTS_IN_MATRIX1D(v)
        outFile << VEC_ELEM(v,i) << " ";
    outFile << std::endl;
}

std::string ProgForwardZernikeVol::readNthLine(int N) const
{
	std::ifstream in(fn_sph.getString());
	std::string s;  

	//skip N lines
	for(int i = 0; i < N; ++i)
		std::getline(in, s);

	std::getline(in,s);
	return s;
}

std::vector<double> ProgForwardZernikeVol::string2vector(std::string const &s) const
{
	std::stringstream iss(s);
    double number;
    std::vector<double> v;
    while (iss >> number)
        v.push_back(number);
    return v;
}

void ProgForwardZernikeVol::volume2Blobs(MultidimArray<double> &vol, MultidimArray<double> &vol2, const MultidimArray<double> &mV, const MultidimArray<int> &mask)
{
	// blob.radius = 2 * blob_r;
	vol.initZeros(mV);
	vol.setXmippOrigin();
	vol2.initZeros(mV);
	vol2.setXmippOrigin();
	for (int k=STARTINGZ(mV); k<=FINISHINGZ(mV); k+=loop_step) {
        for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i+=loop_step) {
            for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j+=loop_step) {
				if (A3D_ELEM(mask,k,i,j) == 1) {
					auto pos = std::array<double, 3>{};
					pos[0] = j;
					pos[1] = i;
					pos[2] = k;
					double voxel_mV= A3D_ELEM(mV,k,i,j);
					splattingAtPos(pos, voxel_mV, vol, vol2);
				}
            }
        }
    }
	// blob.radius = 2 * blob_r;
}

void ProgForwardZernikeVol::volume2Mask(MultidimArray<double> &vol, double thr)
{
	for (int k = STARTINGZ(vol); k <= FINISHINGZ(vol); k ++)
	{
		for (int i = STARTINGY(vol); i <= FINISHINGY(vol); i ++)
		{
			for (int j = STARTINGX(vol); j <= FINISHINGX(vol); j ++)
			{
				if (A3D_ELEM(vol, k, i, j) >= thr)
					A3D_ELEM(vol, k, i, j) = 1.0;
				else
					A3D_ELEM(vol, k, i, j) = 0.0;
			}
		}
	}
}

void ProgForwardZernikeVol::rmsd(MultidimArray<double> vol1, MultidimArray<double> vol2, double &val)
{
	const MultidimArray<double> mVI = VI();
	const MultidimArray<double> mVR = VR();
	const MultidimArray<double> mVR2 = VR2();
	double N = 0.0;
	for (int k = STARTINGZ(vol1); k <= FINISHINGZ(vol1); k +=loop_step)
	{
		for (int i = STARTINGY(vol1); i <= FINISHINGY(vol1); i +=loop_step)
		{
			for (int j = STARTINGX(vol1); j <= FINISHINGX(vol1); j +=loop_step)
			{
				// double diff1 = (A3D_ELEM(vol1, k, i, j) - A3D_ELEM(mVR, k, i, j)) / (abs(A3D_ELEM(mVR, k, i, j)) + 0.0001);
				double diffv1 = A3D_ELEM(vol1, k, i, j) - A3D_ELEM(mVR, k, i, j);
				double diffv2 = A3D_ELEM(vol2, k, i, j) - A3D_ELEM(mVR2, k, i, j);
				double diff2v1 = 1.0;
				if (A3D_ELEM(mVR, k, i, j) == 0.0 || A3D_ELEM(mVI, k, i, j) == 0.0)
					double diff2v1 = A3D_ELEM(mVR, k, i, j) - A3D_ELEM(mVI, k, i, j);
				double diff2v2 = 1.0;
				if (A3D_ELEM(mVR2, k, i, j) == 0.0 || A3D_ELEM(mVI, k, i, j) == 0.0)
					double diff2v2 = A3D_ELEM(mVR2, k, i, j) - A3D_ELEM(mVI, k, i, j);
				// double diff = A3D_ELEM(vol1, k, i, j) - A3D_ELEM(vol2, k, i, j);
				// val += 0.5 * ((diffv1 * diffv1) * (diff2v1 * diff2v1) + (diffv2 * diffv2) * (diff2v2 * diff2v2));
				val += 0.5 * ((diffv1 * diffv1) + (diffv2 * diffv2));
				// val += diff1 * diff1;
				// N += abs(A3D_ELEM(mVR, k, i, j)) + 0.0001;
				// N += 0.5 * (diff2v1 + diff2v2);
				N++;
			}
		}
	}
	val = std::sqrt(val / N);
}
