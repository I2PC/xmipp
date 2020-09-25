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

#include <fstream>
#include <iterator>
#include "volume_deform_sph.h"
#include "data/fourier_filter.h"
#include "data/normalize.h"

// Params definition =======================================================
void ProgVolDeformSph::defineParams() {
	addUsageLine("Compute the deformation that properly fits two volumes using spherical harmonics");
	addParamsLine("   -i <volume>                         : Volume to deform");
	addParamsLine("   -r <volume>                         : Reference volume");
	addParamsLine("  [-o <volume=\"\">]                   : Output volume which is the deformed input volume");
	addParamsLine("  [--oroot <rootname=\"Volumes\">]     : Root name for output files");
	addParamsLine("                                       : By default, the input file is rewritten");
	addParamsLine("  [--sigma <Matrix1D=\"\">]	          : Sigma values to filter the volume to perform a multiresolution analysis");
	addParamsLine("  [--analyzeStrain]                    : Save the deformation of each voxel for local strain and rotation analysis");
	addParamsLine("  [--optimizeRadius]                   : Optimize the radius of each spherical harmonic");
	addParamsLine("  [--l1 <l1=3>]                        : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                        : Harmonical depth of the deformation=1,2,3,...");
	addParamsLine("  [--regularization <l=0.00025>]       : Regularization weight");
	addParamsLine("  [--Rmax <r=-1>]                      : Maximum radius for the transformation");
	addExampleLine("xmipp_volume_deform_sph -i vol1.vol -r vol2.vol -o vol1DeformedTo2.vol");
}

// Read arguments ==========================================================
void ProgVolDeformSph::readParams() {
    std::string aux;
	fnVolI = getParam("-i");
	fnVolR = getParam("-r");
	L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	fnRoot = getParam("--oroot");
	
	aux = getParam("--sigma");
	// Transform string ov values separated by white spaces into substrings stored in a vector
	std::stringstream ss(aux);
	std::istream_iterator<std::string> begin(ss);
	std::istream_iterator<std::string> end;
	std::vector<std::string> vstrings(begin, end);
	sigma.resize(vstrings.size());
	std::transform(vstrings.begin(), vstrings.end(), sigma.begin(), [](const std::string& val)
	{
    	return std::stod(val);
	});

	fnVolOut = getParam("-o");
	if (fnVolOut=="")
		fnVolOut=fnVolI;
	analyzeStrain=checkParam("--analyzeStrain");
	optimizeRadius=checkParam("--optimizeRadius");
	lambda = getDoubleParam("--regularization");
	Rmax = getDoubleParam("--Rmax");
	applyTransformation = false;
}

// Show ====================================================================
void ProgVolDeformSph::show() {
	if (verbose==0)
		return;
	std::cout
	        << "Volume to deform:     " << fnVolI         << std::endl
			<< "Reference volume:     " << fnVolR         << std::endl
			<< "Output volume:        " << fnVolOut       << std::endl
			<< "Zernike Degree:       " << L1             << std::endl
			<< "SH Degree:            " << L2             << std::endl
			<< "Save deformation:     " << analyzeStrain  << std::endl
			<< "Regularization:       " << lambda         << std::endl
	;

}

// Distance function =======================================================
// #define DEBUG
double ProgVolDeformSph::distance(double *pclnm)
{
	if (applyTransformation)
	{
		VO().initZeros(VR());
		VO().setXmippOrigin();
	}
	// int maxl1 = L1;
	int l1,n,l2,m;
	size_t idxY0=VEC_XSIZE(clnm)/3;
	size_t idxZ0=2*idxY0;
	// size_t idxR=3*idxY0;
	double Ncount=0.0;
	// double totalVal=0.0;
	double diff2=0.0;
	sumVD = 0.0;
	const MultidimArray<double> &mVR=VR();
	const MultidimArray<double> &mVI=VI();
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];
#ifdef DEBUG
	std::cout << "Starting to evaluate\n" << clnm << std::endl;
#endif
	double modg=0.0;
	// double absVoxelR;
	double voxelR, voxelI, diff;
	double Rmax2=Rmax*Rmax;
	double iRmax=1.0/Rmax;
	for (int k=STARTINGZ(mVR); k<=FINISHINGZ(mVR); k++)
	{
		for (int i=STARTINGY(mVR); i<=FINISHINGY(mVR); i++)
		{
			for (int j=STARTINGX(mVR); j<=FINISHINGX(mVR); j++)
			{
				double gx=0.0, gy=0.0, gz=0.0;
				for (size_t idx=0; idx<idxY0; idx++)
				{
					if (VEC_ELEM(steps_cp,idx) == 1)
					{
						// double Rmax=VEC_ELEM(clnm,idx+idxR);
						double k2=k*k;
						double kr=k*iRmax;
						double k2i2=k2+i*i;
						double ir=i*iRmax;
						double r2=k2i2+j*j;
						double jr=j*iRmax;
						double rr=std::sqrt(r2)*iRmax;
						double zsph=0.0;
						if (r2<Rmax2)
						{
							l1 = VEC_ELEM(vL1,idx);
							n = VEC_ELEM(vN,idx);
							l2 = VEC_ELEM(vL2,idx);
							m = VEC_ELEM(vM,idx);
							zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
						}

#ifdef NEVERDEFINED
						if (ir!=0&jr!=0&rr!=0)
						{
							x = zsph*(ir/std::sqrt(ir*ir+jr*jr))*(kr/rr);
							y = zsph*(ir/rr);
							z = zsph*(jr/std::sqrt(ir*ir+jr*jr));
							gx += VEC_ELEM(clnm,idx)      *x;
							gy += VEC_ELEM(clnm,idx+idxY0)*y;
							gz += VEC_ELEM(clnm,idx+idxZ0)*z;
						}
						else
						{
							gx += VEC_ELEM(clnm,idx)      *zsph;
							gy += VEC_ELEM(clnm,idx+idxY0)*zsph;
							gz += VEC_ELEM(clnm,idx+idxZ0)*zsph;
						}
#endif
						// if (rr>0 || (l2==0 && l1==0))
						if (rr>0 || l2==0)
						{
							gx += VEC_ELEM(clnm,idx)        *(zsph);
							gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
							gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
						}
					}
				}
				if (applyTransformation)
				{
					// absVoxelR=fabs(voxelR);
					voxelR=A3D_ELEM(VR(),k,i,j);
					voxelI=VI().interpolatedElement3D(j+gx,i+gy,k+gz);
					if (voxelI >= 0.0)
						sumVD += voxelI;
					VO(k,i,j)=voxelI;
					diff=voxelR-voxelI;
					diff2+=diff*diff;
					modg+=gx*gx+gy*gy+gz*gz;
					Ncount++;
					// totalVal += absVoxelR;
				}
				for (int idv=0; idv<volumesR.size(); idv++)
				{
					voxelR=A3D_ELEM(volumesR[idv](),k,i,j);
					// absVoxelR=fabs(voxelR);
					voxelI=volumesI[idv]().interpolatedElement3D(j+gx,i+gy,k+gz);
					if (voxelI >= 0.0)
						sumVD += voxelI;
					diff=voxelR-voxelI;
					// diff2+=absMaxR_vec[idv]*diff*diff;
					diff2+=diff*diff;
					// modg+=absVoxelR*(gx*gx+gy*gy+gz*gz);
					modg+=gx*gx+gy*gy+gz*gz;
					Ncount++;
					// totalVal += absVoxelR;
				}

				if (saveDeformation) 
				{
					Gx(k,i,j)=gx;
					Gy(k,i,j)=gy;
					Gz(k,i,j)=gz;
				}
			}
		}
	}

	// deformation=std::sqrt(modg/(totalVal));
	deformation=std::sqrt(modg/(Ncount));

#ifdef DEBUG
	Image<double> save;
	save() = VI();
	save.write(fnRoot+"_PPPIdeformed.vol");
	save()-=VR();
	save.write(fnRoot+"_PPPdiff.vol");
	save()=VR();
	save.write(fnRoot+"_PPPR.vol");
	if (saveDeformation)
	{
		save() = Gx();
		save.write(fnRoot+"_PPPGx.vol");
		save() = Gy();
		save.write(fnRoot+"_PPPGy.vol");
		save() = Gz();
		save.write(fnRoot+"_PPPGz.vol");
	}
	std::cout << "Error=" << deformation << " " << std::sqrt(diff2/totalVal) << std::endl;
	std::cout << "Press any key\n";
	char c; std::cin >> c;
#endif
	if (applyTransformation)
		VO.write(fnVolOut);
	// return std::sqrt(diff2/totalVal);
	//// sumVD /= volumesR.size();
	double massDiff=std::abs(sumVI-sumVD)/sumVI;
	return std::sqrt(diff2/Ncount)+lambda*(deformation+massDiff);
}
#undef DEBUG

double volDeformSphGoal(double *p, void *vprm)
{
    ProgVolDeformSph *prm=(ProgVolDeformSph *) vprm;
	return prm->distance(p);
}

// Run =====================================================================
void ProgVolDeformSph::run() {
	// Matrix1D<int> nh;
	// nh.resize(depth+2);
	// nh.initConstant(0);
	// nh(1)=1;

	saveDeformation=false;
	// Numsph(nh);

	VI.read(fnVolI);
	VR.read(fnVolR);
	sumVI = 0.0;
	if (Rmax<0)
		Rmax=XSIZE(VI())/2;

	VI().setXmippOrigin();
	VR().setXmippOrigin();

	// Filter input and reference volumes according to the values of sigma
	FourierFilter filter;
    filter.FilterShape = REALGAUSSIAN;
    filter.FilterBand = LOWPASS;
	filter.generateMask(VI());

	// We need also to normalized the filtered volumes to compare them appropiately
	double maxVoxelR;
	Image<double> auxI = VI;
	Image<double> auxR = VR;

	MultidimArray<int> bg_mask;
	bg_mask.resizeNoCopy(VI().zdim, VI().ydim, VI().xdim);
    bg_mask.setXmippOrigin();
	normalize_Robust(auxI(), bg_mask, true);
	// auxI.write("input_filt.vol");
	bg_mask *= 0;
	normalize_Robust(auxR(), bg_mask, true);
	// auxR.write("ref_filt.vol");

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(auxI())
	{
		if (DIRECT_A3D_ELEM(auxI(),k,i,j) >= 0.0)
    		sumVI += DIRECT_A3D_ELEM(auxI(),k,i,j);
	}

	if (sigma.size() == 1 && sigma[0] == 0)
	{
		volumesI.push_back(auxI());
		volumesR.push_back(auxR());
		// maxVoxelR = auxR().computeMax();
		// absMaxR_vec.push_back(fabs(maxVoxelR));
	}
	else
	{
		volumesI.push_back(auxI());
		volumesR.push_back(auxR());
		// maxVoxelR = auxR().computeMax();
		// absMaxR_vec.push_back(fabs(maxVoxelR));
		for (int ids=0; ids<sigma.size(); ids++)
		{
			Image<double> auxI = VI;
			Image<double> auxR = VR;
			filter.w1 = sigma[ids];

			// Filer input vol
			filter.do_generate_3dmask = true;
			filter.applyMaskSpace(auxI());
			bg_mask *= 0;
			normalize_Robust(auxI(), bg_mask, true);
			volumesI.push_back(auxI);
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(auxI())
			{
				if (DIRECT_A3D_ELEM(auxI(),k,i,j) >= 0.0)
    				sumVI += DIRECT_A3D_ELEM(auxI(),k,i,j);
			}
			// auxI.write("input_filt.vol");

			// Filter ref vol
			filter.applyMaskSpace(auxR());
			bg_mask *= 0;
			normalize_Robust(auxR(), bg_mask, true);
			volumesR.push_back(auxR);
			// auxR.write("ref_filt.vol"); 
			// maxVoxelR = auxR().computeMax();
			// absMaxR_vec.push_back(fabs(maxVoxelR));
		}
	}

    Matrix1D<double> steps, x;
	numCoefficients(L1,L2,vecSize);
	size_t totalSize = 3*vecSize;
	fillVectorTerms(L1,L2,vL1,vN,vL2,vM);
	clnm.initZeros(totalSize);
	x.initZeros(totalSize);
    for (int h=0;h<=L2;h++)
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
    	// if (h==0)
    	// {

    	// }
    	// else
    	// {
    	// 	x.resize(VEC_XSIZE(steps),false);
    	// 	copyvectors(clnm,x);
    	// 	clnm=x;
    	// }

        // for(int d=VEC_XSIZE(x)-L+prevL;d<VEC_XSIZE(x);d++)
    	// {
    	// 	x(d)=Rmax;
    	// 	clnm(d)=Rmax;
    	// }
#ifdef NEVERDEFINED
        for (int d=0;d<VEC_XSIZE(x);d++)
        {
        	if (x(d)==0)
        	{
        		steps(d) = 1;
        	}
        	else if (d>=VEC_XSIZE(steps)+nh(h)-nh(h+1)&d<VEC_XSIZE(steps)+nh(h+1))
        	{
        		steps(d) = 1;
        	}
        	else
        	{
        		steps(d) = 0;
        	}
        	//std::cout<<steps(d)<<" ";
        }
        //std::cout<<std::endl;
#endif
        // if (h!=0)
        // {
        //     minimizepos(steps,prevsteps);
        // }
        // else
        // {
        // 	steps(VEC_XSIZE(steps)-1)=0;
        // }
        int iter;
        double fitness;
        powellOptimizer(x, 1, totalSize, &volDeformSphGoal, this,
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
// void ProgVolDeformSph::copyvectors(Matrix1D<double> &oldvect,Matrix1D<double> &newvect)
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
// void ProgVolDeformSph::minimizepos(Matrix1D<double> &vectpos, Matrix1D<double> &prevpos)
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
// void ProgVolDeformSph::minimizepos(Matrix1D<double> &vectpos, int &current_l2)
// {
// 	size_t currentSize = std::floor((4+4*L1+std::pow(L1,2))/4)*std::pow(current_l2+1,2);
// 	for (int i=0;i<currentSize;i++)
// 	{
// 		VEC_ELEM(vectpos,i) = 1;
// 		VEC_ELEM(vectpos,i+vecSize) = 1;
// 		VEC_ELEM(vectpos,i+2*vecSize) = 1;
// 	}
// }

void ProgVolDeformSph::minimizepos(int L1, int l2, Matrix1D<double> &steps)
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
// void ProgVolDeformSph::Numsph(Matrix1D<int> &sphD)
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
// void ProgVolDeformSph::numCoefficients(int l1, int l2, int &vecSize)
// {
// 	vecSize = std::floor((4+4*l1+std::pow(l1,2))/4)*std::pow(l2+1,2);
// }

void ProgVolDeformSph::numCoefficients(int l1, int l2, int &vecSize)
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

void ProgVolDeformSph::fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN, 
									   Matrix1D<int> &vL2, Matrix1D<int> &vM)
{
    int idx = 0;
	vL1.initZeros(vecSize);
	vN.initZeros(vecSize);
	vL2.initZeros(vecSize);
	vM.initZeros(vecSize);
    for (int h=0; h<=l2; h++)
    {
        int totalSPH = 2*h+1;
        int aux = std::floor(totalSPH/2);
        for (int l=h; l<=l1; l+=2)
        {
            for (int m=0; m<totalSPH; m++)
            {
                VEC_ELEM(vL1,idx) = l;
                VEC_ELEM(vN,idx) = h;
                VEC_ELEM(vL2,idx) = h;
                VEC_ELEM(vM,idx) = m-aux;
                idx++;
            }
        }
    }
}

// void ProgVolDeformSph::fillVectorTerms(Matrix1D<int> &vL1, Matrix1D<int> &vN, Matrix1D<int> &vL2, Matrix1D<int> &vM)
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

void ProgVolDeformSph::computeStrain()
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

void ProgVolDeformSph::writeVector(std::string outPath, Matrix1D<double> v, bool append)
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


