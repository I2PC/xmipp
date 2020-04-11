/***************************************************************************
 *
 * Authors:    Carlos Oscar             coss@cnb.csic.es
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

#include "volume_deform_sph.h"
#include <data/numerical_tools.h>
#include <data/basis.h>
#include <data/fourier_filter.h>
#include <data/normalize.h>

// Params definition =======================================================
void ProgVolDeformSph::defineParams() {
	addUsageLine("Compute the deformation that properly fits two volumes using spherical harmonics");
	addParamsLine("   -i <volume>                         : Volume to deform");
	addParamsLine("   -r <volume>                         : Reference volume");
	addParamsLine("  [-o <volume=\"\">]                   : Output volume which is the deformed input volume");
	addParamsLine("                                       : By default, the input file is rewritten");
	addParamsLine("  [--sigma <Matrix1D=\"\">]	      : Sigma values to filter the volume to perform a multiresolution analysis");
	addParamsLine("  [--analyzeStrain]                    : Save the deformation of each voxel for local strain and rotation analysis");
	addParamsLine("  [--optimizeRadius]                   : Optimize the radius of each spherical harmonic");
	addParamsLine("  [--depth <d=1>]                      : Harmonical depth of the deformation=1,2,3,...");
	addParamsLine("  [--Rmax <r=-1>]                      : Maximum radius for the transformation");
	addExampleLine("xmipp_volume_deform_sph -i vol1.vol -r vol2.vol -o vol1DeformedTo2.vol");
}

// Read arguments ==========================================================
void ProgVolDeformSph::readParams() {
    std::string aux;
	fnVolI = getParam("-i");
	fnVolR = getParam("-r");
	depth = getIntParam("--depth");
	
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
	Rmax = getDoubleParam("--Rmax");
	applyTransformation = false;
}

// Show ====================================================================
void ProgVolDeformSph::show() {
	if (verbose==0)
		return;
	std::cout
	        << "Volume to deform:     " << fnVolI       << std::endl
			<< "Reference volume:     " << fnVolR       << std::endl
			<< "Output volume:        " << fnVolOut     << std::endl
			<< "Depth:                " << depth        << std::endl
			<< "Save deformation:     " << analyzeStrain << std::endl
	;

}

// Distance function =======================================================
//#define DEBUG
double ProgVolDeformSph::distance(double *pclnm)
{
	if (applyTransformation)
	{
		VO().initZeros(VR());
		VO().setXmippOrigin();
	}
	int l,n,m;
	size_t idxY0=VEC_XSIZE(clnm)/4;
	size_t idxZ0=2*idxY0;
	size_t idxR=3*idxY0;
//	double Ncount=0.0;
	double totalVal=0.0;
	double diff2=0.0;
	const MultidimArray<double> &mVR=VR();
	const MultidimArray<double> &mVI=VI();
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];
#ifdef DEBUG
	std::cout << "Starting to evaluate\n" << clnm << std::endl;
#endif
	double modg=0.0;
	double voxelR, absVoxelR, voxelI, diff;
	for (int k=STARTINGZ(mVR); k<=FINISHINGZ(mVR); k++)
	{
		for (int i=STARTINGY(mVR); i<=FINISHINGY(mVR); i++)
		{
			for (int j=STARTINGX(mVR); j<=FINISHINGX(mVR); j++)
			{
				double gx=0.0, gy=0.0, gz=0.0;
				for (size_t idx=0; idx<idxY0; idx++)
				{
					double Rmax=VEC_ELEM(clnm,idx+idxR);
					double Rmax2=Rmax*Rmax;
					double iRmax=1.0/Rmax;
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
						spherical_index2lnm(idx,l,n,m);
						zsph=ZernikeSphericalHarmonics(l,n,m,jr,ir,kr,rr);
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
					if (rr>0 || l==0)
					{
						gx += VEC_ELEM(clnm,idx)        *(zsph);
						gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
						gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
					}
				}

				for (int idv=0; idv<volumesR.size(); idv++)
				{
					voxelR=A3D_ELEM(volumesR[idv](),k,i,j);
					absVoxelR=fabs(voxelR);
					voxelI=volumesI[idv]().interpolatedElement3D(j+gx,i+gy,k+gz);
					if (applyTransformation && idv == 0)
						VO(k,i,j)=voxelI;
					diff=voxelR-voxelI;
					diff2+=absVoxelR*diff*diff;
					modg+=absVoxelR*(gx*gx+gy*gy+gz*gz);
	//				Ncount++;
					totalVal += absVoxelR;
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

	deformation=std::sqrt(modg/(totalVal));

#ifdef DEBUG
	//save.write("PPPIdeformed.vol");
	//save()-=VR();
	//save.write("PPPdiff.vol");
	//save()=VR();
	//save.write("PPPR.vol");
	std::cout << "Error=" << deformation << " " << std::sqrt(diff2/totalVal) << std::endl;
	//std::cout << "Press any key\n";
	//char c; std::cin >> c;
#endif
	if (applyTransformation)
		VO.write(fnVolOut);
	return std::sqrt(diff2/totalVal);
}
#undef DEBUG

double volDeformSphGoal(double *p, void *vprm)
{
    ProgVolDeformSph *prm=(ProgVolDeformSph *) vprm;
	return prm->distance(p);
}

// Run =====================================================================
void ProgVolDeformSph::run() {
	Matrix1D<int> nh;
	nh.resize(depth+2);
	nh.initConstant(0);
	nh(1)=1;

	saveDeformation=false;
	Numsph(nh);

	VI.read(fnVolI);
	VR.read(fnVolR);
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
	MultidimArray<int> bg_mask;
	Image<double> auxI = VI;
	Image<double> auxR = VR;
	bg_mask.resizeNoCopy(VI().zdim, VI().ydim, VI().xdim);
    bg_mask.setXmippOrigin();
	BinaryCircularMask(bg_mask, Rmax, OUTSIDE_MASK);

	normalize_Robust(auxI(), bg_mask, true);
	normalize_Robust(auxR(), bg_mask, true);

	volumesI.push_back(auxI());
	volumesR.push_back(auxR());

	for (int ids=0; ids<sigma.size(); ids++)
	{
		Image<double> auxI = VI;
		Image<double> auxR = VR;
		filter.w1 = sigma[ids];

		// Filer input vol
		filter.do_generate_3dmask = true;
		filter.applyMaskSpace(auxI());
		normalize_Robust(auxI(), bg_mask, true);
		volumesI.push_back(auxI);

		// Filter ref vol
		filter.applyMaskSpace(auxR());
		normalize_Robust(auxR(), bg_mask, true);
		volumesR.push_back(auxR);
	}

    Matrix1D<double> steps, x, prevsteps;
    for (int h=0;h<VEC_XSIZE(nh)-1;h++)
    {
    	L = nh(h+1);
    	prevL = nh(h);
    	prevsteps=steps;
    	steps.clear();
    	std::cout<<std::endl;
    	std::cout<<"-------------------------- Spherical harmonic depth: "<<h<<" --------------------------"<<std::endl;
        steps.initConstant(4*L,1);
    	if (h==0)
    	{
    		clnm.initZeros(VEC_XSIZE(steps));
    		x.initZeros(VEC_XSIZE(steps));

    	}
    	else
    	{
    		x.resize(VEC_XSIZE(steps),false);
    		copyvectors(clnm,x);
    		clnm=x;
    	}

        for(int d=VEC_XSIZE(x)-L+prevL;d<VEC_XSIZE(x);d++)
    	{
    		x(d)=Rmax;
    		clnm(d)=Rmax;
    	}
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
        if (h!=0)
        {
            minimizepos(steps,prevsteps);
        }
        else
        {
        	steps(VEC_XSIZE(steps)-1)=0;
        }
        int iter;
        double fitness;
        powellOptimizer(x, 1, VEC_XSIZE(steps), &volDeformSphGoal, this,
		                0.01, fitness, iter, steps, true);

        std::cout<<std::endl;
        std::cout << "Deformation " << deformation << std::endl;
        std::ofstream deformFile;
        deformFile.open ("./deformation.txt");
        deformFile << deformation;
        deformFile.close();

    }
    applyTransformation=true;
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

    if (analyzeStrain)
    	computeStrain();
}

// Copy Vectors ============================================================
void ProgVolDeformSph::copyvectors(Matrix1D<double> &oldvect,Matrix1D<double> &newvect)
{
	size_t groups = 4;
	size_t olditems = VEC_XSIZE(oldvect)/groups;
	size_t newitems = VEC_XSIZE(newvect)/groups;
	for (int g=0;g<groups;g++)
	{
		for (int i=0;i<olditems;i++)
			{
			    newvect(g*newitems+i) = oldvect(g*olditems+i);
			}
	}
}

// Minimize Positions ======================================================
void ProgVolDeformSph::minimizepos(Matrix1D<double> &vectpos, Matrix1D<double> &prevpos)
{
	size_t groups = 4;
	size_t olditems = VEC_XSIZE(prevpos)/groups;
	size_t newitems = VEC_XSIZE(vectpos)/groups;
	for (int i=0;i<olditems;i++)
	{
		vectpos(3*newitems+i) = 0;
	}
	if (!optimizeRadius)
	{
		for (int j=olditems;j<newitems;j++)
		{
			vectpos(3*newitems+j) = 0;
		}
	}
}

// Number Spherical Harmonics ==============================================
void ProgVolDeformSph::Numsph(Matrix1D<int> &sphD)
{
	for (int d=1;d<(VEC_XSIZE(sphD)-1);d++)
	{
	    if (d%2==0)
	    {
	    	sphD(d+1) = sphD(d)+((d/2)+1)*(2*d+1);
	    }
	    else
	    {
	    	sphD(d+1) = sphD(d)+(((d-1)/2)+1)*(2*d+1);
	    }
	}
}

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

	Gx.write("PPPGx.vol");
	Gy.write("PPPGy.vol");
	Gz.write("PPPGz.vol");

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


