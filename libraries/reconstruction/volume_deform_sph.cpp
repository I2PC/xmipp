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
#include "data/numerical_tools.h"
#include "data/basis.h"

// Params definition =======================================================
void ProgVolDeformSph::defineParams() {
	addUsageLine("Compute the deformation that properly fits two volumes using spherical harmonics");
	addParamsLine("   -i <volume>                         : Volume to deform");
	addParamsLine("   -r <volume>                         : Reference volume");
	addParamsLine("  [-o <volume=\"\">]                   : Output volume which is the deformed input volume");
	addParamsLine("                                       : By default, the input file is rewritten");
	addParamsLine("  [--alignVolumes]                     : Align the deformed volume to the reference volume before comparing");
	addParamsLine("                                       : You need to compile Xmipp with SHALIGNMENT support (see install.sh)");
	addParamsLine("  [--depth <d=1>]                      : Harmonical depth of the deformation=1,2,3,...");
	addParamsLine("  [--Rmax <r=-1>]                      : Maximum radius for the transformation");
	addExampleLine("xmipp_volume_deform_sph -i vol1.vol -r vol2.vol -o vol1DeformedTo2.vol");
}

// Read arguments ==========================================================
void ProgVolDeformSph::readParams() {
	fnVolI = getParam("-i");
	fnVolR = getParam("-r");
	depth = getIntParam("--depth");
	fnVolOut = getParam("-o");
	if (fnVolOut=="")
		fnVolOut=fnVolI;
	alignVolumes=checkParam("--alignVolumes");
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
			<< "Align volumes:        " << alignVolumes << std::endl
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
	//std::cout << "Starting to evaluate\n" << clnm << std::endl;
#endif
	double modg=0.0;
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
					double rr=sqrt(r2)*iRmax;
					spherical_index2lnm(idx,l,n,m);
					double zsph=ZernikeSphericalHarmonics(l,n,m,jr,ir,kr,rr);

#ifdef NEVERDEFINED
					if (ir!=0&jr!=0&rr!=0)
					{
						x = zsph*(ir/sqrt(ir*ir+jr*jr))*(kr/rr);
						y = zsph*(ir/rr);
						z = zsph*(jr/sqrt(ir*ir+jr*jr));
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
				double voxelR=A3D_ELEM(mVR,k,i,j);
				double absVoxelR=abs(voxelR);
				double voxelI=mVI.interpolatedElement3D(j+gx,i+gy,k+gz);
                if (applyTransformation)
                	VO(k,i,j)=voxelI;
				double diff=voxelR-voxelI;
				diff2+=absVoxelR*diff*diff;
				modg+=absVoxelR*(gx*gx+gy*gy+gz*gz);
//				Ncount++;
				totalVal = totalVal+absVoxelR;
			}
		}
	}
	deformation=sqrt(modg/(totalVal));

#ifdef DEBUG
	save.write("PPPIdeformed.vol");
	save()-=VR();
	save.write("PPPdiff.vol");
	save()=VR();
	save.write("PPPR.vol");
	std::cout << "Error=" << sqrt(diff2/Ncount) << " " << sqrt(modg/Ncount) << std::endl;
	std::cout << "Press any key\n";
	char c; std::cin >> c;
#endif
	if (applyTransformation)
		VO.write(fnVolOut);
	return sqrt(diff2/totalVal);
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

	Numsph(nh);

	VI.read(fnVolI);
	VR.read(fnVolR);
	if (Rmax<0)
		Rmax=XSIZE(VI());

	VI().setXmippOrigin();
	VR().setXmippOrigin();

    Matrix1D<double> steps, x, prevsteps;
    for (int h=0;h<VEC_XSIZE(nh)-1;h++)
    {
    	L = nh(h+1);
    	prevL = nh(h);
    	prevsteps=steps;
    	steps.clear();
    	std::cout<<std::endl;
    	std::cout<<"Spherical harmonic depth: "<<h<<" --------------------------"<<std::endl;
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
		                0.5, fitness, iter, steps, true);

        std::cout<<std::endl;
        std::cout << "Deformation " << deformation << std::endl;
    }
    applyTransformation=true;
    distance(x.adaptForNumericalRecipes()); // To save the output volume
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


