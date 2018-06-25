/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
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

// Params definition ============================================================
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
	alignVolumes=checkParam("--alignVolumes");
	Rmax = getDoubleParam("--Rmax");
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
double ProgVolDeformSph::distance(double *pclnm) const
{
	int l,n,m;
	//double Rmax2=Rmax*Rmax;
	//double iRmax=1.0/Rmax;
	size_t idxY0=VEC_XSIZE(clnm)/4;
	size_t idxZ0=2*idxY0;
	size_t idxR=3*idxY0;
	double Ncount=0.0;
	double diff2=0.0;
	int id=1;
	const MultidimArray<double> &mVR=VR();
	const MultidimArray<double> &mVI=VI();
	//double maxVR = sqrt(FINISHINGZ(mVR)*FINISHINGZ(mVR)+FINISHINGY(mVR)*FINISHINGY(mVR)+FINISHINGX(mVR)*FINISHINGX(mVR));
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];
	// std::cout << "Starting to evaluate\n" << clnm << std::endl;
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
					//if (Rmax<=0.0)
					//{
//						for (int d=VEC_XSIZE(clnm)-4;d<VEC_XSIZE(clnm);d++)
//						{
//							VEC_ELEM(clnm,d)=-VEC_ELEM(clnm,d);
//						}
						//Rmax = -Rmax;
						//return 0.0;
						//if (Rmax>maxVR)
						//{
							//VEC_ELEM(clnm,idx+idxR)=maxVR;
						    //Rmax = 0.45*XSIZE(VI());
						//}
					//}
					double Rmax2=Rmax*Rmax;
					double iRmax=1.0/Rmax;
					double k2=k*k;
//					if (k2>Rmax2)
//					    continue;
					double kr=k*iRmax;
					double k2i2=k2+i*i;
//					if (k2i2>Rmax2)
//					    continue;
					double ir=i*iRmax;
					double r2=k2i2+j*j;
//					if (r2>Rmax2)
//						continue;
					double jr=j*iRmax;
					double rr=sqrt(r2)*iRmax;
					spherical_index2lnm(idx,l,n,m);
					double zsph=ZernikeSphericalHarmonics(l,n,m,jr,ir,kr,rr);
					gx += VEC_ELEM(clnm,idx)      *zsph;
					gy += VEC_ELEM(clnm,idx+idxY0)*zsph;
					gz += VEC_ELEM(clnm,idx+idxZ0)*zsph;
//					if (k==0 && i==0 & j==10)
//					std::cout << "k=" << k << " i=" << i << " j=" << j << " kr=" << kr << " ir=" << ir << " jr=" << jr << " rr=" << rr << std::endl
//						      << "   idx=" << idx << " l=" << l << " n=" << n << " m=" << m << " zsph=" << zsph << std::endl
//							  << "   cx=" << VEC_ELEM(clnm,idx) << " cy=" << VEC_ELEM(clnm,idx+idxY0) << " cz=" << VEC_ELEM(clnm,idx+idxZ0) << std::endl
//						      << "   gx=" << gx << " gy=" << gy << " gz=" << gz << std::endl;
				}
				double voxelR=A3D_ELEM(mVR,k,i,j);
				double voxelI=mVI.interpolatedElement3D(j+gx,i+gy,k+gz);
//				if (k==0 && i==0 & j==0)
//			       std::cout << "    VR=" << voxelR << " VI=" << A3D_ELEM(mVI,k,i,j) << " VIg=" << voxelI << std::endl;
				double diff=voxelR-voxelI;
				diff2+=diff*diff;
				modg+=gx*gx+gy*gy+gz*gz;
				Ncount++;
			}
		}
	}
	//std::cout<<"Rmax "<<Rmax<<std::endl;
	//std::cout << "Error=" << sqrt(diff2/Ncount) << " " << sqrt(modg/Ncount) << std::endl;
	return sqrt(diff2/Ncount);
}

double volDeformSphGoal(double *p, void *vprm)
{
    ProgVolDeformSph *prm=(ProgVolDeformSph *) vprm;
	return prm->distance(p);
}

// Run =====================================================================
void ProgVolDeformSph::run() {
	VI.read(fnVolI);
	VR.read(fnVolR);
	if (Rmax<0)
		Rmax=XSIZE(VI());

	VI().setXmippOrigin();
	VR().setXmippOrigin();

    Matrix1D<double> steps;
    steps.resize(4*6);
    steps.initConstant(1);
    clnm.initZeros(VEC_XSIZE(steps));
    for(int d=VEC_XSIZE(clnm)-3;d<VEC_XSIZE(clnm);d++)
	{
		VEC_ELEM(clnm,d)=Rmax;
		//VEC_ELEM(steps,d)=0;
	}
    for(int h=0;h<VEC_XSIZE(clnm)-3;h++)
    {
        VEC_ELEM(clnm,h)=0.5;
    }
    int iter;
    double fitness;
    powellOptimizer(clnm, 1, VEC_XSIZE(steps), &volDeformSphGoal, this,
    		        0.25, fitness, iter, steps, true);
}
