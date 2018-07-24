/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

#include "angular_sph_alignment.h"
#include "program_image_residuals.h"
#include <data/mask.h>
#include <data/numerical_tools.h>

// Empty constructor =======================================================
ProgAngularSphAlignment::ProgAngularSphAlignment()
{
    produces_a_metadata = true;
    each_image_produces_an_output = true;
//    debug = false;
    penalization = true;
//    ctfImage = NULL;
}

ProgAngularSphAlignment::~ProgAngularSphAlignment()
{
//	delete ctfImage;
}

// Read arguments ==========================================================
void ProgAngularSphAlignment::readParams()
{
	XmippMetadataProgram::readParams();
	fnVolR = getParam("--ref");
    maxShift = getDoubleParam("--max_shift");
    maxAngularChange = getDoubleParam("--max_angular_change");
    maxResol = getDoubleParam("--max_resolution");
    Ts = getDoubleParam("--sampling");
    Rmax = getIntParam("--Rmax");
    optimizeAlignment = checkParam("--optimizeAlignment");
    optimizeDeformation = checkParam("--optimizeDeformation");
    phaseFlipped = checkParam("--phaseFlipped");
    depth = getIntParam("--depth");
}

// Show ====================================================================
void ProgAngularSphAlignment::show()
{
    if (!verbose)
        return;
	XmippMetadataProgram::show();
    std::cout
    << "Reference volume:    " << fnVolR             << std::endl
    << "Max. Shift:          " << maxShift           << std::endl
    << "Max. Angular Change: " << maxAngularChange   << std::endl
    << "Max. Resolution:     " << maxResol           << std::endl
    << "Sampling:            " << Ts                 << std::endl
    << "Max. Radius:         " << Rmax               << std::endl
	<< "Depth:               " << depth              << std::endl
    << "Optimize alignment:  " << optimizeAlignment  << std::endl
	<< "Optimize deformation " << optimizeDeformation<< std::endl
    << "Phase flipped:       " << phaseFlipped       << std::endl
    ;
}

// usage ===================================================================
void ProgAngularSphAlignment::defineParams()
{
    addUsageLine("Make a continuous angular assignment with deformations");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Metadata with the angular alignment and deformation parameters");
    XmippMetadataProgram::defineParams();
    addParamsLine("   --ref <volume>              : Reference volume");
    addParamsLine("  [--max_shift <s=-1>]         : Maximum shift allowed in pixels");
    addParamsLine("  [--max_angular_change <a=5>] : Maximum angular change allowed (in degrees)");
    addParamsLine("  [--max_resolution <f=4>]     : Maximum resolution (A)");
    addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine("  [--Rmax <R=-1>]              : Maximum radius (px). -1=Half of volume size");
    addParamsLine("  [--depth <depth=1>]          : Harmonical depth of the deformation=1,2,3,...");
    addParamsLine("  [--optimizeAlignment]        : Optimize alignment");
    addParamsLine("  [--optimizeDeformation]      : Optimize deformation");
    addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_angular_sph_alignment -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --optimizeAlignment --optimizeDeformation --depth 1");
}

// Produce side information ================================================
void ProgAngularSphAlignment::preProcess()
{
    // Read the reference volume
    V.read(fnVolR);
    V().setXmippOrigin();
    Xdim=XSIZE(V());
    Vdeformed().initZeros(V());

    Ip().initZeros(Xdim,Xdim);
    Ifilteredp().initZeros(Xdim,Xdim);
    Ifilteredp().setXmippOrigin();

    // Construct mask
    if (Rmax<0)
    	Rmax=Xdim/2;
    Mask mask;
    mask.type = BINARY_CIRCULAR_MASK;
    mask.mode = INNER_MASK;
    mask.R1 = Rmax;
    mask.generate_mask(Xdim,Xdim);
    mask2D=mask.get_binary_mask();
    iMask2Dsum=1.0/mask2D.sum();

    // Low pass filter
    filter.FilterBand=LOWPASS;
    filter.w1=Ts/maxResol;
    filter.raised_w=0.02;

    // Transformation matrix
    A.initIdentity(3);
}

//#define DEBUG
double ProgAngularSphAlignment::tranformImageSph(ProgAngularSphAlignment *prm,double *pclnm, double rot, double tilt, double psi,
		Matrix2D<double> &A)
{
	const MultidimArray<double> &mV=V();
	MultidimArray<double> &mVD=Vdeformed();
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];
	deformVol(mVD, mV);
	MultidimArray<double> &mV1=Vdeformed();
	projectVolume(mV1, prm->P, (int)XSIZE(prm->I()), (int)XSIZE(prm->I()),  rot, tilt, psi);
    double cost=0;
	if (prm->old_flip)
	{
		MAT_ELEM(A,0,0)*=-1;
		MAT_ELEM(A,0,1)*=-1;
		MAT_ELEM(A,0,2)*=-1;
	}

	applyGeometry(LINEAR,prm->Ifilteredp(),prm->Ifiltered(),A,IS_NOT_INV,DONT_WRAP,0.);
	const MultidimArray<double> &mP=prm->P();
	const MultidimArray<int> &mMask2D=prm->mask2D;
	MultidimArray<double> &mIfilteredp=prm->Ifilteredp();
	cost=-correlationIndex(mIfilteredp,mP,&mMask2D);

#ifdef NEVERDEFINED
   if (debug)
   {
		std::cout << "A=" << A << std::endl;
		Image<double> save;
		save()=prm->P();
		save.write("PPPtheo.xmp");
		save()=prm->Ifilteredp();
		save.write("PPPfilteredp.xmp");
		save()=prm->Ifiltered();
		save.write("PPPfiltered.xmp");
		Vdeformed.write("PPPVdeformed.vol");
		std::cout << "Cost=" << cost << std::endl;
		std::cout << "Press any key" << std::endl;
		char c; std::cin >> c;
    }
#endif

    double avgClnm=0.0;
    double numClnm=0.0;
    if (!penalization)
    {
	avgVector(clnm, avgClnm);
	computeNum(clnm, avgClnm, numClnm);
	if (avgClnm!=0)
	{
		normClnm = numClnm/avgClnm;
		normClnm /= 3*L;
	}
	else
		normClnm = 0.0;
    cost = cost+normClnm;
    }

	return cost;
}

double continuousSphCost(double *x, void *_prm)
{
	ProgAngularSphAlignment *prm=(ProgAngularSphAlignment *)_prm;
	double deltax=x[prm->pos+1];
	double deltay=x[prm->pos+2];
	double deltaRot=x[prm->pos+3];
	double deltaTilt=x[prm->pos+4];
	double deltaPsi=x[prm->pos+5];
	if (prm->maxShift>0 && deltax*deltax+deltay*deltay>prm->maxShift*prm->maxShift)
		return 1e38;
	if (fabs(deltaRot)>prm->maxAngularChange || fabs(deltaTilt)>prm->maxAngularChange || fabs(deltaPsi)>prm->maxAngularChange)
		return 1e38;

	MAT_ELEM(prm->A,0,2)=prm->old_shiftX+deltax;
	MAT_ELEM(prm->A,1,2)=prm->old_shiftY+deltay;
	return prm->tranformImageSph(prm,x,prm->old_rot+deltaRot, prm->old_tilt+deltaTilt, prm->old_psi+deltaPsi,
			prm->A);
}

// Predict =================================================================
//#define DEBUG
void ProgAngularSphAlignment::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	if (depth==0)
		depth = 1;
	Matrix1D<int> nh;
	nh.resize(depth+2);
	nh.initConstant(0);
	Numsph(nh);

	normClnm = 0.0;

	L = nh(2);
	prevL = nh(1);
	pos = 4*L;
	Matrix1D<double> p(pos+5), steps(pos+5);
	for (int h=1;h<VEC_XSIZE(nh)-1;h++)
	{
		if (h!=1)
		{
			L = nh(h+1);
			prevL = nh(h);
			pos = 4*L;
		}

		rowOut=rowIn;

		// Read input image and initial parameters
	//  ApplyGeoParams geoParams;
	//	geoParams.only_apply_shifts=false;
	//	geoParams.wrap=DONT_WRAP;

		rowIn.getValue(MDL_ANGLE_ROT,old_rot);
		rowIn.getValue(MDL_ANGLE_TILT,old_tilt);
		rowIn.getValue(MDL_ANGLE_PSI,old_psi);
		rowIn.getValue(MDL_SHIFT_X,old_shiftX);
		rowIn.getValue(MDL_SHIFT_Y,old_shiftY);
		rowIn.getValue(MDL_FLIP,old_flip);

		std::cout << "Processing " << fnImg << std::endl;
		I.read(fnImg);
		I().setXmippOrigin();

		Ifiltered()=I();
		filter.applyMaskSpace(Ifiltered());

    	if (h!=1)
    	{
    		p.resize(pos+5,false);
    		copyvectors(clnm,p);
    		clnm=p;
    	}
    	else
        	clnm=p;

        for(int d=VEC_XSIZE(p)-5-L+prevL;d<VEC_XSIZE(p)-5;d++)
    	{
        	if (Rmax==Xdim/2)
        	{
        		p(d)=2*Rmax;
        		clnm(d)=2*Rmax;
        	}
        	else
        	{
        		p(d)=Rmax;
        		clnm(d)=Rmax;
        	}
    	}

		// Optimize
		double cost=-1;
		try
		{
			cost=1e38;
			int iter;
			steps.initZeros();
			if (optimizeAlignment)
				steps(pos)=steps(pos+1)=steps(pos+2)=steps(pos+3)=steps(pos+4)=1.;
			if (optimizeDeformation)
			{
//				if (h!=1)
//				{
//					for (int i=0;i<3*pos/4;i++)
//					{
//						steps(i)=1.;
//					}
//					minimizepos(steps);
//				}
//				else
//				{
					for (int i=0;i<pos;i++)
					{
						steps(i) = 1.;
					}
//				}
			}
			penalization = true;
			if (penalization)
			{
				powellOptimizer(p, 1, pos+5, &continuousSphCost, this, 0.01, cost, iter, steps, true);
				penalization = false;
			}
			powellOptimizer(p, 1, pos+5, &continuousSphCost, this, 0.01, cost, iter, steps, true);

//#ifdef DEBUG
//			debug = true;
//			continuousSphCost(p.adaptForNumericalRecipes(),this);
//			debug = false;
//#endif
			if (cost>0)
			{
				rowOut.setValue(MDL_ENABLED,-1);
				p.initZeros();
			}
			cost=-cost;
			if (verbose>=2)
				std::cout<<std::endl;
				for (int j=1;j<5;j++)
				{
					switch (j)
					{
					case 1:
						std::cout << "X Coefficients=(";
						break;
					case 2:
						std::cout << "Y Coefficients=(";
						break;
					case 3:
						std::cout << "Z Coefficients=(";
						break;
					case 4:
						std::cout << "Radius=(";
						break;
					}
					for (int i=(j-1)*L;i<j*L;i++)
					{
						std::cout << p(i) << ",";
					}
					std::cout << ")" << std::endl;
				}
				std::cout << " Dshift=(" << p(pos) << "," << p(pos+1) << ") "
						  << "Drot=" << p(pos+2) << " Dtilt=" << p(pos+3) << " Dpsi=" << p(pos+4) << std::endl;
		}
		catch (XmippError XE)
		{
			std::cerr << XE << std::endl;
			std::cerr << "Warning: Cannot refine " << fnImg << std::endl;
			rowOut.setValue(MDL_ENABLED,-1);
		}

		rowOut.setValue(MDL_ANGLE_ROT,  old_rot+p(pos+2));
		rowOut.setValue(MDL_ANGLE_TILT, old_tilt+p(pos+3));
		rowOut.setValue(MDL_ANGLE_PSI,  old_psi+p(pos+4));
		rowOut.setValue(MDL_SHIFT_X,    old_shiftX+p(pos+0));
		rowOut.setValue(MDL_SHIFT_Y,    old_shiftY+p(pos+1));
		rowOut.setValue(MDL_FLIP,       old_flip);
		rowOut.setValue(MDL_COST,       cost);
//		p.clear();
//		steps.clear();
//		clnm.clear();
//		nh.clear();
	}
}
//#undef DEBUG

void ProgAngularSphAlignment::Numsph(Matrix1D<int> &sphD)
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

void ProgAngularSphAlignment::minimizepos(Matrix1D<double> &vectpos)
{
	size_t groups = 4;
	size_t olditems = (4*prevL)/groups;
	size_t newitems = (4*L)/groups;
//	size_t initial = 3*newitems+olditems;
	size_t initial = 3*newitems;
	for (int i=initial;i<pos;i++)
		{
			vectpos(i) = 1;
		}
//	for (int g=0;g<groups;g++)
//	{
//		vectpos(g*newitems) = 0;
//	}
}

void ProgAngularSphAlignment::copyvectors(Matrix1D<double> &oldvect,Matrix1D<double> &newvect)
{
	size_t groups = 4;
	size_t olditems = (4*prevL)/groups;
	size_t newitems = (4*L)/groups;
	for (int g=0;g<groups;g++)
	{
		for (int i=0;i<olditems;i++)
			{
			    newvect(g*newitems+i) = oldvect(g*olditems+i);
			}
	}
	for (int i=0;i<5;i++)
	{
		newvect(groups*newitems+i) = oldvect(groups*olditems+i);
	}
}

void ProgAngularSphAlignment::deformVol(MultidimArray<double> &mVD, const MultidimArray<double> &mV)
{
	int l,n,m;
	size_t idxY0=(VEC_XSIZE(clnm)-5)/4;
	size_t idxZ0=2*idxY0;
	size_t idxR=3*idxY0;
	for (int k=STARTINGZ(mV); k<=FINISHINGZ(mV); k++)
	{
		for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i++)
		{
			for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j++)
			{

				double gx=0.0, gy=0.0, gz=0.0;
				for (size_t idx=0; idx<idxY0; idx++)
				{
					double RmaxF=VEC_ELEM(clnm,idx+idxR);
					double RmaxF2=RmaxF*RmaxF;
					double iRmaxF=1.0/RmaxF;
					double k2=k*k;
					double kr=k*iRmaxF;
					double k2i2=k2+i*i;
					double ir=i*iRmaxF;
					double r2=k2i2+j*j;
					double jr=j*iRmaxF;
					double rr=sqrt(r2)*iRmaxF;
					spherical_index2lnm(idx+1,l,n,m);
					double zsph=ZernikeSphericalHarmonics(l,n,m,jr,ir,kr,rr);

					if (rr>0 || l==0)
					{
						gx += VEC_ELEM(clnm,idx)        *(zsph);
						gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
						gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
					}
				}
				mVD(k,i,j) = mV.interpolatedElement3D(j+gx,i+gy,k+gz);
			}
		}
	}
}

void ProgAngularSphAlignment::avgVector(Matrix1D<double> &vect, double &avg)
{
	size_t groups = 4;
	size_t newitems = (4*L)/groups;
	avg = 0.0;
	for (int i=0;i<3*newitems;i++)
	{
		avg += fabs(vect(i));
	}
	avg = avg/(3*newitems);
}

void ProgAngularSphAlignment::computeNum(Matrix1D<double> &vect, double &mod, double &num)
{
	num = 0.0;
	size_t groups = 4;
	size_t newitems = (4*L)/groups;
	double aux=0.0;
		for(int i=0; i<newitems; i++)
		{
			aux = fabs(clnm(i))+fabs(clnm(newitems+i))+fabs(clnm(2*newitems+i));
			num += fabs(aux-mod);
		}
}

