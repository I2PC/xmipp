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
#include <numeric>
#include "data/cpu.h"
#include "volume_deform_sph.h"
#include "data/fourier_filter.h"
#include "data/normalize.h"

// Params definition =======================================================
// -i ---> V2 (paper) / -r --> V1 (paper)
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
	addParamsLine("   [--thr <N=-1>]                      : Maximal number of the processing CPU threads");
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
    int threads = getIntParam("--thr");
    if (0 >= threads) {
        threads = CPU::findCores();
    }
    m_threadPool.resize(threads);
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

void ProgVolDeformSph::computeShift(int k) {
    const auto &mVR = VR();
    const double Rmax2=Rmax*Rmax;
    const double iRmax = 1.0 / Rmax;
    size_t vec_idx = mVR.yxdim * (k - STARTINGZ(mVR));
    const auto &clnm_c = m_clnm;
    const auto &zsh_vals_c = m_zshVals;
        for (int i=STARTINGY(mVR); i<=FINISHINGY(mVR); i++) {
            for (int j=STARTINGX(mVR); j<=FINISHINGX(mVR); j++, ++vec_idx) {
                const auto r_vals = Radius_vals(i, j, k, iRmax);
                if (r_vals.r2 < Rmax2) {
                    auto &g = m_shifts[vec_idx] = 0;
                    for (size_t idx=0; idx<clnm_c.size(); idx++) {
                        if (clnm_c[idx].x != 0) {
                            auto &tmp = zsh_vals_c[idx];
                            if (r_vals.rr>0 || tmp.l2==0) {
                                double zsph=ZernikeSphericalHarmonics(tmp.l1,tmp.n,tmp.l2,tmp.m,
                                        r_vals.jr, r_vals.ir, r_vals.kr, r_vals.rr);
                                auto &c = clnm_c[idx];
                                g.x += c.x * zsph;
                                g.y += c.y * zsph;
                                g.z += c.z * zsph;
                            }
                        }
                    }
                }
            }
        }
//    }
}

template<bool APPLY_TRANSFORM, bool SAVE_DEFORMATION>
void ProgVolDeformSph::computeDistance(Distance_vals &vals) {
    const auto &mVR = VR();
    const auto &VR_c = VR;
    const auto &VI_c = VI;
    const auto &VO_c = VO;
    const auto &volumesR_c = volumesR;
    const auto &volumesI_c = volumesI;
    size_t vec_idx = 0;
    for (int k=STARTINGZ(mVR); k<=FINISHINGZ(mVR); k++) {
        for (int i=STARTINGY(mVR); i<=FINISHINGY(mVR); i++) {
            for (int j=STARTINGX(mVR); j<=FINISHINGX(mVR); j++, ++vec_idx) {
                const auto &g = m_shifts[vec_idx];
                if (APPLY_TRANSFORM) {
                    double voxelR = A3D_ELEM(VR_c(),k,i,j);
                    double voxelI = VI_c().interpolatedElement3D(j+g.x,i+g.y,k+g.z);
                    if (voxelI >= 0.0)
                        vals.VD += voxelI;
                    VO_c(k,i,j) = voxelI;
                    double diff = voxelR-voxelI;
                    vals.diff += diff * diff;
                    vals.modg += (g.x * g.x) + (g.y * g.y) + (g.z * g.z);
                    vals.count++;
                }
                for (int idv=0; idv<volumesR_c.size(); idv++) {
                    double voxelR = A3D_ELEM(volumesR_c[idv](),k,i,j);
                    double voxelI = volumesI_c[idv]().interpolatedElement3D(j+g.x,i+g.y,k+g.z);
                    if (voxelI >= 0.0)
                        vals.VD += voxelI;
                    double diff = voxelR - voxelI;
                    vals.diff += diff * diff;
                    vals.modg += (g.x * g.x) + (g.y * g.y) + (g.z * g.z);
                    vals.count++;
                }

                if (SAVE_DEFORMATION) {
                    Gx(k,i,j) = g.x;
                    Gy(k,i,j) = g.y;
                    Gz(k,i,j) = g.z;
                }
            }
        }
    }
}

void ProgVolDeformSph::computeDistance(int k, Distance_vals &vals) {
    const auto &mVR = VR();
    for (int idv=0; idv<volumesR.size(); idv++) {
        size_t voxel_idx = mVR.yxdim * (k - STARTINGZ(mVR));
        const auto &volR = volumesR[idv]();
        const auto &volI = volumesI[idv]();
            for (int i=STARTINGY(mVR); i<=FINISHINGY(mVR); i++) {
                for (int j=STARTINGX(mVR); j<=FINISHINGX(mVR); j++, ++voxel_idx) {
                    const auto &g = m_shifts[voxel_idx];
                    double voxelR = volR.data[voxel_idx];
                    double voxelI = volI.interpolatedElement3D(j+g.x,i+g.y,k+g.z);
                    if (voxelI >= 0.0) // background could be smaller than 0
                        vals.VD += voxelI;
                    double diff = voxelR - voxelI;
                    vals.diff += diff * diff;
                    vals.modg += (g.x * g.x) + (g.y * g.y) + (g.z * g.z);
                    vals.count++;
                }
            }
        }
//    }
}

ProgVolDeformSph::Distance_vals ProgVolDeformSph::computeDistance() {
    const bool usual_case = ( ! applyTransformation) && ( ! saveDeformation);
    // parallelize at the level of Z slices
    const auto &mVR = VR();
    m_shifts.resize(mVR.zyxdim);
    auto futures = std::vector<std::future<void>>();
    futures.reserve(mVR.zdim);
    auto vals = std::vector<Distance_vals>(m_threadPool.size());
    // most common routine, run during the computation
    // for each Z slice, compute shift and distance
    auto usual_routine = [this, &vals](int thrId, int k) {
        computeShift(k);
        computeDistance(k, vals[thrId]);
    };
    // special routine, run at the end for the computation
    // for each Z slice, compute shift. Distance will be computed later
    auto special_routine = [this](int thrId, int k) {
        computeShift(k);
    };
    for (int k=STARTINGZ(mVR); k<=FINISHINGZ(mVR); ++k) {
        if (usual_case) {
            futures.emplace_back(m_threadPool.push(usual_routine, k));
        } else {
            futures.emplace_back(m_threadPool.push(special_routine, k));
        }
    }
    // wait till all slices are processed
    for (auto &f : futures) {
        f.get();
    }
    if ( ! usual_case) {
        // shifts have been already computed, now compute the final distance
        if (applyTransformation) {
            if (saveDeformation) {
                computeDistance<true, true>(vals[0]);
            } else {
                computeDistance<true, false>(vals[0]);
            }
        } else {
            if (saveDeformation) {
                computeDistance<false, true>(vals[0]);
            } else {
                computeDistance<false, false>(vals[0]);
            }
        }
    }
    // merge results
    return std::accumulate(vals.begin(), vals.end(), Distance_vals());
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
	const MultidimArray<double> &mVR=VR();
	const MultidimArray<double> &mVI=VI();
	const size_t size = m_clnm.size();
	for (size_t i = 0; i < size; ++i) {
	    auto &p = m_clnm[i];
	    p.x = pclnm[i + 1];
	    p.y = pclnm[i + size + 1];
	    p.z = pclnm[i + size + size + 1];
	}

    const auto distance_vals = computeDistance();
	deformation=std::sqrt(distance_vals.modg / (double)distance_vals.count);
	sumVD = distance_vals.VD;

	if (applyTransformation)
		VO.write(fnVolOut);

	double massDiff=std::abs(sumVI-sumVD)/sumVI;
	return std::sqrt(distance_vals.diff / (double)distance_vals.count)+lambda*(deformation+massDiff);
}
#undef DEBUG

double volDeformSphGoal(double *p, void *vprm)
{
	auto *prm=(ProgVolDeformSph *) vprm;
	return prm->distance(p);
}

// Run =====================================================================
void ProgVolDeformSph::run() {
	saveDeformation=false;

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
	Image<double> auxI = VI;
	Image<double> auxR = VR;

	MultidimArray<int> bg_mask;
	bg_mask.resizeNoCopy(VI().zdim, VI().ydim, VI().xdim);
    bg_mask.setXmippOrigin();
	normalize_Robust(auxI(), bg_mask, true);
	bg_mask *= 0;
	normalize_Robust(auxR(), bg_mask, true);

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(auxI())
	{
		if (DIRECT_A3D_ELEM(auxI(),k,i,j) >= 0.0)
    		sumVI += DIRECT_A3D_ELEM(auxI(),k,i,j);
	}

	volumesI.push_back(auxI());
	volumesR.push_back(auxR());
	if (sigma.size() > 1 || sigma[0] != 0)
	{
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

			// Filter ref vol
			filter.applyMaskSpace(auxR());
			bg_mask *= 0;
			normalize_Robust(auxR(), bg_mask, true);
			volumesR.push_back(auxR);
		}
	}

    Matrix1D<double> steps;
	Matrix1D<double> x;
	vecSize = 0;
	numCoefficients(L1,L2,vecSize);
	size_t totalSize = 3*vecSize;
	fillVectorTerms(L1,L2);
	m_clnm.resize(vecSize);
	x.initZeros(totalSize);
    for (int h=0;h<=L2;h++)
    {
		steps.clear();
    	steps.initZeros(totalSize);
		minimizepos(h,steps);

    	std::cout<<std::endl;
    	std::cout<<"-------------------------- Basis Degrees: ("<<L1<<","<<h<<") --------------------------"<<std::endl;
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
        int iter;
        double fitness;
        powellOptimizer(x, 1, (int)(totalSize), &volDeformSphGoal, this,
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

void ProgVolDeformSph::minimizepos(int l2, Matrix1D<double> &steps) const
{
    int size = 0;
	numCoefficients(L1,l2,size);
    auto totalSize = (int)(steps.size()/3);
    for (int idx=0; idx<size; idx++)
    {
        VEC_ELEM(steps,idx) = 1;
        VEC_ELEM(steps,idx+totalSize) = 1;
        VEC_ELEM(steps,idx+2*totalSize) = 1;
    }	
}

void ProgVolDeformSph::numCoefficients(int l1, int l2, int &nc) const
{
	// l1 -> Degree Zernike
	// l2 & h --> Degree SPH
    for (int h=0; h<=l2; h++)
    {
		// For the current SPH degree (h), determine the number of SPH components/equations
        int numSPH = 2*h+1;
		// Finf the total number of radial components with even degree for a given l1 and h
        int count=l1-h+1;
        int numEven=(count>>1)+(count&1 && !(h&1));
		// Total number of components is the number of SPH as many times as Zernike components
        if (h%2 == 0)
            nc += numSPH*numEven;
        else
        	nc += numSPH*(l1-h+1-numEven);
    }
}

void ProgVolDeformSph::fillVectorTerms(int l1, int l2)
{
    m_zshVals.resize(vecSize);
    int idx = 0;
    for (int h=0; h<=l2; h++)
    {
        int totalSPH = 2*h+1;
        auto aux = (int)(std::floor(totalSPH/2));
        for (int l=h; l<=l1; l+=2)
        {
            for (int m=0; m<totalSPH; m++)
            {
                auto &t = m_zshVals[idx];
                t.l1 = l;
                t.n = h;
                t.l2 = h;
                t.m = m - aux;
                idx++;
            }
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

void ProgVolDeformSph::writeVector(std::string const &outPath, Matrix1D<double> const &v, 
								   bool append) const
{
    std::ofstream outFile;
    if (append)
		outFile.open(outPath, std::ios_base::app);
    else
        outFile.open(outPath);
    FOR_ALL_ELEMENTS_IN_MATRIX1D(v)
        outFile << VEC_ELEM(v,i) << " ";
    outFile << std::endl;
}


