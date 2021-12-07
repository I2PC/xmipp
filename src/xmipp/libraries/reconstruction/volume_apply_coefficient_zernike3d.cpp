/***************************************************************************
 *
 * Authors:    David Herreros Calero             dherreros@cnb.csic.es
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

#include <data/numerical_tools.h>
#include <data/basis.h>
#include "volume_apply_coefficient_zernike3d.h"
#include <data/numerical_tools.h>
#include <fstream>
#include "data/mask.h"

void ProgApplyCoeffZernike3D::defineParams()
{
	addUsageLine("Deform a PDB according to a list of SPH deformation coefficients");
	addParamsLine("-i <volume>                    : Volume to deform");
	addParamsLine("  [--mask <m=\"\">]            : Reference volume");
	addParamsLine("--clnm <metadata_file>         : List of deformation coefficients");
	addParamsLine("-o <volume>                    : Deformed volume");
	addParamsLine("  [--step <step=1>]            : Voxel index step");
	addParamsLine("  [--blobr <b=4>]              : Blob radius for forward mapping splatting");
	addExampleLine("xmipp_apply_deform_sph -i input.vol -o volume_deformed.vol --clnm coefficients.txt");
}

void ProgApplyCoeffZernike3D::readParams()
{
	fn_vol=getParam("-i");
	fn_sph=getParam("--clnm");
	fn_mask = getParam("--mask");
	blob_r = getDoubleParam("--blobr");
	loop_step = getIntParam("--step");
	fn_out=getParam("-o");
}

void ProgApplyCoeffZernike3D::show() const
{
	if (verbose==0)
		return;
	std::cout
	<< "Volume:              "  << fn_vol     << std::endl
	<< "Reference mask:      "  << fn_mask    << std::endl
	<< "Coefficient list:    "  << fn_sph     << std::endl
	<< "Step:                "  << loop_step  << std::endl
	<< "Blob radius:         "  << blob_r     << std::endl
	<< "Output:              "  << fn_out     << std::endl
	;
}

void ProgApplyCoeffZernike3D::run()
{
	Image<double> VI;
	Image<double> VO;
    MultidimArray<int> V_mask;
	VI.read(fn_vol);
	VI().setXmippOrigin();
	VO().initZeros(VI());
	VO().setXmippOrigin();

	// Blob
	blob.radius = blob_r;   // Blob radius in voxels
	blob.order  = 2;        // Order of the Bessel function
    blob.alpha  = 3.6;      // Smoothness parameter

	std::string line;
	line = readNthLine(0);
	basisParams = string2vector(line);
	line = readNthLine(1);
	clnm = string2vector(line);
	fillVectorTerms();
	size_t idxY0=(clnm.size())/3;
	size_t idxZ0=2*idxY0;
	const MultidimArray<double> &mVI=VI();
	MultidimArray<double> &mVO=VO();
	double voxelI=0.0;
	double Rmax=basisParams[2];
	double Rmax2=Rmax*Rmax;
	double iRmax=1.0/Rmax;

	Mask mask;
	mask.type = BINARY_CIRCULAR_MASK;
	mask.mode = INNER_MASK;
	if (fn_mask != "") {
		Image<double> aux;
		aux.read(fn_mask);
		typeCast(aux(), V_mask);
		V_mask.setXmippOrigin();
		for (int k=STARTINGZ(V_mask); k<=FINISHINGZ(V_mask); k++) {
			for (int i=STARTINGY(V_mask); i<=FINISHINGY(V_mask); i++) {
				for (int j=STARTINGX(V_mask); j<=FINISHINGX(V_mask); j++) {
					double r2 = k*k + i*i + j*j;
					if (r2>=Rmax2)
						A3D_ELEM(V_mask,k,i,j) = 0;
				}
			}
		}
	}
	else {
		mask.R1 = Rmax;
		mask.generate_mask(VI());
		V_mask = mask.get_binary_mask();
		V_mask.setXmippOrigin();
	}

	for (int k=STARTINGZ(mVI); k<=FINISHINGZ(mVI); k+=loop_step)
	{
		for (int i=STARTINGY(mVI); i<=FINISHINGY(mVI); i+=loop_step)
		{
			for (int j=STARTINGX(mVI); j<=FINISHINGX(mVI); j+=loop_step)
			{
				if (A3D_ELEM(V_mask,k,i,j) == 1) {
					double gx=0.0;
					double gy=0.0;
					double gz=0.0;
					double k2=k*k;
					double kr=k*iRmax;
					double k2i2=k2+i*i;
					double ir=i*iRmax;
					double r2=k2i2+j*j;
					double jr=j*iRmax;
					double rr=std::sqrt(r2)*iRmax;
					if (r2<Rmax2)
					{
						for (size_t idx=0; idx<idxY0; idx++)
						{
							auto l1 = VEC_ELEM(vL1,idx);
							auto n = VEC_ELEM(vN,idx);
							auto l2 = VEC_ELEM(vL2,idx);
							auto m = VEC_ELEM(vM,idx);
							auto zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
							if (rr>0 && l2!=0)
							{
								gx += clnm[idx]        *zsph;
								gy += clnm[idx+idxY0]  *zsph;
								gz += clnm[idx+idxZ0]  *zsph;
							}
						}
					}

					if (fn_mask == "") 
					{
						voxelI=mVI.interpolatedElement3D(j+gx,i+gy,k+gz);
						A3D_ELEM(VO(), k, i, j)=voxelI;
					}
					else 
					{
						auto pos = std::array<double, 3>{};
						pos[0] = j + gx;
						pos[1] = i + gy;
						pos[2] = k + gz;
						double voxel_mV = A3D_ELEM(mVI,k,i,j);
						splattingAtPos(pos, voxel_mV, mVO);
					}

				}
			}
		}
	}
	VO.write(fn_out);
}

std::string ProgApplyCoeffZernike3D::readNthLine(int N) const
{
	std::ifstream in(fn_sph.getString());
	std::string s;  

	//skip N lines
	for(int i = 0; i < N; ++i)
		std::getline(in, s);

	std::getline(in,s);
	return s;
}

std::vector<double> ProgApplyCoeffZernike3D::string2vector(std::string const &s) const
{
	std::stringstream iss(s);
    double number;
    std::vector<double> v;
    while (iss >> number)
        v.push_back(number);
    return v;
}

void ProgApplyCoeffZernike3D::fillVectorTerms()
{
    int idx = 0;
	auto vecSize = (int)((clnm.size())/3);
	vL1.initZeros(vecSize);
	vN.initZeros(vecSize);
	vL2.initZeros(vecSize);
	vM.initZeros(vecSize);
    for (int h=0; h<=basisParams[1]; h++)
    {
        int totalSPH = 2*h+1;
        auto aux = (int)(std::floor(totalSPH/2));
        for (int l=h; l<=basisParams[0]; l+=2)
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

void ProgApplyCoeffZernike3D::splattingAtPos(std::array<double, 3> r, double weight, const MultidimArray<double> &mVO) {
	// Find the part of the volume that must be updated
	double x_pos = r[0];
	double y_pos = r[1];
	double z_pos = r[2];
	int k0 = XMIPP_MAX(FLOOR(z_pos - blob_r), STARTINGZ(mVO));
	int kF = XMIPP_MIN(CEIL(z_pos + blob_r), FINISHINGZ(mVO));
	int i0 = XMIPP_MAX(FLOOR(y_pos - blob_r), STARTINGY(mVO));
	int iF = XMIPP_MIN(CEIL(y_pos + blob_r), FINISHINGY(mVO));
	int j0 = XMIPP_MAX(FLOOR(x_pos - blob_r), STARTINGX(mVO));
	int jF = XMIPP_MIN(CEIL(x_pos + blob_r), FINISHINGX(mVO));
	// Perform splatting at this position r
	for (int k = k0; k <= kF; k++)
		for (int i = i0; i <= iF; i++)
			for (int j = j0; j <= jF; j++)
			{
				double mod = sqrt((x_pos - j) * (x_pos - j) + (y_pos - i) * (y_pos - i) + (z_pos - k) * (z_pos - k));
				A3D_ELEM(mVO,k, i, j) += weight * blob_val(mod, blob);
			}
}