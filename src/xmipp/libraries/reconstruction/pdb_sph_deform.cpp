/***************************************************************************
 *
 * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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

#include "pdb_sph_deform.h"
#include <data/numerical_tools.h>

void ProgPdbSphDeform::defineParams()
{
	addUsageLine("Deform a PDB according to a list of SPH deformation coefficients");
	addParamsLine("--pdb <file>            : PDB to deform");
	addParamsLine("--clnm <metadata_file>  : List of deformation coefficients");
	addParamsLine("--l1 <metadata_file>  : : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("-o <file>               : Deformed PDB");
	addExampleLine("xmipp_pdb_sph_deform --pdb 2tbv.pdb -o 2tbv_deformed.pdb --clnm coefficients.txt");
}

void ProgPdbSphDeform::readParams()
{
	fn_pdb=getParam("--pdb");
	fn_sph=getParam("--clnm");
	maxl1 = getIntParam("--l1");
	fn_out=getParam("-o");
}

void ProgPdbSphDeform::show()
{
	if (verbose==0)
		return;
	std::cout
	<< "PDB:                  " << fn_pdb << std::endl
	<< "Coefficient list:     " << fn_sph << std::endl
	<< "Output:               " << fn_out << std::endl
	;
}

void ProgPdbSphDeform::run()
{
	PDBRichPhantom pdb;
	pdb.read(fn_pdb);
	int nCoeff = numberCoefficients();
	clnm.resize(nCoeff);
	clnm.read(fn_sph);
	int l1,n,l2,m;
	size_t idxY0=VEC_XSIZE(clnm)/3;
	size_t idxZ0=2*idxY0;
	size_t idxR=3*idxY0;
	for (size_t a=0; a<pdb.getNumberOfAtoms(); a++)
	{
		double gx=0.0, gy=0.0, gz=0.0;
		RichAtom& atom_i=pdb.atomList[a];
		int k = atom_i.z;
		int i = atom_i.y;
		int j = atom_i.x;
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
				spherical_index2lnm(idx,l1,n,l2,m,maxl1);
				zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
			}
			if (rr>0 || (l2==0 && l1==0))
			{
				gx += VEC_ELEM(clnm,idx)        *(zsph);
				gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
				gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
			}
		}
		atom_i.x += gx;
		atom_i.y += gy;
		atom_i.z += gz;
	}
	pdb.write(fn_out);
}

int ProgPdbSphDeform::numberCoefficients()
{
	int nCoeff = 0;
	std::ifstream coeff_file;
	coeff_file.open(fn_sph.getString());
	float i;
	while (coeff_file >> i)
	{
		nCoeff++;
	}
	return nCoeff;
}
