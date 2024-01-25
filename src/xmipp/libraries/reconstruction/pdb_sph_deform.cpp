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
#include <fstream>

void ProgPdbSphDeform::defineParams()
{
	addUsageLine("Deform a PDB according to a list of SPH deformation coefficients");
	addParamsLine("--pdb <file>            : PDB to deform");
	addParamsLine("--clnm <metadata_file>  : List of deformation coefficients");
	addParamsLine("-o <file>               : Deformed PDB");
	addParamsLine("[--center_mass]         : Center the PDB according with its center of mass");
	addParamsLine("[--boxsize <b=0> ]      : Box size of the volume where the PDB was fitted");
	addParamsLine("[--sr <s=1> ]           : Sampling rate of the volume where the PDB was fitted");
	addExampleLine("xmipp_pdb_sph_deform --pdb 2tbv.pdb -o 2tbv_deformed.pdb --clnm coefficients.txt");
}

void ProgPdbSphDeform::readParams()
{
	fn_pdb = getParam("--pdb");
	fn_sph = getParam("--clnm");
	fn_out = getParam("-o");
	center = checkParam("--center_mass");
	boxSize= 0.5 * getDoubleParam("--boxsize") * getDoubleParam("--sr");
}

void ProgPdbSphDeform::show() const
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
	std::string line;
	line = readNthLine(0);
	basisParams = string2vector(line);
	line = readNthLine(1);
	clnm = string2vector(line);
	fillVectorTerms();
	size_t idxY0=clnm.size()/3;
	size_t idxZ0=2*idxY0;
	Matrix1D<double> cm;
	cm.initZeros(3);
	centerOfMass(pdb, cm);
	for (size_t a=0; a<pdb.getNumberOfAtoms(); a++)
	{
		double gx=0.0; 
		double gy=0.0; 
		double gz=0.0;
		RichAtom& atom_i=pdb.atomList[a];
		auto k = atom_i.z;
		auto i = atom_i.y;
		auto j = atom_i.x;
		if (center)
		{
			k -= VEC_ELEM(cm, 2);
			i -= VEC_ELEM(cm, 1);
			j -= VEC_ELEM(cm, 0);
		}
		else
		{
			k -= boxSize;
			i -= boxSize;
			j -= boxSize;
		}
		for (size_t idx=0; idx<idxY0; idx++)
		{
			double Rmax=basisParams[2];
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
				int l1 = VEC_ELEM(vL1,idx);
				int n = VEC_ELEM(vN,idx);
				int l2 = VEC_ELEM(vL2,idx);
				int m = VEC_ELEM(vM,idx);
				zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
				if (rr>0 || l2==0)
				{
					gx += clnm[idx]        *zsph;
					gy += clnm[idx+idxY0]  *zsph;
					gz += clnm[idx+idxZ0]  *zsph;
				}
			}
		}
		atom_i.x += gx;
		atom_i.y += gy;
		atom_i.z += gz;
	}
	pdb.write(fn_out);
}

std::string ProgPdbSphDeform::readNthLine(int N) const
{
	std::ifstream in(fn_sph.getString());
	std::string s;  

	//skip N lines
	for(int i = 0; i < N; ++i)
		std::getline(in, s);

	std::getline(in,s);
	return s;
}

std::vector<double> ProgPdbSphDeform::string2vector(std::string const &s) const
{
	std::stringstream iss(s);
    double number;
    std::vector<double> v;
    while (iss >> number)
        v.push_back(number);
    return v;
}

void ProgPdbSphDeform::fillVectorTerms()
{
    int idx = 0;
	auto vecSize = (int)(clnm.size()/3);
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

void ProgPdbSphDeform::centerOfMass(PDBRichPhantom pdb, Matrix1D<double> &cm) 
{
	size_t numAtoms = pdb.getNumberOfAtoms();
	for (size_t a=0; a<numAtoms; a++)
	{
		RichAtom& atom_i=pdb.atomList[a];
		auto z = atom_i.z;
		auto y = atom_i.y;
		auto x = atom_i.x;
		VEC_ELEM(cm, 0) += x;
		VEC_ELEM(cm, 1) += y;
		VEC_ELEM(cm, 2) += z;

	}
	VEC_ELEM(cm, 0) /= numAtoms;
	VEC_ELEM(cm, 1) /= numAtoms;
	VEC_ELEM(cm, 2) /= numAtoms;
}