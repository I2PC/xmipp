/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 		jlvilas@cnb.csic.es
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

#include "resolution_pdb_bfactor.h"



void ProgResBFactor::readParams()
{
	fn_pdb = getParam("--atmodel");
	fn_locres = getParam("--vol");
	sampling = getDoubleParam("--sampling");
	medianTrue = checkParam("--hasMedian");
	fscResolution = getDoubleParam("--fscResolution");
	fnOut = getParam("-o");
}


void ProgResBFactor::defineParams()
{
	addUsageLine("The local b-factor of an atomic model (from xray or cryoem) are compared with the local resolution of a cryoEM map. To do that, the atom positions are selected and the local resolution around that atom is taken to determine an estimation of the local resolution of the residue. Only alpha carbons are selected to carry out this task. Tha output is a metadata with the residues, local resolution estimation of the residue, and local bfactor");
	addParamsLine("  --atmodel <pdb_file=\"\">   		: Atomic model (pdb). Ensure it is aligned/fitted to the local resolution map");
	addParamsLine("  --vol <vol_file=\"\">			: Local resolution map");
	addParamsLine("  [--sampling <sampling=1>]		: Sampling Rate (A)");
	addParamsLine("  [--hasMedian]			        : The resolution an bfactor per residue are averaged instead of computed the median");
	addParamsLine("  [--fscResolution <fscResolution=-1>]	: If this is provided, the FSC resolution, R, in Angstrom is used to normalized the local resolution, LR, as (LR-R)/R, where LR is the local resoluion and R is the global resolution");
	addParamsLine("  -o <output=\"amap.mrc\">		: Output of the algorithm");
}


template <typename T>
std::vector<size_t> ProgResBFactor::sort_indexes(const std::vector<T> &v)
{
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}



void ProgResBFactor::sweepByResidue(std::vector<double> &residuesToChimera)
{
	// Reading Local Resolution Map
	Image<double> imgResVol;
	imgResVol.read(fn_locres);
	MultidimArray<double> resvol;
	resvol = imgResVol();

	// The mask is initialized with the same dimentions of the local resolution map
        MultidimArray<int> mask;
	mask.resizeNoCopy(resvol);
	mask.initZeros();

	// Getting dimensions of th local resolution map to match atom position and local resolution information
	size_t xdim, ydim, zdim, ndim;
	resvol.getDimensions(xdim, ydim, zdim, ndim);

	if (zdim == 1)
		zdim = ndim;

	// idx_residue determines the positional index of the residue
	// Note that residues are stored in the vector at_pos.residue.
	// Then, the vector at_pos.residue is sorted from low to high
	// The new vector idx_residue, determine the position of each
	// residue in the original vector.
	std::vector<size_t> idx_residue;
	idx_residue = sort_indexes(at_pos.residue);

	// The resolution of each residue is stored in each component (as many as components as residues)
	std::vector<double> resolution_per_residue(0);

	// The contributing voxels to a single residue are stored
	std::vector<double> resolution_to_estimate(0);

	// the b-factor per residue
	std::vector<double> bfactor_per_residue(0), resNumberList(0);

	// vector to estimate the bfactor of each residue
	std::vector<double> bfactor_to_estimate(0);

	// vectors to estimate the moving average
	std::vector<double> ma_l(0);
	std::vector<double> ma_c(0);

	// Setting the first residue before the loop that sweeps all residues
	size_t r = 0;
	size_t idx = idx_residue[r];

	// The b-factor and resolution mean per residue, as well as number of voxels contributing to the mean
	double bfactor_mean = 0;
	double resolution_mean = 0;
	int N_elems = 0;

	MetaData md;
	size_t objId;

	// Selecting the residue
	int resi, last_resi;

	size_t first_index = 0;
	size_t last_index = idx_residue.size()-1;

	last_resi = at_pos.residue[idx_residue[first_index]];

	for (size_t r=first_index; r<numberOfAtoms; ++r)
	{
		// Index of the residue in the residue vector
		idx = idx_residue[r];

		// Selecting the residue
		resi = at_pos.residue[r];

		if (resi != last_resi)
		{
			double res_resi, bfactor_resi;
			// The resolution per residue is estimated using the mean of the median
			if (medianTrue)
			{
				std::sort(resolution_to_estimate.begin(), resolution_to_estimate.end());
				std::sort(bfactor_to_estimate.begin(), bfactor_to_estimate.end());

				res_resi = resolution_to_estimate[size_t(resolution_to_estimate.size()*0.5)];
				bfactor_resi = bfactor_to_estimate[size_t(bfactor_to_estimate.size()*0.5)];

			}else
			{
				res_resi = resolution_mean/N_elems;
				bfactor_resi = bfactor_mean/bfactor_to_estimate.size();
			}

                        // If the fsc is provided, then, the local resolution is normalized
			if (fscResolution>0)
			{
				res_resi -= fscResolution;
				res_resi /= fscResolution;
			}

			// The resolution of the residue is stored
			resolution_per_residue.push_back(res_resi);
			bfactor_per_residue.push_back(bfactor_resi);

			// 0.3 and 0.4 are weigths to smooth the output
			ma_l.push_back(res_resi*0.3);
			ma_c.push_back(res_resi*0.4);
			resNumberList.push_back(last_resi);

			last_resi = resi;

			// The means are reseted
			bfactor_mean = 0;
			resolution_mean = 0;
			N_elems = 0;
			resolution_to_estimate.clear();
			bfactor_to_estimate.clear();
		}

		// Getting the atom position
		int k = round(at_pos.z[idx]/sampling) + floor(zdim/2);
		int i = round(at_pos.y[idx]/sampling) + floor(ydim/2);
		int j = round(at_pos.x[idx]/sampling) + floor(xdim/2);

		// Covalent Radius of the atom
		double covRad = at_pos.atomCovRad[idx];

		// Thermal displacement
		double bfactorRad = at_pos.b[idx];

		// Storing the bfactor per residue to compute the median
		bfactor_to_estimate.push_back(bfactorRad);

		// Updating the mean of bfactor
		bfactor_mean += bfactorRad;

		// Total Displacement in voxels
		int totRad = 3; //round( (covRad + bfactorRad)/sampling );

		// All resolutions around the atom position are stored in a vector to estimate
                // the local resolution of the residue
		estimatingResolutionOfResidue(k, i, j, totRad, mask, resvol, resolution_mean, N_elems, resolution_to_estimate);
		

	}

	// Smoothing the output
	std::vector<double> smoothedResolution(ma_c.size());
	smoothedResolution[0]=(ma_c[0]+ma_l[1])/(0.3+0.4); //0.3 left weight,, 0.4 central weight

	for (size_t i=1; i<(ma_c.size()-1); ++i)
	{
		smoothedResolution[i] = ma_l[i-1] + ma_c[i] + ma_l[i+1];
	}
	smoothedResolution[ma_c.size()-1] = (ma_c[ma_c.size()-1]+ma_l[ma_c.size()-2])/(0.3+0.4);

	std::vector<double> residuesToChimera_aux(at_pos.residue[idx_residue[last_index]]);

	// Creation of the output metadata with the local resolution per residue
	MetaData mdSmooth;
	size_t objsmth;
	for (size_t nn = 0; nn<resolution_per_residue.size(); ++nn)
	{
		double bf, lr;
		int resnumber;

		lr = resolution_per_residue[nn];
		bf = bfactor_per_residue[nn];
		resnumber = resNumberList[nn];

		int idx_resnumber = (int) resnumber;

		lr = smoothedResolution[nn];
		residuesToChimera_aux[idx_resnumber-1] = lr;

		objsmth = md.addObject();
		md.setValue(MDL_BFACTOR, bf, objsmth);
		md.setValue(MDL_RESIDUE, resnumber, objsmth);
		md.setValue(MDL_RESOLUTION_LOCAL_RESIDUE, lr, objsmth);
	}

	FileName fn;

	fn = fnOut + "/bfactor_resolution.xmd";
	md.write(fn);

        residuesToChimera = residuesToChimera_aux;
	Image<int> imMask;
	imMask() = mask;
        fn = fnOut + "/mask.mrc";
	imMask.write(fn);

}

void ProgResBFactor::estimatingResolutionOfResidue(int k, int i, int j, int totRad, MultidimArray<int> &mask, MultidimArray<double> &resvol, double &resolution_mean, int &N_elems, std::vector<double> &resolution_to_estimate)
{
    int dim = totRad*totRad;

    for (size_t kk = 0; kk<totRad; ++kk)
	{
	size_t kk2 = kk * kk;
	for (size_t jj = 0; jj<totRad; ++jj)
	{
		size_t jj2kk2 = jj * jj + kk2;
		for (size_t ii = 0; ii<totRad; ++ii)
		{
			size_t dist2 = (ii)*(ii) + (jj)*(jj) + (kk)*(kk);
			if (dist2 <= dim)
			{
				A3D_ELEM(mask, k-kk, i-ii, j-jj) = 1;
				A3D_ELEM(mask, k-kk, i-ii, j+jj) = 1;
				A3D_ELEM(mask, k-kk, i+ii, j-jj) = 1;
				A3D_ELEM(mask, k-kk, i+ii, j+jj) = 1;
				A3D_ELEM(mask, k+kk, i-ii, j-jj) = 1;
				A3D_ELEM(mask, k+kk, i-ii, j+jj) = 1;
				A3D_ELEM(mask, k+kk, i+ii, j-jj) = 1;
				A3D_ELEM(mask, k+kk, i+ii, j+jj) = 1;

				double &z1 = A3D_ELEM(resvol, k-kk, i-ii, j-jj);
				resolution_to_estimate.push_back(z1);
				resolution_mean += z1;
				N_elems++;

				double &z2 = A3D_ELEM(resvol, k-kk, i-ii, j+jj);
				resolution_to_estimate.push_back(z2);
				resolution_mean += z2;
				N_elems++;

				double &z3 = A3D_ELEM(resvol, k-kk, i+ii, j-jj);
				resolution_to_estimate.push_back(z3);
				resolution_mean += z3;
				N_elems++;

				double &z4 = A3D_ELEM(resvol, k-kk, i+ii, j+jj);
				resolution_to_estimate.push_back(z4);
				resolution_mean += z4;
				N_elems++;

				double &z5 = A3D_ELEM(resvol, k+kk, i-ii, j-jj);
				resolution_to_estimate.push_back(z5);
				resolution_mean += z5;

				double &z6 = A3D_ELEM(resvol, k+kk, i-ii, j+jj);
				resolution_to_estimate.push_back(z6);
				resolution_mean += z6;
				N_elems++;

				double &z7 = A3D_ELEM(resvol, k+kk, i+ii, j-jj);
				resolution_to_estimate.push_back(z7);
				resolution_mean += z7;
				N_elems++;

				double &z8 = A3D_ELEM(resvol, k+kk, i+ii, j+jj);
				resolution_to_estimate.push_back(z8);
				resolution_mean += z8;
				N_elems++;
			}
		}
	}
    }
}



void ProgResBFactor::generateOutputPDB(const std::vector<double> &residuesToChimera)
{
	//Open the pdb file
	std::ifstream f2parse;
	f2parse.open(fn_pdb.c_str());

	std::ofstream pdbToChimera;
	pdbToChimera.open(fnOut+"/chimeraPDB.pdb");

	int last_resi = 0;

	while (!f2parse.eof())
	{
		std::string line;
		getline(f2parse, line);

//		// The type of record (line) is defined in the first 6 characters of the pdb
		std::string typeOfline = line.substr(0,4);

		if ( (typeOfline == "ATOM") || (typeOfline == "HETA"))
		{
			std::string lineInit = line.substr(0,61);
			std::string lineEnd = line.substr(66,  std::string::npos);

                        // The residue is read
			int resi = (int) textToFloat(line.substr(23,5));
			std::string lineMiddle;
			int digitNumber = 5;
			
			std::string auxstr = std::to_string(residuesToChimera[resi-1]);

			// The bfactor column has 5 digits so we set the normalized resolution to 
                        // 5 digits
			auxstr = auxstr.substr(0, 5);
                        std::stringstream ss;
			ss << std::setfill('0') << std::setw(5) << residuesToChimera[resi-1];
                        lineMiddle = ss.str();

			std::string linePDB;
			linePDB = lineInit + lineMiddle + lineEnd;
			pdbToChimera << linePDB << std::endl;
		}
		else
		{
			pdbToChimera << line << std::endl;
		}
	}
	pdbToChimera.close();
}



void ProgResBFactor::run()
{
	std::cout << "Start" << std::endl;

        // Reading the atomic model and getting the atom positions
	//analyzePDB();

        std::string typeOfAtom = "CA";
        numberOfAtoms = 0;
        analyzePDBAtoms(fn_pdb, typeOfAtom, numberOfAtoms, at_pos);

	// Estimating the local resolution per residue
	std::vector<double> residuesToChimera;
	sweepByResidue(residuesToChimera);
	
	// Output generation: a new atomic model is generated by substituting the bfactor column by
        // the local resolution of the residue. Also a smoothing is carried out
	generateOutputPDB(residuesToChimera);

	std::cout << "Finished" << std::endl;
}
