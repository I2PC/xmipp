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

#include "classify_first_split3.h"
#include <core/symmetries.h>
#include <data/filters.h>

// Read arguments ==========================================================
void ProgClassifyFirstSplit3::readParams()
{
    fnClasses = getParam("-i");
    fnRoot = getParam("--oroot");
    Niter = getIntParam("--Niter");
    fnSym = getParam("--sym");
    mask.allowed_data_types = INT_MASK;
    if ((externalMask=checkParam("--mask")))
        mask.readParams(this);
    if ((mpiUse=checkParam("--mpiCommand")))
    	mpiCommand = getParam("--mpiCommand");
}

// Show ====================================================================
void ProgClassifyFirstSplit3::show()
{
    if (!verbose)
        return;
    std::cout
    << "Input classes:       " << fnClasses          << std::endl
    << "Output root:         " << fnRoot             << std::endl
    << "N. iterations:       " << Niter               << std::endl
    << "Symmetry:            " << fnSym              << std::endl
    ;
}

// usage ===================================================================
void ProgClassifyFirstSplit3::defineParams()
{
    addUsageLine("Produce a first volume split from a set of directional classes using K-means");
    addParamsLine("   -i <metadata>                  : Metadata with the list of directional classes with angles");
    addParamsLine("  [--oroot <fnroot=split>]        : Rootname for the output");
    addParamsLine("  [--Niter <n=5000>]              : Number of iterations");
    addParamsLine("  [--sym <sym=c1>]                : Symmetry");
    addParamsLine("  [--mpiCommand <mystr>]   : MPI command to parallelize the reconstruction process");
    mask.defineParams(this,INT_MASK);
}


void ProgClassifyFirstSplit3::updateVolume(const std::vector<size_t> &objIds, const FileName &fnRoot, FourierProjector &projector)
{
	MetaData mdOut;
	MDRow row;
	for(size_t i=0; i<objIds.size(); i++){
		md.getRow(row, objIds[i]);
		mdOut.addRow(row);
	}
	mdOut.write(fnRoot+".xmd");

	String command;
	if (!mpiUse){
		command=formatString("xmipp_reconstruct_fourier -i %s.xmd -o %s.vol --max_resolution 0.25 --sym %s -v 0",
				fnRoot.c_str(),fnRoot.c_str(), fnSym.c_str());
	}else{
		command=formatString(" `which xmipp_mpi_reconstruct_fourier` -i %s.xmd -o %s.vol --max_resolution 0.25 --sym %s -v 0",
		    				fnRoot.c_str(),fnRoot.c_str(), fnSym.c_str());
		command = mpiCommand + command;
	}
    //std::cout << command << std::endl;
    int retval=system(command.c_str());
    V.read(fnRoot+".vol");
    V().setXmippOrigin();

    projector.updateVolume(V());
}


void ProgClassifyFirstSplit3::calculateProjectedIms (size_t id, double &corrI_P1, double &corrI_P2){

	//Project the first volume with the parameters in the randomly selected image
	double rot, tilt, psi, x, y;
	bool flip;
	MDRow currentRow;
	FileName fnImg;
	Matrix2D<double> A;

	md.getRow(currentRow, id);
	currentRow.getValue(MDL_IMAGE,fnImg);
	imgV.read(fnImg);
	currentRow.getValue(MDL_ANGLE_ROT,rot);
	currentRow.getValue(MDL_ANGLE_TILT,tilt);
	currentRow.getValue(MDL_ANGLE_PSI,psi);
	currentRow.getValue(MDL_SHIFT_X,x);
	currentRow.getValue(MDL_SHIFT_Y,y);
	currentRow.getValue(MDL_FLIP,flip);
	A.initIdentity(3);
	MAT_ELEM(A,0,2)=x;
	MAT_ELEM(A,1,2)=y;
	if (flip)
	{
		MAT_ELEM(A,0,0)*=-1;
		MAT_ELEM(A,0,1)*=-1;
		MAT_ELEM(A,0,2)*=-1;
	}
	int xdim = (int)XSIZE(V());
	projectVolume(*projectorV1, PV, xdim, xdim,  rot, tilt, psi);
	applyGeometry(LINEAR,projV,PV(),A,IS_INV,DONT_WRAP,0.);
	corrI_P1 = correlation(imgV(), projV);

	projectVolume(*projectorV2, PV, xdim, xdim,  rot, tilt, psi);
	applyGeometry(LINEAR,projV,PV(),A,IS_INV,DONT_WRAP,0.);
	corrI_P2 = correlation(imgV(), projV);
}

void ProgClassifyFirstSplit3::run()
{
    show();

    randomize_random_generator();

    countSwap=0;
    countRandomSwap=0;
    countNormalSwap=0;
    double th=0.05;

    // Generate initial volumes
    md.read(fnClasses);
    std::vector<size_t> objIds1, objIds2;

	FOR_ALL_OBJECTS_IN_METADATA(md){
		if(rnd_unif()<0.5)
			objIds1.push_back(__iter.objId);
		else
			objIds2.push_back(__iter.objId);
	}

	projectorV1 = new FourierProjector(2,0.5,BSPLINE3);
	projectorV2 = new FourierProjector(2,0.5,BSPLINE3);

	updateVolume(objIds1, fnRoot+"_avg1", *projectorV1);
	updateVolume(objIds2, fnRoot+"_avg2", *projectorV2);

    init_progress_bar(Niter);
    Image<double> img1, img2;
	MultidimArray<double> imgProjectedV1, imgProjectedV2;

	for (int n=0; n<Niter; n++)
    {
		// Select a random image from the subset of every volume
		size_t idx1 = floor(rnd_unif(0,objIds1.size()));
		size_t idx2 = floor(rnd_unif(0,objIds2.size()));
		size_t id1 = objIds1[idx1];
		size_t id2 = objIds2[idx2];

		bool swap=false;
		if(rnd_unif()<th){
			swap = true;
			countRandomSwap++;
			//std::cout << "RANDOM" << std::endl;
		}
		else
		{
			//Calculating correlations
			double corrI1_P1, corrI2_P1, corrI2_P2, corrI1_P2;
			calculateProjectedIms(id1, corrI1_P1, corrI1_P2);
			calculateProjectedIms(id2, corrI2_P1, corrI2_P2);

			//std::cout << " corrI1_P1 = " << corrI1_P1 << " corrI2_P1 = " << corrI2_P1
			//		  << " corrI2_P2 = " << corrI2_P2 << " corrI1_P2 = " << corrI1_P2 << std::endl;
			if (corrI1_P2>corrI1_P1 && corrI2_P1>corrI2_P2){
				swap=true;
				countNormalSwap++;
			}

		}

		if(swap)
		{
			countSwap++;
			//std::cout << "SWAPPING" << std::endl;
			//Generate metadata with swapped images
			objIds1.erase(objIds1.begin()+idx1);
			objIds2.erase(objIds2.begin()+idx2);
			objIds1.push_back(id2);
			objIds2.push_back(id1);
			updateVolume(objIds1, fnRoot+"_avg1", *projectorV1);
			updateVolume(objIds2, fnRoot+"_avg2", *projectorV2);

	    	//char c;
	    	//std::cout << "Press any key" << std::endl;
	    	//std::cin >> c;
		}

		if (countSwap>0){
			th = (double)countSwap/(double)(n*10); //to make the th lower with time
			//std::cout << " th = " << th << std::endl;
		}

    	progress_bar(n);

    }
    progress_bar(Niter);
    std::cout << " Niter = " << Niter << " Images in set1 = " << objIds1.size() << " Images in set2 = " << objIds2.size() << " countRandomSwap = " << countRandomSwap << " countNormalSwap = " << countNormalSwap << std::endl;
    //deleteFile(fnSubset);
    //deleteFile(fnSubsetVol);

    // Save volumes
    //V1.write(fnRoot+"_v1.vol");
    //V2.write(fnRoot+"_v2.vol");
}
