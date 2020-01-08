/***************************************************************************
 *
 * Authors:    Jose Luis Vilas,                     jlvilas@cnb.csic.es
 * 			   Carlos Oscar Sorzano					coss@cnb.csic.es
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

#include "volume_directional_sharpening.h"
#include "resolution_directional.h"
//#define DEBUG
//#define DEBUG_MASK

void ProgDirSharpening::readParams()
{
        fnVol = getParam("--vol");
        fnMask = getParam("--mask");
        sampling = getDoubleParam("--sampling");
        res_step = getDoubleParam("--resStep");
        significance = getDoubleParam("--significance");
        R = getDoubleParam("--volumeRadius");
        lambda = getDoubleParam("-l");
        K= getDoubleParam("-k");
        Niter = getIntParam("-i");
        Nthread = getIntParam("-n");
        fnOut = getParam("-o");
        test = checkParam("--test");
}

void ProgDirSharpening::defineParams()
{
        addUsageLine("This function performs local sharpening");
        addParamsLine("  --vol <vol_file=\"\">   : Input volume");
        addParamsLine("  --mask <vol_file=\"\">  : Binary mask");
        addParamsLine("  --sampling <s=1>: sampling");
        addParamsLine("  [--volumeRadius <s=100>]                : This parameter determines the radius of a sphere where the volume is");
        addParamsLine("  [--significance <s=0.95>]               : The level of confidence for the hypothesis test.");
        addParamsLine("  [--resStep <s=0.5>]  		             : Resolution step (precision) in A");
        addParamsLine("  -o <output=\"Sharpening.vol\">: sharpening volume");
        addParamsLine("  [--test]: 								 :Launch the test of the algorithm");
        addParamsLine("  [-l <lambda=1>]: regularization param");
        addParamsLine("  [-k <K=0.025>]: K param");
        addParamsLine("  [-i <Niter=50>]: iteration");
        addParamsLine("  [-n <Nthread=1>]: threads number");
}


void ProgDirSharpening::produceSideInfo()
{
        std::cout << "Starting..." << std::endl;
        //TODO: Use the local resolution map as input
        Monogenic mono;
        MultidimArray<double> inputVol;
        if (test)
		{
        	std::cout << "Preparing test data ..." << std::endl;
        	size_t xdim = 300, ydim = 300, zdim = 300;
        	double wavelength = 10.0, mean = 0.0, std = 0.5;
        	int maskrad = 125;
        	inputVol = mono.createDataTest(xdim, ydim, zdim, wavelength, mean, 0.0);
//        	inputVol.setXmippOrigin();
        	mask() = mono.createMask(inputVol, maskrad);
        	mask().setXmippOrigin();
        	mono.applyMask(inputVol, mask());
        	mono.addNoise(inputVol, 0, std);
        	FileName fn;
        	Image<double> saveImg;
        	fn = formatString("inputVol.vol");
        	saveImg() = inputVol;
        	saveImg.write(fn);
		}
        else
        {
        	std::cout << "Reading data..." << std::endl;
        	Image<double> V;
			V.read(fnVol);
			V().setXmippOrigin();
			inputVol = V();
			mask.read(fnMask);
			mask().setXmippOrigin();
        }

    	FourierTransformer transformer;

    	VRiesz.resizeNoCopy(inputVol);
    	maxRes = ZSIZE(inputVol);
    	minRes = 2*sampling;

    	//TODO: remove Nthr
    	int Nthr = 1;
//    	transformer_inv.setThreadsNumber(Nthr);
//    	Vorig = inputVol;

    	transformer.FourierTransform(inputVol, fftV);
    	iu.initZeros(fftV);

    	// Frequency volume
    	iu = mono.fourierFreqs_3D(fftV, inputVol, freq_fourier_x, freq_fourier_y, freq_fourier_z);

    	// Prepare mask
    	MultidimArray<int> &pMask=mask();
    	N_smoothing = 7;
    	NVoxelsOriginalMask = 0;
    	double radius = 0;
    	//TODO: create a function with this
    	FOR_ALL_ELEMENTS_IN_ARRAY3D(pMask)
    	{
    		if (A3D_ELEM(pMask, k, i, j) == 1)
    		{
    			if ((k*k + i*i + j*j)>radius)
    				radius = k*k + i*i + j*j;
    		}
    		if (A3D_ELEM(pMask, k, i, j) == 1)
    			++NVoxelsOriginalMask;
    		if (i*i+j*j+k*k > (R-N_smoothing)*(R-N_smoothing))
    			A3D_ELEM(pMask, k, i, j) = -1;
    	}

    	Rparticle = round(sqrt(radius));

    	std::cout << "particle radius = " << Rparticle << std::endl;

    	size_t xrows = angles.mdimx;

    	resolutionMatrix.initConstant(xrows, NVoxelsOriginalMask, maxRes);


    	#ifdef DEBUG_MASK
    	std::cout << "-------------DEBUG-----------" <<std::endl;
    	std::cout << "Next number ought to be the same than number of directions"
    			<< std::endl;
    	std::cout << "number of angles" << xrows << std::endl;
    	std::cout << "---------END-DEBUG--------" <<std::endl;
    	mask.write("mask.vol");
    	#endif

    	maxRes = 18;
    	minRes = 1;
}

void ProgDirSharpening::icosahedronVertex(Matrix2D<double> &vertex)
{
	std::cout << "Defining Icosahedron vertex..." << std::endl;

	//The icosahedron vertex are located in (0, +-1, +-phi), (+-1, +-phi, 0), (+-phi, 0, +-1) with phi = (1+sqrt(5))/2
	double phi =  (1+sqrt(5))/2;

	vertex.initZeros(12,3);

	MAT_ELEM(vertex, 0,0) = 0;    		MAT_ELEM(vertex, 0,1) = 1;    		MAT_ELEM(vertex, 0,2) = phi;
	MAT_ELEM(vertex, 1,0) = 0;    		MAT_ELEM(vertex, 1,1) = 1;    		MAT_ELEM(vertex, 1,2) = -phi;
	MAT_ELEM(vertex, 2,0) = 0;    		MAT_ELEM(vertex, 2,1) = -1;    		MAT_ELEM(vertex, 2,2) = phi;
	MAT_ELEM(vertex, 3,0) = 0;    		MAT_ELEM(vertex, 3,1) = -1;    		MAT_ELEM(vertex, 3,2) = -phi;
	MAT_ELEM(vertex, 4,0) = 1;    		MAT_ELEM(vertex, 4,1) = phi;    		MAT_ELEM(vertex, 4,2) = 0;
	MAT_ELEM(vertex, 5,0) = 1;    		MAT_ELEM(vertex, 5,1) = -phi;    		MAT_ELEM(vertex, 5,2) = 0;
	MAT_ELEM(vertex, 6,0) = -1;    		MAT_ELEM(vertex, 6,1) = phi;    		MAT_ELEM(vertex, 6,2) = 0;
	MAT_ELEM(vertex, 7,0) = -1;    		MAT_ELEM(vertex, 7,1) = -phi;    		MAT_ELEM(vertex, 7,2) = 0;
	MAT_ELEM(vertex, 8,0) = phi;    		MAT_ELEM(vertex, 8,1) = 0;    		MAT_ELEM(vertex, 8,2) = 1;
	MAT_ELEM(vertex, 9,0) = phi;    		MAT_ELEM(vertex, 9,1) = 0;    		MAT_ELEM(vertex, 9,2) = -1;
	MAT_ELEM(vertex, 10,0) = -phi;    		MAT_ELEM(vertex, 10,1) = 0;    		MAT_ELEM(vertex, 10,2) = 1;
	MAT_ELEM(vertex, 11,0) = -phi;    		MAT_ELEM(vertex, 11,1) = 0;    		MAT_ELEM(vertex, 11,2) = -1;

	vertex = vertex*(1/sqrt(1+phi*phi));
}

void ProgDirSharpening::icosahedronFaces(Matrix2D<int> &faces, Matrix2D<double> &vertex)
{
	std::cout << " Defining the faces of the icosahedron ..." << std::endl;
	//Each face is defined by three vertex

	//An icosahedron has 20 faces.
	faces.initZeros(20,3);

	int v1, v2, v3, v1_bis, v2_bis, v3_bis;
	double x1, x2, x3, y1, y2, y3, z1, z2, z3, x1_bis, x2_bis, x3_bis, y1_bis, y2_bis, y3_bis, z1_bis, z2_bis, z3_bis;

	int xdim = MAT_YSIZE(vertex); //Number of vertex
	int counter = 0;

	for (int i = 0; i<(xdim-2); ++i)
	{
	    for (int j = (i+1); j<(xdim-1); ++j)
	    {
	        for (int k = (j+1); k<(xdim); ++k)
	        {
	            double dotprodutij, dotprodutjk, dotprodutik;
	            dotprodutij = (MAT_ELEM(vertex, i,0)*MAT_ELEM(vertex, j,0) + \
	            		MAT_ELEM(vertex, i,1)*MAT_ELEM(vertex, j,1) +\
						MAT_ELEM(vertex, i,2)*MAT_ELEM(vertex, j,2));///norm_vertex;

	            dotprodutjk = (MAT_ELEM(vertex, k,0)*MAT_ELEM(vertex, j,0) + \
	            	            		MAT_ELEM(vertex, k,1)*MAT_ELEM(vertex, j,1) + \
	            						MAT_ELEM(vertex, k,2)*MAT_ELEM(vertex, j,2));///norm_vertex;

	            dotprodutik = (MAT_ELEM(vertex, i,0)*MAT_ELEM(vertex, k,0) + \
	            	            		MAT_ELEM(vertex, i,1)*MAT_ELEM(vertex, k,1) + \
	            						MAT_ELEM(vertex, i,2)*MAT_ELEM(vertex, k,2));///norm_vertex;

	            // the number 65 comes because is greater than 60 that is the exact angle between two icosahedron vertex
	            if ((acos(dotprodutij)< 65*PI/180) && (acos(dotprodutjk)< 65*PI/180) && (acos(dotprodutik)< 65*PI/180) )
	            {
	            	MAT_ELEM(faces, counter, 0) = i;
	            	MAT_ELEM(faces, counter, 1) = j;
	            	MAT_ELEM(faces, counter, 2) = k;

	            	z1 = MAT_ELEM(vertex,i, 2);
					z2 = MAT_ELEM(vertex,j, 2);
					z3 = MAT_ELEM(vertex,k, 2);

					if ( ((z1+z2+z3) < 0) )
					{
						MAT_ELEM(faces,counter, 0) = -1; MAT_ELEM(faces,counter, 1) = -1; MAT_ELEM(faces,counter, 2) = -1;
					}

	            	++counter;
	            }

	        }
	    }
	}
	//TODO: Check if both loops can be written together

	//However, only the half of the sphere is used, so 10 faces must be considered
	for (int f1 = 0; f1<(MAT_YSIZE(faces)-1); ++f1)
	{
		if (MAT_ELEM(faces,f1, 0) < 0)
			continue;

		v1 = MAT_ELEM(faces,f1, 0); v2 = MAT_ELEM(faces,f1, 1); v3 = MAT_ELEM(faces,f1, 2);

		for (int f2 = f1+1; f2<MAT_YSIZE(faces); ++f2)
		{
			if (MAT_ELEM(faces,f2, 0) < 0)
				continue;

			v1_bis = MAT_ELEM(faces,f2, 0); v2_bis = MAT_ELEM(faces,f2, 1); v3_bis = MAT_ELEM(faces,f2, 2);

			x1 = MAT_ELEM(vertex,v1, 0); y1 = MAT_ELEM(vertex,v1, 1); z1 = MAT_ELEM(vertex,v1, 2);
			x2 = MAT_ELEM(vertex,v2, 0); y2 = MAT_ELEM(vertex,v2, 1); z2 = MAT_ELEM(vertex,v2, 2);
			x3 = MAT_ELEM(vertex,v3, 0); y3 = MAT_ELEM(vertex,v3, 1); z3 = MAT_ELEM(vertex,v3, 2);

			x1_bis = MAT_ELEM(vertex,v1_bis, 0); y1_bis = MAT_ELEM(vertex,v1_bis, 1); z1_bis = MAT_ELEM(vertex,v1_bis, 2);
			x2_bis = MAT_ELEM(vertex,v2_bis, 0); y2_bis = MAT_ELEM(vertex,v2_bis, 1); z2_bis = MAT_ELEM(vertex,v2_bis, 2);
			x3_bis = MAT_ELEM(vertex,v3_bis, 0); y3_bis = MAT_ELEM(vertex,v3_bis, 1); z3_bis = MAT_ELEM(vertex,v3_bis, 2);

			double x_tot = x1 + x2 + x3;
			double y_tot = y1 + y2 + y3;
			double z_tot = z1 + z2 + z3;
			double norm_tot, norm_tot_bis;

			norm_tot = sqrt(x_tot*x_tot + y_tot*y_tot + z_tot*z_tot);

			double x_tot_bis = x1_bis + x2_bis + x3_bis;
			double y_tot_bis = y1_bis + y2_bis + y3_bis;
			double z_tot_bis = z1_bis + z2_bis + z3_bis;

			norm_tot_bis = sqrt(x_tot_bis*x_tot_bis + y_tot_bis*y_tot_bis + z_tot_bis*z_tot_bis);

			double dotproduct;
			dotproduct = (x_tot*x_tot_bis + y_tot*y_tot_bis + z_tot*z_tot_bis)/(norm_tot*norm_tot_bis);

			if ( (fabs(dotproduct)>0.9 ) )
			{
				MAT_ELEM(faces,f2, 0) = -1;
				MAT_ELEM(faces,f2, 1) = -1;
				MAT_ELEM(faces,f2, 2) = -1;
			}
		}
	}
}

void ProgDirSharpening::getFaceVector(int face_number, Matrix2D<int> &faces,
		Matrix2D<double> &vertex, double &x1, double &y1, double &z1)
{

	double x2, x3, y2, y3, z2, z3;
	int v1, v2, v3;
	//Selecting the vertex number for each face
	v1 = MAT_ELEM(faces, face_number, 0); v2 = MAT_ELEM(faces, face_number, 1); v3 = MAT_ELEM(faces,face_number, 2);

	//Coordinates of each vertex
	x1 = MAT_ELEM(vertex,v1, 0); y1 = MAT_ELEM(vertex,v1, 1); z1 = MAT_ELEM(vertex,v1, 2);
	x2 = MAT_ELEM(vertex,v2, 0); y2 = MAT_ELEM(vertex,v2, 1); z2 = MAT_ELEM(vertex,v2, 2);
	x3 = MAT_ELEM(vertex,v3, 0); y3 = MAT_ELEM(vertex,v3, 1); z3 = MAT_ELEM(vertex,v3, 2);

	//x1, y1, z1 are used instead of defining a new variable to calculate the norm
	x1 = x1 + x2 + x3;
	y1 = y1 + y2 + y3;
	z1 = z1 + z2 + z3;

	double norm_ = sqrt(x1*x1 + y1*y1 + z1*z1);
	x1 /= norm_;
	y1 /= norm_;
	z1 /= norm_;
}


void ProgDirSharpening::defineIcosahedronCone(int face_number, Matrix2D<int> &faces, Matrix2D<double> &vertex,
		MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &conefilter, double coneAngle)
{
//	getFaceVector(face_number, faces, vertex, x1, y1, z1);
	double x1, x2, x3, y1, y2, y3, z1, z2, z3, ang_con;

	int v1, v2, v3;
	//Selecting the vertex number for each face
	v1 = MAT_ELEM(faces, face_number, 0); v2 = MAT_ELEM(faces, face_number, 1); v3 = MAT_ELEM(faces,face_number, 2);

	//Coordinates of each vertex
	x1 = MAT_ELEM(vertex,v1, 0); y1 = MAT_ELEM(vertex,v1, 1); z1 = MAT_ELEM(vertex,v1, 2);
	x2 = MAT_ELEM(vertex,v2, 0); y2 = MAT_ELEM(vertex,v2, 1); z2 = MAT_ELEM(vertex,v2, 2);
	x3 = MAT_ELEM(vertex,v3, 0); y3 = MAT_ELEM(vertex,v3, 1); z3 = MAT_ELEM(vertex,v3, 2);

	std::cout << "........." << std::endl;
	std::cout << vertex << std::endl;

	//x1, y1, z1 are used instead of defining a new variable to calculate the norm
	x1 = x1 + x2 + x3;
	y1 = y1 + y2 + y3;
	z1 = z1 + z2 + z3;

	double norm_ = sqrt(x1*x1 + y1*y1 + z1*z1);
	x1 /= norm_;
	y1 /= norm_;
	z1 /= norm_;

	std::cout << x1 << " " << y1 << " " << z1 << std::endl;

//	MultidimArray<double> conetest;
//	conetest.resizeNoCopy(myfftV);

	conefilter.initZeros(myfftV);

	double uz, uy, ux;
	long n = 0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz *= z1;

		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uy *= y1;

			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);
				ux = VEC_ELEM(freq_fourier_x, j);
				ux *= x1;

				iun *= (ux + uy + uz);
				double acosine_v1 = acos(fabs(iun));
//				DIRECT_MULTIDIM_ELEM(conefilter, n) = 1;
				if ((acosine_v1<=coneAngle))
				{
					DIRECT_MULTIDIM_ELEM(conefilter, n) = 1;
//					DIRECT_MULTIDIM_ELEM(conetest, n) = 0;
				}
				else
				{
//					DIRECT_MULTIDIM_ELEM(conetest, n) = DIRECT_MULTIDIM_ELEM(conefilter, n);
				}
				++n;
			}
		}
	}

	Image<double> icosahedronMasked;
	icosahedronMasked = conefilter;
	FileName fnmasked;
	fnmasked = formatString("maskCone_%i.mrc",face_number);
	icosahedronMasked.write(fnmasked);

}


void ProgDirSharpening::directionalNoiseEstimation(double &x_dir, double &y_dir, double &z_dir,
		MultidimArray<double> &amplitudeMS, MultidimArray<int> &mask, double &cone_angle,
		int &particleRadius, double &NS, double &NN, double &sumS, double &sumS2, double &sumN2, double &sumN,
		double &thresholdNoise)
{
	double uz, uy, ux;
	int n=0;

	int z_size = ZSIZE(amplitudeMS);
	int x_size = XSIZE(amplitudeMS);
	int y_size = YSIZE(amplitudeMS);

	double amplitudeValue;
	std::vector<float> noiseValues;
	NS = 0;
	NN = 0;
	for(int k=0; k<z_size; ++k)
	{
		for(int i=0; i<y_size; ++i)
		{
			for(int j=0; j<x_size; ++j)
			{
				if (DIRECT_MULTIDIM_ELEM(mask, n)>=1)
				{
					amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
//					std::cout << amplitudeValue << std::endl;
					sumS  += amplitudeValue;
					sumS2 += amplitudeValue*amplitudeValue;
					++NS;
				}
				else
				{
					if (DIRECT_MULTIDIM_ELEM(mask, n)==0)
					{
						uz = (k - z_size*0.5);
						ux = (j - x_size*0.5);
						uy = (i - y_size*0.5);

						double rad = sqrt(ux*ux + uy*uy + uz*uz);
						double iun = 1/rad;

						//BE CAREFULL with the order
						double dotproduct = (uy*y_dir + ux*x_dir + uz*z_dir)*iun;

						double acosine = acos(dotproduct);

						//TODO: change efficiency the if condition by means of fabs(cos(angle))
						if (((acosine<(cone_angle)) || (acosine>(PI-cone_angle)) )
								&& (rad>particleRadius))
						{
//							std::cout << "rad " << rad << std::endl;
	//						DIRECT_MULTIDIM_ELEM(coneVol, n) = 1;
							amplitudeValue = DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
							noiseValues.push_back((float) amplitudeValue);
							sumN  += amplitudeValue;
							sumN2 += amplitudeValue*amplitudeValue;
							++NN;
						}
					}
				}
				++n;
			}
		}
	}
//	std::cout << "after loop directionalNoiseEstimation" << std::endl;
//	std::cout << "NS " << NS << "  NN " << NN << " sumS " << sumS << std::endl;
	std::sort(noiseValues.begin(),noiseValues.end());
	thresholdNoise = (double) noiseValues[size_t(noiseValues.size()*significance)];

	//std::cout << "thr="<< thresholdNoise << " " << meanN+criticalZ*sqrt(sigma2N) << " " << NN << std::endl;
	noiseValues.clear();

}


void ProgDirSharpening::directionalResolutionStep(int face_number, Matrix2D<int> &faces,
		Matrix2D<double> &vertex, MultidimArray< std::complex<double> > &conefilter,
		MultidimArray<int> &mask, MultidimArray<double> &localResolutionMap,
		double &cone_angle)
{
	std::cout << "Computing local-directional resolution" << std::endl;

	//Setting parameters
	double cut_value = 0.025; //percentage of voxels to stop the frequency analysis

	bool continueIter, breakIter;
	bool doNextIteration = true;
	double freq, freqL, freqH, counter, resolution_2, resolution, step = res_step;
	double last_resolution = 0;
	int fourier_idx, last_fourier_idx = -1, iter = 0, fourier_idx_2, v1, v2, v3;
	std::vector<double> list;

	FileName fnDebug = "Signal";

	MultidimArray<double> amplitudeMS;
	localResolutionMap.initZeros(mask);
	MultidimArray<double> &plocalResolutionMap = localResolutionMap;
	MultidimArray<int> mask_aux = mask;
	MultidimArray<int> &pMask = mask_aux;

	ProgResDir resolutionSweep;

	int aux_idx, volsize;

	volsize = XSIZE(mask);

	DIGFREQ2FFT_IDX(sampling/18.0, volsize, aux_idx);

	fourier_idx = aux_idx;

	std::cout << "fourier_idx = " << fourier_idx << std::endl;

	//Calculating the average of amplitudes
	Monogenic mono;

	//TODO: Change number of threads
	int numberOfThreads = 1;
	amplitudeMS.resizeNoCopy(mask);
	mono.monogenicAmplitude_3D_Fourier(fftV, iu, amplitudeMS, numberOfThreads);

//	Image<int> icosahedronMasked;
//	icosahedronMasked = mask;
//	FileName fnmasked;
//	fnmasked = formatString("amplitude%i.mrc",face_number);
//	icosahedronMasked.write(fnmasked);

	double AvgNoise;
	double max_meanS = -1e38;
	AvgNoise = mono.averageInMultidimArray(amplitudeMS, pMask);
	double criticalZ=icdf_gauss(significance);
	do
	{
		continueIter = false;
		breakIter = false;

		mono.resolution2eval(fourier_idx, step, sampling, volsize,
						resolution, last_resolution, last_fourier_idx,
						freq, freqL, freqH,
						continueIter, breakIter, doNextIteration);

		if (breakIter)
			break;

		if (continueIter)
			continue;

		list.push_back(resolution);

		if (iter<2)
			resolution_2 = list[0];
		else
			resolution_2 = list[iter - 2];

		//TODO: Remove fnDebug
		fnDebug = "Signal";

		std::cout << "res = " << resolution << " freq " << sampling/freq << "  freqH " << sampling/freqH << "  freqL " << sampling/freqL << std::endl;



		mono.amplitudeMonoSigDir3D_LPF(conefilter,
		transformer_inv, fftVRiesz, fftVRiesz_aux, VRiesz, freq, freqH, freqL, iu,
		freq_fourier_x, freq_fourier_y, freq_fourier_z, amplitudeMS,
		 iter, face_number, fnDebug, N_smoothing);



		double x1, y1, z1, amplitudeValue;
		getFaceVector(face_number, faces, vertex, x1, y1, z1);



		double thresholdNoise, sumS=0, sumS2=0, sumN=0, sumN2=0, NN = 0, NS = 0;
		directionalNoiseEstimation(x1, y1, z1, amplitudeMS, pMask, cone_angle,
				Rparticle, NS, NN, sumS, sumS2, sumN2, sumN, thresholdNoise);

		if (NS == 0)
		{
			std::cout << "There are no points to compute inside the mask" << std::endl;
			std::cout << "If the number of computed frequencies is low, perhaps the provided"
					"mask is not enough tight to the volume, in that case please try another mask" << std::endl;
			break;
		}

		if ( (NS/(double) NVoxelsOriginalMask) < cut_value ) //when the 2.5% is reached then the iterative process stops
		{
			std::cout << "Search of resolutions stopped due to mask has been completed" << std::endl;
			doNextIteration =false;
		}

		double meanS=sumS/NS;
		double sigma2S=sumS2/NS-meanS*meanS;
		double meanN=sumN/NN;
		double sigma2N=sumN2/NN-meanN*meanN;

		double thresholdNoiseGauss = meanN+criticalZ*sqrt(sigma2N);
		double z=(meanS-meanN)/sqrt(sigma2S/NS+sigma2N/NN);
		if (meanS>max_meanS)
			max_meanS = meanS;

		if (meanS<0.001*AvgNoise)//0001*max_meanS)
		{
			std::cout << "Search of resolutions stopped due to too low signal" << std::endl;
			std::cout << "\n" << std::endl;
			break;
		}

		std::cout << "It= " << iter << ",   Res= " << resolution << ",   Sig = " << meanS << ",  Thr = " << thresholdNoise << std::endl;
		std::cout << "thresholdNoiseGauss= " << thresholdNoiseGauss << ",   z= " << z << ",   criticalZ = " << criticalZ << std::endl;
		std::cout << "        " << std::endl;

		mono.setLocalResolutionMap(amplitudeMS, pMask, plocalResolutionMap,
				thresholdNoise, resolution, resolution_2);

		if (doNextIteration)
			if (resolution <= (minRes-0.001))
				doNextIteration = false;

		++iter;
		last_resolution = resolution;
	}while(doNextIteration);

	FileName fn;
	Image<double> saveImg;
	fn = formatString("dirMap_%i.vol", face_number);
	saveImg() = plocalResolutionMap;
	saveImg.write(fn);

}


void ProgDirSharpening::bandPassDirectionalFilterFunction(int face_number, Matrix2D<int> &faces,
		Matrix2D<double> &vertex, MultidimArray< std::complex<double> > &myfftV,
		MultidimArray<double> &Vorig, MultidimArray<double> &iu, FourierTransformer &transformer_inv,
        double w, double wL, MultidimArray<double> &filteredVol, int count)
{
	double coneAngle;
	coneAngle = PI/6;
	MultidimArray<double> maskCone;
	defineIcosahedronCone(face_number, faces, vertex, myfftV, maskCone, coneAngle);

	MultidimArray< std::complex<double> > fftVfilter;
	fftVfilter.initZeros(myfftV);

	double delta = wL-w;
	double w_inf = w-delta;
	// Filter the input volume and add it to amplitude
	long n=0;
	double ideltal=PI/(delta);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
	{
		if (DIRECT_MULTIDIM_ELEM(maskCone, n) == 1)
		{
			double un=DIRECT_MULTIDIM_ELEM(iu,n);
			if (un>=w && un<=wL)
			{
					DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVfilter, n) *= 0.5*(1+cos((un-w)*ideltal));//H;
			} else{
				if (un<=w && un>=w_inf)
				{
					DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVfilter, n) *= 0.5*(1+cos((un-w)*ideltal));//H;
				}
			}
		}
	}

	filteredVol.resizeNoCopy(Vorig);

	transformer_inv.inverseFourierTransform(fftVfilter, filteredVol);
}

void ProgDirSharpening::localDirectionalfiltering(Matrix2D<int> &faces,
		Matrix2D<double> &vertex, MultidimArray< std::complex<double> > &myfftV,
        MultidimArray<double> &localfilteredVol, MultidimArray<double> &Vorig,
        double &minRes, double &maxRes, double &step)
{
        MultidimArray<double> filteredVol, lastweight, weight;
        localfilteredVol.initZeros(Vorig);
        weight.initZeros(Vorig);
        lastweight.initZeros(Vorig);
        Monogenic mono;

        double freq, lastResolution=1e38;
        int idx, lastidx = -1;
        Image<double> resVol;
        MultidimArray<double> &presVol=resVol();

        for (size_t face_number = 0; face_number<MAT_YSIZE(faces); ++face_number)
		{
			//TODO: remove repeated faces
			//Repeated faces are skipped
			if (MAT_ELEM(faces, face_number, 0) < 0)
				continue;
			FileName fn;
			fn = formatString("dirMap_%i.vol", face_number);
			resVol.read(fn);


			for (double res = minRes; res<maxRes; res+=step)
			{
				freq = sampling/res;

				DIGFREQ2FFT_IDX(freq, ZSIZE(myfftV), idx);

				if (idx == lastidx)
					continue;

				double wL = sampling/(res - step);

				//TODO: Check performance in the mask
//				mono.bandPassFilterFunction(myfftV, Vorig, iu,
//						transformer_inv, freq, wL, filteredVol, idx);
				bandPassDirectionalFilterFunction(face_number, faces, vertex, myfftV, Vorig, iu,
										transformer_inv, freq, wL, filteredVol, idx);

				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(filteredVol)
				{

				   if (DIRECT_MULTIDIM_ELEM(presVol, n) < 2*sampling)
						{
						   DIRECT_MULTIDIM_ELEM(filteredVol, n)=0;
						}
				   else
						{
						   double res_map = DIRECT_MULTIDIM_ELEM(presVol, n);//+1e-38;
						   DIRECT_MULTIDIM_ELEM(weight, n) = (exp(-K*(res-res_map)*(res-res_map)));
						   DIRECT_MULTIDIM_ELEM(filteredVol, n) *= DIRECT_MULTIDIM_ELEM(weight, n);
						}
				}

				localfilteredVol += filteredVol;
				lastweight += weight;
				lastResolution = res;
				lastidx = idx;
			}


			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(localfilteredVol)
			{
				if (DIRECT_MULTIDIM_ELEM(lastweight, n)>0)
					DIRECT_MULTIDIM_ELEM(localfilteredVol, n) /=DIRECT_MULTIDIM_ELEM(lastweight, n);
			}
		}
}


void ProgDirSharpening::localdeblurStep(MultidimArray<double> &vol, MultidimArray<int> &mask,
		Matrix2D<double> &vertex, Matrix2D<int> &faces)
{
	std::cout << "                               " << std::endl;
	std::cout << "Starting directional sharpening" << std::endl;
	std::cout << "                               " << std::endl;

	//TODO Set number of processors in
	//transformer_inv and transformer

	MultidimArray<double> Vorig;
	Vorig = vol;

	transformer.FourierTransform(vol, fftV);

	//TODO: check if exist inf in next multidimarray
	//Frequencies are redefined
	iu =1/iu;

	vol.clear();
	Monogenic mono;
	double desvOutside_Vorig, mean;
	mono.statisticsInBinaryMask(Vorig, mask, mean, desvOutside_Vorig);
	//std::cout << "desvOutside_Vorig = " << desvOutside_Vorig << std::endl;


//	Image<double> resolutionVolume;
//	MultidimArray<double> resVol;
//	resolutionVolume.read(fnRes);
//
//	resVol = resolutionVolume();
//	resolutionVolume().clear();
//	resVol.setXmippOrigin();

	maxRes = 18;
	minRes = 2*sampling;


	MultidimArray<double> auxVol;
	MultidimArray<double> operatedfiltered, Vk, filteredVol;
	double lastnorm = 0, lastporc = 1;
	double freq;
	double step = 0.2;
	int idx, bool1=1, bool2=1;
	int lastidx = -1;

	maxRes = maxRes + 2;

	//std::cout << "Resolutions between " << minRes << " and " << maxRes << std::endl;

	filteredVol = Vorig;

	MultidimArray<double> sharpenedMap;
	sharpenedMap.resizeNoCopy(Vorig);
	double normOrig=0;

	for (size_t i = 1; i<=Niter; ++i)
	{
		std::cout << "----------------Iteration " << i << "----------------" << std::endl;
		auxVol = filteredVol;
		transformer.FourierTransform(auxVol, fftV);

		//TODO: Here the directional filtering should be carried out
		localDirectionalfiltering(faces, vertex,
				fftV, operatedfiltered, Vorig, minRes, maxRes, step);

		filteredVol = Vorig;

		filteredVol -= operatedfiltered;

		//calculate norm for Vorig
		if (i==1)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vorig)
				normOrig +=(DIRECT_MULTIDIM_ELEM(Vorig,n)*DIRECT_MULTIDIM_ELEM(Vorig,n));

			normOrig = sqrt(normOrig);
			//std::cout << "norma del original  " << normOrig << std::endl;
		}


		//calculate norm for operatedfiltered
		double norm=0;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(operatedfiltered)
			norm +=(DIRECT_MULTIDIM_ELEM(operatedfiltered,n)*DIRECT_MULTIDIM_ELEM(operatedfiltered,n));

		norm=sqrt(norm);


		double porc=lastnorm*100/norm;
		//std::cout << "norm " << norm << " percetage " << porc << std::endl;

		double subst=porc-lastporc;

		if ((subst<1)&&(bool1==1)&&(i>2))
			bool1=2;
			//std::cout << "-----iteration completed-----" << std::endl;


		lastnorm=norm;
		lastporc=porc;

		if (i==1 && lambda==1)
		{
			lambda=(normOrig/norm)/12;
			std::cout << "  lambda  " << lambda << std::endl;
		}

		////Second operator
		transformer.FourierTransform(filteredVol, fftV);
		localDirectionalfiltering(faces, vertex,
						fftV, filteredVol, Vorig, minRes, maxRes, step);

		if (i == 1)
				Vk = Vorig;
		else
				Vk = sharpenedMap;

		//sharpenedMap=Vk+lambda*(filteredVol);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sharpenedMap)
		{
			DIRECT_MULTIDIM_ELEM(sharpenedMap,n)=DIRECT_MULTIDIM_ELEM(Vk,n)+
								 lambda*DIRECT_MULTIDIM_ELEM(filteredVol,n);
								 //-0.01*DIRECT_MULTIDIM_ELEM(Vk,n)*SGN(DIRECT_MULTIDIM_ELEM(Vk,n));
			if (DIRECT_MULTIDIM_ELEM(sharpenedMap,n)<-4*desvOutside_Vorig)
				DIRECT_MULTIDIM_ELEM(sharpenedMap,n)=-4*desvOutside_Vorig;
		}

//        		double desv_sharp=0;
//                computeAvgStdev_within_binary_mask(resVol, sharpenedMap, desv_sharp);
//                std::cout << "desv_sharp = " << desv_sharp << std::endl;

		filteredVol = sharpenedMap;

		if (bool1==2)
		{
			Image<double> filteredvolume;
			filteredvolume() = sharpenedMap;
			filteredvolume.write(fnOut);
			break;
		}
	}

	Image<double> filteredvolume;
	filteredvolume() = sharpenedMap;
	filteredvolume.write(fnOut);

}


void ProgDirSharpening::run()
{
	bool stopError = false;
	if (test)
	{
		Monogenic Mono;
		stopError = Mono.TestmonogenicAmplitude_3D_Fourier();
		if (stopError == false)
			exit(0);
	}

	//Defining general information to be used
	produceSideInfo();
	std::cout << "Reading data..." << std::endl;
	//Defining the number of vertex and faces of the icosahedron
	Matrix2D<double> vertex;
	Matrix2D<int> faces;
	double coneAngle;
	coneAngle = PI/6;
	MultidimArray< std::complex<double> > fftCone;
	MultidimArray<double> conefilter, localResolutionMap;
	Monogenic mono;

	icosahedronVertex(vertex);
	icosahedronFaces(faces, vertex);

	//TODO: Clean faces
	//icosahedronFaces_test();
	std::cout << "vertex = " << vertex << std::endl;
	std::cout << "faces = " << faces << std::endl;

	//TODO: Define a real icosahedron

	for (size_t face_number = 0; face_number<MAT_YSIZE(faces); ++face_number)
	{
		//TODO: remove repeated faces
		//Repeated faces are skipped
		if (MAT_ELEM(faces, face_number, 0) < 0)
			continue;

		//TODO: Modify this function to use real pyramids
		defineIcosahedronCone(face_number, faces, vertex, fftV, conefilter, coneAngle);
		//defineIcosahedronCone_test();

		fftCone = mono.applyMaskFourier(fftV, conefilter);
				//defineIcosahedronCone_test();

		std::cout << "Computing local-directional resolution along face " << face_number << std::endl;
		directionalResolutionStep(face_number, faces, vertex, fftCone, mask(), localResolutionMap, coneAngle);
//		//directionalResolutionStep_test();
	}

	Image<double> Vin;
	Vin.read(fnVol);
	Vin().setXmippOrigin();
	MultidimArray<double> Vorig = Vin();
	localdeblurStep(Vorig, mask(), vertex, faces);
	//TODO: Think a test...
	//	testlocaldeblur()
}



