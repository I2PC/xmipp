/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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

 #include "resolution_fscoh.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_base.h"
 #include "core/xmipp_fftw.h"
 #include <iostream>
 #include <string>
 #include <chrono>



// I/O methods ===================================================================
void ProgFSCoh::readParams()
{
    fn_mapPool = getParam("-i");
    fn_oroot = getParam("--oroot");
    sampling_rate = getDoubleParam("--sampling_rate");
}

void ProgFSCoh::defineParams()
{
	//Usage
    addUsageLine("This algorithm calculate the Fourier Shell Coherence from a input map pool.");

    //Parameters
    addParamsLine("-i <i=\"\">                              : Input metadata containing the map pool.");
    addParamsLine("--oroot <oroot=\"\">                     : Location for saving output.");
    addParamsLine("--sampling_rate <sampling_rate=1.0>      : Sampling rate of the input of maps.");
}

void ProgFSCoh::show() const
{
    if (!verbose)
        return;
	std::cout
	<< "Input metadata containing map pool:\t" << fn_mapPool << std::endl
	<< "Output location for FSCoh:\t" << fn_oroot << std::endl;
}


// Main method ===================================================================
void ProgFSCoh::run()
{
	auto t1 = std::chrono::high_resolution_clock::now();

	// Compose side freq map 
	composefreqMap();

    // Calculate statistical map
    Ndim = mapPoolMD.size();
    FourierShellCoherence(mapPoolMD);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

 	std::cout << "Execution time: " << ms_int.count() << " ms" << std::endl;
}


// Core methods ===================================================================
void ProgFSCoh::FourierShellCoherence(MetaDataVec mapPoolMD)
{
    std::cout << "Calculating Fourier Shell Coherence..." << std::endl;

    FourierTransformer ft;
    MultidimArray<std::complex<double>> V_ft;

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef DEBUG_STAT_MAP
        std::cout << "  Processing volume " << fn_V << " For FSC calculation" << std::endl;
        #endif

        V.read(fn_V); 

        if (!dimInitialized)
        {
            // Read dim
            Xdim = XSIZE(V());
            Ydim = YSIZE(V());
            Zdim = ZSIZE(V());

            #ifdef DEBUG_DIM
            std::cout 
            << "Xdim: " << Xdim << std::endl
            << "Ydim: " << Ydim << std::endl
            << "Zdim: " << Zdim << std::endl
            << "Ndim: " << Ndim << std::endl;
            #endif

            // Initialize maps
            avgVolume().initZeros(Zdim, Ydim, Xdim);
            stdVolume().initZeros(Zdim, Ydim, Xdim);

            dimInitialized = true;
        }

        ft.FourierTransform(V(), V_ft, false);

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_ft)
        {
            DIRECT_MULTIDIM_ELEM(mFSC_map,  n) +=  DIRECT_MULTIDIM_ELEM(V_ft,n);
            DIRECT_MULTIDIM_ELEM(mFSC_map2, n) += (DIRECT_MULTIDIM_ELEM(V_ft,n) * std::conj(DIRECT_MULTIDIM_ELEM(V_ft,n))).real();
        }
    }

    // Compute mFSC map
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mFSC_map2)
    {
        DIRECT_MULTIDIM_ELEM(mFSC_map2, n) = (DIRECT_MULTIDIM_ELEM(mFSC_map,n) * std::conj(DIRECT_MULTIDIM_ELEM(mFSC_map,n))).real() / (Ndim * DIRECT_MULTIDIM_ELEM(mFSC_map2, n));
    }

    #ifdef DEBUG_OUTPUT_FILES
	Image<double> saveImage;
    std::string debugFileFn = fn_oroot + "mFSC.mrc";

	saveImage() = mFSC_map2;
	saveImage.write(debugFileFn);
    #endif

    // Coherence per fequency
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mFSC_map2)
	{
        int freqIdx = (int)(DIRECT_MULTIDIM_ELEM(freqMap,n));

        // Consider only up to Nyquist (remove corners from analysis)
        if (freqIdx < NZYXSIZE(mFSC))
        {
            DIRECT_MULTIDIM_ELEM(mFSC,         freqIdx) += DIRECT_MULTIDIM_ELEM(mFSC_map2,n);
            DIRECT_MULTIDIM_ELEM(mFSC_counter, freqIdx) += 1;
        }
	}

    // Save output metadata
	MetaDataVec md;
	size_t id;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mFSC)
	{
        double value = DIRECT_MULTIDIM_ELEM(mFSC,n) / DIRECT_MULTIDIM_ELEM(mFSC_counter,n);

        DIRECT_MULTIDIM_ELEM(mFSC,n) = value;

		id = md.addObject();
		md.setValue(MDL_X, DIRECT_MULTIDIM_ELEM(mFSC,n), id);
		md.setValue(MDL_Y, DIRECT_MULTIDIM_ELEM(mFSC_counter,n), id);
	}

	std::string outputMD = fn_oroot + "mFSC.xmd";
	md.write(outputMD);

	std::cout << "  Fourier shell coherence written at: " << outputMD << std::endl;

    // Define Coherence threhold (demonstration in notebook)
    float m = 7;  // Rosenthal and Henderson 2003. If SNR = 1/m: m=7 for FSC=0.143
    double thr = (m+Ndim)/(Ndim*(m+1));
    
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mFSC)
    {
        if (DIRECT_MULTIDIM_ELEM(mFSC, n) < thr)
        {
            indexThr = n;
            break;           
        }
    }

    std::cout << "  Frequency (normalized) thresholded at (for FSCoh > " << thr << "): " << sampling_rate*((2*(float)NZYXSIZE(mFSC))/(float)indexThr) << std::endl;
    std::cout << "  indexThr " << indexThr << std::endl;
}



// Utils methods ===================================================================
void ProgFSCoh::composefreqMap()
{
	// Calculate FT
	MultidimArray<std::complex<double>> V_ft; // Volume FT

	FourierTransformer ft;
	ft.FourierTransform(V(), V_ft, false);

	// FT dimensions
	int Xdim_ft = XSIZE(V_ft);
	int Ydim_ft = YSIZE(V_ft);
	int Zdim_ft = ZSIZE(V_ft);
	int Ndim_ft = NSIZE(V_ft);

    // Use this dimension to initialize mFSC auxiliary maps
    mFSC.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    mFSC_counter.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    mFSC_map2.initZeros(Zdim_ft, Ydim_ft, Xdim_ft);
    mFSC_map.initZeros(Zdim_ft, Ydim_ft, Xdim_ft);

	if (Zdim_ft == 1)
	{
		Zdim_ft = Ndim_ft;
	}

	int maxRadius = std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft));	// Restric analysis to Nyquist

	#ifdef DEBUG_FREQUENCY_MAP
	std::cout << "FFT map dimensions: " << std::endl;  
	std::cout << "FT xSize " << Xdim_ft << std::endl;
	std::cout << "FT ySize " << Ydim_ft << std::endl;
	std::cout << "FT zSize " << Zdim_ft << std::endl;
	std::cout << "FT nSize " << Ndim_ft << std::endl;
	std::cout << "maxRadius " << maxRadius << std::endl;
	#endif

	// Construct frequency map and initialize the frequency vectors
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;

	freq_fourier_x.initZeros(Xdim_ft);
	freq_fourier_y.initZeros(Ydim_ft);
	freq_fourier_z.initZeros(Zdim_ft);

	double u;	// u is the frequency

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_z,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Zdim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Zdim, u);
		VEC_ELEM(freq_fourier_z, k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Ydim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Ydim, u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Xdim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Xdim, u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(V_ft);

	// Directional frequencies along each direction
	double uz;
	double uy;
	double ux;
	double uz2;
	double uz2y2;
	long n=0;
	int idx = 0;

	for(size_t k=0; k<Zdim_ft; ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2 = uz*uz;
		
		for(size_t i=0; i<Ydim_ft; ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<Xdim_ft; ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				ux = sqrt(uz2y2 + ux*ux);

				idx = (int) round(ux * Xdim);
				DIRECT_MULTIDIM_ELEM(freqMap,n) = idx;

				++n;
			}
		}
	}

    #ifdef DEBUG_OUTPUT_FILES
	Image<double> saveImage;
	std::string debugFileFn = fn_oroot + "freqMap.mrc";
	saveImage() = freqMap;
	saveImage.write(debugFileFn);
    #endif
}