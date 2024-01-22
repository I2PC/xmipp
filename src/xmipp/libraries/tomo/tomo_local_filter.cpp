/***************************************************************************
 *
 * Authors:    Jose Luis Vilas,              .jlvilas@cnb.csic.es
 *
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

#include "tomo_local_filter.h"
//#include <data/cpu.h>
#include <data/fft_settings.h>
#include "data/fftwT.h"
#include <cmath>


void ProgTomoLocalFilter::readParams()
{
        fnVol      = getParam("--tomo");
        fnRes      = getParam("--resTomo");
        sampling   = getDoubleParam("--sampling");
        K          = getDoubleParam("-k");
        Nthread    = getIntParam("-n");
        fnOut      = getParam("-o");
}

void ProgTomoLocalFilter::defineParams()
{
        addUsageLine("This function performs a local filtering of a tomogram based on the local resolution of the tomogram");
        addParamsLine("  --tomo <vol_file=\"\">                       : Input volume");
        addParamsLine("  --resTomo <vol_file=\"\">                    : Resolution map");
        addParamsLine("  --sampling <s=1>                             : Sampling rate");
        addParamsLine("  -o <output=\"Sharpening.vol\">               : Sharpening volume");
        addParamsLine("  [-k <K=0.025>]                               : K param");
        addParamsLine("  [-n <Nthread=1>]                             : Threads number");
}

void ProgTomoLocalFilter::produceSideInfo()
{
        std::cout << "Starting..." << std::endl;
        Image<float> V;
        V.read(fnVol);
        V().setXmippOrigin();
        auto &tomo = V();

        size_t xdim = XSIZE(tomo);
        size_t ydim = YSIZE(tomo);
        size_t zdim = ZSIZE(tomo);

        float normConst = xdim * ydim * zdim;

        std::cout  << xdim << " " << ydim << " " << zdim << std::endl;
        const FFTSettings<float> fftSettingForward(xdim, ydim, zdim);
        const auto fftSettingBackward = fftSettingForward.createInverse();

        auto hw = CPU(Nthread);
        forward_transformer = std::make_unique<FFTwT<float>>();
        backward_transformer = std::make_unique<FFTwT<float>>();
        forward_transformer->init(hw, fftSettingForward);
        backward_transformer->init(hw, fftSettingBackward);

        const auto &fdim = fftSettingForward.fDim();
        fourierData.resize(fdim.size());
        forward_transformer->fft(MULTIDIM_ARRAY(tomo), fourierData.data());

        std::cout << "fftSettingForward.fDim().x() " << fftSettingForward.fDim().x() << std::endl;

        xdimFT = fdim.x();
        ydimFT = fdim.y();
        zdimFT = fdim.z();

        float uz, uy, ux, uz2, u, uz2y2;
        size_t n=0;
        for (size_t k=0; k<zdimFT; ++k)
        {
            FFT_IDX2DIGFREQ(k, zdim, uz);
			uz2 = uz*uz;

			for (size_t i=0; i<ydimFT; ++i)
			{
				FFT_IDX2DIGFREQ(i, ydim, uy);
				uz2y2 = uz2 + uy*uy;
				for (size_t j=0; j<xdimFT; ++j)
				{
						FFT_IDX2DIGFREQ(j, xdim, ux);
						u = std::sqrt(uz2y2 + ux*ux);
						if (u <= 0.5)
						{
							idxMap.push_back(n);
							freqTomo.push_back(u);
						}
						++n;
				}
			}
        }

		Image<float> resolutionVolume;
		resolutionVolume.read(fnRes);

		resVol = resolutionVolume();

		maxMinResolution(resVol, maxRes, minRes);

		resVol.setXmippOrigin();

}


void ProgTomoLocalFilter::maxMinResolution(MultidimArray<float> &resVol, float &maxRes, float &minRes)
{
		float nyquist = 2*sampling;
        // Count number of voxels with resolution
        float lastLowestRes=1e38, lastHighestRes=1e-38, value;
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resVol)
        {
                value = DIRECT_MULTIDIM_ELEM(resVol, n);

                if (value<nyquist)
                {
                	DIRECT_MULTIDIM_ELEM(resVol, n) = nyquist;
                	value = nyquist;
                }

                if (value>lastHighestRes)
                	lastHighestRes = value;
                if (value<lastLowestRes)
                	lastLowestRes = value;
        }

        maxRes = lastHighestRes;
        minRes = lastLowestRes;
}


void ProgTomoLocalFilter::bandPassFilterFunction(const std::vector<std::complex<float>> &fftin,
                float w, float wL, MultidimArray<float> &filteredVol, int count)
{
	    //TODO: check if this declaration can go to the produce sideinfo
	    std::vector<std::complex<float>> fourierDataFiltered(fftin.size());

        float delta = wL-w;
        float w_inf = w-delta;
        float ideltal=PI/delta;

        const auto numberOfIdx = idxMap.size();
        std::cout << "numberOfIdx = " << numberOfIdx << std::endl;
        for (size_t n = 0; n<numberOfIdx; n++)
        {
        	auto idx = idxMap[n];
        	auto freq = freqTomo[n];
        	if (freq>=w_inf && freq<=wL)
        	{
        		fourierDataFiltered[idx] = fourierData[idx] * 0.5f*(1.0f+  std::cos((freq-w)*ideltal));
        	}
        	else
        	{
        		fourierDataFiltered[idx] = 0.0f;
        	}

        }

        backward_transformer->ifft(fourierDataFiltered.data(), MULTIDIM_ARRAY(filteredVol));

}


void ProgTomoLocalFilter::localfiltering(std::vector<std::complex<float>>  &myfftV, MultidimArray<float> &localfilteredVol,float minRes, float maxRes, float step)
{
        MultidimArray<float> filteredVol, lastweight, weight;
        localfilteredVol.initZeros(resVol);
        filteredVol.initZeros(resVol);
        weight.initZeros(resVol);
        lastweight.initZeros(resVol);

        float freq;
        int idx, lastidx = -1;

        //TODO:Validate step to avoid go below nyquist
	
		for (float res = minRes; res<maxRes; res+=step)
		{
                freq = sampling/res;

                std::cout << "freq = " << freq << std::endl;
                DIGFREQ2FFT_IDX(freq, zdimFT, idx);

                if (idx == lastidx)
                {
                        continue;
                        std::cout << "The program exited because of a coding error"  << std::endl;
                }

                double wL = sampling/(res - step);

                bandPassFilterFunction(fourierData, freq, wL, filteredVol, idx);

                localfilteredVol += filteredVol;
                lastweight += weight;
                lastidx = idx;
        }

//        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(localfilteredVol)
//        {
//        	if (DIRECT_MULTIDIM_ELEM(lastweight, n)>0)
//                DIRECT_MULTIDIM_ELEM(localfilteredVol, n) *=0.01*sigmaBefore/sigmaAfter;
//        }
		Image<float> filteredvolume;
		filteredvolume() = localfilteredVol;
		filteredvolume.write(fnOut);
}


void ProgTomoLocalFilter::run()
{
        produceSideInfo();

        MultidimArray<float> auxVol;
        MultidimArray<float> filteredTomo;

		localfiltering(fourierData, filteredTomo, minRes, maxRes, 1.0);


}
