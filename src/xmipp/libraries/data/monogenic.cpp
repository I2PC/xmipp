/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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
#include "monogenic.h"


//This function determines a multidimArray with the frequency values in Fourier space;
MultidimArray<double> Monogenic::fourierFreqs_3D(const MultidimArray< std::complex<double> > &myfftV,
		const MultidimArray<double> &inputVol,
		Matrix1D<double> &freq_fourier_x,
		Matrix1D<double> &freq_fourier_y,
		Matrix1D<double> &freq_fourier_z)
{
	double u;

	freq_fourier_z.initZeros(ZSIZE(myfftV));
	freq_fourier_x.initZeros(XSIZE(myfftV));
	freq_fourier_y.initZeros(YSIZE(myfftV));

	VEC_ELEM(freq_fourier_z,0) = 1e-38;
	for(size_t k=1; k<ZSIZE(myfftV); ++k){
		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_z,k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = 1e-38;
	for(size_t k=1; k<YSIZE(myfftV); ++k){
		FFT_IDX2DIGFREQ(k,YSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_y,k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = 1e-38;
	for(size_t k=1; k<XSIZE(myfftV); ++k){
		FFT_IDX2DIGFREQ(k,XSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_x,k) = u;
	}


	MultidimArray<double> iu;

	iu.initZeros(myfftV);

	double uz, uy, ux, uz2, u2, uz2y2;
	long n=0;
	//  TODO: reasign uz = uz*uz to save memory
	//  TODO: Take ZSIZE(myfftV) out of the loop
	//	TODO: Use freq_fourier_x instead of calling FFT_IDX2DIGFREQ

	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol),uz);
		uz2 = uz*uz;
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			FFT_IDX2DIGFREQ(i,YSIZE(inputVol),uy);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				FFT_IDX2DIGFREQ(j,XSIZE(inputVol), ux);
				u2 = uz2y2 + ux*ux;

				if ((k != 0) || (i != 0) || (j != 0))
				{
					DIRECT_MULTIDIM_ELEM(iu,n) = 1/sqrt(u2);
				}
				else
				{
					DIRECT_MULTIDIM_ELEM(iu,n) = 1e38;
				}
				++n;
			}
		}
	}


	return iu;
}


//This function has been checked
MultidimArray<double> Monogenic::fourierFreqs_2D(const MultidimArray< std::complex<double> > &myfftImg,
		const MultidimArray<double> &inputImg)
{
	MultidimArray<double> iu;

	iu.initZeros(myfftImg);

	double uy, ux, uy2, u2;
	long n=0;

	for(size_t i=0; i<YSIZE(myfftImg); ++i)
	{
		FFT_IDX2DIGFREQ(i,YSIZE(inputImg),uy);
		uy2 = uy*uy;

		for(size_t j=0; j<XSIZE(myfftImg); ++j)
		{
			FFT_IDX2DIGFREQ(j,XSIZE(inputImg), ux);
			u2=uy2+ux*ux;
			if ((i != 0) || (j != 0))
			{
				A2D_ELEM(iu,i,j) = 1.0/sqrt(u2);
			}
			else
				A2D_ELEM(iu,i,j) = 1e38;
			++n;
		}
	}

	return iu;
}

double Monogenic::averageInMultidimArray(const MultidimArray<double> &vol, MultidimArray<int> &mask)
{
	double sum = 0;
	double N = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
	{
		if (DIRECT_MULTIDIM_ELEM(mask, n) == 1)
		{
			sum += DIRECT_MULTIDIM_ELEM(vol, n);
			N++;
		}
	}

	double avg;
	avg = sum/N;

	return avg;
}


void Monogenic::statisticsInBinaryMask(const MultidimArray<double> &vol, MultidimArray<int> &mask, double &mean, double &sd)
{
	double sum = 0, sum2 = 0;;
	double N = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
	{
		if (DIRECT_MULTIDIM_ELEM(mask, n) == 1)
		{
			double value = DIRECT_MULTIDIM_ELEM(vol, n);
			sum += value;
			sum2 += value*value;
			N++;
		}
	}
	mean = sum/N;
	sd = sum2/N - mean*mean;
}


// TODO: Take this function out of this class, and create a class resolution
void Monogenic::setLocalResolutionMap(const MultidimArray<double> &amplitudeMS,
		MultidimArray<int> &pMask, MultidimArray<double> &plocalResolutionMap,
		double &thresholdNoise, double &resolution, double &resolution_2)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
		{
			if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
				if (DIRECT_MULTIDIM_ELEM(amplitudeMS, n)>thresholdNoise)
				{
					DIRECT_MULTIDIM_ELEM(pMask, n) = 1;
					DIRECT_MULTIDIM_ELEM(plocalResolutionMap, n) = resolution;//sampling/freq;
				}
				else{
					DIRECT_MULTIDIM_ELEM(pMask, n) += 1;
					if (DIRECT_MULTIDIM_ELEM(pMask, n) >2)
					{
						DIRECT_MULTIDIM_ELEM(pMask, n) = -1;
						DIRECT_MULTIDIM_ELEM(plocalResolutionMap, n) = resolution_2;//maxRes - counter*R_;
					}
				}
		}
	}


void Monogenic::resolution2eval(int &fourier_idx, double min_step, double sampling, int volsize,
								double &resolution, double &last_resolution,
								int &last_fourier_idx,
								double &freq, double &freqL, double &freqH,
								bool &continueIter, bool &breakIter, bool &doNextIteration)
{
	FFT_IDX2DIGFREQ(fourier_idx, volsize, freq);

	resolution = sampling/freq;
//	std::cout << "res = " << resolution << std::endl;
//	std::cout << "min_step = " << min_step << std::endl;
	if (resolution>8)
		min_step =1;

	if ( fabs(resolution - last_resolution)<min_step )
	{
		freq = sampling/(last_resolution-min_step);
		DIGFREQ2FFT_IDX(freq, volsize, fourier_idx);
		FFT_IDX2DIGFREQ(fourier_idx, volsize, freq);

		if (fourier_idx == last_fourier_idx)
		{
			continueIter = true;
			++fourier_idx;
			return;
		}
	}

	resolution = sampling/freq;
	last_resolution = resolution;

	double step = 0.05*resolution;

	double resolution_L, resolution_H;

	if ( step < min_step)
	{
		resolution_L = resolution - min_step;
		resolution_H = resolution + min_step;
	}
	else
	{
		resolution_L = 0.95*resolution;
		resolution_H = 1.05*resolution;
	}

	freqH = sampling/(resolution_H);
	freqL = sampling/(resolution_L);

	if (freqH>0.5 || freqH<0)
		freqH = 0.5;

	if (freqL>0.5 || freqL<0)
		freqL = 0.5;
	int fourier_idx_H, fourier_idx_L;

	DIGFREQ2FFT_IDX(freqH, volsize, fourier_idx_H);
	DIGFREQ2FFT_IDX(freqL, volsize, fourier_idx_L);

	if (fourier_idx_H == fourier_idx)
		fourier_idx_H = fourier_idx - 1;

	if (fourier_idx_L == fourier_idx)
		fourier_idx_L = fourier_idx + 1;

	FFT_IDX2DIGFREQ(fourier_idx_H, volsize, freqH);
	FFT_IDX2DIGFREQ(fourier_idx_L, volsize, freqL);

//	std::cout << "freq_H = " << freqH << std::endl;
//	std::cout << "freq_L = " << freqL << std::endl;

	if (freq>0.49 || freq<0)
	{
		std::cout << "Nyquist limit reached" << std::endl;
		breakIter = true;
		doNextIteration = false;
		return;
	}
	else
	{
		breakIter = false;
		doNextIteration = true;
	}
//	std::cout << "resolution = " << resolution << "  resolutionL = " <<
//				sampling/(freqL) << "  resolutionH = " << sampling/freqH
//				<< "  las_res = " << last_resolution << std::endl;
	last_fourier_idx = fourier_idx;
	++fourier_idx;
}

void Monogenic::monogenicAmplitude_3D_Fourier(const MultidimArray< std::complex<double> > &myfftV,
		MultidimArray<double> iu, MultidimArray<double> &amplitude, int numberOfThreads)
{
	Matrix1D<double> freq_fourier_z, freq_fourier_y, freq_fourier_x;

	iu = fourierFreqs_3D(myfftV, amplitude, freq_fourier_x, freq_fourier_y, freq_fourier_z);

	// Filter the input volume and add it to amplitude
	MultidimArray< std::complex<double> > fftVRiesz, fftVRiesz_aux;
	fftVRiesz.initZeros(myfftV);
	fftVRiesz_aux.initZeros(myfftV);
	std::complex<double> J(0,1);

	long n=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
	{
		//double H=0.5*(1+cos((un-w1)*ideltal));
		DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
		DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
		DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
		DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(iu, n);

	}
	MultidimArray<double> VRiesz;
	VRiesz.resizeNoCopy(amplitude);

	FourierTransformer transformer_inv;
	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) = DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	// Calculate first component of Riesz vector
	double uz, uy, ux;
	n=0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier_x,j);
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	// Calculate second and third components of Riesz vector
	n=0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z,k);
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y,i);
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uz*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}
	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	transformer_inv.inverseFourierTransform(fftVRiesz_aux, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
		DIRECT_MULTIDIM_ELEM(amplitude,n) = sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
	}
}


MultidimArray<int> Monogenic::createMask(MultidimArray<double> &vol, int radius)
{
	MultidimArray<int> mask;
	mask.initZeros(vol);

	size_t xcenter, ycenter, zcenter, N;
	vol.getDimensions(xcenter, ycenter, zcenter, N);

	xcenter /=2;
	ycenter /=2;
	zcenter /=2;

	radius = radius*radius;

	FOR_ALL_ELEMENTS_IN_ARRAY3D(mask)
	{
		if (((k-zcenter)*(k-zcenter) + (i-ycenter)*(i-ycenter) + (j-xcenter)*(j-xcenter) )<=radius)
			A3D_ELEM(mask, k, i, j) = 1;
	}

	return mask;
}

void Monogenic::addNoise(MultidimArray<double> &vol, double mean, double stddev)
{

	std::default_random_engine generator;
	std::normal_distribution<double> dist(mean, stddev);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
		DIRECT_MULTIDIM_ELEM(vol,n) += dist(generator);
}


void Monogenic::applyMask(MultidimArray<double> &vol, MultidimArray<int> &mask)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
	{
		DIRECT_MULTIDIM_ELEM(vol,n) *= DIRECT_MULTIDIM_ELEM(mask,n);
	}
}


MultidimArray< std::complex<double> > Monogenic::applyMaskFourier(MultidimArray< std::complex<double> > &vol, MultidimArray<double> &mask)
{
	MultidimArray< std::complex<double> > filtered;
	filtered.resizeNoCopy(vol);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
	{
		DIRECT_MULTIDIM_ELEM(filtered,n) = DIRECT_MULTIDIM_ELEM(vol,n) * DIRECT_MULTIDIM_ELEM(mask,n);
	}

	return filtered;
}

MultidimArray<double> Monogenic::createDataTest(size_t xdim, size_t ydim, size_t zdim,
		double wavelength, double mean, double sigma)
{
	int siz_z, siz_y, siz_x;
	double x, y, z;

	siz_z = xdim/2;
	siz_y = ydim/2;
	siz_x = xdim/2;
	MultidimArray<double> testmap;
	testmap.initZeros(zdim, ydim, xdim);

	long n=0;
	for(int k=0; k<zdim; ++k)
	{
		z = (k - siz_z);
		z= z*z;
		for(int i=0; i<ydim; ++i)
		{
			y = (i - siz_y);
			y = y*y;
			y = z + y;
			for(int j=0; j<xdim; ++j)
			{
				x = (j - siz_x);
				x = x*x;
				x = sqrt(x + y);
				DIRECT_MULTIDIM_ELEM(testmap, n) = cos(2*PI/(wavelength)*x);
				++n;
			}
		}
	}

	FileName fn;
	fn = formatString("fringes.vol");
	Image<double> saveImg;
	saveImg() = testmap;
	saveImg.write(fn);

	return testmap;
}

void Monogenic::amplitudeMonoSigDir3D_LPF(const MultidimArray< std::complex<double> > &myfftV,
		FourierTransformer &transformer_inv,
		MultidimArray< std::complex<double> > &fftVRiesz,
		MultidimArray< std::complex<double> > &fftVRiesz_aux, MultidimArray<double> &VRiesz,
		double freq, double freqH, double freqL, MultidimArray<double> &iu,
		Matrix1D<double> &freq_fourier_x, Matrix1D<double> &freq_fourier_y,
		Matrix1D<double> &freq_fourier_z, MultidimArray<double> &amplitude,
		int count, int dir, FileName fnDebug, int N_smoothing)
{
	fftVRiesz.initZeros(myfftV);
	fftVRiesz_aux.initZeros(myfftV);
	std::complex<double> J(0,1);

	// Filter the input volume and add it to amplitude
	long n=0;
	double ideltal=PI/(freq-freqH);

	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);
				double un=1.0/iun;
				if (freqH<=un && un<=freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= 0.5*(1+cos((un-freq)*ideltal));//H;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= iun;
				} else if (un>freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= iun;
				}
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, amplitude);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) *= DIRECT_MULTIDIM_ELEM(amplitude,n);

	//TODO: create a macro with these kind of code
	// Calculate first component of Riesz vector
	double ux;
	n=0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier_x,j);
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
	}

	// Calculate second and third component of Riesz vector
	n=0;
	double uy, uz;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z,k);
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y,i);
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = uz*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	transformer_inv.inverseFourierTransform(fftVRiesz_aux, VRiesz);

//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
//		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);


//	amplitude.setXmippOrigin();
	int z_size = ZSIZE(amplitude);
	int siz = z_size*0.5;

	double limit_radius = (siz-N_smoothing);
	n=0;
	for(int k=0; k<z_size; ++k)
	{
		uz = (k - siz);
		uz *= uz;
		for(int i=0; i<z_size; ++i)
		{
			uy = (i - siz);
			uy *= uy;
			for(int j=0; j<z_size; ++j)
			{
				ux = (j - siz);
				ux *= ux;
				DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
				DIRECT_MULTIDIM_ELEM(amplitude,n)=sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
				double radius = sqrt(ux + uy + uz);
				if ((radius>=limit_radius) && (radius<=siz))
					DIRECT_MULTIDIM_ELEM(amplitude, n) *= 0.5*(1+cos(PI*(limit_radius-radius)/(N_smoothing)));
				else if (radius>siz)
					DIRECT_MULTIDIM_ELEM(amplitude, n) = 0;
				++n;
			}
		}
	}

	//TODO: change (k - z_size*0.5)

	transformer_inv.FourierTransform(amplitude, fftVRiesz, false);

	double raised_w = PI/(freqL-freq);

	n=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftVRiesz)
	{
		double un=1.0/DIRECT_MULTIDIM_ELEM(iu,n);
		if ((freqL)>=un && un>=freq)
		{
			DIRECT_MULTIDIM_ELEM(fftVRiesz,n) *= 0.5*(1 + cos(raised_w*(un-freq)));
		}
		else
		{
			if (un>freqL)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz,n) = 0;
			}
		}
	}

	transformer_inv.inverseFourierTransform();

}


bool Monogenic::TestmonogenicAmplitude_3D_Fourier()
{
	size_t xdim, ydim, zdim;

	double wavelength = 10.0;
	xdim = 300;
	ydim = 300;
	zdim = 300;
	double mean =0.0, sigma = 0.0;

	MultidimArray<double> testmap;
	testmap = createDataTest(xdim, ydim, zdim, wavelength, mean, sigma);

	MultidimArray< std::complex<double> > myfftV;
	FourierTransformer transformer;

	transformer.FourierTransform(testmap, myfftV);

	MultidimArray<double> iu;
	MultidimArray<double> amplitude;
	amplitude.resizeNoCopy(testmap);

	monogenicAmplitude_3D_Fourier(myfftV, iu, amplitude, 1);

	double Sum =0;
	double N=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		Sum += DIRECT_MULTIDIM_ELEM(amplitude, n);
		N++;
	}

	double avg = Sum/N;

	bool testResult;
	if ( (avg < 1.1) && (avg > 0.9))
	{
		testResult = true;
	}
	else
	{
		testResult = false;
		std::cout << "Error in the calculus of the monogenic signal check " << std::endl;
		std::cout << "the function  monogenicAmplitude_3D_Fourier in the monogenic class" << std::endl;
	}

//	FileName fn;
//	Image<double> saveImg;
//	fn = formatString("MA.vol");
//	saveImg() = amplitude;
//	saveImg.write(fn);

	return testResult;

}



//void Monogenic::monogenicAmplitude_3D(FourierTransformer &transformer, MultidimArray<double> &iu,
//		Matrix1D<double> &freq_fourier_z, Matrix1D<double> &freq_fourier_y, Matrix1D<double> &freq_fourier_x,
//		MultidimArray< std::complex<double> > &fftVRiesz, MultidimArray< std::complex<double> > &fftVRiesz_aux,
//		MultidimArray<double> &VRiesz, const MultidimArray<double> &inputVol,
//		MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &amplitude)
//{
//	VRiesz.resizeNoCopy(inputVol);
//
//	std::complex<double> J(0,1);
//
//	fftVRiesz.initZeros(myfftV);
//	fftVRiesz_aux.initZeros(myfftV);
//
//	//Original volume in real space and preparing Riesz components
//	long n=0;
//	for(size_t k=0; k<ZSIZE(myfftV); ++k)
//	{
//		for(size_t i=0; i<YSIZE(myfftV); ++i)
//		{
//			for(size_t j=0; j<XSIZE(myfftV); ++j)
//			{
//				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
//				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
//				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
//				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(iu,n);
//				++n;
//			}
//		}
//	}
//
//	transformer.inverseFourierTransform(fftVRiesz, amplitude);
//
////	#ifdef DEBUG_DIR
////		Image<double> filteredvolume;
////		filteredvolume = VRiesz;
////		filteredvolume.write(formatString("Volumen_filtrado_%i_%i.vol", dir+1,count));
////	#endif
//
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
//		DIRECT_MULTIDIM_ELEM(amplitude,n) *= DIRECT_MULTIDIM_ELEM(amplitude,n);
//
//
//	// Calculate first component of Riesz vector
//	double ux;
//	n=0;
//	for(size_t k=0; k<ZSIZE(myfftV); ++k)
//	{
//		for(size_t i=0; i<YSIZE(myfftV); ++i)
//		{
//			for(size_t j=0; j<XSIZE(myfftV); ++j)
//			{
//				ux = VEC_ELEM(freq_fourier_x,j);
//				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
//				++n;
//			}
//		}
//	}
//
//	transformer.inverseFourierTransform(fftVRiesz, VRiesz);
//
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
//		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
//
//	// Calculate second and third component of Riesz vector
//	n=0;
//	double uy, uz;
//	for(size_t k=0; k<ZSIZE(myfftV); ++k)
//	{
//		uz = VEC_ELEM(freq_fourier_z,k);
//		for(size_t i=0; i<YSIZE(myfftV); ++i)
//		{
//			uy = VEC_ELEM(freq_fourier_y,i);
//			for(size_t j=0; j<XSIZE(myfftV); ++j)
//			{
//				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = uz*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
//				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
//				++n;
//			}
//		}
//	}
//
//	transformer.inverseFourierTransform(fftVRiesz, VRiesz);
//
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
//		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
//
//	transformer.inverseFourierTransform(fftVRiesz_aux, VRiesz);
//
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
//		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
//		DIRECT_MULTIDIM_ELEM(amplitude,n) = sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
//}
//

//void Monogenic::monogenicAmplitudeHPF_3D(const MultidimArray<double> &inputVol, MultidimArray<double> &amplitude, int numberOfThreads=1)
//{
//	MultidimArray< std::complex<double> > iu, fftVRiesz, fftVRiesz_aux, myfftV;
//	MultidimArray<double> VRiesz;
//	VRiesz.resizeNoCopy(inputVol);
//
//	FourierTransformer transformer;
//	transformer.setThreadsNumber(numberOfThreads);
//
//	transformer.FourierTransform(inputVol, myfftV);
//
//	double u;
//	Matrix1D<double> freq_fourier_z, freq_fourier_y, freq_fourier_x;
//
//	freq_fourier_z.initZeros(ZSIZE(myfftV));
//	freq_fourier_x.initZeros(XSIZE(myfftV));
//	freq_fourier_y.initZeros(YSIZE(myfftV));
//
//	VEC_ELEM(freq_fourier_z,0) = 1e-38;
//	for(size_t k=0; k<ZSIZE(myfftV); ++k)
//	{
//		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol), u);
//		VEC_ELEM(freq_fourier_z,k) = u;
//	}
//	VEC_ELEM(freq_fourier_y,0) = 1e-38;
//	for(size_t k=0; k<YSIZE(myfftV); ++k)
//	{
//		FFT_IDX2DIGFREQ(k,YSIZE(inputVol), u);
//		VEC_ELEM(freq_fourier_y,k) = u;
//	}
//	VEC_ELEM(freq_fourier_x,0) = 1e-38;
//	for(size_t k=0; k<XSIZE(myfftV); ++k)
//	{
//		FFT_IDX2DIGFREQ(k,XSIZE(inputVol), u);
//		VEC_ELEM(freq_fourier_x,k) = u;
//	}
//
//	fftVRiesz.initZeros(myfftV);
//	fftVRiesz_aux.initZeros(myfftV);
//
//	iu = fourierFreqs_3D(myfftV, inputVol);
//
//	monogenicAmplitude_3D(transformer, iu, freq_fourier_z, freq_fourier_y, freq_fourier_x,
//							fftVRiesz, fftVRiesz_aux, VRiesz, inputVol, myfftV, amplitude);
//}

void Monogenic::monogenicAmplitude_2D(const MultidimArray<double> &inputImg, MultidimArray<double> &amplitude, int numberOfThreads)
{
	MultidimArray< std::complex<double> > fftVRiesz, fftVRiesz_aux, myfftImg;
	MultidimArray<double> VRiesz, iu;
	VRiesz = inputImg;

	FourierTransformer transformer;
	transformer.setThreadsNumber(numberOfThreads);

	transformer.FourierTransform(VRiesz, myfftImg);

	double u;
	Matrix1D<double> freq_fourier_y, freq_fourier_x;

	freq_fourier_x.initZeros(XSIZE(myfftImg));
	freq_fourier_y.initZeros(YSIZE(myfftImg));

	VEC_ELEM(freq_fourier_y,0) = 1e-38;
	for(size_t k=0; k<YSIZE(myfftImg); ++k)
	{
		FFT_IDX2DIGFREQ(k,YSIZE(inputImg), u);
		VEC_ELEM(freq_fourier_y,k) = u;
	}
	VEC_ELEM(freq_fourier_x,0) = 1e-38;
	for(size_t k=0; k<XSIZE(myfftImg); ++k)
	{
		FFT_IDX2DIGFREQ(k,XSIZE(inputImg), u);
		VEC_ELEM(freq_fourier_x,k) = u;
	}

	fftVRiesz.initZeros(myfftImg);
	fftVRiesz_aux.initZeros(myfftImg);

	iu = fourierFreqs_2D(myfftImg, inputImg);

	monogenicAmplitude_2D(transformer, iu, freq_fourier_y, freq_fourier_x,
							fftVRiesz, fftVRiesz_aux, VRiesz, inputImg, myfftImg, amplitude);
}

void Monogenic::monogenicAmplitude_2D(FourierTransformer &transformer, MultidimArray<double> &iu,
		Matrix1D<double> &freq_fourier_y, Matrix1D<double> &freq_fourier_x,
		MultidimArray< std::complex<double> > &fftVRiesz, MultidimArray< std::complex<double> > &fftVRiesz_aux,
		MultidimArray<double> &VRiesz, const MultidimArray<double> &inputImg,
		MultidimArray< std::complex<double> > &myfftImg, MultidimArray<double> &amplitude)
{
	VRiesz.resizeNoCopy(inputImg);

	std::complex<double> J(0,1);

	fftVRiesz.initZeros(myfftImg);
	fftVRiesz_aux.initZeros(myfftImg);

	//Original volume in real space and preparing Riesz components
	long n=0;
	for(size_t i=0; i<YSIZE(myfftImg); ++i)
	{
		for(size_t j=0; j<XSIZE(myfftImg); ++j)
		{
			DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftImg, n);
			DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
			DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
			DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(iu,n);
			++n;
		}
	}

	transformer.inverseFourierTransform(fftVRiesz, amplitude);

//	#ifdef DEBUG_DIR
//		Image<double> filteredvolume;
//		filteredvolume = VRiesz;
//		filteredvolume.write(formatString("Volumen_filtrado_%i_%i.vol", dir+1,count));
//	#endif

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) *= DIRECT_MULTIDIM_ELEM(amplitude,n);


	// Calculate first and second component of Riesz vector
	double ux, uy;
	n=0;
	for(size_t i=0; i<YSIZE(myfftImg); ++i)
	{
		uy = VEC_ELEM(freq_fourier_y,i);
		for(size_t j=0; j<XSIZE(myfftImg); ++j)
		{
			ux = VEC_ELEM(freq_fourier_x,j);
			DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
			DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
			++n;
		}
	}

	transformer.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	transformer.inverseFourierTransform(fftVRiesz_aux, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
		DIRECT_MULTIDIM_ELEM(amplitude,n)=sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
}


void Monogenic::monogenicAmplitude_2DHPF(FourierTransformer &transformer, MultidimArray<double> &iu,
		double freqH, double freq, double freqL,
		Matrix1D<double> &freq_fourier_y, Matrix1D<double> &freq_fourier_x,
		MultidimArray< std::complex<double> > &fftVRiesz,
		MultidimArray< std::complex<double> > &fftVRiesz_aux,
		MultidimArray<double> &VRiesz,
		const MultidimArray<double> &inputImg,
		MultidimArray< std::complex<double> > &myfftImg,
		MultidimArray<double> &amplitude, int count, FileName fnDebug)
{


	std::complex<double> J(0,1);

	fftVRiesz.initZeros(myfftImg);
	fftVRiesz_aux.initZeros(myfftImg);

	//Original volume in real space and preparing Riesz components
	long n=0;

	double ideltal=PI/(freq-freqH);

	for(size_t i=0; i<YSIZE(myfftImg); ++i)
	{
		for(size_t j=0; j<XSIZE(myfftImg); ++j)
		{
			double iun=A2D_ELEM(iu, i, j);
			double un=1.0/iun;
			if (freqH<=un && un<=freq)
			{
				//double H=0.5*(1+cos((un-w1)*ideltal));
				A2D_ELEM(fftVRiesz, i, j) = A2D_ELEM(myfftImg, i, j);
				A2D_ELEM(fftVRiesz, i, j) *= 0.5*(1+cos((un-freq)*ideltal));//H;
				A2D_ELEM(fftVRiesz_aux, i, j) = -J;
				A2D_ELEM(fftVRiesz_aux, i, j) *= A2D_ELEM(fftVRiesz, i, j);
				A2D_ELEM(fftVRiesz_aux, i, j) *= iun;
			} else if (un>freq)
			{
				A2D_ELEM(fftVRiesz, i, j) = A2D_ELEM(myfftImg, i, j);
				A2D_ELEM(fftVRiesz_aux, i, j) = -J;
				A2D_ELEM(fftVRiesz_aux, i, j) *= A2D_ELEM(fftVRiesz, i, j);
				A2D_ELEM(fftVRiesz_aux, i, j) *= iun;
			}
			n++;
		}
	}

	amplitude.resizeNoCopy(inputImg);
	transformer.inverseFourierTransform(fftVRiesz, amplitude);


////	#ifdef DEBUG_DIR
	Image<double> auxImg;
	auxImg() = amplitude;
	FileName fntest;

	fntest = formatString("Filtered_%i.mrc", count);
	auxImg.write(fnDebug+fntest);
////	#endif

	n=0;
	for(size_t i=0; i<YSIZE(amplitude); ++i){
		for(size_t j=0; j<XSIZE(amplitude); ++j){
			A2D_ELEM(amplitude, i, j) = A2D_ELEM(amplitude, i, j)*A2D_ELEM(amplitude, i, j);
			//		DIRECT_MULTIDIM_ELEM(amplitude,n) = DIRECT_MULTIDIM_ELEM(amplitude,n)*DIRECT_MULTIDIM_ELEM(amplitude,n);
		}
	}

	// Calculate first and second component of Riesz vector
	double ux, uy;
	n=0;
	for(size_t i=0; i<YSIZE(myfftImg); ++i)
	{
		uy = VEC_ELEM(freq_fourier_y,i);
		for(size_t j=0; j<XSIZE(myfftImg); ++j)
		{
			ux = VEC_ELEM(freq_fourier_x,j);
			A2D_ELEM(fftVRiesz, i, j) =  ux*A2D_ELEM(fftVRiesz_aux, i, j);
			A2D_ELEM(fftVRiesz_aux, i, j) = uy*A2D_ELEM(fftVRiesz_aux, i, j);
//			DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
//			DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
//			++n;
		}
	}
	VRiesz.resizeNoCopy(inputImg);
	transformer.inverseFourierTransform(fftVRiesz, VRiesz);


//	//	Image<double> auxImg;
//		auxImg() = VRiesz;
//	//	FileName fntest;
//
//		fntest = formatString("AmplitudeAMx_%i.mrc", count);
//		std::cout << fntest << std::endl;
//		auxImg.write(fntest);


	for(size_t i=0; i<YSIZE(amplitude); ++i){
		for(size_t j=0; j<XSIZE(amplitude); ++j)
		{
//			DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
			A2D_ELEM(amplitude, i, j) += A2D_ELEM(VRiesz, i, j)*A2D_ELEM(VRiesz, i, j);
		}
	}
 //
	transformer.inverseFourierTransform(fftVRiesz_aux, VRiesz);

	for(size_t i=0; i<YSIZE(amplitude); ++i){
		for(size_t j=0; j<XSIZE(amplitude); ++j)
		{
//		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
//		DIRECT_MULTIDIM_ELEM(amplitude,n) = sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
			A2D_ELEM(amplitude, i, j) += A2D_ELEM(VRiesz, i, j)*A2D_ELEM(VRiesz, i, j);
			A2D_ELEM(amplitude, i, j) = sqrt(A2D_ELEM(amplitude, i, j));
		}
	}


	auxImg() = amplitude;

	fntest = formatString("Amplitud_%i.mrc", count);

	auxImg.write(fnDebug+fntest);
//	if (freqL-freq >0)
//	{
//		// Low pass filter the monogenic amplitude
//		transformer.FourierTransform(amplitude, fftVRiesz, false);
//		double raised_w = PI/(freqL-freq + 1e-38);
//
//		n=0;
//
//		FOR_ALL_ELEMENTS_IN_ARRAY2D(fftVRiesz)
//		{
//			double un=1.0/A2D_ELEM(iu, i, j);
//	//		std::cout << "un = " << un << "  freqL = " << freqL << " freq = " << freq << std::endl;
//			if ((freqL)>=un && un>=freq)
//			{
//				A2D_ELEM(fftVRiesz, i, j) *= 0.5*(1 + cos(raised_w*(un-freq)));
//			}
//			else
//			{
//				if (un>freqL)
//				{
//					A2D_ELEM(fftVRiesz, i, j) = 0;
//				}
//			}
//		}
//		transformer.inverseFourierTransform();
//	}
//
//	auxImg() = amplitude;
//
//	fntest = formatString("AmplitudFiltered_%i.mrc", count);
//
//	auxImg.write(fnDebug+fntest);

}

//void Monogenic::bandPassFilterFunction(FourierTransformer &transformer, MultidimArray< std::complex<double> > &iu,
//		const MultidimArray< std::complex<double> > &myfftV,
//                double w, double wL, MultidimArray<double> &filteredVol, int count)
//{
//		MultidimArray< std::complex<double> > &fftVfilter;
//        fftVfilter.initZeros(myfftV);
//        size_t xdim, ydim, zdim, ndim;
//
//        double delta = wL-w;
//        double w_inf = w-delta;
//        // Filter the input volume and add it to amplitude
//        long n=0;
//        double ideltal=PI/(delta);
//        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
//        {
//                double un=DIRECT_MULTIDIM_ELEM(iu,n);
//                if (un>=w && un<=wL)
//                {
//                        DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
//                        DIRECT_MULTIDIM_ELEM(fftVfilter, n) *= 0.5*(1+cos((un-w)*ideltal));//H;
//                } else if (un<=w && un>=w_inf)
//                {
//                        DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
//                        DIRECT_MULTIDIM_ELEM(fftVfilter, n) *= 0.5*(1+cos((un-w)*ideltal));//H;
//                }
//        }
//
//        filteredVol.resizeNoCopy(Vorig);
//
//        transformer.inverseFourierTransform(fftVfilter, filteredVol);
//
////        #ifdef DEBUG_FILTER
////        Image<double> filteredvolume;
////        filteredvolume() = filteredVol;
////        filteredvolume.write(formatString("Volumen_filtrado_%i.vol", count));
////        #endif
//}
