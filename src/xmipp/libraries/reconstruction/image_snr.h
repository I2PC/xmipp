/***************************************************************************
 *
 * Authors:    Jose Luis Vilas (joseluis.vilas-prieto@yale.edu)
 *                             or (jlvilas@cnb.csic.es)
 *              Hemant. D. Tagare (hemant.tagare@yale.edu)
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

#ifndef _PROG_RES_FSO
#define _PROG_RES_FSO

#include <core/xmipp_program.h>
#include <core/matrix2d.h>
#include <core/xmipp_fftw.h>
#include <core/xmipp_filename.h>
#include <core/metadata_vec.h>


class ProgEstimateSNR : public XmippProgram
{
private:
    // Filenames
    FileName fnIn1, fnIn2, fnOut;

    size_t Nthreads, xvoldim;

    bool normalize;

    double sampling;

	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;

	MultidimArray<double> freqMap;
	MultidimArray<std::complex<double>> FT1, FT2;

private:
        /* Once done, the result of the computation is stored in freqMap (class field)
        * freqMap is a multidim array that define INVERSE of the value of the frequency in Fourier Space
        * To do that, it makes use of myfftV to detemine the size of the output map (mapfftV and
        * the output will have the same size), and the vectors freq_fourier_x, freq_fourier_y, 
        * and freq_fourier_z that defines the frequencies along the 3 axis. The output will be
        * sqrt(freq_fourier_x^2+freq_fourier_y^2+freq_fourier_x^2) */
        void defineFrequencies(const MultidimArray< std::complex<double> > &mapfftV,
    		                      const MultidimArray<double> &inputVol);

        /* Estimates the global FSC between two half maps FT1 and FT2 (in Fourier Space)
         * ARRANGEFSC_AND_FSCGLOBAL: This function estimates the global resolution FSC FSC=real(z1*conj(z2)/(||z1||·||z2||)
         * and precomputes the products z1*conj(z2),  ||z1||, ||z2||, to calculate faster the directional FSC. The input are
         * the Fourier transform of two half maps (FT1, FT2) defined in the .h, the sampling_rate, the used threshold (thrs)
         * to that define the resolution (FSC-threshold). The output are: 1) The resolution of the map in Angstrom (In the terminal).
         * 2) three vectors defined in the .h, real_z1z2, absz1_vec, absz2_vec, with z1*conj(z2),  ||z1||, ||z2||, defined in
         * float to speed up the computation and reduce the use of memory. These vector make use of the two half maps FT1, and FT2.
         */
		void arrangeFSC_and_fscGlobal(double sampling_rate,
				                    	double &thrs, MultidimArray<double> &freq);

        void normalizeHalf(MultidimArray<double> &half);

        void prepareHalf(MultidimArray<double> &half, FileName &fn);

        void computeSNR(MultidimArray<std::complex<double>> &FT1, MultidimArray<std::complex<double>> &FT2, double &SNR, double &FC);

        void computeHalves(FileName &fn1, FileName &fn2, double &FC, double SNR);

public:
        /* Defining the params and help of the algorithm */
        void defineParams();

        /* It reads the input parameters */
        void readParams();

        /* Run the program */
        void run();
};

#endif
