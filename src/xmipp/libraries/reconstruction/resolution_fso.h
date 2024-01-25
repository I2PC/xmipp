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


class ProgFSO : public XmippProgram
{
private:
    // Filenames
    FileName fnhalf1, fnhalf2, fnmask, fnOut;

    // Double Params
    double sampling, ang_con, thrs;

    // Int params
    int Nthreads;
       
    // Bool params
    bool do_3dfsc_filter;

    // Matrix2d for the projection angles
    Matrix2D<float> angles;

    // Int params
    size_t xvoldim, yvoldim, zvoldim, fscshellNum;

	// Frequency vectors and frequency map
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;
	MultidimArray<float> fx, fy, fz;
	MultidimArray<float> threeD_FSC, normalizationMap, aniFilter;
    MultidimArray< double > freqMap;

	// Half maps
	MultidimArray< std::complex< double > > FT1, FT2;
	MultidimArray<float> real_z1z2, absz1_vec, absz2_vec;

	//Access indices
	MultidimArray<long> freqElems, cumpos, freqidx, arr2indx;


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

        /* 
        * */
        double incompleteGammaFunction(double &x);

        /* Defines a Matrix2D with coordinates Rot and tilt achieving a uniform coverage of the
        * projection sphere. Bool alot = True, implies a dense coverage */
        void generateDirections(Matrix2D<float> &angles, bool alot);

        /* ANISOTROPYPARAMETER: Given a directional FSC it is determined how many 
        * frequencies/points of the FSC has a greater fsc than the fsc threshold, thrs,
        * This is carried out in aniParam, . */
        void anistropyParameter(const MultidimArray<float> &FSC,
			                    MultidimArray<float> &aniParam, double thrs);

        /* Estimates the directional FSC between two half maps FT1 and FT2 (in Fourier Space)
        * requires the sampling rate, and the frequency vectors,  */
		void fscDir_fast(MultidimArray<float> &fsc, double rot, double tilt,
				                      MultidimArray<float> &threeD_FSC, 
						              MultidimArray<float> &normalizationMap,
						              double &thrs, double &resol, std::vector<Matrix2D<double>> &freqMat, size_t dirnum);

        /* PREPAREDATA: Data are prepared to be taken by the algorithm. 
        * The half maps will be read and stored in the multidimarray half1, and half2.
        * Also de mask (if provided) is read and stored in mask (defined in .h) */
        void prepareData(MultidimArray<double> &half1, MultidimArray<double> &half2);

        /* SAVEANISOTROPYTOMETADATA - The FSO is stored in metadata file. The FSO comes from
        * anisotropy and the frequencies from freq. */
        void saveAnisotropyToMetadata(MetaDataVec &mdAnisotropy,
    		                    	const MultidimArray<double> &freq,
			                      	const MultidimArray<float> &anisotropy, const MultidimArray<double> &isotropyMatrix);

        /* DIRECTIONALFILTER: The half maps are summed to get the full map, and are filtered
        * by an anisotropic filter with cutoff the isosurface of the fsc at the given threshold
        * introduced by the user. */
        void directionalFilter(MultidimArray<std::complex<double>> &FThalf1,
    		                    MultidimArray<double> &threeDfsc, MultidimArray<double> &filteredMap,
                            	int m1sizeX, int m1sizeY, int m1sizeZ);

        void directionalFilterHalves(MultidimArray<std::complex<double>> &FThalf1,
    			MultidimArray<double> &threeDfsc);

        /* RESOLUTIONDISTRIBUTION: This function stores in a metadata the resolution distribution on the
        * projection sphere. Thus the metadata contains the resolution of each direction.
        * To do that, a matrix with rows the rot angle and columns the tilt angle is created.
        * the rot angle goes from from 0-360 and tilt from 0-90 both in steps of 1 degree. */
        void resolutionDistribution(MultidimArray<double> &resDirFSC, FileName &fn);

        /* GETCOMPLETEFOURIER: Because of the hermitician symmetry of the Fourier space, xmipp works with the half
        *  of the space. This function recover the whole Fourier space (both halfs). */
        void getCompleteFourier(MultidimArray<double> &V, MultidimArray<double> &newV,
    		                    int m1sizeX, int m1sizeY, int m1sizeZ);

        /* CREATEFULLFOURIER: The inpur is a Fourier Transform, the function will compute the full Fourier
        *  and will save it in disc, with the name of fnMap m1sizeX, m1sizeY, m1sizeZ, define the size. */
        void createFullFourier(MultidimArray<double> &fourierHalf, FileName &fnMap,
    	                      	int m1sizeX, int m1sizeY, int m1sizeZ);

        /* FSCINTERPOLATION: The exact resolution of the the FSC = thrs is estimated. thrs is a global variable */
        void fscInterpolation(const MultidimArray<double> &freq, const MultidimArray< double > &frc);

public:
        /* Defining the params and help of the algorithm */
        void defineParams();

        /* It reads the input parameters */
        void readParams();

        /* Run the program */
        void run();
};

#endif
