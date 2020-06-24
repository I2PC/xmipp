/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela,          eramirez@cnb.csic.es
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

#ifndef _PROG_DIR_SHARPENING
#define _PROG_DIR_SHARPENING

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/geometry.h>
#include <core/metadata.h>
#include <core/xmipp_hdf5.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <math.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/monogenic.h>
#include <data/filters.h>
#include <string>
#include "symmetrize.h"
#include "resolution_directional.h"
#include "volume_correct_bfactor.h"

/**@defgroup Directional Sharpening
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class Prog3dDecomp : public XmippProgram
{
public:
     /** Filenames */
    FileName fnVol, fnHalf1, fnHalf2, fnMask;

    /** sampling rate, minimum resolution, and maximum resolution */
    double sampling;
    int Nthread;
    bool  icosahedron, wfsc, mask_exist, local;
    float resol;

    /** values for guinier adjusment */
    int  xsize;
    double apply_maxres, fit_minres, fit_maxres, sampling_rate;
    double x, y, w;

public:

    void defineParams();
    void readParams();
    void produceSideInfo();
    void icosahedronVertex(Matrix2D<double> &angles);
    void icosahedronFaces(Matrix2D<int> &faces, Matrix2D<double> &vertex);

    void getFaceVectorIcosahedron(Matrix2D<int> &faces,
            Matrix2D<double> &vertex, Matrix2D<double> &facesVector);

    void defineComplexCaps(Matrix2D<double> &facesVector,
            MultidimArray< std::complex<double> > &myfftV, MultidimArray<int> &coneMask);

    void getFaceVectorSimple(Matrix2D<double> &facesVector, Matrix2D<double> &faces);

    void defineIcosahedronCone(int face_number, double &x1, double &y1, double &z1,
            MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &conefilter,
            double coneAngle);

    void simpleGeometryFaces(Matrix2D<double> &faces, Matrix2D<double> &limts);

    void defineSimpleCaps(MultidimArray<int> &coneMask, Matrix2D<double> &limits,
            MultidimArray< std::complex<double> > &myfftV);

    void FilterFunction(size_t &Nfaces, MultidimArray<int> &maskCone, MultidimArray<double> &vol,
            MultidimArray<double> &hmap1, MultidimArray<double> &hmap2, FourierTransformer &transformer_inv);


     //************Calculation of the FSC******************
//    void FscCalculation(MultidimArray<double> &half1, MultidimArray<double> &half2);
    void FscCalculation(MultidimArray< std::complex<double> > &half1,
                        MultidimArray< std::complex<double> > &half2,
                        MultidimArray<double> &frc);

//    void VolumesFsc(MultidimArray<double> &FThalf1, MultidimArray<double> &FThalf2);
    void VolumesFsc(MultidimArray< std::complex<double> > &FThalf1,
                    MultidimArray< std::complex<double> > &FThalf2);

    //************Calculation of the b-factor******************
    void snrWeights(std::vector<double> &snr);
    void apply_snrWeights(MultidimArray< std::complex< double > > &FT1,
                          std::vector<double> &snr);
   void make_guinier_plot(MultidimArray< std::complex< double > > &FT1,
            std::vector<fit_point2D> &guinier);
   void apply_bfactor(MultidimArray< std::complex< double > > &FT1,
                                           double bfactor);

   void directionalFilter(size_t &Nfaces, MultidimArray<int> &maskCone, MultidimArray<double> &vol,
           MultidimArray< std::complex< double > > &fftV, MultidimArray< std::complex< double > > &fftM1,
           MultidimArray< std::complex< double > > &fftM2, FourierTransformer &transformer_inv, MultidimArray<double> dsharpVolFinal);


    void run();
public:
    MultidimArray<double> iu; // Inverse of the frequency
    MultidimArray<double> frc;
    MultidimArray< std::complex<double> > fftV, fftVfilter; // Fourier transform of the input volume
    FourierTransformer transformer, transformer_inv;
    Matrix1D<double> freq_fourier_x, freq_fourier_y, freq_fourier_z;
};
//@}
#endif
