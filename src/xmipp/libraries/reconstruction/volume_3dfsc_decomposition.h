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

/**@defgroup Directional Sharpening
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class Prog3dDecomp : public XmippProgram
{
public:
     /** Filenames */
    FileName fnOut, fnVol, fnRes, fnMD, fnMask;

    /** sampling rate, minimum resolution, and maximum resolution */
    double sampling, maxRes, minRes, lambda, K, maxFreq, minFreq, desv_Vorig, R, significance, res_step;
    int Niter, Nthread;
    bool test, icosahedron;

public:

    void defineParams();
    void readParams();
    void produceSideInfo();
    void icosahedronVertex(Matrix2D<double> &angles);
    void icosahedronFaces(Matrix2D<int> &faces, Matrix2D<double> &vertex);

    double averageInMultidimArray(MultidimArray<double> &amplitude, MultidimArray<int> &mask);

    void getFaceVectorIcosahedron(Matrix2D<int> &faces,
            Matrix2D<double> &vertex, Matrix2D<double> &facesVector);

    void defineComplexCaps(Matrix2D<double> &facesVector,
            MultidimArray< std::complex<double> > &myfftV, MultidimArray<int> &coneMask);

    void getFaceVectorSimple(Matrix2D<double> &facesVector, Matrix2D<double> &faces);

    void FilterFunction(size_t &Nfaces, MultidimArray<int> &maskCone,
            MultidimArray<double> &vol, FourierTransformer &transformer_inv);

    void defineIcosahedronCone(int face_number, double &x1, double &y1, double &z1,
            MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &conefilter,
            double coneAngle);

    void simpleGeometryFaces(Matrix2D<double> &faces, Matrix2D<double> &limts);

    void defineSimpleCaps(MultidimArray<int> &coneMask, Matrix2D<double> &limits,
            MultidimArray< std::complex<double> > &myfftV);

    void createFullFourier(MultidimArray<double> &fourierHalf, FileName &fnMap,
            int m1sizeX, int m1sizeY, int m1sizeZ, MultidimArray<double> &fullMap);

    void getCompleteFourier(MultidimArray<double> &V, MultidimArray<double> &newV,
            int m1sizeX, int m1sizeY, int m1sizeZ);


    void run();
public:
    //, VsoftMask;
    Image<int> mask;
    MultidimArray<double> iu, VRiesz, sharpenedMap; // Inverse of the frequency
    MultidimArray< std::complex<double> > fftV, fftVfilter; // Fourier transform of the input volume
    FourierTransformer transformer, transformer_inv;
    MultidimArray< std::complex<double> > fftVRiesz, fftVRiesz_aux;
    Matrix2D<double> angles, resolutionMatrix, maskMatrix, trigProducts;
    Matrix1D<double> freq_fourier_x, freq_fourier_y, freq_fourier_z;
    int N_smoothing, Rparticle;
    long NVoxelsOriginalMask;
};
//@}
#endif
