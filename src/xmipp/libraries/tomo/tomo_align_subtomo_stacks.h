/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
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

#ifndef _PROG_TOMO_ALIGN_SUBTOMO_STACKS
#define _PROG_TOMO_ALIGN_SUBTOMO_STACKS

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/metadata_vec.h>
#include "data/fourier_projection.h"
#include <limits>
#include <complex>
#include <string>


class ProgTomoAlignSubtomoStacks : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut;
    FileName fnIn;
    FileName fnIniVol;
    FileName fnGal;

    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
    size_t Xts, Yts;
    double tilt, rot, tx, ty;

    bool invertContrast, normalize, swapXY, setCTF, defocusPositive;
    std::vector<double> tsTiltAngles, tsRotAngles, tsShiftX, tsShiftY, tsDefU, tsDefV, tsDefAng, tsDose;
    std::vector<MultidimArray<double> > tsImages;
    std::vector<size_t> freqIdx;

    double scaleFactor, sampling, maxFreq;

    FourierProjector projector;

	int boxsize, nbest;
    int nthrs;

public:

    void defineParams();

    void alignAgainstGallery(MultidimArray<double> &img0, MetaDataVec &mdGallery);

    void corr2(MultidimArray<double> &img0, MultidimArray<double> &imgGal, double &corrVal);

    void createCTFImage(MultidimArray<double> &ctfImage);

    void listofParticles(std::vector<size_t> &listparticlesIds);

    void validateAlignment(MultidimArray<double> &initVol, MultidimArray<double> &imgExp, double &rot, double &tilt, double &psi, int ydim, int xdim);

    void fourierDistance(MultidimArray<double> &img0, MultidimArray<double> &imgGal, double &dist2);

    void freqMask(const MultidimArray< std::complex<double> > &mapfftV, const MultidimArray<double> &inputVol, double &maxDigFreq);

    void sortAndRemoveMdParticles(MetaDataVec &mdParticles);

    template <typename T>
    std::vector<size_t> sort_indexes(const std::vector<T> &v);

    void readParams();

    void run();
};
//@}
#endif
