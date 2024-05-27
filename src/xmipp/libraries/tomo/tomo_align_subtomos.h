/***************************************************************************
 *
 * Authors:    Jose Luis Vilas (jlvilas@cnb.csic.es)
 *             Oier Lauzirika  (olauzirika@cnb.csic.es)

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

#ifndef _PROG_ALIGN_SUBTOMOS
#define _PROG_ALIGN_SUBTOMOS

#include <core/xmipp_program.h>
#include <core/matrix2d.h>
#include <core/xmipp_fftw.h>
#include <core/xmipp_filename.h>
#include <core/metadata_vec.h>
#include <data/fourier_filter.h>


class ProgAlignSubtomos : public XmippProgram
{
private:
    // Filenames
    FileName fnIn, fnRef, fnOut;

    // Double Params
    double sampling, angularSampling;

    // Int params
    int Nthreads;

    // Estimate average of aligned subtomograms
    bool doSTA;
       

public:
	template<typename T>
	void readReference(MultidimArray<T> &ref);

	void readSubtomos(std::vector<FileName> &fnVec, double &randomSubset);

	template<typename T>
	void initialReference(MultidimArray<T> &ref, std::vector<FileName> &fnVec);

	template<typename T>
	void alignSubtomos(MultidimArray<T> &ref, std::vector<FileName> &fnVec);

	template<typename T>
	void applyAlignmentAndAverage(MultidimArray<T> &ref, MetaDataVec &md, bool saveAligned=false);

	template<typename T>
	double twofoldAlign(MultidimArray<T> &ref, MultidimArray<T> &subtomo,
												    double &rot, double &tilt, double &psi);

	/* Defining the params and help of the algorithm */
	void defineParams();

	/* It reads the input parameters */
	void readParams();

	/* Run the program */
	void run();
};

#endif
