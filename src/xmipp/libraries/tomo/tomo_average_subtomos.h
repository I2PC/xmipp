/***************************************************************************
 *
 * Authors:    Jose Luis Vilas (jlvilas@cnb.csic.es)
 *             Oier Lauzirika  (olauzirika@cnb.csic.es)

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

#ifndef _PROG_AVERAGE_SUBTOMOS
#define _PROG_AVERAGE_SUBTOMOS

#include <core/xmipp_program.h>
#include <core/xmipp_filename.h>
#include <core/metadata_vec.h>


class ProgAverageSubtomos : public XmippProgram
{
private:
    /* Filenames of the input subtomograms (.xmd file)
	and output file (volume of the average) */
    FileName fnSubtomos, fnOut;

    /* Sampling rate */
    double sampling;

    /* Number of threads */
    int Nthreads;

    /* Flag to avoid applying the alignment of hte subtomograms
     * and for normalizing the subtomograms, or save each aligned
     * subtomogram as a file */
    bool notapplyAlignment, saveAligned;


public:
	    /* This function takes a set of subtomograms with or without alignment, and
	     * estimates the average of all subtomograms. The flag saveAligned allows
	     * to saved the applied alignment of each subtomogram
	     */
		void averageSubtomograms(MetaDataVec &md, bool saveAligned=false);

        /* Defining the params and help of the algorithm */
        void defineParams();

        /* It reads the input parameters */
        void readParams();

        /* Run the program */
        void run();
};

#endif
