/***************************************************************************
 *
 * Authors:    Tomas Majtner           tmajtner@cnb.csic.es (2017)
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
#ifndef _PROG_KMEANS2DCLUSTERING
#define _PROG_KMEANS2DCLUSTERING

#include <numeric>
#include <core/xmipp_program.h>
#include <data/filters.h>
#include <data/morphology.h>
#include "classify_extract_features.h"

class ProgClassifyKmeans2D: public XmippProgram
{
public:
    /** Filename selection file containing the images */
    FileName fnSel;

    /**  Filename output root */
    FileName fnOut;

    /**  Filename clusters */
    FileName fnClusters;

    /**  Filename points */
    FileName fnPoints;

    /**  Number of clusters */
    int K;

    /**  Maximum number of objects used for clustering */
    int maxObjects;

public:
    /// Read argument
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /// Main routine
    void run();
};
#endif