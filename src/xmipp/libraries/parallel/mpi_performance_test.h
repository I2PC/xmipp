/***************************************************************************
 *
 * Authors:    Carlos Oscar            coss@cnb.csic.es (2002)
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
#ifndef _PROG_PERFORMANCE_TEST
#define _PROG_PERFORMANCE_TEST

#include <parallel/xmipp_mpi.h>
#include <classification/pca.h>

/// @defgroup ProgPerformanceTest MPI Performance Test
/// @ingroup ParallelLibrary
//@{
class ProgPerformanceTest: public XmippProgram
{
public:
	/** Input selfile */
	FileName fnIn;
public:
    // Mpi node
    MpiNode *node=nullptr;
public:
    /// Empty constructor
    ProgPerformanceTest(int argc, char **argv);
    ProgPerformanceTest(const ProgPerformanceTest&)=delete;
    ProgPerformanceTest(const ProgPerformanceTest&&)=delete;

    /// Destructor
    ~ProgPerformanceTest();
    ProgPerformanceTest & operator =(const ProgPerformanceTest &)=delete;
    ProgPerformanceTest & operator =(const ProgPerformanceTest &&)=delete;

    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Usage
    void defineParams();

    /// Produce side info
    void produceSideInfo();

    /** Run. */
    void run();
};
//@}
#endif
