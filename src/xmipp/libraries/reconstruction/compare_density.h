/***************************************************************************
 *
 * Authors:    Carlos Oscar             coss@cnb.csic.es
 *             David Herreros Calero    dherreros@cnb.csic.es
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

#ifndef _PROG_COMPARE_DENSITY
#define _PROG_COMPARE_DENSITY

#include <vector>
#include <ctpl_stl.h>
#include "core/xmipp_program.h"
#include "core/xmipp_image.h"
#include "data/point3D.h"

/**@defgroup VolDeformSph Deform a volume using spherical harmonics
   @ingroup ReconsLibrary */
//@{
/** Sph Alignment Parameters. */
class ProgCompareDensity: public XmippProgram
{
public:
	/// Volumes to compare
	FileName fnVol1;
    FileName fnVol2;

    /// Output corelation image
    FileName fnImgOut;

    /// Degree step
    double degstep;

public:
    /// Images
	Image<double> V1, V2, CorrImg;

    /// Rot and tilt vectors
    std::vector<double> tilt_v, rot_v;

public:
    /// Define params
    void defineParams();

    /// Read arguments from command line
    void readParams();

    /// Show
    void show();

    /// Run
    void run();

    /// Compute corr image
    void computeCorrImage(int i);

    /// Compare binary images
    void compare(Image<double> &op1, const Image<double> &op2);

private:
    ctpl::thread_pool m_threadPool;
};

//@}
#endif
