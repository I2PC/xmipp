/***************************************************************************
 *
 * Authors:    Carlos Oscar             coss@cnb.csic.es
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

#ifndef _PROG_VOL_DEFORM_SPH
#define _PROG_VOL_DEFORM_SPH

#include <vector>

#include "core/xmipp_program.h"
#include "core/xmipp_image.h"

/**@defgroup VolDeformSph Deform a volume using spherical harmonics
   @ingroup ReconsLibrary */
//@{
/** Sph Alignment Parameters. */
class ProgVolDeformSph: public XmippProgram
{
public:
	/// Volume to deform
	FileName fnVolI;

    /// Reference volume
    FileName fnVolR;

    /// Output Volume (deformed input volume)
    FileName fnVolOut;

    /// Save the deformation of each voxel for local strain and rotation analysis
    bool analyzeStrain;

    /// Radius optimization
    bool optimizeRadius;

    /// Degree of Zernike polynomials and spherical harmonics
    int depth;

    /// Gaussian width to filter the volumes
    std::vector<double> sigma;

    /// Image Vector
    std::vector<Image<double>> volumesI, volumesR;

    /// Maximum radius for the transformation
	double Rmax;

public:
	/// Degree of the spherical harmonic
	int L, prevL;

	Image<double> VI, VR, VO, Gx, Gy, Gz;

	//Vector containing the degree of the Zernike-Spherical harmonics
	Matrix1D<double> clnm;

	//Deformation in pixels
	double deformation;

	// Save output volume
	bool applyTransformation;

	// Save the values of gx, gy and gz for local strain and rotation analysis
	bool saveDeformation;

public:
    /// Define params
    void defineParams();

    /// Read arguments from command line
    void readParams();

    /// Show
    void show();

    /// Distance
    double distance(double *pclnm);

    /// Run
    void run();

    /// Copy the coefficients from harmonical depth n-1 vector to harmonical depth n vector
    void copyvectors(Matrix1D<double> &oldvect,Matrix1D<double> &newvect);

    /// Determine the positions to be minimize of a vector containing spherical harmonic coefficients
    void minimizepos(Matrix1D<double> &vectpos, Matrix1D<double> &prevpos);

    ///Compute the number of spherical harmonics in l=0,1,...,depth
    void Numsph(Matrix1D<int> &sphD);

    /// Compute strain
    void computeStrain();
};

//@}
#endif
