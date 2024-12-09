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

#ifndef _PROG_VOL_DEFORM_SPH
#define _PROG_VOL_DEFORM_SPH

#include <vector>
#include <ctpl_stl.h>
#include "core/xmipp_program.h"
#include "core/xmipp_image.h"
#include "data/point3D.h"

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

    /// Root name for several output files
    FileName fnRoot;

    /// Save the deformation of each voxel for local strain and rotation analysis
    bool analyzeStrain;

    /// Radius optimization
    bool optimizeRadius;

    /// Degree of Zernike polynomials and spherical harmonics
    int L1; 
    int L2;

    /// Gaussian width to filter the volumes
    std::vector<double> sigma;

    /// Image Vector
    std::vector<Image<double>> volumesI;
    std::vector<Image<double>> volumesR;

    /// Maximum radius for the transformation
	double Rmax;

public:
	/// Coefficient vector size
	int vecSize;

    /// Images
	Image<double> VI;
    Image<double> VR;
    Image<double> VO;
    Image<double> Gx;
    Image<double> Gy;
    Image<double> Gz;

    /// Maxima of reference volumes (in absolute value)
    std::vector<double> absMaxR_vec;

	//Deformation in pixels, sumVI, sumVD
	double deformation;
    double sumVI;
    double sumVD;

    // Regularization
    double lambda;

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

    /// Determine the positions to be minimize of a vector containing spherical harmonic coefficients
    void minimizepos(int l2, Matrix1D<double> &steps) const;

    /// Length of coefficients vector
    void numCoefficients(int l1, int l2, int &nc) const;

    /// Zernike and SPH coefficients allocation
    void fillVectorTerms(int l1, int l2);

    /// Compute strain
    void computeStrain();

    /// Save vector to file
    void writeVector(std::string const &outPath, Matrix1D<double> const &v, bool append) const;

private:
    ctpl::thread_pool m_threadPool;
    struct ZSH_vals {
        int l1;
        int n;
        int l2;
        int m;
    };

    struct Radius_vals {
        Radius_vals(int i, int j, int k, double iRmax) {
            double k2 = k * k;
            double k2i2 = k2 + i * i;
            r2 = k2i2 + j * j;
            ir = i * iRmax;
            jr = j * iRmax;
            kr = k * iRmax;
            rr = std::sqrt(r2) * iRmax;
        }
        double r2;
        double jr;
        double ir;
        double kr;
        double rr;

    };

    struct Distance_vals {
        double VD;
        double diff;
        double modg;
        size_t count;
        Distance_vals& operator+=(const Distance_vals& rhs) {
              this->VD += rhs.VD;
              this->diff += rhs.diff;
              this->modg += rhs.modg;
              this->count += rhs.count;
              return *this;
        }
        friend Distance_vals operator+(Distance_vals lhs, const Distance_vals& rhs) {
            lhs += rhs;
            return lhs;
        }
    };

    // Zernike and SPH coefficients vectors
    std::vector<ZSH_vals> m_zshVals;

    //Vector containing the degree of the Zernike-Spherical harmonics
    std::vector<Point3D<double>> m_clnm;

    void computeShift(int k);

    void computeDistance(int k, Distance_vals &vals);

    template<bool APPLY_TRANSFORM, bool SAVE_DEFORMATION>
    void computeDistance(Distance_vals &vals);

    Distance_vals computeDistance();

    std::vector<Point3D<double>> m_shifts;
};

//@}
#endif
