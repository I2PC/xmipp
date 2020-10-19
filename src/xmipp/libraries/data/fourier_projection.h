/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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
 * 02111-1307  USAcd ..
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef _CORE_FOURIER_FOURIER_PROJECTION_H
#define _CORE_FOURIER_FOURIER_PROJECTION_H

#include "core/matrix2d.h"
#include "core/xmipp_fftw.h"
#include "core/xmipp_image.h"

/**@defgroup FourierProjection Fourier projection
   @ingroup ReconsLibrary */
//@{

/// @defgroup Projections Projections (2D Image + Euler angles)
/// @ingroup DataLibrary
//@{
/** Projection class.
 *
 * A projection is a 2D, double Image plus some information (about the direction
 * of projection) which makes it suitable for 3D reconstruction. A projection
 * is supposed to have the point (0,0) at the center of the image and not in
 * the corners as usual matrices have.
 *
 * The normal use of a projection is:
 *
 * @code
 * Projection P; // Create variable
 * P.reset(65, 65); // Init with zeros and set right origin
 * P.set_angles(30, 45, -50); // Set Euler angles
 * @endcode
 *
 * From now on, the projection can be treated as any other Image.
 */
class Projection: public Image<double>
{
public:
    /** Empty constructor */
    Projection();

    /** Vector perpendicular to the projection plane.
     * It is calculated as a function of rot and tilt.
     */
    Matrix1D< double > direction;

    /** Matrix used to pass from the Universal coordinate system to the
     * projection coordinate system.
     *
     * @code
     * Rp = euler * Ru
     * @endcode
     */
    Matrix2D< double > euler;

    /** Just the opposite.
     *
     * @code
     * Ru = eulert * Rp.
     * @endcode
     */
    Matrix2D< double > eulert;

    /** Init_zeros and move origin to center.
     *
     * This function initialises the projection plane with 0's, and then moves
     * the logical origin of the image to the physical center of the image
     * (using the Xmipp conception of image center).
     */
    void reset(int Ydim, int Xdim);

    /** Set Euler angles for this projection.
     *
     * The Euler angles are stored in the Xmipp header, then the pass matrices
     * (Universe <---> Projection coordinate system) are computed, and the
     * vector perpendicular to this projection plane is also calculated.
     */
    void setAngles(double _rot, double _tilt, double _psi);

    /** Read a Projection from file.
      *
      * When a projection is read, the Euler matrices and perpendicular
      * direction is computed and stored in the Projection structure.
      */
    void read(const FileName& fn, const bool only_apply_shifts = false,
              DataMode datamode = DATA, MDRow * row = NULL );

    /** Assignment.
     */
    Projection& operator=(const Projection& P);

    /** Another function for assignment.
     */
    void assign(const Projection& P);
};

/** Program class to create projections in Fourier space */
class FourierProjector
{
public:
    /// Padding factor
    double paddingFactor;
    /// Maximum Frequency for pixels
    double maxFrequency;
    /// The order of B-Spline for interpolation
    double BSplineDeg;

public:
    // Auxiliary FFT transformer
    FourierTransformer transformer2D;

    // Volume to project
    MultidimArray<double> *volume;

    // Real and imaginary B-spline coefficients for Fourier of the volume
    MultidimArray< double > VfourierRealCoefs, VfourierImagCoefs;

    // Projection in Fourier space
    MultidimArray< std::complex<double> > projectionFourier;

    // Projection in real space
    Image<double> projection;

    // Phase shift image
    MultidimArray<double> phaseShiftImgB, phaseShiftImgA;

    // Original volume size
    int volumeSize;

    // Volume padded size
    int volumePaddedSize;

    // Euler matrix
    Matrix2D<double> E;
public:
    /* Empty constructor */
    FourierProjector(double paddFactor, double maxFreq, int degree);

    /*
     * The constructor of the class
     */
    FourierProjector(MultidimArray<double> &V, double paddFactor, double maxFreq, int BSplinedegree);

    /**
     * This method gets the volume's Fourier and the Euler's angles as the inputs and interpolates the related projection
     */
    void project(double rot, double tilt, double psi, const MultidimArray<double> *ctf=NULL);

    /** Update volume */
    void updateVolume(MultidimArray<double> &V);
public:
    /// Prepare the Spline coefficients and projection space
    void produceSideInfo();

    /// Prepare projection space
    void produceSideInfoProjection();
};

/*
 * This function gets an object form the FourierProjection class and makes the desired projection in Fourier space
 */
void projectVolume(FourierProjector &projector, Projection &P, int Ydim, int Xdim,
                   double rot, double tilt, double psi, const MultidimArray<double> *ctf=NULL);

//@}

#endif
