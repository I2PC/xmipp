/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
 *             James Krieger                jamesmkrieger@gmail.com
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

#ifndef _PROG_CONTINUOUS_CREATE_RESIDUALS
#define _PROG_CONTINUOUS_CREATE_RESIDUALS

#include "core/xmipp_metadata_program.h"
#include "core/multidim_array.h"
#include "core/xmipp_image.h"
#include "data/fourier_filter.h"
#include "data/fourier_projection.h"

class FourierProjector;

/**@defgroup ProgContinuousCreateResiduals continuous_create_residuals (Apply continuous assigned parameters to projections to make better residuals)
   @ingroup ReconsLibrary */
//@{

constexpr int  CONTCOST_CORR = 0;
constexpr int  CONTCOST_L1 = 1;


/** Predict Continuous Parameters without filter and shifting projection not image. */
/** This should create better residuals for log likelihood calculations. */
class ProgContinuousCreateResiduals: public XmippMetadataProgram
{
public:
    /** Filename of the reference volume */
    FileName fnVol;
    /** Filename of residuals */
    FileName fnResiduals;
    /** Filename of projections */
    FileName fnProjections;
    /** Maximum shift allowed */
    double maxShift;
    /** Maximum scale allowed */
    double maxScale;
    /** Maximum angular change allowed */
    double maxAngularChange;
    /** Maximum defocus change (A) */
    double maxDefocusChange;
    /** Maximum gray scale change */
    double maxA;
    /** Maximum gray shift change */
    double maxB;
    /** Sampling rate */
    double Ts;
    /** Maximum radius */
    int Rmax;
    /** Padding factor */
    int pad;
    // Optimize gray
    bool optimizeGrayValues;
    // Optimize shift
    bool optimizeShift;
    // Optimize scale
    bool optimizeScale;
    // Optimize angles
    bool optimizeAngles;
    // Optimize defocus
    bool optimizeDefocus;
    // Ignore CTF
    bool ignoreCTF;
    // Phase Flipped
    bool phaseFlipped;
    // Penalization for the average
    // double penalization;
    
    // Force defocusU = defocusV
    bool sameDefocus;

public:
    // Rank (used for MPI version)
    int rank;

    // 2D mask in real space
    MultidimArray<int> mask2D;
    // Inverse of the sum of Mask2D
    double iMask2Dsum;
    // Fourier projector
    FourierProjector *projector;
    // Volume size
    size_t Xdim;
    // Input image
	Image<double> I, Ip, E, Pp;
	// Theoretical projection
	Projection P;
    // Transformation matrix
    Matrix2D<double> A;
    // Original angles
    double old_rot, old_tilt, old_psi;
    // Original shift
	double old_shiftX, old_shiftY;
	// Original flip
	bool old_flip;
	// Original gray scale
	double old_grayA, old_grayB;
	// Has CTF
	bool hasCTF;
	// Original defocus
	double old_defocusU, old_defocusV, old_defocusAngle;
	// CTF
	CTFDescription ctf;
	// Covariance matrices
	Matrix2D<double> C0, C;
	// Image stddev
	double Istddev;
	// Continuous cost function
	int contCost;
	// Current defoci
	double currentDefocusU, currentDefocusV, currentAngle;
	// CTF image
	MultidimArray<double> *ctfImage;
    std::unique_ptr<MultidimArray<double>> ctfEnvelope;
	// Fourier Transformer
	FourierTransformer fftTransformer;
	// Fourier transforms
	MultidimArray< std::complex<double> > fftE;
public:
    /// Empty constructor
    ProgContinuousCreateResiduals();

    /// Destructor
    ~ProgContinuousCreateResiduals();

    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /** Start processing */
    void startProcessing();

    /** Produce side info.
        An exception is thrown if any of the files is not found*/
    void preProcess();

    /** Predict angles and shift.
        At the input the pose parameters must have an initial guess of the
        parameters. At the output they have the estimated pose.*/
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    /** Update CTF image */
    void updateCTFImage(double defocusU, double defocusV, double angle);

    /** Post process */
    void postProcess();
};
//@}
#endif
