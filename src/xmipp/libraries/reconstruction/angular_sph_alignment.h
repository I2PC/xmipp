/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es)
 *             David Herreros Calero (dherreros@cnb.csic.es)
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

#ifndef _PROG_ANGULAR_SPH_ALIGNMENT
#define _PROG_ANGULAR_SPH_ALIGNMENT

#include "core/xmipp_metadata_program.h"
#include "core/matrix1d.h"
#include "core/xmipp_image.h"
#include "data/fourier_filter.h"
#include "data/fourier_projection.h"

/**@defgroup AngularPredictContinuous2 angular_continuous_assign2 (Continuous angular assignment)
   @ingroup ReconsLibrary */
//@{

/** Predict Continuous Parameters. */
class ProgAngularSphAlignment: public XmippMetadataProgram
{
public:
    /** Filename of the reference volume */
    FileName fnVolR;
    /** Filename of the reference volume mask */
    FileName fnMaskR;
    /// Output directory
    FileName fnOutDir;
    // Metadata with already processed images
    FileName fnDone;
    /** Degrees of Zernike polynomials and spherical harmonics */
    int L1;
    int L2;
    /** Zernike and SPH coefficients vectors */
    Matrix1D<int> vL1;
    Matrix1D<int>vN;
    Matrix1D<int> vL2;
    Matrix1D<int> vM;
    /** Maximum shift allowed */
    double maxShift;
    /** Maximum angular change allowed */
    double maxAngularChange;
    /** Maximum frequency (A) */
    double maxResol;
    /** Sampling rate */
    double Ts;
    /** Maximum radius */
    int Rmax;
    // Optimize alignment
    bool optimizeAlignment;
    //Optimize deformation
    bool optimizeDeformation;
    //Optimize defocus
    bool optimizeDefocus;
    // Ignore CTF
    bool ignoreCTF;
    //Radius optimization
    bool optimizeRadius;
    // Phase Flipped
    bool phaseFlipped;
    // Regularization weight
    double lambda;
    // Maximum radius for the deformation
    int RmaxDef;

    Matrix1D<double> p;
    int flagEnabled;

public:
    /** Resume computations */
    bool resume;
    // 2D and 3D masks in real space
    MultidimArray<int> mask2D;
    MultidimArray<int> V_mask;
    // Volume size
    size_t Xdim;
    // Input images
	Image<double> V;
    Image<double> Vdeformed;
    Image<double> I;
    Image<double> Ip;
    Image<double> Ifiltered;
    Image<double> Ifilteredp;
	// Theoretical projection
	Image<double> P;
	// Filter
    FourierFilter filter;
    // Transformation matrix
    Matrix2D<double> A;
    // Original angles
    double old_rot;
    double old_tilt;
    double old_psi;
    // Original shift
	double old_shiftX;
    double old_shiftY;
	// Original flip
	bool old_flip;
    // CTF Check
    bool hasCTF;
    // Original defocus
	double old_defocusU;
    double old_defocusV;
    double old_defocusAngle;
    // Current defoci
	double currentDefocusU;
    double currentDefocusV;
    double currentAngle;
	// CTF
	CTFDescription ctf;
    // CTF filter
    FourierFilter FilterCTF;
	// Vector Size
	int vecSize;
	// Vector containing the degree of the spherical harmonics
	Matrix1D<double> clnm;
    //Copy of Optimizer steps
    Matrix1D<double> steps_cp;
	//Total Deformation, sumV, sumVd
	double totalDeformation;
    double sumV;
    double sumVd;
	// Show optimization
	bool showOptimization;
	// Correlation
	double correlation;

public:
    /// Empty constructor
	ProgAngularSphAlignment();

    /// Destructor
    ~ProgAngularSphAlignment();

    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /** Produce side info.
        An exception is thrown if any of the files is not found*/
    void preProcess();

    /** Create the processing working files.
     * The working files are:
     * nmaTodo.xmd for images to process (nmaTodo = mdIn - nmaDone)
     * nmaDone.xmd image already processed (could exists from a previous run)
     */
    virtual void createWorkFiles();

    /** Predict angles and shift.
        At the input the pose parameters must have an initial guess of the
        parameters. At the output they have the estimated pose.*/
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    /// Length of coefficients vector
    void numCoefficients(int l1, int l2, int &nc) const;

    /// Determine the positions to be minimize of a vector containing spherical harmonic coefficients
    void minimizepos(int l2, Matrix1D<double> &steps) const;

    /// Zernike and SPH coefficients allocation
    void fillVectorTerms(int l1, int l2);

    ///Deform a volumen using Zernike-Spherical harmonic basis
    void deformVol(MultidimArray<double> &mVD, const MultidimArray<double> &mV, double &def,
                   double rot, double tilt, double psi);

    void applyCTFImage(double const &deltaDefocusU, double const &deltaDefocusV, 
					   double const &deltaDefocusAngle);

    void updateCTFImage(double defocusU, double defocusV, double angle);

    double tranformImageSph(double *pclnm, double rot, double tilt, double psi,
    		                double deltaDefocusU, double deltaDefocusV, double deltaDefocusAngle);

    /** Write the final parameters. */
    virtual void finishProcessing();

    /** Write the parameters found for one image */
    virtual void writeImageParameters(const FileName &fnImg);

};
//@}
#endif
