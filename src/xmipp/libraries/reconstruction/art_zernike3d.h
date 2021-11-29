/***************************************************************************
 *
 * Authors:    David Herreros Calero dherreros@cnb.csic.es
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


#ifndef _PROG_ART_ZERNIKE3D
#define _PROG_ART_ZERNIKE3D

#include "core/xmipp_metadata_program.h"
#include "core/matrix1d.h"
#include "core/xmipp_image.h"
#include "data/fourier_filter.h"
#include "data/fourier_projection.h"
#include <core/xmipp_error.h>


/** Predict Continuous Parameters. */
class ProgArtZernike3D: public XmippMetadataProgram
{
public:
     /** Filename of the reference volume */
    FileName fnVolR;
    /** Filename of the refined volume */
    FileName fnVolO;
    /// Output directory
    FileName fnOutDir;
    // Metadata with already processed images
    // FileName fnDone;
    /** Degrees of Zernike polynomials and spherical harmonics */
    int L1, L2;
    /** Zernike and SPH coefficients vectors */
    Matrix1D<int> vL1, vN, vL2, vM;
    /** Sampling rate */
    double Ts;
    /** Maximum radius */
    int RmaxDef;
    // Phase Flipped
    bool phaseFlipped;
    // Ignore CTF
    bool ignoreCTF;
    // Regularization ART
    double lambda;
    // Save each # iter
    int save_iter;
    // Correct CTF
    bool useCTF;
    // Apply Zernike
    bool useZernike;

    int flagEnabled;

public:
    /** Resume computations */
    bool resume;
    // Number of ART iterations
    int niter;
    // Sort last N projections
    int sort_last_N;
    // 2D and 3D masks in real space
    MultidimArray<int> mask2D;
    // Volume size
    size_t Xdim;
    // Input image
	Image<double> V, Vrefined, I, Ip, Ifiltered, Ifilteredp;
    // Spherical mask
    MultidimArray<int> Vmask;
	// Theoretical projection
	Image<double> P;
    // Weight Image
    Image<double> W;
    // Difference Image
    Image<double> Idiff;
    // Transformation matrix
    Matrix2D<double> A;
    // Original angles
    double rot, tilt, psi;
    // Original shift
	double shiftX, shiftY;
	// Original flip
	bool flip;
    // CTF Check
    bool hasCTF;
    // Original defocus
	double defocusU, defocusV, defocusAngle;
	// CTF
	CTFDescription ctf;
    // CTF filter
    FourierFilter FilterCTF;
	// Vector Size
	int vecSize;
	// Vector containing the degree of the spherical harmonics
	Matrix1D<double> clnm;
	// Show optimization
	bool showOptimization;
    double w_i;
    MultidimArray<size_t> ordered_list;
    int current_save_iter;
    int num_images;
    int current_iter;
public:
    /// Empty constructor
	ProgArtZernike3D();

    /// Destructor
    ~ProgArtZernike3D();

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
    // virtual void createWorkFiles();

    /** Predict angles and shift.
        At the input the pose parameters must have an initial guess of the
        parameters. At the output they have the estimated pose.*/
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    /// Length of coefficients vector
    void numCoefficients(int l1, int l2, int &vecSize);

    /// Zernike and SPH coefficients allocation
    void fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN, 
                         Matrix1D<int> &vL2, Matrix1D<int> &vM);

    ///Deform a volumen using Zernike-Spherical harmonic basis
    void deformVol(MultidimArray<double> &mP, MultidimArray<double> &mW,
                   const MultidimArray<double> &mV,
                   double rot, double tilt, double psi);

    void updateCTFImage(double defocusU, double defocusV, double angle);

    void artModel(int direction);

    template<bool USESZERNIKE, int DIRECTION>
    void zernikeModel();

    void weightsInterpolation3D(double x, double y, double z, Matrix1D<double> &w);

    void removeOverdeformation();

    // virtual void checkPoint();
    
    virtual void finishProcessing();

    virtual void run();

    void sortOrthogonal();

};
//@}
#endif