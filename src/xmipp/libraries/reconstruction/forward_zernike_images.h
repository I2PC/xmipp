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

#ifndef _PROG_FORWARD_ZERNIKE_IMAGES
#define _PROG_FORWARD_ZERNIKE_IMAGES

#include "core/xmipp_metadata_program.h"
#include "core/rerunable_program.h"
#include "core/matrix1d.h"
#include <data/blobs.h>
#include "core/xmipp_image.h"
#include "data/fourier_filter.h"
#include "data/fourier_projection.h"

/**@defgroup AngularPredictContinuous2 angular_continuous_assign2 (Continuous angular assignment)
   @ingroup ReconsLibrary */
//@{

/** Predict Continuous Parameters. */
class ProgForwardZernikeImages: public XmippMetadataProgram, public Rerunable
{
public:
    /** Filename of the reference volume */
    FileName fnVolR;
    /** Filename of the reference volume mask */
    FileName fnMaskR;
    /// Output directory
    FileName fnOutDir;
    /** Degrees of Zernike polynomials and spherical harmonics */
    int L1, L2;
    /** Zernike and SPH coefficients vectors */
    Matrix1D<int> vL1, vN, vL2, vM;
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
    // Number of images (execution mode) (single image == 1, pair == 2, or triplet == 3)
    int num_images;
    // Number alignment parameters to minimize
    int algn_params;
    // Number CTF parameters to minimize
    int ctf_params;
    Matrix1D<double> p;
    int flagEnabled;
    int image_mode;
    bool useCTF;

public:
    /** Resume computations */
    bool resume;
    // 2D and 3D masks in real space
    MultidimArray<int> mask2D, V_mask;
    // Volume size
    size_t Xdim;
    // Images Filename
    std::vector<FileName> fnImage;
    // Input image
	Image<double> V, Vdeformed;
    std::vector<Image<double>> I;
    std::vector<Image<double>> Ip;
    std::vector<Image<double>> Ifiltered;
    std::vector<Image<double>> Ifilteredp;
	// Theoretical projections
    std::vector<Image<double>> P;
	// Filter
    FourierFilter filter;
    // Transformation matrix
    Matrix2D<double> A1, A2, A3;
    // Original angles
    std::vector<double> old_rot, old_tilt, old_psi, deltaRot, deltaTilt, deltaPsi;
    // Original shift
	std::vector<double> old_shiftX, old_shiftY, deltaX, deltaY;
	// Original flip
	bool old_flip;
    // CTF Check
    bool hasCTF;
    // Original defocus
	std::vector<double> old_defocusU, old_defocusV, old_defocusAngle, deltaDefocusU, deltaDefocusV, deltaDefocusAngle;
    // Current defoci
	std::vector<double> currentDefocusU, currentDefocusV, currentAngle;
	// CTF
	CTFDescription ctf;
    // CTF filter
    FourierFilter FilterCTF1;
    FourierFilter FilterCTF2;
    FourierFilter FilterCTF3;
	// Vector Size
	int vecSize;
	// Vector containing the degree of the spherical harmonics
	Matrix1D<double> clnm;
    //Copy of Optimizer steps
    Matrix1D<double> steps_cp;
	//Total Deformation, sumV, sumVd
	double totalDeformation, sumV, sumVd;
	// Show optimization
	bool showOptimization;
	// Correlation
	double correlation;
    // Loop step
    int loop_step;
    // Blob
    struct blobtype blob;
    double blob_r;

public:
    /// Empty constructor
	ProgForwardZernikeImages();

    /// Destructor
    ~ProgForwardZernikeImages();

    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /** Produce side info.
        An exception is thrown if any of the files is not found*/
    void preProcess();

    /** Predict angles and shift.
        At the input the pose parameters must have an initial guess of the
        parameters. At the output they have the estimated pose.*/
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    /// Length of coefficients vector
    void numCoefficients(int l1, int l2, int &vecSize);

    /// Determine the positions to be minimize of a vector containing spherical harmonic coefficients
    // void minimizepos(Matrix1D<double> &vectpos);
    void minimizepos(int L1, int l2, Matrix1D<double> &steps);

    /// Zernike and SPH coefficients allocation
    void fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN, 
                         Matrix1D<int> &vL2, Matrix1D<int> &vM);

    ///Deform a volumen using Zernike-Spherical harmonic basis
    void deformVol(MultidimArray<double> &mVD, const MultidimArray<double> &mV, double &def,
                   double rot, double tilt, double psi);

    void updateCTFImage(double defocusU, double defocusV, double angle);

    double tranformImageSph(double *pclnm);

    void rotateCoefficients();

    //AJ new
    /** Write the final parameters. */
    virtual void finishProcessing();

    /** Write the parameters found for one image */
    virtual void writeImageParameters(MDRow &row);
    //END AJ

    virtual void checkPoint();

    // void removePixels();

    Matrix1D<double> weightsInterpolation3D(double x, double y, double z);
    
    void splattingAtPos(std::array<double, 3> r, double weight, MultidimArray<double> &mP, const MultidimArray<double> &mV);

protected:
    void createWorkFiles() { return Rerunable::createWorkFiles(resume, getInputMd()); }

private:
    std::vector<MDLabel> getLabelsForEmpty() override
    {
        std::vector<MDLabel> labels = getInputMd()->getActiveLabels();
        labels.push_back(MDL_SPH_DEFORMATION);
        labels.push_back(MDL_SPH_COEFFICIENTS);
        labels.push_back(MDL_COST);
        return labels;
    }
};
//@}
#endif
