/***************************************************************************
 *
 * Authors:    Slavica Jonic            slavica.jonic@epfl.ch (2004)
 *             Carlos Oscar             coss@cnb.csic.es
 *
 * Biomedical Imaging Group, EPFL.
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

#ifndef _PROG_ANGULAR_PREDICT_CONTINOUOS
#define _PROG_ANGULAR_PREDICT_CONTINUOUS

#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <data/mask.h>
#include <core/metadata.h>

/**@defgroup AngularPredictContinuous angular_continuous_assign (Continuous angular assignment)
   @ingroup ReconsLibrary */
//@{
/** Predict Continuous Parameters. */
class ProgAngularContinuousAssign: public XmippMetadataProgram
{
public:
    /** Filename of the reference volume */
    FileName fn_ref;
    /** Gaussian weight sigma in Fourier space. */
    double   gaussian_DFT_sigma;
    /** Gaussian weight sigma in real space. */
    double   gaussian_Real_sigma;
    /** Weight for zero frequency. */
    double   weight_zero_freq;
    /** Maximum number of iterations */
    int      max_no_iter;
    /** Do not produce any output */
    bool quiet;
    /** Maximum shift allowed */
    double max_shift;
    /** Maximum angular change allowed */
    double max_angular_change;
public:
    // Real part of the Fourier transform
    MultidimArray<double> reDFTVolume;
    // Imaginary part of the Fourier transform
    MultidimArray<double> imDFTVolume;
    // Weighting mask in Fourier space
    Mask      mask_Fourier;
    // Weighting mask in Real space
    Mask      mask_Real;
public:
    /// Empty constructor
    ProgAngularContinuousAssign();

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
};

/** Assign pose parameters for 1 image.
    The weight must be an image of the size of the input image
    with the appropriate weighting in frequency (normally a gaussian).
    The pose parameters at the input must have the initial guess
    of the pose. At the output they contain the parameters estimated
    by CST Spline Assignment. The maximum number of iterations
    controls the optimization process. */
double CSTSplineAssignment(
    MultidimArray<double> &ReDFTVolume,
    MultidimArray<double> &ImDFTVolume,
    MultidimArray<double> &image,
    MultidimArray<double> &weight,
    Matrix1D<double> &pose_parameters,
    int               max_no_iter = 60
);

/* Data structure for Continuous Angular assignment.
    All output vectors, matrices, volumes, ... must be previously
    allocated by the caller. */
struct cstregistrationStruct
{
    /* Real part of the DFT of the volume.
        The size must be given in the fields:
        nx_ReDftVolume, ny_ReDftVolume and nz_ReDftVolume.of this structure.
    */
    double   *ReDftVolume;
    // X dimension of the input real part of the DFT volume
    long    nx_ReDftVolume;
    // Y dimension of the input real part of the DFT volume
    long    ny_ReDftVolume;
    // Z dimension of the input real part of the DFT volume
    long    nz_ReDftVolume;

    /* Imaginary part of the DFT of the volume.
        The size must be given in the fields:
        nx_ImDftVolume, ny_ImDftVolume and nz_ImDftVolume.of this structure.
    */
    double   *ImDftVolume;
    // X dimension of the input imaginary part of the DFT volume
    long    nx_ImDftVolume;
    // Y dimension of the input imaginary part of the DFT volume
    long    ny_ImDftVolume;
    // Z dimension of the input imaginary part of the DFT volume
    long    nz_ImDftVolume;

    /* Real part of the DFT of the image.
        The size must be given in the fields:
        nx_ReDftImage and ny_ReDftImage of this structure.
    */
    double   *ReDftImage;
    // X dimension of the input real part of the DFT image
    long    nx_ReDftImage;
    // Y dimension of the input real part of the DFT image
    long    ny_ReDftImage;

    /* Imaginary part of the DFT of the image.
        The size must be given in the fields:
        nx_ReDftImage and ny_ImDftImage.of this structure.
    */
    double   *ImDftImage;
    // X dimension of the input imaginary part of the DFT image
    long    nx_ImDftImage;
    // Y dimension of the input imaginary part of the DFT image
    long    ny_ImDftImage;

    /* 2D weighting function in the frequency domain.
        Normally a gaussian. Its size is given by nx_Weight and ny_Weight
        in this same structure. */
    double   *Weight;
    // X dimension of the weight in frequency
    long    nx_Weight;
    // Y dimension of the weight in frequency
    long    ny_Weight;

    /* Vector with the voxel size in physical units (A/pix) in each direction.
        Its length is given by nx_VoxelSize. Normally it is 3. */
    double   *VoxelSize;
    // Length of the VoxelSize vector
    long    nx_VoxelSize;

    /* Vector with the pixel size in physical units (A/pix) in each direction.
        Its length is given by nx_PixelSize. Normally it is 2 */
    double   *PixelSize;
    // Length of the PixelSize vector
    long    nx_PixelSize;

    /* Initial pose parameters.
        Vector with the initial pose (rot,tilt,psi,X shift, Y shift).
        Its length (5) must be set in nx_Parameters. */
    double   *Parameters;
    // Length of the initial pose parameters vector
    long    nx_Parameters;

    /* Final pose parameters.
        This is a matrix whose columns are the different pose parameters
        after each iteration of the optimization. Column 0 is the pose at time 0,
        column 1 is the pose at time 1, ... The order in the column is
        (rot,tilt,psi,shift x,shift y). The size of the matrix is given by
        nx_OutputParameters and ny_OutputParameters. */
    double   *OutputParameters;
    // X dimension of the outputParameters matrix
    long    nx_OutputParameters;
    // Y dimension of the outputParameters matrix
    long    ny_OutputParameters;

    /* Vector with the cost function at each iteration.
        Its size is nx_Cost */
    double   *Cost;
    // Size of the vector Cost
    long    nx_Cost;

    /* Vector with the time consumed at each iteration.
        Its size is nx_TimePerIter */
    double   *TimePerIter;
    // Size of TimePerIter
    long    nx_TimePerIter;

    /* Number of iterations performed. */
    long    *NumberIterPerformed;
    /* Number of successful iterations */
    long    *NumberSuccPerformed;
    /* Number of failed iterations. */
    long    *NumberFailPerformed;

    /* Vector with the number of the failed iterations.
        Its size should be the maximum number of iterations. Normally
        all values are 0s except at those iterations where it failed.
        */
    double   *Failures;
    // Size of the Failures vector
    long    nx_Failures;

    /* Interpolated section of the Fourier volume.
        It is a two slice volume. The first slice is the real part and
        the second slice the imaginary part. No */
    double   *dftProj;

    // X dimension of the interpolated section
    long    nx_dftProj;
    // Y dimension of the interpolated section
    long    ny_dftProj;
    // Z dimension of the interpolated section (should be 2)
    long    nz_dftProj;

    /* Scaling factor for changing the parameter lambda in the LM optimizer.
        If a success/failure occurs, lambda is changed by this factor. */
    double  ScaleLambda;
    /* Initial lambda. */
    double  LambdaInitial;
    /* Maximum number of iterations allowed. */
    long    MaxNoIter;
    /* Maximum number of failures allowed. */
    long    MaxNoFailure;
    /* Maximum number of successes allowed. */
    long    SatisfNoSuccess;
    /* Project or Refine.
        If this variable is 0, the routine refines the angles.
        If it is 1, it simply interpolates the central section slice,
        with the initial pose parameters. */
    long    MakeDesiredProj;
    /* Tolerance in the angle to accept a certain solution.
        If the angular parameters change less than this value,
        the optimizer will stop. */
    double  ToleranceAngle;
    /* Tolerance in the shift to accept a certain solution.
        If the shift parameters change less than this value,
        the optimizer will stop. */
    double  ToleranceShift;
};
//@}
#endif
