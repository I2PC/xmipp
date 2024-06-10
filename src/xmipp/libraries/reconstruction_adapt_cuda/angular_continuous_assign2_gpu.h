/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
 *             Martin Pernica, Masaryk University
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

#ifndef _PROG_ANGULAR_PREDICT_CONTINUOUS2_GPU
#define _PROG_ANGULAR_PREDICT_CONTINUOUS2_GPU

#include "core/xmipp_metadata_program.h"
#include "core/multidim_array.h"
#include "core/xmipp_image.h"
#include "data/fourier_filter.h"
#include "reconstruction_cuda/cuda_fourier_projection.h"

#include <thread>
//#include "data/fourier_projection.h"

class CudaFourierProjector;

/**@defgroup AngularPredictContinuous2 angular_continuous_assign2_gpu (Continuous angular assignment)
   @ingroup ReconsLibrary */
//@{

constexpr int  CONTCOST_CORR = 0;
constexpr int  CONTCOST_L1 = 1;


/** Predict Continuous Parameters. */
class ProgCudaAngularContinuousAssign2: public XmippMetadataProgram
{
public:
    /** Filename of the reference volume */
    std::mutex thread_mutex;

    int nThreads=3;

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
    /** Maximum frequency (A) */
    double maxResol;
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
    // Apply transformation to this image
    String originalImageLabel;
    // Phase Flipped
    bool phaseFlipped;
    // Penalization for the average
    // double penalization;

    // Force defocusU = defocusV
    bool sameDefocus;

    double skipThreshold=0.000001;

public:
    // Rank (used for MPI version)
    int rank;
    double *last_transform[5];
    FileName *fnImg;
    FileName *fnImgOut;
    Image<double> *I;
    Image<double> *Ip;
    Image<double> *E;
    Image<double> *Ifiltered;
    Image<double> *Ifilteredp;
    MultidimArray<std::complex<double>> *FourierIfiltered;
    MultidimArray<std::complex<double>> *FourierIfilteredp;
    MultidimArray<std::complex<double>> *FourierProjection;
    Projection *P;
    FourierFilter *filter;
    Matrix2D<double> *A;
    double *old_rot;
    double *old_tilt;
    double *old_psi;
    double *old_shiftX;
    double *old_shiftY;
    bool *old_flip;
    double *old_grayA;
    double *old_grayB;
    double *old_defocusU;
    double *old_defocusV;
    double *old_defocusAngle;
    CTFDescription *ctf;
    double *Istddev;
    double *currentDefocusU;
    double *currentDefocusV;
    double *currentAngle;
    MultidimArray<double> **ctfImage;
    std::unique_ptr<MultidimArray<double>> *ctfEnvelope;
    FourierTransformer *fftTransformer;
    MultidimArray<std::complex<double>> *fftE;
    //device
    int device;
    int contCost;

    Matrix2D<double> C0, C;
    bool hasCTF;
    // 2D mask in real space
    MultidimArray<int> mask2D;
    // Inverse of the sum of Mask2D
    double iMask2Dsum;
    // Fourier projector
    //CudaFourierProjector *projector;
    CudaFourierProjector *projector;
    // Volume size
    size_t Xdim;
    FileName fullBaseName;

public:
    /// Empty constructor
    ProgCudaAngularContinuousAssign2();

    /// Destructor
    ~ProgCudaAngularContinuousAssign2();

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
    void processImage_t(const FileName &locfnImg, const FileName &locfnImgOut, const MDRow &rowIn, MDRow &rowOut, int thread);

    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    /** Update CTF image */
    void updateCTFImage(double defocusU, double defocusV, double angle, int thread, bool gen_env);

    /** Post process */
    void postProcess();

    void run();

    bool getImageToProcess(size_t &objId, size_t &objIndex);

    void runThread(int threadIdx);
};
//@}
#endif
