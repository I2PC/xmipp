/***************************************************************************
 *
 * Authors:     Jeison Méndez García (jmendez@utp.edu.co)
 *
 * Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas -- IIMAS
 * Universidad Nacional Autónoma de México -UNAM
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

#ifndef __ANGULAR_ASSIGNMENT_MAG_GPU_H
#define __ANGULAR_ASSIGNMENT_MAG_GPU_H

#include <core/xmipp_program.h>
#include <core/xmipp_fftw.h>
#include <core/metadata_extension.h>
#include <core/multidim_array.h>
#include <core/symmetries.h>
#include <data/mask.h>
#include <data/filters.h>
#include <reconstruction/project_real_shears.h>

#include <vector>
#include <fstream> 
#include <ctime>

#include "reconstruction_cuda/gpu.h"
#include "reconstruction_cuda/cuda_fft.h"
#include "reconstruction_cuda/cuda_asserts.h"

/**@defgroup AngularAssignmentMagGPU ***
   @ingroup ReconsLibrary */
//@{

/** Angular_Assignment_mag parameters. */
class ProgAngularAssignmentMagGpu: public XmippMetadataProgram
{
	using Complex = std::complex<double>;
    static constexpr size_t type_size = sizeof(double);
    static constexpr size_t complex_size = sizeof(Complex);

private:
    const GPU *m_gpu;

    // device memory
    std::complex<double> *m_d_single_FD;
    std::complex<double> *m_d_batch_FD;
    double *m_d_single_SD;
    double *m_d_batch_SD;

    double *d_data;
    Complex *d_f_data;

    // host memory
    double *m_h_centers;

    // FT plans
    cufftHandle *m_singleToFD;
    cufftHandle *m_batchToFD;
    cufftHandle *m_batchToSD;

	/*
	* Plans for CuFFT
	*/
	cufftHandle* planForward;
	cufftHandle* planBackward;
	GPU gpu;

	/*
	* Sizes of allocated arrays on gpu
	*/
	size_t xdim;
	size_t ydim;
	size_t zdim;
	size_t image_size;
	size_t fourier_size; // size of complex array used in FFT
	size_t memsize;
	size_t fourier_memsize;

	void setSizes(MultidimArray<double> &data);
	void createFFTPlans();

public:
    /** Filenames */
    FileName fnIn;
    FileName fnOut;
    FileName fnDir;
    FileName fnSym;
    FileName fnRef;

    size_t rank, Nprocessors;

    // Metadata with input images and input volumes
    MetaData mdIn;
    MetaData mdRef;
    MetaData mdOut;

    // vector of reference images
    std::vector< MultidimArray<double> > vecMDaRef;

    // vector of Fourier of reference images
    std::vector< MultidimArray< std::complex<double> > > vecMDaRefF;

    // vector of Fourier of polar representation of reference image in real space
    std::vector< MultidimArray< std::complex<double> > > vecMDaRef_polarF;

    // vector of Fourier of polar representation of magnitude spectrum of reference images
    std::vector< MultidimArray< std::complex<double> > > vecMDaRefFMs_polarF;

    // Size of the images
    size_t Xdim;
    size_t Ydim;

    // Transformers
    FourierTransformer transformerImage;
    FourierTransformer transformerPolarImage;
    FourierTransformer transformerPolarRealSpace;

    MultidimArray<double> C; // circular mask

    // eigenDecomposition
	Matrix1D<double> eigenvalues;
	Matrix2D<double> eigenvectors;

	int      peaksFound = 0; // peaksFound in ccVectorRot

    // matrix for neighbors and angular distance
    std::vector< std::vector<int> > neighborsMatrix; // this should be global
    std::vector< std::vector<double> > neighboursDistance; // not sure if necessary this global
    std::vector< std::vector<double> > neighborsWeights; // this variable should be global

    //reference image - projection coordinates
    std::vector<double>               referenceRot;
    std::vector<double>               referenceTilt;

    /**reference volume to be projected */
    FileName inputReference_volume;
    Image<double> refVol;
    // Size of the reference volume
    int refXdim;
    int refYdim;
    int refZdim;

    // some constants
    int sizeMdRef;
    int sizeMdIn;
    size_t n_bands;
    size_t startBand;
    size_t finalBand;
    size_t n_rad;
    size_t n_ang;
    size_t n_ang2;
    double maxShift;
    double sampling;
    double angStep;

    // Symmetry list
    SymList SL;
    // Left matrices for the symmetry transformations
    std::vector< Matrix2D<double> > L;
    // Right matrices for the symmetry transformations
    std::vector< Matrix2D<double> > R;

    /** Use it for validation */
    bool useForValidation;

    int Nsim;

    ProgAngularAssignmentMagGpu();

    ~ProgAngularAssignmentMagGpu();

    /// Read arguments from command line
    void defineParams();
    void readParams();

    void show();

    void startProcessing();

    void preProcess();

    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    void postProcess();

    void applyCircularMask(const MultidimArray<double> &in, MultidimArray<double> &out);
    void applyFourierImage(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData);
    void applyFourierImageGpu(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData);
    void applyFourierImage(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    void applyFourierImage2(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData);
    void applyFourierImage2(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    void applyFourierImage3(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    void applyRotation(const MultidimArray<double> &MDaRef, double &rot, MultidimArray<double> &MDaRefRot);
    void applyRotation(MultidimArray<double> &MDaRef, double &rot, MultidimArray<double> &MDaRefRot);
    void applyShift(MultidimArray<double> &input, double &tx, double &ty, MultidimArray<double> &output);
    void applyShift(const MultidimArray<double> &input, double &tx, double &ty, MultidimArray<double> &output);
    void applyRotationAndShift(const MultidimArray<double> &MDaRef, double &rot, double &tx, double &ty, MultidimArray<double> &MDaRefRot);
    void applyShiftAndRotation(const MultidimArray<double> &MDaRef, double &rot, double &tx, double &ty, MultidimArray<double> &MDaRefRot);

    void bestCand(const MultidimArray<double> &MDaIn, const MultidimArray<std::complex<double> > &MDaInF, const MultidimArray<double> &MDaRef, std::vector<double> &cand, double &bestCandRot, double &shift_x, double &shift_y, double &bestCoeff);

    void completeFourierShift(MultidimArray<double> &in, MultidimArray<double> &out);
    void ccMatrix(const MultidimArray<std::complex<double> > &F1, const MultidimArray<std::complex<double> > &F2, MultidimArray<double> &result);
    void computingNeighborGraph();
    void computeLaplacianMatrix(Matrix2D<double> &L, const std::vector< std::vector<int> > &allNeighborsjp, const std::vector< std::vector<double> > &allWeightsjp);
    void computeCircular();
    void circularWindow(MultidimArray<double> &in);

    void getComplexMagnitude(MultidimArray<std::complex<double> > &FourierData, MultidimArray<double> &FourierMag);
    void getShift(MultidimArray<double> &ccVector, double &shift, const size_t &size);
    void graphFourierFilter(Matrix1D<double> &ccVecIn, Matrix1D<double> &ccVecOut);

    MultidimArray<double> imToPolar(MultidimArray<double> &cartIm, size_t &start, size_t &end);
    double interpolate(MultidimArray<double> &cartIm, double &x_coord, double &y_coord);

    void meanByColumn(MultidimArray<double> &in, MultidimArray<double> &out);
    void maxByColumn(MultidimArray<double> &in, MultidimArray<double> &out);
    void meanByRow(MultidimArray<double> &in, MultidimArray<double> &out);
    void maxByRow(MultidimArray<double> &in, MultidimArray<double> &out);

    void psiCandidates(MultidimArray<double> &in, std::vector<double> &cand, const size_t &size);


    /// Synchronize with other processors
    virtual void synchronize() {}

private:


};
//@}


#endif
