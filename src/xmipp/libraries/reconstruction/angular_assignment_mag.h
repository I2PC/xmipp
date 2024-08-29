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

#ifndef __ANGULAR_ASSIGNMENT_MAG_H
#define __ANGULAR_ASSIGNMENT_MAG_H

#include "core/xmipp_metadata_program.h"
#include "core/matrix1d.h"
#include "core/metadata_vec.h"
#include <core/xmipp_fftw.h>
#include <core/multidim_array.h>
#include <core/symmetries.h>
#include <core/xmipp_image.h>
#include <data/mask.h>
#include <data/filters.h>
#include "ctpl_stl.h"


#include <vector>

/**@defgroup AngularAssignmentMag ***
   @ingroup ReconsLibrary */
//@{

/** Angular_Assignment_mag parameters. */
class ProgAngularAssignmentMag: public XmippMetadataProgram
{
public:
	void defineParams() override;
	void readParams() override;
	void show() const override;
	void preProcess() override;
	void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) override;
	void postProcess() override;
    ProgAngularAssignmentMag();
    ~ProgAngularAssignmentMag();
    void applyRotation(const MultidimArray<double> &, double &, MultidimArray<double> &) const;
    void applyShiftAndRotation(const MultidimArray<double> &MDaRef, double &rot, double &tx, double &ty, MultidimArray<double> &MDaRefRot) const;

    /// Synchronize with other processors
    virtual void synchronize() {}

	size_t rank;
	size_t Nprocessors;

private:
	double angDistance(int &, int &, Matrix1D<double> &, Matrix1D<double> &);
    void bestCand(const MultidimArray<double> &MDaIn, const MultidimArray<std::complex<double> > &MDaInF, const MultidimArray<double> &MDaRef, std::vector<double> &cand, double &bestCandRot, double &shift_x, double &shift_y, double &bestCoeff);
    void completeFourierShift(const MultidimArray<double> &in, MultidimArray<double> &out) const;
    void ccMatrix(const MultidimArray<std::complex<double> > &F1, const MultidimArray<std::complex<double> > &F2, MultidimArray<double> &result, FourierTransformer &transformer);
    void checkStorageSpace();
    void computingNeighborGraph();
    void computeEigenvectors();
    void computeLaplacianMatrix (Matrix2D<double> &L, const std::vector< std::vector<int> > &allNeighborsjp, const std::vector< std::vector<double> > &allWeightsjp) const;
    void computeCircular();
    void circularWindow(MultidimArray<double> &in) const;
    void getShift(MultidimArray<double> &ccVector, double &shift, const size_t &size) const;
    void graphFourierFilter(const Matrix1D<double> &ccVecIn, Matrix1D<double> &ccVecOut) const;
    MultidimArray<double> imToPolar(MultidimArray<double> &cartIm, size_t &start, size_t &end);
    double interpolate(MultidimArray<double> &cartIm, double &x_coord, double &y_coord) const;
    void maxByColumn(const MultidimArray<double> &in, MultidimArray<double> &out) const;
    void maxByRow(const MultidimArray<double> &in, MultidimArray<double> &out) const;

    void processGallery(FileName &);
    void psiCandidates(const MultidimArray<double> &in, std::vector<double> &cand, const size_t &size);
    void validateAssignment(int &, int &, double &, double &, const MDRow &, MDRow &, MultidimArray<double> &, Matrix1D<double> &);

    ctpl::thread_pool threadPool;

    /** Filenames */
    FileName fnIn;
    FileName fnOut;
    FileName fnDir;
    FileName fnSym;
    FileName fnRef;

    // Metadata with input images and input volumes
    MetaDataVec mdIn;
    MetaDataVec mdRef;
    MetaDataVec mdOut;

    // Transformers
    std::vector<FourierTransformer> transformersForImages;
    FourierTransformer transformerPolarImage;
    FourierTransformer transformerImage;
    FourierTransformer transformerPolarRealSpace;
    std::vector<FourierTransformer> ccMatrixBestCandidTransformers;
    FourierTransformer ccMatrixProcessImageTransformer;

    std::vector<MultidimArray<double>> ccMatrixShifts;

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

    // threads to use
    int threads;

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

};
//@}


#endif
