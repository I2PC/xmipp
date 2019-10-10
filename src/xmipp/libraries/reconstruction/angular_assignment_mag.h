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

#include <core/xmipp_program.h>
#include <core/xmipp_fftw.h>
#include <core/metadata_extension.h>
#include <core/multidim_array.h>
#include <data/mask.h>
#include <data/filters.h>

#include <vector>
#include <fstream> 
#include <ctime>


/**@defgroup AngularAssignmentMag ***
   @ingroup ReconsLibrary */
//@{

/** Angular_Assignment_mag parameters. */
class ProgAngularAssignmentMag: public XmippMetadataProgram
{
public:
    /** Filenames */
    FileName fnIn;
    FileName fnOut;
    FileName fnDir;
    FileName fnSym;
    FileName fnRef;

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

    // Transformer
    FourierTransformer transformerImage;
    FourierTransformer transformerPolarImage;
    FourierTransformer transformerPolarRealSpace;

    MultidimArray<double> W; // Hann window

    MultidimArray<double> C; // circular mask

    // eigenDecomposition
	Matrix1D<double> eigenvalues;
	Matrix2D<double> eigenvectors;

	int      peaksFound = 0; // peaksFound in ccVectorRot

    // matrix for neighbors and angular distance
    std::vector< std::vector<int> > neighborsMatrix; // this should be global
    std::vector< std::vector<double> > neighboursDistance; // not sure if necessary this global
    std::vector< std::vector<double> > neighborsWeights; // this variable should be global
    int N_neighbors;

    int testCounter = 0;
    int testCounter2=0;

    //reference values
    std::vector<double>               referenceRot;
    std::vector<double>               referenceTilt;

    // some constants
    int sizeMdRef;
    int sizeMdIn;
    // some constants
    size_t n_bands;
    size_t startBand;
    size_t finalBand;
    size_t n_rad;
    size_t n_ang;
    size_t n_ang2;
    double maxShift;
    double sampling;
    double angStep;

    int Nsim;

    ProgAngularAssignmentMag();

    ~ProgAngularAssignmentMag();

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
    void arithmetic_mean_and_stddev(const MultidimArray<double> &data, double &avg, double &stddev);
    void arithmetic_mean_and_stddev(MultidimArray<double> &data, double &avg, double &stddev);

    void bestCand(const MultidimArray<double> &MDaIn, const MultidimArray<std::complex<double> > &MDaInF, const MultidimArray<double> &MDaRef, std::vector<double> &cand, double &bestCandRot, double &shift_x, double &shift_y, double &bestCoeff);
    void bestCand2(const MultidimArray<double> &MDaIn, const MultidimArray<std::complex<double> > &MDaInF, const MultidimArray<double> &MDaRef, std::vector<double> &cand, double &bestCandRot, double &shift_x, double &shift_y, double &bestCoeff);

    void completeFourierShift(MultidimArray<double> &in, MultidimArray<double> &out);
    void ccMatrix(const MultidimArray<std::complex<double> > &F1, const MultidimArray<std::complex<double> > &F2, MultidimArray<double> &result);
    void ccMatrixPCO(const MultidimArray<std::complex<double> > &F1,const  MultidimArray<std::complex<double> > &F2, MultidimArray<double> &result);
    void computeHann();
    void computingNeighborGraph();
    void computingNeighborGraph2();
    void computeLaplacianMatrix(Matrix2D<double> &L, const std::vector< std::vector<int> > &allNeighborsjp, const std::vector< std::vector<double> > &allWeightsjp);
    void computeCircular();
    void circularWindow(MultidimArray<double> &in);

    double energyDistribution(const Matrix1D<double> &dirj,const std::vector<unsigned int> &Idx, const std::vector<double> &ccVector, const std::vector<unsigned int> &candidates);
    double energyDistribution(const Matrix1D<double> &dirj,const std::vector<unsigned int> &Idx, const Matrix1D<double> &ccVector);


    void getComplexMagnitude(MultidimArray<std::complex<double> > &FourierData, MultidimArray<double> &FourierMag);
    void getShift(MultidimArray<double> &ccVector, double &shift, const size_t &size);
    void getShift2(MultidimArray<double> &ccVector, std::vector<double> &cand, const size_t &size);
    void graphFourierFilter(Matrix1D<double> &ccVecIn, Matrix1D<double> &ccVecOut);

    void halfFourierShift(MultidimArray<double> &in, MultidimArray<double> &out);
    void hannWindow(MultidimArray<double> &in);

    MultidimArray<double> imToPolar2(MultidimArray<double> &cartIm, const size_t &rad, const size_t &ang);
    MultidimArray<double> imToPolar(MultidimArray<double> &cartIm, size_t &start, size_t &end);
    double interpolate(MultidimArray<double> &cartIm, double &x_coord, double &y_coord);
    void imNormalized_cc(const MultidimArray<double>& I1, const MultidimArray<double>& I2, double &value);
    void imZNCC(const MultidimArray<double>& I1, const MultidimArray<double>& I2, double &value);

    double mean_of_products(const MultidimArray<double> &data1, MultidimArray<double> &data2);
    double mean_of_products(MultidimArray<double> &data1, MultidimArray<double> &data2);
    void meanByColumn(MultidimArray<double> &in, MultidimArray<double> &out);
    void maxByColumn(MultidimArray<double> &in, MultidimArray<double> &out);
    void meanByRow(MultidimArray<double> &in, MultidimArray<double> &out);
    void maxByRow(MultidimArray<double> &in, MultidimArray<double> &out);

    void normalized_cc(MultidimArray<double> &X, MultidimArray<double> &Y, double &valor);
    void normalized_cc(const MultidimArray<double> &X, MultidimArray<double> &Y, double &value);

    void pearsonCorr(const MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);

    void rotCandidates3(MultidimArray<double> &in, std::vector<double> &cand, const size_t &size);
    void rotCandidates(MultidimArray<double> &in, std::vector<double>& cand, const size_t &size /*,int *nPeaksFound*/);

    void ssimIndex(MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);
    void ssimIndex(const MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);

    void _writeTestFile(const MultidimArray<double> &data, const char* fileName,size_t nFil, size_t nCol);
    void _writeTestFile(MultidimArray<double> &data, const char *fileName);
    void _writeTestFile(MultidimArray<double> &data, const char *fileName, size_t nFil, size_t nCol);

private:


};
//@}


#endif
