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
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef __ANGULAR_ASSIGNMENT_MAG_H
#define __ANGULAR_ASSIGNMENT_MAG_H

 /*
#include <../data/xmipp_program.h>
#include <../data/xmipp_fftw.h>
#include <../data/metadata_extension.h>
#include <../data/multidim_array.h>
#include <../data/mask.h>
// */


// /*
#include <core/xmipp_program.h>
#include <core/xmipp_fftw.h>
#include <core/metadata_extension.h>
#include <core/multidim_array.h>
#include <data/mask.h>
#include <data/filters.h>
// */

#include <vector>
#include <fstream> 
#include <ctime>


/**@defgroup AngularAssignmentMag ***
   @ingroup ReconsLibrary */
//@{

/** Angular Assignment mag parameters. */
class ProgAngularAssignmentMag: public XmippMetadataProgram
{
public:
    /** Filenames */
    FileName fnIn, fnOut, fnDir, fnSym, fnRef;
public: // Internal members
    // Metadata with input images and input volumes
    MetaData mdIn, mdRef, mdOut;

    // vector of reference images
    std::vector< MultidimArray<double> > vecMDaRef;

    // vector of Fourier of reference images
    std::vector< MultidimArray< std::complex<double> > > vecMDaRefF;

    // vector of Fourier of polar representation of reference image in real space
    std::vector< MultidimArray< std::complex<double> > > vecMDaRef_polarF;

    // vector of Fourier of polar representation of magnitude spectrum of reference images
    std::vector< MultidimArray< std::complex<double> > > vecMDaRefFMs_polarF;

    // Size of the images
    size_t Xdim, Ydim;

    // Transformer
    FourierTransformer transformerImage;
    FourierTransformer transformerPolarImage;
    FourierTransformer transformerPolarRealSpace;

    // "delay axes"
    MultidimArray<double> axRot;
    MultidimArray<double> axTx;
    MultidimArray<double> axTy;

    // Hann window
    MultidimArray<double> W;
    // circular mask
    MultidimArray<double> C;

    // CCV result matrix
    MultidimArray<double>                   ccMatrixRot;
    MultidimArray<double>                   ccVectorRot;
    int                                     peaksFound = 0; // peaksFound in ccVectorRot

    // matrix for neighbors and angular distance
    std::vector< std::vector<int> > neighboursMatrix; // this should be global
    std::vector< std::vector<double> > neighboursDistance; // not sure if necessary this global
    std::vector< std::vector<double> > neighboursWeights; // this variable should be global
    int N_neighbors;

    size_t idxOut; // index for metadata output file

    int testCounter = 0;
    int testCounter2=0;

    // candidates for each loop
    std::vector<unsigned int>               candidatesFirstLoop;
    std::vector<unsigned int>               Idx;
    std::vector<double>                     candidatesFirstLoopCoeff;
    std::vector<double>                     bestTx;
    std::vector<double>                     bestTy;
    std::vector<double>                     bestPsi;

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
    //    size_t startBandRef;
    //    size_t finalBandRef;
    size_t n_rad;
    size_t n_ang;
    size_t n_ang2;
    double maxShift;
    double sampling;

    int Nsim;


public:
    // constructor
    ProgAngularAssignmentMag();

    // destructor
    ~ProgAngularAssignmentMag();

    /// Read arguments from command line
    void defineParams();
    void readParams();

    void show();

    /*startProcessing() */
    void startProcessing();

    /// Produce side info: fill arrays with relevant transformation matrices
    void preProcess();

    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    void postProcess();

    //borrar luego
    void arithmetic_mean_and_stddev(const MultidimArray<double> &data, double &avg, double &stddev);
    void arithmetic_mean_and_stddev(MultidimArray<double> &data, double &avg, double &stddev);
    void _applyCircularMask(const MultidimArray<double> &in, MultidimArray<double> &out);
    void _applyFourierImage(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData);
    void _applyFourierImage(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    void _applyFourierImage2(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData);
    void _applyFourierImage2(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    void _applyFourierImage3(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    void _applyRotation(const MultidimArray<double> &MDaRef, double &rot, MultidimArray<double> &MDaRefRot);
    void _applyRotation(MultidimArray<double> &MDaRef, double &rot, MultidimArray<double> &MDaRefRot);
    void _applyShift(MultidimArray<double> &input, double &tx, double &ty, MultidimArray<double> &output);
    void _applyShift(const MultidimArray<double> &input, double &tx, double &ty, MultidimArray<double> &output);
    void _applyRotationAndShift(const MultidimArray<double> &MDaRef, double &rot, double &tx, double &ty, MultidimArray<double> &MDaRefRot);
    void _applyShiftAndRotation(const MultidimArray<double> &MDaRef, double &rot, double &tx, double &ty, MultidimArray<double> &MDaRefRot);

    void bestCand(const MultidimArray<double> &MDaIn, const MultidimArray<std::complex<double> > &MDaInF, const MultidimArray<double> &MDaRef, std::vector<double> &cand, double &bestCandRot, double &shift_x, double &shift_y, double &bestCoeff);
    void bestCand2(const MultidimArray<double> &MDaIn, const MultidimArray<std::complex<double> > &MDaInF, const MultidimArray<double> &MDaRef, std::vector<double> &cand, double &bestCandRot, double &shift_x, double &shift_y, double &bestCoeff);

    void completeFourierShift(MultidimArray<double> &in, MultidimArray<double> &out);
    void ccMatrix(const MultidimArray<std::complex<double> > &F1, const MultidimArray<std::complex<double> > &F2, MultidimArray<double> &result);
    void ccMatrixPCO(MultidimArray<std::complex<double> > &F1, MultidimArray<std::complex<double> > &F2, MultidimArray<double> &result);
    void computeHann();
    void computingNeighborGraph();
    void computingNeighborGraph2();
    void computeCircular();
    void circularWindow(MultidimArray<double> &in);

    void _delayAxes(const size_t &Ydim, const size_t &Xdim, const size_t &n_ang);

    void _getComplexMagnitude(MultidimArray<std::complex<double> > &FourierData, MultidimArray<double> &FourierMag);
    void getShift(MultidimArray<double> &ccVector, double &shift, const size_t &size);
    void getShift2(MultidimArray<double> &ccVector, std::vector<double> &cand, const size_t &size);
    void getRot(MultidimArray<double> &ccVector, double &rot, const size_t &size);

    void halfFourierShift(MultidimArray<double> &in, MultidimArray<double> &out);
    void hannWindow(MultidimArray<double> &in);

    MultidimArray<double> imToPolar2(MultidimArray<double> &cartIm, const size_t &rad, const size_t &ang);
    MultidimArray<double> imToPolar(MultidimArray<double> &cartIm, size_t &start, size_t &final);
    double interpolate(MultidimArray<double> &cartIm, double &x_coord, double &y_coord);
    void imNormalized_cc(const MultidimArray<double>& I1, const MultidimArray<double>& I2, double &value);
    void imZNCC(const MultidimArray<double>& I1, const MultidimArray<double>& I2, double &value);

    double mean_of_products(const MultidimArray<double> &data1, MultidimArray<double> &data2);
    double mean_of_products(MultidimArray<double> &data1, MultidimArray<double> &data2);
    void meanByColumn(MultidimArray<double> &in, MultidimArray<double> &out);
    void maxByColumn(MultidimArray<double> &in, MultidimArray<double> &out);
    void meanByRow(MultidimArray<double> &in, MultidimArray<double> &out);
    void maxByRow(MultidimArray<double> &in, MultidimArray<double> &out);

    void newApplyGeometry(MultidimArray<double> &in, MultidimArray<double> &out, const double &a, const double &b, const double &c, const double &d, const double &tx, const double &ty);
    void normalized_cc(MultidimArray<double> &X, MultidimArray<double> &Y, double &valor);
    void normalized_cc(const MultidimArray<double> &X, MultidimArray<double> &Y, double &value);

    void printSomeValues(MultidimArray<double> & MDa);
    void pearsonCorr(const MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);

    void rotCandidates3(MultidimArray<double> &in, std::vector<double> &cand, const size_t &size);
    void rotCandidates(MultidimArray<double> &in, std::vector<double>& cand, const size_t &size /*,int *nPeaksFound*/);

    void ssimIndex(MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);
    void ssimIndex(const MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);
    void selectBands(MultidimArray<double> &in, MultidimArray<double> &out);

    void _writeTestFile(const MultidimArray<double> &data, const char* fileName,size_t nFil, size_t nCol);
    void _writeTestFile(MultidimArray<double> &data, const char *fileName);
    void _writeTestFile(MultidimArray<double> &data, const char *fileName, size_t nFil, size_t nCol);

private:
    //    void arithmetic_mean_and_stddev(const MultidimArray<double> &data, double &avg, double &stddev);
    //    void arithmetic_mean_and_stddev(MultidimArray<double> &data, double &avg, double &stddev);
    //    void _applyCircularMask(const MultidimArray<double> &in, MultidimArray<double> &out);
    //    void _applyFourierImage(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData);
    //    void _applyFourierImage(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    //    void _applyFourierImage2(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData);
    //    void _applyFourierImage2(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    //    void _applyFourierImage3(MultidimArray<double> &data, MultidimArray<std::complex<double> > &FourierData, const size_t &ang);
    //    void _applyRotation(const MultidimArray<double> &MDaRef, double &rot, MultidimArray<double> &MDaRefRot);
    //    void _applyRotation(MultidimArray<double> &MDaRef, double &rot, MultidimArray<double> &MDaRefRot);
    //    void _applyShift(MultidimArray<double> &input, double &tx, double &ty, MultidimArray<double> &output);
    //    void _applyShift(const MultidimArray<double> &input, double &tx, double &ty, MultidimArray<double> &output);
    //    void _applyRotationAndShift(const MultidimArray<double> &MDaRef, double &rot, double &tx, double &ty, MultidimArray<double> &MDaRefRot);
    //    void _applyShiftAndRotation(const MultidimArray<double> &MDaRef, double &rot, double &tx, double &ty, MultidimArray<double> &MDaRefRot);
    //
    //    void bestCand(const MultidimArray<double> &MDaIn, const MultidimArray<std::complex<double> > &MDaInF, const MultidimArray<double> &MDaRef, std::vector<double> &cand, /*int &peaksFound,*/ double *bestCandRot, double *shift_x, double *shift_y, double *bestCoeff);
    //    void bestCand2(MultidimArray<double> &MDaIn, MultidimArray<std::complex<double> > &MDaInF, MultidimArray<double> &MDaRef, std::vector<double> &cand, int &peaksFound, double *bestCandRot, double *shift_x, double *shift_y, double *bestCoeff);
    //
    //    void completeFourierShift(MultidimArray<double> &in, MultidimArray<double> &out);
    //    void ccMatrix(const MultidimArray<std::complex<double> > &F1, const MultidimArray<std::complex<double> > &F2, MultidimArray<double> &result);
    //    void ccMatrixPCO(MultidimArray<std::complex<double> > &F1, MultidimArray<std::complex<double> > &F2, MultidimArray<double> &result);
    //    void computeHann();
    //    void computingNeighborGraph();
    //    void computeCircular();
    //    void circularWindow(MultidimArray<double> &in);
    //
    //    void _delayAxes(const size_t &Ydim, const size_t &Xdim, const size_t &n_ang);
    //
    //    void _getComplexMagnitude(MultidimArray<std::complex<double> > &FourierData, MultidimArray<double> &FourierMag);
    //    void getShift(MultidimArray<double> &ccVector, double &shift, const size_t &size);
    //    void getRot(MultidimArray<double> &ccVector, double &rot, const size_t &size);
    //
    //    void halfFourierShift(MultidimArray<double> &in, MultidimArray<double> &out);
    //    void hannWindow(MultidimArray<double> &in);
    //
    //    MultidimArray<double> imToPolar(MultidimArray<double> &cartIm, size_t &start, size_t &final);
    //    double interpolate(MultidimArray<double> &cartIm, double &x_coord, double &y_coord);
    //    MultidimArray<double> imToPolar2(MultidimArray<double> &cartIm, const size_t &rad, const size_t &ang);
    //
    //    double mean_of_products(const MultidimArray<double> &data1, MultidimArray<double> &data2);
    //    double mean_of_products(MultidimArray<double> &data1, MultidimArray<double> &data2);
    //    void meanByColumn(MultidimArray<double> &in, MultidimArray<double> &out);
    //    void maxByColumn(MultidimArray<double> &in, MultidimArray<double> &out);
    //    void meanByRow(MultidimArray<double> &in, MultidimArray<double> &out);
    //    void maxByRow(MultidimArray<double> &in, MultidimArray<double> &out);
    //
    //    void newApplyGeometry(MultidimArray<double> &in, MultidimArray<double> &out, const double &a, const double &b, const double &c, const double &d, const double &tx, const double &ty);
    //
    //    void printSomeValues(MultidimArray<double> & MDa);
    //    void pearsonCorr(const MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);
    //
    //    void rotCandidates2(MultidimArray<double> &in, std::vector<double> &cand, const size_t &size, int *nPeaksFound);
    //    void rotCandidates3(MultidimArray<double> &in, std::vector<double> &cand, const size_t &size);
    //    void rotCandidates(MultidimArray<double> &in, std::vector<double>& cand, const size_t &size /*,int *nPeaksFound*/);
    //
    //    void shiftCandidates(MultidimArray<double> &in, std::vector<double> &cand, const size_t &size, int *nPeaksFound);
    //    void ssimIndex(MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);
    //    void ssimIndex(const MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);
    //    void selectBands(MultidimArray<double> &in, MultidimArray<double> &out);
    //
    //    void _writeTestFile(const MultidimArray<double> &data, const char* fileName,size_t nFil, size_t nCol);
    //    void _writeTestFile(MultidimArray<double> &data, const char *fileName);
    //    void _writeTestFile(MultidimArray<double> &data, const char *fileName, size_t nFil, size_t nCol);
    //
    //    void zncc_coeff(const MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff);

};
//@}


#endif
