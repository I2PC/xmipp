/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez         me.fernandez@cnb.csic.es (2019)
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
#ifndef _PROG_SUBTOMO_SUBTRACTION
#define _PROG_SUBTOMO_SUBTRACTION

#include "core/xmipp_program.h"
#include <core/xmipp_fftw.h>
#include "data/fourier_filter.h"
#include "core/xmipp_metadata_program.h"

/**@defgroup ProgSubtomoSubtraction Subtomogram subtraction
   @ingroup ReconsLibrary */
//@{
/** Subtomogram subtraction */

/*This class contains methods that are use to adjust an input volume (V) to a another reference volume (V1) through the use
 of Projectors Onto Convex Sets (POCS) and to perform the subtraction between them. Other methods contained in this class
 are used to pre-process and operate with the volumes*/

class ProgSubtomoSubtraction: public XmippMetadataProgram 
{
public:
  // Input params
  FileName fnVolMd; // Input metadata of volume(s)
  FileName fnVolRef; // Input reference volume (V1)
  FileName fnVol2; // Volume filename 
  FileName fnMask1;
  FileName fnMask2;
  FileName fnMaskSub;
  FileName fnVol1F; // save filtered reference
  FileName fnVol2A; // save process volumes before subtraction

  bool computeE;
  size_t iter;
  double v1min;
  double v1max;
  bool performSubtraction;
  int sigma;
  double cutFreq;
  double lambda;
  bool radavg;
  bool subtomos;

  // Particle metadata
  MetaDataVec mdParticles;
  MDRowVec row;
  Matrix1D<double> roffset; // particle shifts
  struct Angles // particle angles for projection
  {
    double rot;
    double tilt;
    double psi;
  };
  struct Angles part_angles;

  // Data variables
  Image<double> V; // volume to adjust
  Image<double> V1; // reference volume
  Image<double> padv; // padded image when applying aligment
  double pad; // size of padding 
  double std1;
  size_t n;
  FourierTransformer transformer2;
  MultidimArray<std::complex<double>> V2Fourier;
  MultidimArray<double> V1FourierMag;
  MultidimArray<double> mask; 
  Image<double> Vdiff;
  FourierFilter filter2; 
  int rank; // for MPI version

	/// Processing methods
  void POCSmask(const MultidimArray<double> &, MultidimArray<double> &);
  void POCSnonnegative(MultidimArray<double> &);
  void POCSFourierAmplitude(const MultidimArray<double> &, MultidimArray<std::complex<double>> &, double );
  void POCSFourierAmplitudeRadAvg(MultidimArray<std::complex<double>> &, double , const MultidimArray<double> &, int, int, int);
  void POCSMinMax(MultidimArray<double> &, double, double);
  void POCSFourierPhase(const MultidimArray<std::complex<double>> &, MultidimArray<std::complex<double>> &);
	void extractPhase(MultidimArray<std::complex<double>> &) const;
	void computeEnergy(MultidimArray<double> &, const MultidimArray<double> &) const;
	void centerFFTMagnitude(MultidimArray<double> &, MultidimArray<std::complex<double>> &,MultidimArray<double> &) const;
  void radialAverage(const MultidimArray<double> &,	const MultidimArray<double> &, MultidimArray<double> &);
	MultidimArray<double> computeRadQuotient(const MultidimArray<double> &,	const MultidimArray<double> &, const MultidimArray<double> &, 
    const MultidimArray<double> &);
  void createFilter(FourierFilter &, double);
  Image<double> subtraction(Image<double>, Image<double> &, const MultidimArray<double> &, const FileName &,
		const FileName &, FourierFilter &, double );
  MultidimArray<double> computeMagnitude(MultidimArray<double> &);
  MultidimArray<double> createMask(const Image<double> &, const FileName &, const FileName &);
	void filterMask(MultidimArray<double> &) const;
	MultidimArray<std::complex<double>> computePhase(MultidimArray<double> &);
	MultidimArray<double> getSubtractionMask(const FileName &, MultidimArray<double>);

  /// Empty constructor
  ProgSubtomoSubtraction();

  /// Read argument from command line
  void readParams() override;
  /// Show
  void show() const override;
  /// Define parameters
  void defineParams() override;
  /// Read and write methods
  void readParticle(const MDRow &rowIn);
  void writeParticle(MDRow &rowOut, FileName, Image<double> &);
  /// MPI methods
  void preProcess() override;
  void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) override;
  void postProcess() override;

};
//@}
#endif
