/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

 #ifndef _PROG_SUBTRACT_PROJECTION
 #define _PROG_SUBTRACT_PROJECTION

 #include "core/metadata_vec.h"
 #include "core/xmipp_program.h"
 #include "core/xmipp_image.h"
 #include "data/fourier_filter.h"
 #include "data/fourier_projection.h"


 class ProgSubtractProjection: public XmippProgram
 {
 private:
    /** Filename of the reference volume */
    FileName fnVolR;
    FileName fnParticles;
	FileName fnImage;
    FileName fnOut;
    FileName fnMask;
    FileName fnMaskVol;
    FileName fnPart;
    FileName fnProj;
    MetaDataVec mdParticles;
    MDRowVec row;
    bool subtractAll;
	double lambda;
	double Ts;
	double padFourier;
	double maxResol;
	int sigma;
	int iter;
    struct Angles
    {
    	double rot;
    	double tilt;
    	double psi;
    };
    Matrix1D<double> roffset;
 	Image<double> V; // volume
 	Image<double> vM; // mask 3D
 	Image<double> M; // mask projected and smooth
 	Image<double> I; // particle
 	Image<double> iM; // inverse mask
 	Projection P; // projection
 	Projection Pmask; // mask projection
 	Projection PmaskVol;
	FourierFilter FilterG;
	FourierFilter FilterG2;
	FourierFilter Filter2;
	// 	Image<double> PmaskVolI;
	// 	Image<double> maskVol;
    struct Radial
    {
		MultidimArray<double> meanI;
		MultidimArray<double> meanP;
    };
	MultidimArray<double> radQuotient;
	FourierTransformer transformer;
	MultidimArray< std::complex<double> > IFourier;
	MultidimArray< std::complex<double> > PFourier;
	MultidimArray<double> IFourierMag;
	MultidimArray<std::complex<double> > PFourierPhase;
	double Imin;
	double Imax;
	Image<double> Pctf;
	Image<double> Pmaskctf;

    /// Read argument from command line
    void readParams() override;
    /// Show
    void show() const override;
    /// Define parameters
    void defineParams() override;
    /// Processing methods
    void POCSmaskProj(const MultidimArray<double> &, MultidimArray<double> &) const;
    void POCSFourierAmplitudeProj(const MultidimArray<double> &, MultidimArray< std::complex<double> > &, double, const MultidimArray<double> &, int) const;
    void POCSMinMaxProj(MultidimArray<double> &, double, double) const;
    void extractPhaseProj(MultidimArray< std::complex<double> > &) const;
    void POCSFourierPhaseProj(const MultidimArray< std::complex<double> > &, MultidimArray< std::complex<double> > &) const;
    Image<double> createMask(const FileName &, Image<double> &);
    void readParticle(const MDRowVec &);
    void percentileMinMax(const MultidimArray<double> &, double &, double &) const;
    Image<double> applyCTF(const MDRowVec &, Projection &);
    void runIteration();
    Image<double> thresholdMask(Image<double> &);
    Image<double> binarizeMask(Projection &) const;
    void writeParticle(const int &, Image<double> &);
    /// Run
    void run() override;

 };
 //@}
 #endif
