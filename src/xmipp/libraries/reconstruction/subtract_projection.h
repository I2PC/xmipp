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
	double sampling;
	double padFourier;
	double maxResol;
    double fmaskWidth;
	int sigma;
	int iter;
    Matrix1D<double> roffset;
    struct Angles
    {
    	double rot;
    	double tilt;
    	double psi;
    };

 	Image<double> V; // volume
 	Image<double> vM; // mask 3D
 	Image<double> M; // mask projected and smooth
 	Image<double> I; // particle
 	Image<double> iM; // inverse mask
    Image<double> Pctf; // projection with CTF applied
 	Projection P; // projection
 	Projection Pmask; // mask projection for region to keep
 	Projection PmaskVol; // final dilated mask projection
	FourierFilter FilterG; // Gaussian LPF to smooth mask

	FourierTransformer transformer;
	MultidimArray< std::complex<double> > PFourier;

    /// Read argument from command line
    void readParams() override;
    /// Show
    void show() const override;
    /// Define parameters
    void defineParams() override;
    /// Read and write methods
    void readParticle(const MDRowVec &);
    void writeParticle(const int &, Image<double> &);
    /// Processing methods
    Image<double> createMask(const FileName &, Image<double> &);
    Image<double> binarizeMask(Projection &) const;
    Image<double> applyCTF(const MDRowVec &, Projection &);
    /// Run
    void run() override;
 };
 //@}
 #endif
