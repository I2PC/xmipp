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
    // Input params
    FileName fnVolR; // Input reference volume
    FileName fnParticles; // Input metadata
	FileName fnImage; // Particle filename
    FileName fnOut; // Output metadata
    FileName fnMask; // NOT USED but can be final mask? -> DELETE?
    FileName fnMaskVol; // Input 3D mask for reference volume
    FileName fnProj; // JUST FOR SAVING INTERM FILES -> DELETE
    bool subtractAll; // not used now... -> DELETE?
	double lambda; // not used now... -> DELETE?
	double sampling;
	double padFourier;
	double maxResol;
    int fmaskWidth;
	int sigma;
	int iter;

    // Data variables
 	Image<double> V; // volume
 	Image<double> vM; // mask 3D
 	Image<double> M; // mask projected and smooth
 	Image<double> I; // particle
    Image<double> Pctf; // projection with CTF applied
 	Projection P; // projection
 	Projection Pmask; // mask projection for region to keep
 	Projection PmaskVol; // final dilated mask projection
	FourierFilter FilterG; // Gaussian LPF to smooth mask
    std::unique_ptr<FourierProjector> projector;
    const MultidimArray<double> *ctfImage = nullptr;
	FourierTransformer transformer;
	MultidimArray< std::complex<double> > PFourier;

    // Particle metadata
    MetaDataVec mdParticles;
    MDRowVec row;
    Matrix1D<double> roffset;
    struct Angles
    {
    	double rot;
    	double tilt;
    	double psi;
    };
    struct Angles part_angles; // particle angles for projections

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
    Image<double> invertMask(const Image<double> &) const;
    Image<double> applyCTF(const MDRowVec &, Projection &);
    void processParticle(size_t, int, FourierTransformer &);
    MultidimArray< std::complex<double> > computeEstimationImage(const MultidimArray<double> &, const MultidimArray<double> &, FourierTransformer &);
    double evaluateFitting(const MultidimArray<double> &, const MultidimArray<double> &) const;
    void checkBestModel(const MultidimArray<double> &, MultidimArray<double> &) const;

    /// Run
    void run() override;
 };
 //@}
 #endif
