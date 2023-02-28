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
 #include "core/xmipp_metadata_program.h"

/**@defgroup ProgSubtractProjection Subtract projections
   @ingroup ReconsLibrary */
//@{
/** Subtract projections from particles */

class ProgSubtractProjection: public XmippMetadataProgram
 {
 private:
    // Input params
    FileName fnVolR; // Input reference volume
    FileName fnParticles; // Input metadata
	FileName fnImg; // Particle filename
    FileName fnOut; // Output metadata
    FileName fnMaskVol; // Input 3D mask of the reference volume
    FileName fnMask; // Input 3D mask for region to keep
    FileName fnProj; // Path to save intermediate files
	double sampling; 
	double padFourier; 
	double maxResol;
    double cirmaskrad; // Radius of the circular mask
	int sigma;
    int limitfreq;
    int maxwiIdx;
    int i;
    bool nonNegative;
    bool boost;
    bool subtract;
	MultidimArray<int> wi;

    // Data variables
 	Image<double> V; // volume
 	Image<double> vM; // mask 3D
    Image<double> ivM; // invert mask 3D

 	Image<double> M; // mask projected and smooth
 	Image<double> I; // particle
    Image<double> Pctf; // projection with CTF applied
    Image<double> iM; // inverse mask of the region to keep
    Image<double> Mfinal; // final dilated mask
    Image<double> Idiff; // final subtracted image
	Image<double> cirmask; // circular mask to avoid edge artifacts	

 	Projection P; // projection
 	Projection Pmask; // mask projection for region to keep
    Projection PmaskVol; // reference volume mask projection
	FourierFilter FilterG; // Gaussian LPF to smooth mask
    std::unique_ptr<FourierProjector> projector;
    std::unique_ptr<FourierProjector> projectorMask;
    const MultidimArray<double> *ctfImage = nullptr; // needed for FourierProjector
	FourierTransformer transformerP; // Fourier transformer for projection
    FourierTransformer transformerI; // Fourier transformer for particle
    MultidimArray< std::complex<double> > IFourier; // FT(particle)
	MultidimArray< std::complex<double> > PFourier; // FT(projection)
    MultidimArray< std::complex<double> > PFourier0; // FT(projection) estimation of order 0
	MultidimArray< std::complex<double> > PFourier1; // FT(projection) estimation of order 1
    MultidimArray< std::complex<double> > IiMFourier;
	MultidimArray< std::complex<double> > PiMFourier;

    FourierTransformer transformerIiM;
	FourierTransformer transformerPiM;

    CTFDescription ctf;
	FourierFilter FilterCTF;
	Image<double> padp; // padded image when applying CTF
	Image<double> PmaskI; // inverted projected mask
	Image<double> ImgiM; // auxiliary image for computing estimation images
	MultidimArray< std::complex<double> > ImgiMFourier; // FT(ImgiM)

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

    bool disable;
    /// Read and write methods
    void readParticle(const MDRow &rowIn);
    void writeParticle(MDRow &rowOut, Image<double> &, double, double, double);
    /// Processing methods
    void createMask(const FileName &, Image<double> &, Image<double> &);

    Image<double> binarizeMask(Projection &) const;
    Image<double> invertMask(const Image<double> &);
    Image<double> applyCTF(const MDRow &, Projection &);
    void processParticle(const MDRow &rowIn, int, FourierTransformer &, FourierTransformer &);
    MultidimArray< std::complex<double> > computeEstimationImage(const MultidimArray<double> &, 
        const MultidimArray<double> &, FourierTransformer &);
    double evaluateFitting(const MultidimArray< std::complex<double> > &, const MultidimArray< std::complex<double> > &) const;
    Matrix1D<double> checkBestModel(MultidimArray< std::complex<double> > &, const MultidimArray< std::complex<double> > &, 
        const MultidimArray< std::complex<double> > &, const MultidimArray< std::complex<double> > &) const;
public:
    /// Read argument from command line
    void readParams();
    /// Show
    void show() const;
    /// Define parameters
    void defineParams();
    void preProcess();
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);
    void postProcess();
 };
 //@}
#endif

