/***************************************************************************
 *
 * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#ifndef _RECONSTRUCTION_ERROR
#define _RECONSTRUCTION_ERROR

#include <core/xmipp_metadata_program.h>
#include <core/xmipp_filename.h>
#include <core/xmipp_image.h>
#include <core/xmipp_fftw.h>
#include <core/multidim_array.h>
#include <core/metadata_vec.h>
#include <data/fourier_projection.h>
#include <data/ctf.h>

#include <vector>
#include <complex>
#include <memory>

class ProgReconstructNoise : public XmippMetadataProgram
{
public:
    using Real = double;
    using Complex = std::complex<Real>;

    bool useCtf;
    FileName fnReferenceVolume;
    double paddingFactor;
    double maxResolution;
    CTFDescription m_ctfDesc;
    
    virtual void readParams() override;
    virtual void defineParams() override;
    virtual void show() const override;

    virtual void preProcess() override;
    virtual void postProcess() override;
    virtual void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) override;
    virtual void writeOutput() override;

private:
    Image<Real> m_reference;
    std::unique_ptr<FourierProjector> m_referenceProjector;

    FourierTransformer m_fourier;
    Image<Real> m_experimental;
    MultidimArray<Complex> m_experimentalFourier;
    MultidimArray<Real> m_ctf;
    MultidimArray<Real> m_avgExpImagePsd; 
    MultidimArray<Real> m_avgRefImagePsd; 
    MultidimArray<Real> m_avgCtfPsd; 
    MultidimArray<Real> m_avgNoisePsd;

    void readReference();
    void createReferenceProjector();
    void computeNoise();

    const MultidimArray<Complex>& projectReference(double rot, double tilt, double psi, const MultidimArray<double>* ctf);

    static void shiftSpectra(MultidimArray<Complex>& spectra, double shiftX, double shiftY);
    static void updatePsd(MultidimArray<Real>& psd, const MultidimArray<Real>& h);
    static void updatePsd(MultidimArray<Real>& psd, const MultidimArray<Complex>& h);
};


#endif