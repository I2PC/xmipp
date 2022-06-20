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

#ifndef ALIGN_SPECTRAL
#define ALIGN_SPECTRAL

#include <core/xmipp_program.h>
#include <core/xmipp_filename.h>
#include <core/xmipp_image.h>
#include <core/xmipp_fftw.h>
#include <core/multidim_array.h>
#include <core/metadata_vec.h>
#include "../data/online_pca.h"

#include <vector>
#include <string_view>
#include <functional>

/**@defgroup Alignment Alignment
   @ingroup ReconsLibrary */
//@{
namespace Alignment {

class ProgAlignSpectral : public XmippProgram
{
public:
    virtual void readParams() override;
    virtual void defineParams() override;
    virtual void show() const override;
    virtual void run() override;

private:
    class TranslationFilter {
    public:
        TranslationFilter(double dx, double dy, size_t nx, size_t ny)
            : m_dy(dy)
            , m_dx(dx)
            , m_coefficients(ny, toFourierXSize(nx)) //Half FFT as real
        {
            computeCoefficients();
        }
        TranslationFilter(const TranslationFilter& other) = default;
        ~TranslationFilter() = default;

        TranslationFilter& operator=(const TranslationFilter& other) = default;

        void operator()(const MultidimArray<std::complex<double>>& in, 
                        MultidimArray<std::complex<double>>& out) const;

    private:
        double m_dy, m_dx;
        MultidimArray<std::complex<double>> m_coefficients;

        void computeCoefficients();
    };

    class ImageTransformer {
    public:
        template<typename F>
        void forEachInPlaneTransform(   const MultidimArray<double>& img,
                                        size_t nRotations,
                                        const std::vector<TranslationFilter>& translations,
                                        F&& func );

        template<typename F>
        void forEachInPlaneTranslation( const MultidimArray<double>& img,
                                        const std::vector<TranslationFilter>& translations,
                                        F&& func );
    private:
        FourierTransformer m_fourier;
        MultidimArray<double> m_rotated;
        MultidimArray<std::complex<double>> m_dft;
        MultidimArray<std::complex<double>> m_translatedDft;

    };
    
    class BandMap {
    public:
        BandMap() = default;
        explicit BandMap(const MultidimArray<int>& bands);
        BandMap(const BandMap& other) = default;
        BandMap(BandMap&& other) = default;
        ~BandMap() = default;

        BandMap& operator=(const BandMap& other) = default;
        BandMap& operator=(BandMap&& other) = default;

        void reset(const MultidimArray<int>& bands);

        const MultidimArray<int>& getBands() const;
        const std::vector<size_t>& getBandSizes() const;
        void flattenForPca( const MultidimArray<std::complex<double>>& spectrum,
                            std::vector<Matrix1D<double>>& data ) const;
        void flattenForPca( const MultidimArray<std::complex<double>>& spectrum,
                            size_t band,
                            Matrix1D<double>& data ) const;

    private:
        MultidimArray<int> m_bands;
        std::vector<size_t> m_sizes;

        static std::vector<size_t> computeBandSizes(const MultidimArray<int>& bands);

    };

    class SpectralPca {
    public:
        SpectralPca() = default;
        SpectralPca(const std::vector<size_t>& sizes, size_t nPc);
        SpectralPca(const SpectralPca& other) = default;
        SpectralPca(SpectralPca&& other) = default;
        ~SpectralPca() = default;

        SpectralPca& operator=(const SpectralPca& other) = default;
        SpectralPca& operator=(SpectralPca&& other) = default;

        void reset();
        void reset(const std::vector<size_t>& sizes, size_t nPc);
        void learn(const std::vector<Matrix1D<double>>& bands);
        void project(   const std::vector<Matrix1D<double>>& bands, 
                        std::vector<Matrix1D<double>>& projections) const;

    private:
        std::vector<SgaNnOnlinePca<double>> m_bandPcas;
    };

    struct RuntimeParameters {
        FileName fnExperimental;
        FileName fnReference;
        FileName fnOutput;

        size_t nRotations;
        size_t nTranslations;
        double maxShift;
    };



    RuntimeParameters m_parameters;

    MetaDataVec m_mdExperimental;
    MetaDataVec m_mdReference;

    std::vector<TranslationFilter> m_translations;
    BandMap m_bandMap;
    SpectralPca m_pca;



    void readInput();
    void calculateTranslationFilters();
    void calculateBands();
    void initPcas();
    void learnReferences();
    void learnExperimental();
    void projectReferences();
    void projectExperimental();


    static void readMetadata(const FileName& fn, MetaDataVec& result);
    static void readImage(const FileName& fn, Image<double>& result);

    static constexpr size_t toFourierXSize(size_t nx) { return nx/2 + 1; }
    static constexpr size_t fromFourierXSize(size_t nx) { return (nx - 1)*2; }

    static std::vector<TranslationFilter> computeTranslationFilters(size_t nx, 
                                                                    size_t ny, 
                                                                    size_t nTranslations,
                                                                    double maxShift );

    MultidimArray<int> computeBands(const size_t nx, 
                                    const size_t ny, 
                                    const double lowCutoffLimit,
                                    const double highCutoffLimit );

};

}

#endif