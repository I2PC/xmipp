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
#include "../data/basic_pca.h"

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
            , m_coefficients(ny, nx/2+1) //Half FFT as real
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

    struct RuntimeParameters {
        FileName fnExperimental;
        FileName fnReference;
        FileName fnOutput;

        size_t nRotations;
        size_t nTranslations;
        double maxShift;
    };

    struct ThreadData {
        Image<double> reader;
        ImageTransformer transformer;
    };



    RuntimeParameters m_parameters;

    MetaDataVec m_mdExperimental;
    MetaDataVec m_mdReference;

    std::vector<TranslationFilter> m_translations;

    void readInput();
    void calculateTranslations();
    void learnReferences();
    void learnExperimental();
    void projectReferences();
    void projectExperimental();


    static void readMetadata(const FileName& fn, MetaDataVec& result);
    static void readImage(const FileName& fn, Image<double>& result);

    static std::vector<TranslationFilter> computeTranslationFilters(size_t nx, 
                                                                    size_t ny, 
                                                                    size_t nTranslations,
                                                                    double maxShift );

};

}

#endif