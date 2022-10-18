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
#include "../data/sga_nn_online_pca.h"

#include <vector>
#include <string_view>
#include <functional>
#include <mutex>
#include <limits>

/**@defgroup Alignment Alignment
   @ingroup ReconsLibrary */
//@{
namespace Alignment {

class ProgAlignSpectral : public XmippProgram
{
public:
    using Real = double;

    virtual void readParams() override;
    virtual void defineParams() override;
    virtual void show() const override;
    virtual void run() override;

    FileName fnExperimentalMetadata;
    FileName fnReferenceMetadata;
    FileName fnOutputMetadata;
    FileName fnBands;
    FileName fnWeights;
    FileName fnCtf;
    FileName fnPca;
    
    size_t nRotations;
    size_t nTranslations;
    Real maxShift;

    size_t nThreads;
    Real maxMemory;

private:
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
        void flatten(   const MultidimArray<std::complex<Real>>& spectrum,
                        std::vector<Matrix1D<Real>>& data,
                        size_t image = 0 ) const;
        void flattenOddEven(const MultidimArray<Real>& spectrum,
                            std::vector<Matrix1D<Real>>& data,
                            size_t oddEven,
                            size_t image = 0 ) const;
        void flatten(   const MultidimArray<std::complex<Real>>& spectrum,
                        size_t band,
                        Matrix1D<Real>& data,
                        size_t image = 0 ) const;
        void flattenOddEven(const MultidimArray<Real>& spectrum,
                            size_t band,
                            Matrix1D<Real>& data,
                            size_t oddEven,
                            size_t image = 0 ) const;

    private:
        MultidimArray<int> m_bands;
        std::vector<size_t> m_sizes;

        static std::vector<size_t> computeBandSizes(const MultidimArray<int>& bands);

    };

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

        void getTranslation(double& dx, double& dy) const;

        TranslationFilter& operator=(const TranslationFilter& other) = default;

        void operator()(const MultidimArray<std::complex<Real>>& in, 
                        MultidimArray<std::complex<Real>>& out) const;

    private:
        double m_dy, m_dx;
        MultidimArray<std::complex<Real>> m_coefficients;

        void computeCoefficients();
    };

    class ImageTransformer {
    public:
        template<typename F>
        void forEachInPlaneTransform(   const MultidimArray<Real>& img,
                                        size_t nRotations,
                                        const std::vector<TranslationFilter>& translations,
                                        F&& func );

        template<typename F>
        void forEachInPlaneTranslation( const MultidimArray<Real>& img,
                                        const std::vector<TranslationFilter>& translations,
                                        F&& func );
        
    private:
        FourierTransformer m_fourier;
        MultidimArray<Real> m_rotated;
        MultidimArray<std::complex<Real>> m_dft;
        MultidimArray<std::complex<Real>> m_translatedDft;

    };

    class ReferencePcaProjections {
    public:
        ReferencePcaProjections() = default;
        ReferencePcaProjections(size_t nImages, const std::vector<size_t>& bandSizes);
        ReferencePcaProjections(const ReferencePcaProjections& other) = default;
        ~ReferencePcaProjections() = default;

        ReferencePcaProjections& operator=(const ReferencePcaProjections& other) = default;

        void reset(size_t nImages, const std::vector<size_t>& bandSizes);

        size_t getImageCount() const;
        size_t getBandCount() const;
        size_t getComponentCount(size_t i) const;

        void getPcaProjection(size_t i, std::vector<Matrix1D<Real>>& referenceBands);
        size_t matchPcaProjection(const std::vector<Matrix1D<Real>>& experimentalBands, Real& bestDistance) const;
        size_t matchPcaProjectionBaB(const std::vector<Matrix1D<Real>>& experimentalBands, Real& bestDistance) const;

    private:
        std::vector<Matrix2D<Real>> m_projections;

    };

    class ReferenceMetadata {
    public:
        ReferenceMetadata();
        ReferenceMetadata(size_t rowId, double rotation, double shiftx, double shifty, double distance = std::numeric_limits<double>::infinity());
        ReferenceMetadata(const ReferenceMetadata& other) = default;
        ~ReferenceMetadata() = default;

        ReferenceMetadata& operator=(const ReferenceMetadata& other) = default;

        void setRowId(size_t id);
        size_t getRowId() const;

        void setRotation(double rotation);
        double getRotation() const;

        void setShiftX(double sx);
        double getShiftX() const;

        void setShiftY(double sy);
        double getShiftY() const;

        void setDistance(double distance);
        double getDistance() const;

    private:
        size_t m_rowId;
        double m_rotation;
        double m_shiftX;
        double m_shiftY;
        double m_distance;

    };

    enum class CombinationStrategy {
        storeRefRotShift,
        storeRefRot,
        storeRef
    };

    MetaDataVec m_mdExperimental;
    MetaDataVec m_mdReference;
    BandMap m_bandMap;
    std::vector<Matrix2D<Real>> m_bases;
    std::vector<Matrix2D<Real>> m_ctfBases;
    std::vector<TranslationFilter> m_translations;
    ReferencePcaProjections m_references;
    std::vector<ReferenceMetadata> m_referenceData;
    std::vector<ReferenceMetadata> m_classification;

    void readInputMetadata();
    void readBandMap();
    void readBases();
    void applyWeightsToBases();
    void applyCtfToBases();
    void generateTranslations();
    void alignImages();
    void generateOutput();

    void projectReferences(size_t start, size_t count);
    void classifyExperimental();

    void removeFourierSymmetry(MultidimArray<Real>& spectrum) const;
    void multiplyBases( std::vector<Matrix2D<Real>>& bases,
                        const MultidimArray<Real>& spectrum ) const;

    void updateRow(MDRowVec& row, const ReferenceMetadata& data) const;

    template<typename F>
    void processRowsInParallel( const MetaDataVec& md, F&& func, size_t nThreads, 
                                size_t start = 0, size_t count = std::numeric_limits<size_t>::max() );

    static size_t getImageProjectionSize(const std::vector<Matrix2D<Real>>& bases);
    static size_t getGalleryProjectionSize( size_t imageSize, 
                                            size_t nImage, 
                                            size_t nRot, 
                                            size_t nShift, 
                                            CombinationStrategy strategy );
    static CombinationStrategy getCombinationStrategy(  size_t nExp, 
                                                        size_t nRef, 
                                                        size_t nRot, 
                                                        size_t nShift );

    static std::vector<TranslationFilter> computeTranslationFiltersRectangle(   size_t nx, 
                                                                                size_t ny, 
                                                                                size_t nTranslations,
                                                                                double maxShift );
    static std::vector<TranslationFilter> computeTranslationFiltersSunflower(   size_t nx, 
                                                                                size_t ny, 
                                                                                size_t nTranslations,
                                                                                double maxShift );

    static void project(const std::vector<Matrix2D<Real>>& bases,
                        const std::vector<Matrix1D<Real>>& bands,
                        std::vector<Matrix1D<Real>>& projections );

};

}

#endif