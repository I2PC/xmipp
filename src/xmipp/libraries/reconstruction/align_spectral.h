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

        void getTranslation(double& dx, double& dy) const;

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
        
        template<typename F>
        void forFourierTransform(   const MultidimArray<double>& img,
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
        SpectralPca(const std::vector<size_t>& sizes, double initialCompression, double initialBatch);
        SpectralPca(const SpectralPca& other) = default;
        SpectralPca(SpectralPca&& other) = default;
        ~SpectralPca() = default;

        SpectralPca& operator=(const SpectralPca& other) = default;
        SpectralPca& operator=(SpectralPca&& other) = default;

        size_t getBandCount() const;
        
        size_t getBandSize(size_t i) const;
        size_t getProjectionSize(size_t i) const;
        void getMean(size_t i, Matrix1D<double>& v) const;
        void getVariance(size_t i, Matrix1D<double>& v) const;
        void getAxisVariance(size_t i, Matrix1D<double>& v) const;
        void getBasis(size_t i, Matrix2D<double>& b) const;
        double getError(size_t i) const;

        void getErrorFunction(size_t i, Matrix1D<double>& errFn);

        void reset();
        void reset(const std::vector<size_t>& sizes, double initialCompression, double initialBatch);
        void reset(const std::vector<size_t>& sizes, size_t pcaSize, size_t initialBatch);
        void learn(const std::vector<Matrix1D<double>>& bands);
        void learnConcurrent(const std::vector<Matrix1D<double>>& bands);
        void finalize();

        void equalizeError(double precision);

        void projectCentered(   const std::vector<Matrix1D<double>>& bands, 
                                std::vector<Matrix1D<double>>& projections) const;
        void centerAndProject(  std::vector<Matrix1D<double>>& bands, 
                                std::vector<Matrix1D<double>>& projections) const;
        void unprojectAndUncenter(  const std::vector<Matrix1D<double>>& projections,
                                    std::vector<Matrix1D<double>>& bands ) const;
    private:
        std::vector<SgaNnOnlinePca<double>> m_bandPcas;
        std::vector<std::mutex> m_bandMutex;

        static void calculateErrorFunction( Matrix1D<double>& lambdas, 
                                            double totalVariance );

        static size_t calculateRequiredComponents(  const Matrix1D<double>& errFn,
                                                    double precision );

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

        void getPcaProjection(size_t i, std::vector<Matrix1D<double>>& referenceBands);
        size_t matchPcaProjection(const std::vector<Matrix1D<double>>& experimentalBands, const Matrix1D<double>& weights) const;
        size_t matchPcaProjectionBaB(const std::vector<Matrix1D<double>>& experimentalBands, const Matrix1D<double>& weights) const;

    private:
        std::vector<Matrix2D<double>> m_projections;

    };

    class ReferenceMetadata {
    public:
        ReferenceMetadata() = default;
        ReferenceMetadata(size_t rowId, double rotation, double shiftx, double shifty);
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

    private:
        size_t m_rowId;
        double m_rotation;
        double m_shiftX;
        double m_shiftY;

    };

    struct RuntimeParameters {
        FileName fnReference;
        FileName fnExperimental;
        FileName fnOutput;
        FileName fnOroot;

        size_t nRotations;
        size_t nTranslations;
        double maxShift;

        size_t nBands;
        double lowResLimit;
        double highResLimit;

        double pcaEff;
        double training;

        size_t nThreads;
    };

    RuntimeParameters m_parameters;

    MetaDataVec m_mdReference;
    MetaDataVec m_mdExperimental;
    Matrix1D<double> m_weights;

    std::vector<TranslationFilter> m_translations;
    BandMap m_bandMap;
    SpectralPca m_pca;
    ReferencePcaProjections m_references;
    std::vector<ReferenceMetadata> m_referenceData;
    std::vector<size_t> m_classification;
    Matrix2D<double> m_ssnr;

    void readInput();
    void calculateTranslationFilters();
    void calculateBands();
    void trainPcas();
    void calculateBandWeights();
    void projectReferences();
    void classifyExperimental();
    void generateBandSsnr();
    void generateOutput();

    void updateRow(MDRowVec& row, size_t matchIndex) const;

    template<typename F>
    void processRowsInParallel(const MetaDataVec& md, F&& func, size_t nThreads);

    static void readMetadata(const FileName& fn, MetaDataVec& result);
    static void readImage(const FileName& fn, Image<double>& result);

    static constexpr size_t toFourierXSize(size_t nx) { return nx/2 + 1; }
    static constexpr size_t fromFourierXSize(size_t nx) { return (nx - 1)*2; }

    static std::vector<TranslationFilter> computeTranslationFiltersRectangle(   size_t nx, 
                                                                                size_t ny, 
                                                                                size_t nTranslations,
                                                                                double maxShift );
    static std::vector<TranslationFilter> computeTranslationFiltersSunflower(   size_t nx, 
                                                                                size_t ny, 
                                                                                size_t nTranslations,
                                                                                double maxShift );

    static std::vector<double> computeArithmeticBandFrecuencies(double lowResLimit,
                                                                double highResLimit,
                                                                size_t nBands );
    static std::vector<double> computeGeometricBandFrecuencies( double lowResLimit,
                                                                double highResLimit,
                                                                size_t nBands );

    static MultidimArray<int> computeBands( const size_t nx, 
                                            const size_t ny, 
                                            const std::vector<double>& frecuencies );

    static void calculateBandSsnr(  const std::vector<Matrix1D<double>>& reference, 
                                    const std::vector<Matrix1D<double>>& experimental, 
                                    Matrix1D<double>& ssnr );

};

}

#endif