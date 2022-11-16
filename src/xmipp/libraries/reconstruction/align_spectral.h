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
#include <core/matrix2d.h>
#include <data/ball_tree.h>

#include <vector>
#include <string_view>
#include <functional>
#include <mutex>
#include <limits>
#include <complex>
#include <list>

/**@defgroup Alignment Alignment
   @ingroup ReconsLibrary */
//@{
namespace Alignment {

template <typename T>
class ProgAlignSpectral : public XmippProgram
{
public:
    using Real = T;
    using Complex = std::complex<Real>;

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
    double nTranslations;
    Real maxShift;

    size_t nThreads;
    double maxMemory;

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

        template<typename Q, typename P>
        void flatten(   const MultidimArray<Q>& spectrum,
                        std::vector<MultidimArray<P>>& data,
                        size_t image = 0 ) const;

        template<typename Q, typename P>
        void flatten(   const MultidimArray<Q>& spectrum,
                        size_t band,
                        MultidimArray<P>& data,
                        size_t image = 0 ) const;

    private:
        MultidimArray<int> m_bands;
        std::vector<size_t> m_sizes;

        static std::vector<size_t> computeBandSizes(const MultidimArray<int>& bands);

    };
    
    class Rotation {
    public:
        Rotation(double angle = 0);
        Rotation(const Rotation& other) = default;
        ~Rotation() = default;
        
        Rotation& operator=(const Rotation& other) = default;

        void operator()(const MultidimArray<Real>& in, 
                        MultidimArray<Real>& out) const;

        double getAngle() const;
    
    private:
        double m_angle;
        Matrix2D<double> m_matrix;

        void computeMatrix();
    };

    class BandShiftFilters {
    public:
        using Shift = std::array<double, 2>;

        BandShiftFilters() = default;
        BandShiftFilters(std::vector<Shift>&& shifts, const BandMap& bands);
        BandShiftFilters(const BandShiftFilters& other) = default;
        ~BandShiftFilters() = default;


        BandShiftFilters& operator=(const BandShiftFilters& other) = default;

        void operator()(size_t index,
                        const std::vector<MultidimArray<Complex>>& in, 
                        std::vector<MultidimArray<Complex>>& out ) const;

        const Shift& getShift(size_t index) const;
        size_t getShiftCount() const;

    private:
        std::vector<Shift> m_shifts;
        std::vector<MultidimArray<Complex>> m_coefficients;

        static void computeCoefficients(const Shift& shift, MultidimArray<Complex>& result);
        static void computeFlattenedCoefficients(   const std::vector<Shift>& shifts,
                                                    const BandMap& bands,
                                                    std::vector<MultidimArray<Complex>>& result );
    };

    class ImageRotationTransformer {
    public:
        template<typename F>
        void forEachInPlaneRotation(const MultidimArray<Real>& img,
                                    const std::vector<Rotation>& rotations,
                                    F&& func );

    private:
        FourierTransformer m_fourierClean;
        FourierTransformer m_fourierRotated;
        MultidimArray<Real> m_rotated;

    };
    
    class BandShiftTransformer {
    public:
        template<typename F>
        void forEachInPlaneTranslation( const std::vector<MultidimArray<Complex>>& in,
                                        const BandShiftFilters& shifts,
                                        F&& func );
        
    private:
        std::vector<MultidimArray<Complex>> m_shifted;
    };
    
    class ReferencePcaProjections {
    public:
        ReferencePcaProjections() = default;
        ReferencePcaProjections(size_t nImages, const std::vector<size_t>& bandSizes);
        ReferencePcaProjections(const ReferencePcaProjections& other) = default;
        ~ReferencePcaProjections() = default;

        ReferencePcaProjections& operator=(const ReferencePcaProjections& other) = default;

        void reset(size_t nImages, const std::vector<size_t>& bandSizes);
        void finalize();

        size_t getImageCount() const;
        size_t getBandCount() const;
        size_t getComponentCount(size_t i) const;

        void getPcaProjection(size_t i, std::vector<Matrix1D<Real>>& referenceBands);
        size_t matchPcaProjection(const std::vector<Matrix1D<Real>>& experimentalBands, Real& bestDistance, Matrix1D<Real>& distances) const;
        size_t matchPcaProjectionBallTree(const std::vector<Matrix1D<Real>>& experimentalBands, Real& bestDistance) const;
        size_t matchPcaProjectionBnB(const std::vector<Matrix1D<Real>>& experimentalBands, Real& bestDistance, std::list<std::pair<size_t, Real>>& ws) const;

    private:
        std::vector<Matrix2D<Real>> m_projections;
        std::vector<Matrix1D<Real>> m_lengths;
        std::vector<BallTree<Real>> m_trees;

        void computeLengths();
        void computeTrees();
    };

    class ReferenceMetadata {
    public:
        ReferenceMetadata(  size_t rowId = std::numeric_limits<size_t>::max(), 
                            double rotation = 0.0, 
                            double shiftx = 0.0, 
                            double shifty = 0.0, 
                            double distance = std::numeric_limits<double>::infinity() );
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

    MetaDataVec m_mdExperimental;
    MetaDataVec m_mdReference;
    BandMap m_bandMap;
    std::vector<Matrix2D<Real>> m_bases;
    std::vector<Matrix2D<Real>> m_ctfBases;
    std::vector<size_t> m_projectionSizes;
    std::vector<Rotation> m_rotations;
    BandShiftFilters m_translations;
    ReferencePcaProjections m_references;
    std::vector<ReferenceMetadata> m_referenceData;
    std::vector<ReferenceMetadata> m_classification;

    void readInputMetadata();
    void readBandMap();
    void readBases();
    void applyWeightsToBases();
    void applyCtfToBases();
    void generateRotations();
    void generateTranslations();
    void alignImages();
    void generateOutput();

    void projectReferencesRot(size_t start, size_t count);
    void projectReferencesRotShift(size_t start, size_t count);
    void classifyExperimental();
    void classifyExperimentalShift();

    void removeFourierSymmetry(MultidimArray<Real>& spectrum) const;
    void multiplyBases( std::vector<Matrix2D<Real>>& bases,
                        const MultidimArray<Real>& spectrum ) const;

    void updateRow(MDRowVec& row, const ReferenceMetadata& data) const;

    template<typename F>
    void processRowsInParallel( const MetaDataVec& md, F&& func, size_t nThreads, 
                                size_t start = 0, size_t count = std::numeric_limits<size_t>::max() );

    static void getProjectionSizes(const std::vector<Matrix2D<Real>>& bases, std::vector<size_t>& result);
    static size_t getBatchSize(size_t memorySize, size_t imageProjSize, size_t nTransform);

    static std::vector<typename BandShiftFilters::Shift> computeTranslationFiltersRectangle(size_t nx, 
                                                                                            size_t ny, 
                                                                                            size_t nTranslations,
                                                                                            double maxShift );
    static std::vector<typename BandShiftFilters::Shift> computeTranslationFiltersSunflower(size_t nx, 
                                                                                            size_t ny, 
                                                                                            size_t nTranslations,
                                                                                            double maxShift );

    static void project(const std::vector<Matrix2D<Real>>& bases,
                        const std::vector<MultidimArray<Complex>>& bands,
                        std::vector<Matrix1D<Real>>& projections );

    static void composeComplex( const MultidimArray<Real>& re, 
                                const MultidimArray<Real>& im, 
                                MultidimArray<Complex>& result );

    static void aliasComplexElements(MultidimArray<Complex>& x, Matrix1D<Real>& result);

    static double standardizeAngle(double angle);

};

}

#endif