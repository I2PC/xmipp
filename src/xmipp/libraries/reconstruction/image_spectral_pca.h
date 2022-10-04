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

#ifndef _PROG_IMAGE_SPECTRAL_PCA
#define _PROG_IMAGE_SPECTRAL_PCA

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

/**@defgroup SpectralPCA
   @ingroup ReconsLibrary */
//@{

class ProgImageSpectralPca : public XmippProgram
{
public:
    virtual void readParams() override;
    virtual void defineParams() override;
    virtual void show() const override;
    virtual void run() override;

    FileName        fnImages;
    FileName        fnBandMap;
    FileName        fnOroot;
    double          pcaTraining;
    double          pcaEfficiency;
    double          pcaInitialBatch;
    size_t          nThreads;

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
        void flatten(   const MultidimArray<std::complex<double>>& spectrum,
                        std::vector<Matrix1D<double>>& data ) const;
        void flatten(   const MultidimArray<std::complex<double>>& spectrum,
                        size_t band,
                        Matrix1D<double>& data ) const;
        void unflatten( const std::vector<Matrix1D<double>>& data,
                        MultidimArray<std::complex<double>>& spectrum ) const;
        void unflatten( const Matrix1D<double>& data,
                        size_t band,
                        MultidimArray<std::complex<double>>& spectrum ) const;
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

        void reset(     const std::vector<size_t>& sizes,
                        double initialCompression, 
                        double initialBatch );

        size_t getBandCount() const;
        
        size_t getBandSize(size_t i) const;
        size_t getProjectionSize(size_t i) const;
        void getMean(size_t i, Matrix1D<double>& v) const;
        void getVariance(size_t i, Matrix1D<double>& v) const;
        void getProjectionVariance(size_t i, Matrix1D<double>& v) const;
        void getBasis(size_t i, Matrix2D<double>& b) const;
        double getError(size_t i) const;

        void getErrorFunction(size_t i, Matrix1D<double>& errFn);

        void learn(const std::vector<Matrix1D<double>>& bands);
        void learnConcurrent(const std::vector<Matrix1D<double>>& bands);

        void finalize();
        void equalizeError(double precision);

    private:
        std::vector<SgaNnOnlinePca<double>> m_bandPcas;
        std::vector<std::mutex> m_bandMutex;

        static void calculateErrorFunction( Matrix1D<double>& lambdas, 
                                            double totalVariance );

        static size_t calculateRequiredComponents(  const Matrix1D<double>& errFn,
                                                    double precision );


    };



    BandMap m_bandMap;
    SpectralPca m_pca;

    void readBandMap();
    void trainPca();
    void generateOutput();


    template<typename F>
    void processRowsInParallel(const MetaDataVec& md, F&& func, size_t nThreads);

    static void selectSubset(MetaDataVec& md, size_t n);
};

#endif