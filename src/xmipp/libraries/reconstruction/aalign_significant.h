/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *              David Strelak (davidstrelak@gmail.com)
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

#ifndef AALIGN_SIGNIFICANT
#define AALIGN_SIGNIFICANT

#include <core/xmipp_program.h>
#include "data/dimensions.h"
#include <CTPL/ctpl_stl.h>
#include "data/cpu.h"
#include "data/alignment_estimation.h"
#include "core/metadata_extension.h"
#include "core/geometry.h"
#include "reconstruction/single_extrema_finder.h"
#include "core/transformations.h"

namespace Alignment {

template<typename T>
class AProgAlignSignificant : public XmippProgram
{
public:
    virtual void readParams() override;
    virtual void defineParams() override;
    virtual void show() const override;
    virtual void run() override;

protected:
    struct Settings {
        Dimensions refDims = Dimensions(0); // there should be less reference images than experimental images
        Dimensions otherDims = Dimensions(0); // there should be more reference images than experimental images

        unsigned cpuThreads;
    };

    virtual void check() const;
    const Settings &getSettings() {
        return m_settings;
    }
    virtual std::vector<AlignmentEstimation> align(const T *ref, const T *others) = 0;

    ctpl::thread_pool &getThreadPool() {
        return m_threadPool;
    }

private:
    struct DataHelper {
        FileName fn;
        MetaData md;
        Dimensions dims = Dimensions(0);
        std::vector<float> rots;
        std::vector<float> tilts;
        std::unique_ptr<T[]> data;
    };

    struct WeightCompHelper {
        WeightCompHelper(float c, size_t ref, size_t img) :
            merit(c), refIndex(ref), imgIndex(img) {};
        size_t refIndex;
        size_t imgIndex;
        float merit;
    };

    DataHelper m_imagesToAlign;
    DataHelper m_referenceImages;
    FileName m_fnOut;
    FileName m_fnOutUpdatedRefs;
    float m_angDistance;
    Settings m_settings;
    size_t m_noOfBestToKeep;
    bool m_allowDataSwap;
    bool m_useWeightInsteadOfCC;

    std::vector<std::vector<float>> m_weights;

    ctpl::thread_pool m_threadPool;

    void load(DataHelper &h);
    Dimensions crop(const Dimensions &d, DataHelper &h);
    template<bool IS_ESTIMATION_TRANSPOSED>
    void computeWeights(const std::vector<AlignmentEstimation> &est);
    template<bool IS_ESTIMATION_TRANSPOSED>
    void computeWeightsAndSave(
            const std::vector<AlignmentEstimation> &est,
            size_t refIndex);
    void computeWeightsAndSave(
            std::vector<WeightCompHelper> &figsOfMerit,
            size_t refIndex);
    template<bool IS_ESTIMATION_TRANSPOSED, bool USE_WEIGHT>
    void storeAlignedImages(
            const std::vector<AlignmentEstimation> &est);
    void fillRow(MDRow &row,
            const Matrix2D<float> &pose,
            size_t refIndex,
            double weight, double maxVote);
    void extractMax(
            std::vector<float> &data,
            size_t &pos, double &val);

    void updateSettings();
};

} /* namespace Alignment */

#endif /* AALIGN_SIGNIFICANT */
