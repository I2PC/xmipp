/***************************************************************************
 *
 * Authors:     David Strelak (davidstrelak@gmail.com)
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

#include "aalign_significant.h"

namespace Alignment {

template<typename T>
void AProgAlignSignificant<T>::defineParams() {
    addUsageLine("Find alignment of the experimental images in respect to a set of references");

    addParamsLine("   -i <md_file>                : Metadata file with the experimental images");
    addParamsLine("   -r <md_file>                : Metadata file with the reference images");
    addParamsLine("   -o <md_file>                : Resulting metadata file with the aligned images");
    addParamsLine("   [--thr <N=-1>]              : Maximal number of the processing CPU threads");
    addParamsLine("   [--angDistance <a=10>]      : Angular distance");
    addParamsLine("   [--odir <outputDir=\".\">]  : Output directory");
}

template<typename T>
void AProgAlignSignificant<T>::readParams() {
    m_imagesToAlign.fn = getParam("-i");
    m_referenceImages.fn = getParam("-r");
    m_fnOut = std::string(getParam("--odir")) + "/" + std::string(getParam("-o"));
    m_angDistance=getDoubleParam("--angDistance");

    int threads = getIntParam("--thr");
    if (-1 == threads) {
        m_settings.cpuThreads = CPU::findCores();
    } else {
        m_settings.cpuThreads = threads;
    }
}

template<typename T>
void AProgAlignSignificant<T>::show() const {
    if (verbose < 1) return;

    std::cout << "Input metadata              : "  << m_imagesToAlign.fn << "\n";
    std::cout << "Reference metadata          : "  << m_referenceImages.fn <<  "\n";
    std::cout << "Output metadata             : "  << m_fnOut <<  "\n";
    std::cout << "Angular distance            : "  << m_angDistance <<  "\n";
    std::cout.flush();
}

template<typename T>
Dimensions AProgAlignSignificant<T>::load(DataHelper &h) {
    auto &md = h.md;
    md.read(h.fn);
    md.removeDisabled();

    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
    size_t Ndim;
    getImageSize(h.fn, Xdim, Ydim, Zdim, Ndim);
    Ndim = md.size(); // FIXME DS why we  didn't get right Ndim from the previous call?
    auto dims = Dimensions(Xdim, Ydim, Zdim, Ndim);

    h.data = std::unique_ptr<T[]>(new T[dims.size()]);
    auto ptr = h.data.get();
    // routine loading the actual content of the images
    auto routine = [Zdim, Ydim, Xdim, ptr]
            (int thrId, const FileName &fn, size_t storeIndex) {
        size_t offset = storeIndex * Xdim * Ydim * Zdim;
        MultidimArray<T> wrapper(1, Zdim, Ydim, Xdim, ptr + offset);
        auto img = Image<T>(wrapper);
        img.read(fn);
    };
    // load all images in parallel
    auto futures = std::vector<std::future<void>>();
    futures.reserve(Ndim);
    h.rots.reserve(Ndim);
    h.tilts.reserve(Ndim);
    size_t i = 0;
    FOR_ALL_OBJECTS_IN_METADATA(md) {
        FileName fn;
        float rot;
        float tilt;
        md.getValue(MDL_IMAGE, fn, __iter.objId);
        md.getValue(MDL_ANGLE_ROT, rot,__iter.objId);
        md.getValue(MDL_ANGLE_TILT, tilt,__iter.objId);
        h.rots.emplace_back(rot);
        h.tilts.emplace_back(tilt);
        futures.emplace_back(m_threadPool.push(routine, fn, i));
        i++;
    }
    // wait till done
    for (auto &f : futures) {
        f.get();
    }
    return dims;
}

template<typename T>
void AProgAlignSignificant<T>::check() const {
    if ( ! m_settings.otherDims.equalExceptNPadded(m_settings.refDims)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Dimensions of the images to align and reference images do not match");
    }
}

template<typename T>
void AProgAlignSignificant<T>::computeWeightsAndSave(
        const std::vector<AlignmentEstimation> &est,
        size_t refIndex) {
    const size_t noOfRefs = m_settings.refDims.n();
    const size_t noOfSignals = m_settings.otherDims.n();
    auto correlations = std::vector<WeightCompHelper>();
    // and all references that are similar
    for (size_t r = 0; r < noOfRefs; ++r) {
        if (refIndex != r) {
            auto ang=Euler_distanceBetweenAngleSets(
                    m_referenceImages.rots.at(refIndex),
                    m_referenceImages.tilts.at(refIndex),
                    0.0,
                    m_referenceImages.rots.at(r),
                    m_referenceImages.tilts.at(r),
                    0.0,
                    true);
            if (ang > m_angDistance) {
                continue;
            }
        }
        // get correlations of all signals
        correlations.reserve(correlations.size() + noOfSignals);
        auto &cc = est.at(r).correlations;
        for (size_t s = 0; s < noOfSignals; ++s) {
            correlations.emplace_back(cc.at(s), r, s);
        }
    }

    computeWeightsAndSave(correlations, refIndex);
}

template<typename T>
void AProgAlignSignificant<T>::computeWeightsAndSave(
        std::vector<WeightCompHelper> &correlations,
        size_t refIndex) {
    const size_t noOfSignals = m_settings.otherDims.n();
    auto weights = std::vector<float>(noOfSignals);
    const size_t noOfCorrelations = correlations.size();

    // sort ascending using correlation
    std::sort(correlations.begin(), correlations.end(),
            [](WeightCompHelper &l, WeightCompHelper &r) {
        return l.correlation < r.correlation;
    });
    auto invMaxCorrelation = 1.f / correlations.back().correlation;

    // set weight for all images
    for (size_t c = 0; c < noOfCorrelations; ++c) {
        auto &tmp = correlations.at(c);
        if (tmp.refIndex != refIndex) {
            continue; // current record is for different reference
        }
        // FIXME DS check previously set weights, if any
        // cumulative density function - probability of having smaller value then the rest
        float cdf = c / (float)(noOfCorrelations - 1); // <0..1>
        weights.at(tmp.imgIndex) = tmp.correlation * invMaxCorrelation * cdf;
    }
    // store result
    m_weights.at(refIndex) = weights;
}

template<typename T>
void AProgAlignSignificant<T>::computeWeights(const std::vector<AlignmentEstimation> &est) {
    const size_t noOfRefs = m_settings.refDims.n();
    const size_t noOfSignals = m_settings.otherDims.n();
    m_weights.resize(noOfRefs);

    auto workload = [&](int threadId, size_t refIndex) {
        computeWeightsAndSave(est, refIndex);
    };

    auto futures = std::vector<std::future<void>>();
    futures.reserve(noOfRefs);
    // for all references
    for (size_t r = 0; r < noOfRefs; ++r) {
        futures.emplace_back(m_threadPool.push(workload, r));
    }
    // wait till done
    for (auto &f : futures) {
        f.get();
    }
}

template<typename T>
void AProgAlignSignificant<T>::storeAlignedImages(
        const std::vector<AlignmentEstimation> &est) {
    using namespace ExtremaFinder;
    auto &md = m_imagesToAlign.md;
    const size_t noOfRefs = m_settings.refDims.n();
    const auto dims = Dimensions(noOfRefs);

    auto bestEst = AlignmentEstimation(m_settings.otherDims.n());
    size_t i = 0;
    FOR_ALL_OBJECTS_IN_METADATA(md) {
        // find the best matching reference
        auto cc = std::vector<float>();
        cc.reserve(noOfRefs);
        for (size_t r = 0; r < noOfRefs; ++r) {
            cc.emplace_back(est.at(r).correlations.at(i));
        }
        float pos = 0;
        float maxCC = 0;
        SingleExtremaFinder<T>::sFindMax(CPU(), dims, cc.data(), &pos, &maxCC);
        size_t refIndex = std::round(pos);
        // get orientation
        bool flip;
        double scale;
        double shiftX;
        double shiftY;
        double psi;
        auto t = est.at(refIndex).poses.at(i);
        bestEst.poses.at(i) = t;
        transformationMatrix2Parameters2D(
                t.inv(), // we want to store inverse transform
                flip, scale,
                shiftX, shiftY,
                psi);
        // FIXME DS add check of max shift / rotation
        size_t rowId = __iter.objId;
        md.setValue(MDL_ENABLED, 1, rowId);
        md.setValue(MDL_MAXCC, (double)maxCC, rowId);
        md.setValue(MDL_ANGLE_ROT, (double)m_referenceImages.rots.at(refIndex), rowId);
        md.setValue(MDL_ANGLE_TILT, (double)m_referenceImages.tilts.at(refIndex), rowId);
        md.setValue(MDL_WEIGHT_SIGNIFICANT, (double)m_weights.at(refIndex).at(i), rowId);
        md.setValue(MDL_ANGLE_PSI, psi, rowId);
        md.setValue(MDL_SHIFT_X, -shiftX, rowId); // store negative translation
        md.setValue(MDL_SHIFT_Y, -shiftY, rowId); // store negative translation
        md.setValue(MDL_FLIP, flip, rowId);
        i++;
    }
    md.write(m_fnOut);
}

template<typename T>
void AProgAlignSignificant<T>::run() {
    show();
    m_threadPool.resize(m_settings.cpuThreads);
    m_settings.otherDims = load(m_imagesToAlign);
    m_settings.refDims = load(m_referenceImages);
    check();

    auto alignment = align(m_referenceImages.data.get(), m_imagesToAlign.data.get());
    computeWeights(alignment);

    storeAlignedImages(alignment);
}

// explicit instantiation
template class AProgAlignSignificant<float>;

} /* namespace Alignment */

