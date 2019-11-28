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
    addParamsLine("   [--keepBestN <N=1>]         : For each image, store N best alignments to references. N must be smaller than no. of references");
}

template<typename T>
void AProgAlignSignificant<T>::readParams() {
    m_imagesToAlign.fn = getParam("-i");
    m_referenceImages.fn = getParam("-r");
    m_fnOut = std::string(getParam("--odir")) + "/" + std::string(getParam("-o"));
    m_angDistance = getDoubleParam("--angDistance");
    m_noOfBestToKeep = getIntParam("--keepBestN");

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

    std::cout << "Input metadata              : " << m_imagesToAlign.fn << "\n";
    std::cout << "Reference metadata          : " << m_referenceImages.fn <<  "\n";
    std::cout << "Output metadata             : " << m_fnOut <<  "\n";
    std::cout << "Angular distance            : " << m_angDistance <<  "\n";
    std::cout << "Keep N references           : " << m_noOfBestToKeep << "\n";
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
    auto dimsCropped = Dimensions((Xdim / 2) * 2, (Ydim / 2) * 2, Zdim, Ndim);
    bool mustCrop = (dims != dimsCropped);

    // FIXME DS clean up the cropping routine somehow
    h.data = std::unique_ptr<T[]>(new T[dims.size()]);
    auto ptr = h.data.get();
    // routine loading the actual content of the images
    auto routine = [&dims, ptr]
            (int thrId, const FileName &fn, size_t storeIndex) {
        size_t offset = storeIndex * dims.sizeSingle();
        MultidimArray<T> wrapper(1, dims.z(), dims.y(), dims.x(), ptr + offset);
        auto img = Image<T>(wrapper);
        img.read(fn);
    };

    std::vector<Image<T>> tmpImages;
    tmpImages.reserve(m_threadPool.size());
    if (mustCrop) {
        std::cerr << "We need an even input (sizes must be multiple of two). Input will be cropped\n";
        for (size_t t = 0; t < m_threadPool.size(); ++t) {
            tmpImages.emplace_back(Xdim, Ydim);
        }
    }
    // routine loading the actual content of the images
    auto routineCrop = [&dims, &dimsCropped, ptr, &tmpImages]
            (int thrId, const FileName &fn, size_t storeIndex) {
        // load image
        tmpImages.at(thrId).read(fn);
        // copy just the part we're interested in
        const size_t destOffsetN = storeIndex * dimsCropped.sizeSingle();
        for (size_t y = 0; y < dimsCropped.y(); ++y) {
            size_t srcOffsetY = y * dims.x();
            size_t destOffsetY = y * dimsCropped.x();
            memcpy(ptr + destOffsetN + destOffsetY,
                    tmpImages.at(thrId).data.data + srcOffsetY,
                    dimsCropped.x() * sizeof(T));
        }
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
        if (mustCrop) {
            futures.emplace_back(m_threadPool.push(routineCrop, fn, i));
        } else {
            futures.emplace_back(m_threadPool.push(routine, fn, i));
        }
        i++;
    }
    // wait till done
    for (auto &f : futures) {
        f.get();
    }
    return dimsCropped;
}

template<typename T>
void AProgAlignSignificant<T>::check() const {
    if ( ! m_settings.otherDims.equalExceptNPadded(m_settings.refDims)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Dimensions of the images to align and reference images do not match");
    }
    if (m_noOfBestToKeep > m_settings.refDims.n()) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "--keepBestN is higher than number of references");
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
        // cumulative density function - probability of having smaller value then the rest
        float cdf = c / (float)(noOfCorrelations - 1); // <0..1>
        float correlation = tmp.correlation;
        if (correlation <= 0.f) {
            weights.at(tmp.imgIndex) = 0;
        } else {
            weights.at(tmp.imgIndex) = correlation * invMaxCorrelation * cdf;
        }
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
void AProgAlignSignificant<T>::fillRow(MDRow &row,
        const Matrix2D<double> &pose,
        size_t refIndex,
        double weight, double maxCC) {
    // get orientation
    bool flip;
    double scale;
    double shiftX;
    double shiftY;
    double psi;
    transformationMatrix2Parameters2D(
            pose.inv(), // we want to store inverse transform
            flip, scale,
            shiftX, shiftY,
            psi);
    // FIXME DS add check of max shift / rotation
    row.setValue(MDL_ENABLED, 1);
    row.setValue(MDL_MAXCC, (double)maxCC);
    row.setValue(MDL_ANGLE_ROT, (double)m_referenceImages.rots.at(refIndex));
    row.setValue(MDL_ANGLE_TILT, (double)m_referenceImages.tilts.at(refIndex));
    // save both weight and weight significant, so that we can keep track of result of this
    // program, even after some other program re-weights the particle
    row.setValue(MDL_WEIGHT_SIGNIFICANT, weight);
    row.setValue(MDL_WEIGHT, weight);
    row.setValue(MDL_ANGLE_PSI, psi);
    row.setValue(MDL_SHIFT_X, -shiftX); // store negative translation
    row.setValue(MDL_SHIFT_Y, -shiftY); // store negative translation
    row.setValue(MDL_FLIP, flip);
}

template<typename T>
void AProgAlignSignificant<T>::replaceMaxCorrelation(
        std::vector<float> &correlations,
        size_t &pos, double &val) {
    using namespace ExtremaFinder;
    float p = 0;
    float v = 0;
    SingleExtremaFinder<T>::sFindMax(CPU(), Dimensions(correlations.size()), correlations.data(), &p, &v);
    pos = std::round(p);
    val = correlations.at(pos);
    correlations.at(pos) = std::numeric_limits<float>::lowest();
}

template<typename T>
void AProgAlignSignificant<T>::storeAlignedImages(
        const std::vector<AlignmentEstimation> &est) {
    auto &md = m_imagesToAlign.md;
    auto result = MetaData();
    const size_t noOfRefs = m_settings.refDims.n();
    const auto dims = Dimensions(noOfRefs);

    MDRow row;
    size_t i = 0;
    FOR_ALL_OBJECTS_IN_METADATA(md) {
        // get the original row from the input metadata
        md.getRow(row, __iter.objId);
        // collect correlations from all references
        auto cc = std::vector<float>();
        cc.reserve(noOfRefs);
        for (size_t r = 0; r < noOfRefs; ++r) {
            cc.emplace_back(est.at(r).correlations.at(i));
        }
        double maxCC = std::numeric_limits<double>::lowest();
        // for all references that we want to store, starting from the best matching one
        for (size_t nthBest = 0; nthBest < m_noOfBestToKeep; ++nthBest) {
            if (cc.at(nthBest) <= 0) {
                continue; // skip saving the particles which have non-positive correlation to the reference
            }
            size_t refIndex;
            double val;
            // get the weight
            replaceMaxCorrelation(cc, refIndex, val);
            if (0 == nthBest) {
                // set max CC that we found
                maxCC = val;
            }
            // update the row with proper pose info
            fillRow(row,
                    est.at(refIndex).poses.at(i),
                    refIndex,
                    m_weights.at(refIndex).at(i),
                    maxCC); // best cross-correlation
            // store it
            result.addRow(row);
        }
        i++;
    }
    result.write(m_fnOut);
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

