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

    addParamsLine("   -i <md_file>                    : Metadata file with the experimental images");
    addParamsLine("   -r <md_file>                    : Metadata file with the reference images");
    addParamsLine("   -o <md_file>                    : Resulting metadata file with the aligned images");
    addParamsLine("   [--thr <N=-1>]                  : Maximal number of the processing CPU threads");
    addParamsLine("   [--angDistance <a=10>]          : Angular distance");
    addParamsLine("   [--odir <outputDir=\".\">]      : Output directory");
    addParamsLine("   [--keepBestN <N=1>]             : For each image, store N best alignments to references. N must be smaller than no. of references");
    addParamsLine("   [--allowInputSwap]              : Allow swapping reference and experimental images");
    addParamsLine("   [--useWeightInsteadOfCC]        : Select the best reference using weight, instead of CC");
    addParamsLine("   [--oUpdatedRefs <baseName=\"\">]: Update references using assigned experimental images. Store result here");
}

template<typename T>
void AProgAlignSignificant<T>::readParams() {
    m_imagesToAlign.fn = getParam("-i");
    m_referenceImages.fn = getParam("-r");
    m_fnOut = std::string(getParam("--odir")) + "/" + std::string(getParam("-o"));
    m_angDistance = getDoubleParam("--angDistance");
    m_noOfBestToKeep = getIntParam("--keepBestN");
    m_allowDataSwap = checkParam("--allowInputSwap");
    m_useWeightInsteadOfCC = checkParam("--useWeightInsteadOfCC");
    m_updateHelper.doUpdate = checkParam("--oUpdatedRefs");
    if (m_updateHelper.doUpdate) {
        FileName base = std::string(getParam("--odir")) + "/" + std::string(getParam("--oUpdatedRefs"));
        m_updateHelper.fnStk = base + ".stk";
        m_updateHelper.fnXmd = base + ".xmd";
    }

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
    std::cout << "Best references kept        : " << m_noOfBestToKeep << "\n";
    if (m_updateHelper.doUpdate) {
    std::cout << "Update references (to file) : " << m_updateHelper.fnXmd << "\n";
    }
    std::cout.flush();
}

template<typename T>
void AProgAlignSignificant<T>::load(DataHelper &h) {
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
    h.dims = dimsCropped;

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

    // make sure that the files are well-defined
    bool isValid = md.containsLabel(MDL_IMAGE)
        && md.containsLabel(MDL_ANGLE_ROT) && md.containsLabel(MDL_ANGLE_TILT);
    if ( ! isValid) {
        REPORT_ERROR(ERR_MD, h.fn + ": at least one of the following label is missing: MDL_IMAGE, MDL_ANGLE_ROT, MDL_ANGLE_TILT");
    }

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
}

template<typename T>
void AProgAlignSignificant<T>::check() const {
    if ( ! m_referenceImages.dims.equalExceptNPadded(m_imagesToAlign.dims)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Dimensions of the images to align and reference images do not match");
    }
    if (m_noOfBestToKeep > m_referenceImages.dims.n()) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "--keepBestN is higher than number of references");
    }
    if (m_referenceImages.dims.n() <= 1) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "We need at least two references");
    }
}

template<typename T>
template<bool IS_ESTIMATION_TRANSPOSED>
void AProgAlignSignificant<T>::computeWeights(
        const std::vector<AlignmentEstimation> &est) {
    const size_t noOfRefs = m_referenceImages.dims.n();
    m_weights.resize(noOfRefs);

    // for all references
    for (size_t r = 0; r < noOfRefs; ++r) {
        computeWeightsAndSave<IS_ESTIMATION_TRANSPOSED>(est, r);
    }
}

template<typename T>
template<bool IS_ESTIMATION_TRANSPOSED>
void AProgAlignSignificant<T>::computeWeightsAndSave(
        const std::vector<AlignmentEstimation> &est,
        size_t refIndex) {
    const size_t noOfRefs = m_referenceImages.dims.n();
    const size_t noOfSignals = m_imagesToAlign.dims.n();

    // compute angle between two reference orientation
    auto getAngle = [&](size_t index) {
        return Euler_distanceBetweenAngleSets(
                m_referenceImages.rots.at(refIndex),
                m_referenceImages.tilts.at(refIndex),
                0.f,
                m_referenceImages.rots.at(index),
                m_referenceImages.tilts.at(index),
                0.f,
                true);
    };

    // find out which references are sufficiently similar
    size_t count = 0;
    auto mask = std::vector<bool>(noOfRefs, false);
    for (size_t r = 0; r < noOfRefs; ++r) {
        if ((refIndex == r)
            || (getAngle(r) <= m_angDistance)) {
            mask.at(r) = true;
            count++;
        }
    }

    // allocate necessary memory
    auto figsOfMerit = std::vector<WeightCompHelper>();
    figsOfMerit.reserve(count * noOfSignals);

    // for all similar references
    for (size_t r = 0; r < noOfRefs; ++r) {
        if (mask.at(r)) {
            // get figure of merit of all signals
            for (size_t s = 0; s < noOfSignals; ++s) {
                if (IS_ESTIMATION_TRANSPOSED) {
                    figsOfMerit.emplace_back(est.at(s).figuresOfMerit.at(r), r, s);
                } else {
                    figsOfMerit.emplace_back(est.at(r).figuresOfMerit.at(s), r, s);
                }
            }
        }
    }
    computeWeightsAndSave(figsOfMerit, refIndex);
}

template<typename T>
void AProgAlignSignificant<T>::computeWeightsAndSave(
        std::vector<WeightCompHelper> &figsOfMerit,
        size_t refIndex) {
    const size_t noOfSignals = m_imagesToAlign.dims.n();
    auto &weights = m_weights.at(refIndex);
    weights = std::vector<float>(noOfSignals, 0); // zero weight by default
    const size_t noOfNumbers = figsOfMerit.size();

    // sort ascending using figure of merit
    std::sort(figsOfMerit.begin(), figsOfMerit.end(),
            [](const WeightCompHelper &l, const WeightCompHelper &r) {
        return l.merit < r.merit;
    });
    auto invMaxMerit = 1.f / figsOfMerit.back().merit;

    // set weight for all images
    for (size_t c = 0; c < noOfNumbers; ++c) {
        const auto &tmp = figsOfMerit.at(c);
        if (tmp.refIndex != refIndex) {
            continue; // current record is for different reference
        }
        // cumulative density function - probability of having smaller value then the rest
        float cdf = c / (float)(noOfNumbers - 1); // <0..1> // won't work if we have just one reference
        float merit = tmp.merit;
        if (merit > 0.f) {
            weights.at(tmp.imgIndex) = merit * invMaxMerit * cdf;
        }
    }
}

template<typename T>
void AProgAlignSignificant<T>::fillRow(MDRow &row,
        const Matrix2D<float> &pose,
        size_t refIndex,
        double weight) {
    // get orientation
    bool flip;
    float scale;
    float shiftX;
    float shiftY;
    float psi;
    transformationMatrix2Parameters2D(
            pose.inv(), // we want to store inverse transform
            flip, scale,
            shiftX, shiftY,
            psi);
    // FIXME DS add check of max shift / rotation
    row.setValue(MDL_ENABLED, 1);
    row.setValue(MDL_ANGLE_ROT, (double)m_referenceImages.rots.at(refIndex));
    row.setValue(MDL_ANGLE_TILT, (double)m_referenceImages.tilts.at(refIndex));
    // save both weight and weight significant, so that we can keep track of result of this
    // program, even after some other program re-weights the particle
    row.setValue(MDL_WEIGHT_SIGNIFICANT, weight);
    row.setValue(MDL_WEIGHT, weight);
    row.setValue(MDL_ANGLE_PSI, (double)psi);
    row.setValue(MDL_SHIFT_X, (double)-shiftX); // store negative translation
    row.setValue(MDL_SHIFT_Y, (double)-shiftY); // store negative translation
    row.setValue(MDL_FLIP, flip);
    assert(std::numeric_limits<int>::max() >= refIndex);
    row.setValue(MDL_REF, (int)refIndex);
}

template<typename T>
void AProgAlignSignificant<T>::fillRow(MDRow &row,
        const Matrix2D<float> &pose,
        size_t refIndex,
        double weight, double maxVote) {
    fillRow(row, pose, refIndex, weight);
    row.setValue(MDL_MAXCC, (double)maxVote);
}

template<typename T>
void AProgAlignSignificant<T>::extractMax(
        std::vector<float> &data,
        size_t &pos, float &val) {
    using namespace ExtremaFinder;
    float p = 0;
    float v = 0;
    SingleExtremaFinder<T>::sFindMax(CPU(), Dimensions(data.size()), data.data(), &p, &v);
    pos = std::round(p);
    val = data.at(pos);
    data.at(pos) = std::numeric_limits<float>::lowest();
}

template<typename T>
template<bool IS_ESTIMATION_TRANSPOSED, bool USE_WEIGHT>
void AProgAlignSignificant<T>::computeAssignment(
        const std::vector<AlignmentEstimation> &est) {
    const size_t noOfRefs = m_referenceImages.dims.n();
    auto accessor = [&](size_t image, size_t reference) {
        if (USE_WEIGHT) {
            return m_weights.at(reference).at(image);
        } else if (IS_ESTIMATION_TRANSPOSED) {
            return est.at(image).figuresOfMerit.at(reference);
        } else {
            return est.at(reference).figuresOfMerit.at(image);
        }
    };

    const size_t noOfImages = m_imagesToAlign.dims.n();
    m_assignments.reserve(noOfImages * m_noOfBestToKeep);
    for (size_t i = 0; i < noOfImages; ++i) {
        // collect voting from all references
        auto votes = std::vector<float>();
        votes.reserve(noOfRefs);
        for (size_t r = 0; r < noOfRefs; ++r) {
            votes.emplace_back(accessor(i, r));
        }
        // for all references that we want to store, starting from the best matching one
        for (size_t nthBest = 0; nthBest < m_noOfBestToKeep; ++nthBest) {
            size_t refIndex;
            float val;
            // get the max vote
            extractMax(votes, refIndex, val);
            if (val <= 0) {
                continue; // skip saving the particles which have non-positive figure of merit to the reference
            }
            const auto &p = IS_ESTIMATION_TRANSPOSED
                    ? (est.at(i).poses.at(refIndex).inv())
                    : (est.at(refIndex).poses.at(i));
            m_assignments.emplace_back(refIndex, i,
                    m_weights.at(refIndex).at(i), val,
                    p);
        }
    }
}

template<typename T>
template<bool USE_WEIGHT>
void AProgAlignSignificant<T>::storeAlignedImages() {
    auto &md = m_imagesToAlign.md;
    auto result = MetaData();

    std::sort(m_assignments.begin(), m_assignments.end(),
            [](const Assignment &l, const Assignment &r) {
        return (l.imgIndex != r.imgIndex)
                ? (l.imgIndex < r.imgIndex) // sort by image index asc
                : (USE_WEIGHT // then by voting criteria dest
                  ? (l.weight > r.weight)
                  :(l.merit > r.merit));
    });

    MDRow row;
    size_t i = 0;
    FOR_ALL_OBJECTS_IN_METADATA(md) {
        // get the original row from the input metadata
        md.getRow(row, __iter.objId);
        auto maxVote = m_assignments.at(i).merit;
        // for all references that we want to store, starting from the best matching one
        for (size_t nthBest = 0; nthBest < m_noOfBestToKeep; ++nthBest) {
            const auto &a = m_assignments.at(i);
            fillRow(row, a.pose, a.refIndex, a.weight, maxVote);
            result.addRow(row);
            i++;
        }
    }
    result.write(m_fnOut);
}

template<typename T>
void AProgAlignSignificant<T>::updateSettings() {
    m_settings.refDims = m_referenceImages.dims;
    m_settings.otherDims = m_imagesToAlign.dims;
}

template<typename T>
void AProgAlignSignificant<T>::saveRefStk() {
    const Dimensions &dims = m_referenceImages.dims;
    const auto &fn = m_updateHelper.fnStk;
    checkLogDelete(fn);
    for (size_t n = 0; n < dims.n(); ++n ) {
        const size_t indexInStk = n + 1; // within stk file, index images from one (1)
        FileName name;
        name.compose(indexInStk, fn);
        size_t offset = n * dims.sizeSingle();
        MultidimArray<T> wrapper(1, 1, dims.y(), dims.x(), m_referenceImages.data.get() + offset);
        auto img = Image<T>(wrapper);
        img.write(name, n, true, WRITE_APPEND);
    }
}

template<typename T>
void AProgAlignSignificant<T>::saveRefXmd() {
    const auto &fn = m_updateHelper.fnXmd;
    checkLogDelete(fn);
    // write the ref block
    m_updateHelper.refBlock.write("classes@" + fn, MD_APPEND);
    // write the per-ref images blocks
    const size_t noOfBlocks = m_updateHelper.imgBlocks.size();
    unsigned noOfDigits = noOfBlocks > 0 ? log10 ((float) noOfBlocks) + 1 : 1;
    const auto pattern = "class%0" + std::to_string(noOfDigits) + "d_images@%s";
    for (size_t n = 0; n < noOfBlocks; ++n) {
        const auto &md = m_updateHelper.imgBlocks.at(n);
        if (0 == md.size()) {
            continue; // ignore MD for empty references
        }
        auto blockName = formatString(pattern.c_str(), n);
        md.write(blockName + fn, MD_APPEND);
    }
}

template<typename T>
void AProgAlignSignificant<T>::updateRefXmd(size_t zeroBasedIndex, std::vector<Assignment> &images) {
    const size_t indexInStk = zeroBasedIndex + 1; // within stk file, index images from one (1)
    FileName refName;
    auto &refMeta = m_updateHelper.refBlock;
    // name of the reference
    refName.compose(indexInStk, m_updateHelper.fnStk);
    // some info about it
    size_t id = refMeta.addObject();
    assert(std::numeric_limits<int>::max() >= zeroBasedIndex);
    refMeta.setValue(MDL_REF, (int)zeroBasedIndex, id);
    refMeta.setValue(MDL_IMAGE, refName, id);
    refMeta.setValue(MDL_CLASS_COUNT, images.size(), id);

    // create image description block
    std::sort(images.begin(), images.end(), [](const Assignment &l, const Assignment &r) {
       return l.imgIndex < r.imgIndex; // sort by image index
    });
    auto &md = m_updateHelper.imgBlocks.at(zeroBasedIndex);
    MDRow row;
    const size_t noOfImages = images.size();
    for (const auto &a : images) {
        fillRow(row, a.pose, zeroBasedIndex, a.weight);
        md.addRow(row);
    }
}

template<typename T>
void AProgAlignSignificant<T>::checkLogDelete(const FileName &fn) {
    if (fn.exists()) {
        std::cerr << fn << " exists. It will be overwritten.\n";
        fn.deleteFile(); // since we will append, we need to delete original file
    }
}

template<typename T>
void AProgAlignSignificant<T>::updateRefs() {
    if (1 < m_noOfBestToKeep) {
        std::cout << "Each experimental image will contribute to more than one reference image.\n";
    }
    // make sure we start from scratch
    m_updateHelper.imgBlocks.resize(m_referenceImages.dims.n());
    m_updateHelper.refBlock = MetaData();
    // update references. Metadata will be updated on background
    updateRefs(m_referenceImages.data.get(), m_imagesToAlign.data.get(), m_assignments);
    // store result to drive
    saveRefStk();
    saveRefXmd();
}

template<typename T>
void AProgAlignSignificant<T>::run() {
    show();
    m_threadPool.resize(getSettings().cpuThreads);
    // load data
    load(m_imagesToAlign);
    load(m_referenceImages);

    bool hasMoreReferences = m_allowDataSwap
            && (m_referenceImages.dims.n() > m_imagesToAlign.dims.n());
    if (hasMoreReferences) {
        std::cerr << "We are swapping reference images and experimental images. "
                "This will enhance the performance. This might lead to worse results if the experimental "
                "images are not well centered. Use it with care!\n";
        std::swap(m_referenceImages, m_imagesToAlign);
    }

    // for each reference, get alignment of all images
    updateSettings();
    check();
    auto alignment = align(m_referenceImages.data.get(), m_imagesToAlign.data.get());

    // process the alignment and store
    if (hasMoreReferences) {
        std::swap(m_referenceImages, m_imagesToAlign);
        computeWeights<true>(alignment);
        if (m_useWeightInsteadOfCC) {
            computeAssignment<true, true>(alignment);
        } else {
            computeAssignment<true, false>(alignment);
        }
    } else {
        computeWeights<false>(alignment);
        if (m_useWeightInsteadOfCC) {
            computeAssignment<false, true>(alignment);
        } else {
            computeAssignment<false, false>(alignment);
        }
    }
    // at this moment, we can release some memory
    m_weights.clear();
    alignment.clear();

    if (m_useWeightInsteadOfCC) {
        storeAlignedImages<true>();
    } else {
        storeAlignedImages<false>();
    }

    if (m_updateHelper.doUpdate) {
        updateRefs();
    }
}

// explicit instantiation
template class AProgAlignSignificant<float>;

} /* namespace Alignment */

