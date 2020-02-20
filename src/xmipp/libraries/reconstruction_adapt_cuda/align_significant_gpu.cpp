/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
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

#include "align_significant_gpu.h"

namespace Alignment {

template<typename T>
void ProgAlignSignificantGPU<T>::defineParams() {
    AProgAlignSignificant<T>::defineParams();
    this->addParamsLine("  [--device <dev=0>]                 : GPU device to use. 0th by default");
}

template<typename T>
void ProgAlignSignificantGPU<T>::show() const {
    AProgAlignSignificant<T>::show();
    auto gpu = GPU(m_device);
    gpu.set();
    std::cout <<  "Device                      : " << gpu.device() << " (" << gpu.getUUID() << ")" << std::endl;
}

template<typename T>
void ProgAlignSignificantGPU<T>::readParams() {
    AProgAlignSignificant<T>::readParams();
    // read GPU
    m_device = this->getIntParam("--device");
    if (m_device < 0) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Invalid GPU device");
    }
}

template<typename T>
std::vector<AlignmentEstimation> ProgAlignSignificantGPU<T>::align(const T *ref, const T *others) {
    auto s = this->getSettings();

    auto hw = std::vector<HW*>();
    for (size_t i = 0; i < 2; ++i) {
        auto g = new GPU(m_device);
        g->set();
        hw.emplace_back(g);
    }

    auto processDims = s.otherDims.copyForN(std::min(s.otherDims.n(), m_maxBatchSize));

    auto rotEstimator = CudaRotPolarEstimator<T>();
    initRotEstimator(rotEstimator, hw, processDims);
    auto shiftEstimator = CudaShiftCorrEstimator<T>();
    initShiftEstimator(shiftEstimator, hw, processDims);
    CudaBSplineGeoTransformer<T> transformer;
    initTransformer(transformer, hw, processDims);
    CudaCorrelationComputer<T> mc;
    initMeritComputer(mc, hw, processDims);

    auto aligner = IterativeAlignmentEstimator<T>(rotEstimator, shiftEstimator,
            transformer, mc, this->getThreadPool());

    // create local copy of the reference ...
    auto refSize = s.refDims.sizeSingle();
    auto refSizeBytes = refSize * sizeof(T);
    // ... and lock it, so that we can work asynchronously with it
    auto refData = memoryUtils::page_aligned_alloc<T>(refSize, false);
    hw.at(0)->lockMemory(refData, refSizeBytes);

    auto result = std::vector<AlignmentEstimation>();
    for (size_t refId = 0; refId < s.refDims.n(); ++refId) {
        size_t refOffset = refId * refSize;
        // copy reference image to page-aligned memory
        memcpy(refData, ref + refOffset, refSizeBytes);

        if (0 == (refId % 10)) { // FIXME DS remove / replace by proper progress report
            std::cout << "aligning against reference " << refId << "/" << s.refDims.n() << std::endl;
        }

        aligner.loadReference(refData);
        auto est = result.emplace(result.end(), s.otherDims.n());
        for (size_t i = 0; i < s.otherDims.n(); i += processDims.n()) {
            // run on a full-batch subset
            size_t offset = std::min(i, s.otherDims.n() - processDims.n());
            auto tmp = aligner.compute(others + (offset * s.otherDims.sizeSingle()));

            // merge results
            est->figuresOfMerit.insert(est->figuresOfMerit.begin() + offset,
                    tmp.figuresOfMerit.begin(),
                    tmp.figuresOfMerit.end());
            est->poses.insert(est->poses.begin() + offset,
                    tmp.poses.begin(),
                    tmp.poses.end());
        }
    }

    for (auto h : hw) {
        delete h;
    };

    std::cout << "Done" << std::endl;
    return result;
}

template<typename T>
void ProgAlignSignificantGPU<T>::updateRefs(
        T *refs,
        const T *others,
        const std::vector<Assignment> &assignments) {
    const Dimensions &refDims = this->getSettings().refDims;
    const size_t elems = this->getSettings().otherDims.sizeSingle();
    T *workCopy = new T[m_maxBatchSize * refDims.sizeSingle()];
    std::vector<float> workMatrices;
    workMatrices.reserve(9 * m_maxBatchSize);
    std::vector<float> workWeights;
    workWeights.reserve(m_maxBatchSize);
    T *workRef = memoryUtils::page_aligned_alloc<T>(elems, true);

    auto gpu = GPU(m_device);
    gpu.set();
    gpu.pinMemory(workRef, elems * sizeof(T));
    std::vector<HW*> hw{&gpu};

    CudaBSplineGeoTransformer<T> transformer;
    initTransformer(transformer, hw, this->getSettings().otherDims.copyForN(m_maxBatchSize));
    transformer.setSrc(workCopy);
    T norm = 0;

    std::vector<Assignment> workAssignments;
    for (size_t refIndex = 0; refIndex < refDims.n(); ++refIndex) {
        if (0 == (refIndex % 10)) { // FIXME DS remove / replace by proper progress report
            std::cout << "updating reference " << refIndex << "/" << refDims.n() << std::endl;
        }

        // get assignments for this reference
        workAssignments.clear();
        std::copy_if(assignments.begin(), assignments.end(), std::back_inserter(workAssignments),
                [refIndex](const Assignment &a) { return a.refIndex == refIndex; });
        // process assignments in batch
        const size_t noOfAssignments = workAssignments.size();
        auto finalRef = refs + (refIndex * elems);
        if (0 == noOfAssignments) {
            memset(finalRef, 0, elems * sizeof(T)); // clean the result
        }
        for (size_t offset = 0; offset < noOfAssignments; offset += m_maxBatchSize) {
            size_t toProcess = std::min(m_maxBatchSize, noOfAssignments - offset);
            workWeights.clear();
            // copy image data for this batch
            for (size_t i = 0; i < toProcess; ++i) {
                auto &a = workAssignments.at(offset + i);
                workWeights.emplace_back(a.weight);
                memcpy(workCopy + (i * elems), others + (a.imgIndex * elems), elems * sizeof(T));
            }
            // update normalization factor
            norm += std::accumulate(workWeights.begin(), workWeights.end(), 0.f);
            // apply transform
            interpolate(transformer, workCopy, workAssignments, workMatrices, offset, toProcess);
            // sum images
            transformer.sum(workRef, workWeights, toProcess, 1);
            // collect intermediate result (single image per batch)
            if (0 == offset) { // first batch -> copy data
                memcpy(finalRef, workRef, elems * sizeof(T));
            } else { // other batches -> sum data
                for (size_t i = 0; i < elems; ++i) {
                    finalRef[i] += workRef[i];
                }
            }
        }
        // normalize the resulting image
        for (size_t i = 0; i < elems; ++i) {
            finalRef[i] /= norm;
        }
        // store reference
        this->updateRefXmd(refIndex, workAssignments);
    }
    gpu.unpinMemory(workRef);
    delete[] workCopy;
    free(workRef);
}

template<typename T>
void ProgAlignSignificantGPU<T>::interpolate(BSplineGeoTransformer<T> &transformer,
        T *data,
        const std::vector<Assignment> &assignments,
        std::vector<float> &matrices,
        size_t offset,
        size_t toProcess) {
    transformer.setSrc(data);

    matrices.clear();
    auto tmp = Matrix2D<float>(3, 3);
    SPEED_UP_temps0
    for (size_t j = 0; j < m_maxBatchSize; ++j) {
        if (j >= toProcess) {
            tmp.initIdentity();
        } else {
            auto a = assignments.at(offset + j);
            M3x3_INV(tmp, a.pose) // inverse the transformation
        }
        for (int i = 0; i < 9; ++i) {
            matrices.emplace_back(tmp.mdata[i]);
        }
    }
    transformer.interpolate(matrices);
}

template<typename T>
void ProgAlignSignificantGPU<T>::initRotEstimator(CudaRotPolarEstimator<T> &est,
        std::vector<HW*> &hw,
        const Dimensions &dims) {
    // FIXME DS implement properly
    RotationEstimationSetting s;
    size_t batch = m_maxBatchSize;
    auto rotSettings = RotationEstimationSetting();
    s.hw = hw;
    s.type = AlignType::OneToN;
    s.refDims = dims.createSingle();
    s.otherDims = dims;
    s.batch = dims.n();
    s.maxRotDeg = RotationEstimationSetting::getMaxRotation();
    s.firstRing = RotationEstimationSetting::getDefaultFirstRing(dims);
    s.lastRing = RotationEstimationSetting::getDefaultLastRing(dims);
    s.fullCircle = true;
    s.allowTuningOfNumberOfSamples = false; // FIXME DS change to true
    s.allowDataOverwrite = true;
    est.init(s, true);
}

template<typename T>
void ProgAlignSignificantGPU<T>::initTransformer(BSplineGeoTransformer<T> &t,
        std::vector<HW*> &hw,
        const Dimensions &dims) {
    auto s = BSplineTransformSettings<T>();
    s.keepSrcCopy = true;
    s.degree = InterpolationDegree::Linear;
    s.dims = dims;
    s.hw.push_back(hw.at(0));
    s.type = InterpolationType::NToN;
    s.doWrap = false;
    s.defaultVal = (T)0;
    t.init(s, true);
}

template<typename T>
void ProgAlignSignificantGPU<T>::initMeritComputer(AMeritComputer<T> &mc,
        std::vector<HW*> &hw,
        const Dimensions &dims) {
    auto s = MeritSettings();
    s.hw.push_back(hw.at(0));
    s.normalizeResult = true;
    s.otherDims = dims;
    s.refDims = dims.copyForN(1);
    s.type = MeritType::OneToN;
    mc.init(s, true);
}

template<typename T>
void ProgAlignSignificantGPU<T>::initShiftEstimator(CudaShiftCorrEstimator<T> &est,
        std::vector<HW*> &hw,
        const Dimensions &dims) {
    // FIXME DS implement properly
    size_t batch = dims.n();
    size_t maxShift = dims.x() / 4;
    est.init2D(hw,
            AlignType::OneToN,
            FFTSettingsNew<T>(dims, batch),
            maxShift, true, true, true);

}

// explicit instantiation
template class ProgAlignSignificantGPU<float>;

} /* namespace Alignment */
