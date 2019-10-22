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

#ifndef APPLICATIONS_TESTS_FUNCTION_TESTS_ALIGNMENT_TEST_UTILS_H_
#define APPLICATIONS_TESTS_FUNCTION_TESTS_ALIGNMENT_TEST_UTILS_H_

#include <random>
#include <vector>
#include "core/multidim_array.h"
#include "data/point2D.h"
#include "data/dimensions.h"

static size_t getMaxShift(const Dimensions &dims) {
    // max shift must be sharply less than half of the size
    return std::min(dims.x() / 2, dims.y() / 2) - 1;
}

static std::vector<Point2D<float>> generateShifts(const Dimensions &dims, size_t maxShift, std::mt19937 &mt) {
    auto maxShiftSq = maxShift * maxShift;

    std::uniform_int_distribution<> dist(0, maxShift);
    auto shifts = std::vector<Point2D<float>>();
    shifts.reserve(dims.n());
    for(size_t n = 0; n < dims.n(); ++n) {
        // generate shifts so that the Euclidean distance is smaller than max shift
        int shiftX = dist(mt);
        int shiftXSq = shiftX * shiftX;
        int maxShiftY = std::floor(sqrt(maxShiftSq - shiftXSq));
        int shiftY = (0 == maxShiftY) ? 0 : dist(mt) % maxShiftY;
        shifts.emplace_back(shiftX, shiftY);
    }
    return shifts;
}

static float getMaxRotation() {
    return 360.f - std::numeric_limits<float>::min();
}

static std::vector<float> generateRotations(const Dimensions& dims, float maxRotation, std::mt19937 &mt)
{
    std::uniform_int_distribution<> distPos(0, dims.x());
    auto rotations = std::vector<float>();
    rotations.reserve(dims.n());
    std::uniform_real_distribution<> distRot(0, maxRotation);
    for (size_t n = 0;n < dims.n();++n){
        rotations.emplace_back(distRot(mt));
    }
    return rotations;
}

template<typename T>
static void addNoise(T *data, const Dimensions &dims, std::mt19937 &mt) {
    std::normal_distribution<T> dist(0., .5);
    for (size_t i = 0; i < dims.size(); ++i) {
        data[i] += dist(mt);
    }
}

template<typename T>
static void drawClockArms(T *result, const Dimensions &dims, size_t xPos, size_t yPos, float rotDegree) {
    size_t yArmSize = (dims.y() - yPos) / 2;
    size_t xArmSize = (dims.x() - xPos) / 3;

    MultidimArray<T> in(dims.y(), dims.x());
    for (size_t y = yPos; y < yPos + yArmSize; ++y) {
        size_t index = y * dims.x() + xPos;
        in.data[index] = 1;
    }

    for (size_t x = xPos; x < xPos + xArmSize; ++x) {
        size_t index = yPos * dims.x() + x;
        in.data[index] = 1;
    }
    MultidimArray<T> out(1, 1, dims.y(), dims.x(), result);
    rotate(3, out, in, rotDegree);
}


#endif /* APPLICATIONS_TESTS_FUNCTION_TESTS_ALIGNMENT_TEST_UTILS_H_ */
