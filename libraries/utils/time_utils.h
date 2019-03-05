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

#ifndef LIBRARIES_UTILS_TIME_UTILS_H_
#define LIBRARIES_UTILS_TIME_UTILS_H_

#include <chrono>
#include <utility>

namespace timeUtils
{

/**
 * Function to measure execution time of some code
 * @param ToDuration 'resolution' of the time, e.g. std::chrono::seconds
 * @param F funcion to run
 * @param Args arguments of the function
 */
template<typename ToDuration, typename F, typename ...Args>
auto measureTime(F &&func, Args &&...args) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
    return duration_cast<ToDuration>(high_resolution_clock::now() - start).count();
}

/**
 * Helper function, measuring time of some code in seconds
 * Typically, you can wrap the block you want to measure within lambda:
 * timeUtils::measureTimeS([&]{
 *  code_to_measure
 * });
 * @param F funcion to run
 * @param Args arguments of the function
 */
template<typename F, typename... Args>
auto measureTimeS(F &&func, Args &&...args) {
    return measureTime<std::chrono::seconds>(func, args...);
}

/**
 * Helper function, measuring time of some code in milliseconds
 * Typically, you can wrap the block you want to measure within lambda:
 * timeUtils::measureTimeMs([&]{
 *  code_to_measure
 * });
 * @param F funcion to run
 * @param Args arguments of the function
 */
template<typename F, typename... Args>
auto measureTimeMs(F &&func, Args &&...args) {
    return measureTime<std::chrono::milliseconds>(func, args...);
}

/**
 * Helper function, measuring time of some code in seconds and reporting
 * it to std::cout.
 * Typically, you can wrap the block you want to measure within lambda:
 * timeUtils::reportTimeS("myBlock"[&]{
 *  code_to_measure
 * });
 * @param funName some name the block (for identification purposes)
 * @param F funcion to run
 * @param Args arguments of the function
 */
template<typename F, typename... Args>
void reportTimeS(const std::string &funName, F &&func, Args &&...args) {
    std::cout << funName << " took " << measureTime<std::chrono::seconds>(func, args...) << " s\n";
}

/**
 * Helper function, measuring time of some code in milliseconds and reporting
 * it to std::cout.
 * Typically, you can wrap the block you want to measure within lambda:
 * timeUtils::reportTimeMs("myBlock"[&]{
 *  code_to_measure
 * });
 * @param funName some name the block (for identification purposes)
 * @param F funcion to run
 * @param Args arguments of the function
 */
template<typename F, typename... Args>
void reportTimeMs(const std::string &funName, F &&func, Args &&...args) {
    std::cout << funName << " took " << measureTime<std::chrono::milliseconds>(func, args...) << " ms\n";
}

} // timeUtils

#endif /* LIBRARIES_UTILS_TIME_UTILS_H_ */
