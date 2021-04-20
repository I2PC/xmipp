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

#ifndef PYTHON_UTILS_H_
#define PYTHON_UTILS_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // this code is NumPy 1.8 compliant (i.e. we don't need API deprecated in 1.7)

#include <Python.h>
#include <numpy/ndarrayobject.h> // FIXME DS move to numpy utils?
#include <string>
#include "core/xmipp_error.h"
#include "core/multidim_array.h"

namespace Python {
/**
 * This function outputs traceback of the last exception to the standard output
 */
std::string print_traceback();

/**
 * Returns path to the default python intrepreter
 * Equivalent to 'which python'
 */
std::string whichPython();

/**
 * Initialize default Python
 * @path path will point to the python which has been initialized
 */
void initPython(const std::string &path);

/**
 * Initialize default Python
 */
void initPython();

/**
 * Initialize default Python and Numpy
 */
void initPythonAndNumpy();

/**
 * Get a borrowed reference to a Python class
 * This reference should be release using Py_DECREF() or Py_XDECREF()
 * @param moduleName name of the module (script file name)
 * @param className name of the class within the module
 * @return a borrowed reference to the class on success or nullptr on error
 */
PyObject *getClassRef(const std::string &moduleName, const std::string &className);

/**
 * Get a new reference to a Python function
 * This reference should be release using Py_DECREF() or Py_XDECREF()
 * @param moduleName name of the module (script file name)
 * @param funcName name of the class within the module
 * @return a new reference to the function on success or nullptr on error
 */
PyObject * getFunctionRef(const std::string &moduleName, const std::string &funcName);

/**
 * Create an Numpy reference array wrapper around data.
 * This reference should be release using Py_DECREF() or Py_XDECREF()
 * @param array to be wrapped
 * @return a reference to the wrapper
 */
template<typename T>
PyObject* convertToNumpy(const MultidimArray<T> &array);

/**
 * Initialize Numpy arrays. Needs to be called from each compile unit
 * @see https://stackoverflow.com/questions/31971185/segfault-when-import-array-not-in-same-translation-unit
 */
void initNumpy();

} // namespace
#endif /* PYTHON_UTILS_H_ */
