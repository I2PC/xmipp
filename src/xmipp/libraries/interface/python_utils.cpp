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

#include "python_utils.h"

namespace Python {
std::string print_traceback() {
   PyObject* type;
   PyObject* value;
   PyObject* traceback;

   PyErr_Fetch(&type, &value, &traceback);
   PyErr_NormalizeException(&type, &value, &traceback);

   std::string fcn = "";
   fcn += "def get_pretty_traceback(exc_type, exc_value, exc_tb):\n";
   fcn += "    import sys, traceback\n";
   fcn += "    lines = []\n";
   fcn += "    lines = traceback.format_exception(exc_type, exc_value, exc_tb)\n";
   fcn += "    output = '\\n'.join(lines)\n";
   fcn += "    return output\n";

   PyRun_SimpleString(fcn.c_str());
   PyObject* mod = PyImport_ImportModule("__main__");
   PyObject* method = PyObject_GetAttrString(mod, "get_pretty_traceback");
   PyObject* outStr = PyObject_CallObject(method, Py_BuildValue("OOO", type, value, traceback));
   PyObject* str_exc_type = PyObject_Str(outStr); //Now a unicode
   return PyUnicode_AsUTF8(str_exc_type);
}

std::string whichPython() {
    char path[1035];
    FILE *fp = popen("which python", "r");
    if (fp == NULL)
        REPORT_ERROR(ERR_UNCLASSIFIED,"Cannot execute which python");
    if (fgets(path, sizeof(path)-1, fp) == NULL)
        REPORT_ERROR(ERR_UNCLASSIFIED,"Cannot find python");
    pclose(fp);
    return path;
}

void initPython(const std::string &path) {
#if PY_MAJOR_VERSION >= 3
#if PY_MINOR_VERSION > 4
    auto program = Py_DecodeLocale(path.c_str(), NULL);
#else // PY_MINOR_VERSION > 4
    std::vector<wchar_t> tmp(path.begin(), path.end());
    auto program = tmp.data();
#endif // PY_MINOR_VERSION > 4
#else // PY_MAJOR_VERSION >= 3
#error Python version >= 3 expected
#endif // PY_MAJOR_VERSION >= 3
    if (nullptr == program) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Cannot decode the python path\n");
    }
    Py_SetProgramName(program);
    Py_Initialize();
}

void initPython() {
    initPython(whichPython());
}

void initPythonAndNumpy() {
    initPython();
    initNumpy();
}

PyObject *getClassRef(const std::string &moduleName, const std::string &className) {
    PyObject * pName = PyUnicode_FromString(moduleName.c_str());
    PyObject * pModule = PyImport_Import(pName);
    PyObject * pDict = PyModule_GetDict(pModule);
    PyObject * pClass = PyDict_GetItemString(pDict, className.c_str());
    Py_DECREF(pName);
    Py_DECREF(pModule);
    Py_DECREF(pDict);
    return pClass;
}

PyObject * getFunctionRef(const std::string &moduleName, const std::string &funcName) {
    PyObject * pName = PyUnicode_FromString(moduleName.c_str());
    PyObject * pModule = PyImport_Import(pName);
    PyObject * pFunc = PyObject_GetAttrString(pModule, funcName.c_str());
    Py_DECREF(pName);
    Py_DECREF(pModule);
    return pFunc;
}

template<typename T>
PyObject* convertToNumpy(const MultidimArray<T> &array) {
    npy_intp dim[3];
    dim[0]=XSIZE(array);
    dim[1]=YSIZE(array);
    dim[2]=ZSIZE(array);
    if (std::is_same<T, int>::value) {
        return PyArray_SimpleNewFromData(3, dim, NPY_INT, array.data);
    } else if (std::is_same<T, double>::value){
       return PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, array.data);
    } else {
        REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
    }
}

void initNumpy() {
#ifndef NUMPY_IMPORT_ARRAY_RETVAL
#define NUMPY_IMPORT_ARRAY_RETVAL NULL
#endif
    auto init = []() -> std::nullptr_t { // explicit on purpose
        import_array();
    };
    init();
}

// explicit instantiation
template PyObject* convertToNumpy<double>(MultidimArray<double> const&);
template PyObject* convertToNumpy<int>(MultidimArray<int> const&);

} // namespace
