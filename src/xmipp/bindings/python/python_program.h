/***************************************************************************
 *
 * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
 *              Roberto Marabini       (roberto@cnb.csic.es)
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

#ifndef _PYTHON_PROGRAM_H
#define _PYTHON_PROGRAM_H

#include <Python.h>
#include <core/xmipp_program.h>

/***************************************************************/
/*                            Program                         */
/**************************************************************/

#define Program_Check(v) (((v)->ob_type == &ProgramType))
#define Program_Value(v) ((*((ProgramObject*)(v))->program))

/*Program Object*/
typedef struct
{
    PyObject_HEAD
    XmippProgramGeneric * program;
}
ProgramObject;

#define ProgramObject_New() (ProgramObject*)malloc(sizeof(ProgramObject))

/* Destructor */
void Program_dealloc(ProgramObject* self);

/* Constructor */
PyObject *
Program_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);

/* addUsageLine */
PyObject *
Program_addUsageLine(PyObject *obj, PyObject *args, PyObject *kwargs);

/* addExampleLine */
PyObject *
Program_addExampleLine(PyObject *obj, PyObject *args, PyObject *kwargs);

/* addParamsLine */
PyObject *
Program_addParamsLine(PyObject *obj, PyObject *args, PyObject *kwargs);

/* usage */
PyObject *
Program_usage(PyObject *obj, PyObject *args, PyObject *kwargs);

/* endDefinition */
PyObject *
Program_endDefinition(PyObject *obj, PyObject *args, PyObject *kwargs);

/* read */
PyObject *
Program_read(PyObject *obj, PyObject *args, PyObject *kwargs);

/* checkParam */
PyObject *
Program_checkParam(PyObject *obj, PyObject *args, PyObject *kwargs);

/* getParam */
PyObject *
Program_getParam(PyObject *obj, PyObject *args, PyObject *kwargs);

/* getListParam */
PyObject *
Program_getListParam(PyObject *obj, PyObject *args, PyObject *kwargs);

/* Program methods */
extern PyMethodDef Program_methods[];
extern PyTypeObject ProgramType;
#endif
