/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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
/*****************************************************************************/
/* INTERACTION WITH Fast Rotational Matching                                 */
/*****************************************************************************/

#ifndef _XMIPP_FRM_HH
#define _XMIPP_FRM_HH

#include "python_utils.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // this code is NumPy 1.8 compliant (i.e. we don't need API deprecated in 1.7)

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <core/multidim_array.h>

/** Align two volumes using FRM.
 * The first argument is the pointer to the FRM python function. You may obtain it with
 * getPointerToPythonFRMFunction()
 *
 * Imask is a mask in Fourier space for I
 * Maxshift is in pixels.
 * MaxFreq is in digital frequency.
 *
 * A is the transformation matrix that needs to be applied on I to fit Iref.
 *
 * If apply is set, then I is substituted with the aligned volume.
 */
void alignVolumesFRM(PyObject *pFunc, const MultidimArray<double> &Iref, MultidimArray<double> &I,
		PyObject *Imask,
		double &rot, double &tilt, double &psi, double &x, double &y, double &z, double &score,
		Matrix2D<double> &A,
		int maxshift=10, double maxFreq=0.25, const MultidimArray<int> *mask=NULL);

//@}
#endif
