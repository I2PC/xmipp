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

#ifndef _XMIPPMODULE_H
#define _XMIPPMODULE_H

#include "Python.h"
#include <reconstruction/ctf_estimate_from_micrograph.h>
#include <data/fourier_projection.h>

#include "python_fourierprojector.h"

extern PyObject * PyXmippError;


/***************************************************************/
/*                   Some specific utility functions           */
/***************************************************************/

/* calculate enhanced psd and return preview*/
PyObject *
xmipp_fastEstimateEnhancedPSD(PyObject *obj, PyObject *args, PyObject *kwargs);

/* calculate enhanced psd and return preview
* used for protocol preprocess_particles*/
PyObject *
xmipp_bandPassFilter(PyObject *obj, PyObject *args, PyObject *kwargs);

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_gaussianFilter(PyObject *obj, PyObject *args, PyObject *kwargs);

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_realGaussianFilter(PyObject *obj, PyObject *args, PyObject *kwargs);

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_badPixelFilter(PyObject *obj, PyObject *args, PyObject *kwargs);

PyObject *
xmipp_errorBetween2CTFs(PyObject *obj, PyObject *args, PyObject *kwargs);

PyObject *
xmipp_errorMaxFreqCTFs(PyObject *obj, PyObject *args, PyObject *kwargs);

PyObject *
xmipp_errorMaxFreqCTFs2D(PyObject *obj, PyObject *args, PyObject *kwargs);

/* convert an input image to psd */
PyObject *
Image_convertPSD(PyObject *obj, PyObject *args, PyObject *kwargs);

/* I2aligned=align(I1,I2) */
PyObject *
Image_align(PyObject *obj, PyObject *args, PyObject *kwargs);

/* Apply CTF to this image */
PyObject *
Image_applyCTF(PyObject *obj, PyObject *args, PyObject *kwargs);

#endif
