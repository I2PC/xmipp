/***************************************************************************
 *
 * Authors:    José Luis Vilas Prieto
 *             Oier Lauzirika Zarrabeitia
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

#include "tomo_align_subtomo_pairs.h"

#include <core/transformations.h>
#include <core/geometry.h>
#include <interface/frm.h>

#include <iostream>


ProgAlignSubtomoPairs::ProgAlignSubtomoPairs()
{
    produces_a_metadata = true;
    produces_an_output = true;
}


void ProgAlignSubtomoPairs::readParams()
{
    XmippMetadataProgram::readParams();
    maxShift = getDoubleParam("--maxShift");
    maxFreq = getDoubleParam("--maxFreq");
}

void ProgAlignSubtomoPairs::show()
{
    // TODO
}

void ProgAlignSubtomoPairs::defineParams()
{
    XmippMetadataProgram::defineParams();
    addParamsLine("  [--maxShift <shift=16>] : Maximum shift in pixels");
    addParamsLine("  [--maxFreq <freq=0.25>] : Maximum digital frequency");
}

void ProgAlignSubtomoPairs::preProcess()
{
    Python::initPythonAndNumpy();
    pyFrmFunc = Python::getFunctionRef("sh_alignment.frm", "frm_align");
    pyGeneralWedgeClass = Python::getClassRef("sh_alignment.tompy.filter", "GeneralWedge");
}

void ProgAlignSubtomoPairs::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
    const auto &fnReference = rowIn.getValue<String>(MDL_IMAGE_REF);
    const auto &fnReferenceMultiplicity = rowIn.getValue<String>(MDL_MASK); // TODO decide label
    const auto &fnVolumeMultiplicity = rowIn.getValue<String>(MDL_MASK);

    reference.read(fnReference);
    multiplicity.read(fnReferenceMultiplicity);
    volume.read(fnImg);

	PyObject *pyMultiplicityRef = Python::convertToNumpy(multiplicity());
    PyObject *pyGeneralWedgeArgs = Py_BuildValue("(O)", pyMultiplicityRef);
    PyObject *pyGeneralWedge = PyObject_CallObject(pyGeneralWedgeClass, pyGeneralWedgeArgs);

    double rot, tilt, psi, x, y, z, score;
    Matrix2D<double> matrix;
    alignVolumesFRM(
        pyFrmFunc,
        reference(), pyGeneralWedge, 
        volume(), pyGeneralWedge, 
        rot, tilt, psi, 
        x, y, z,
        score,
        matrix,
        maxShift,
        maxFreq,
        nullptr
    );
	Py_DECREF(pyMultiplicityRef);
	Py_DECREF(pyGeneralWedgeArgs);
	Py_DECREF(pyGeneralWedge);

    // Use shifts after rotation
    x = MAT_ELEM(matrix, 0, 3);
    y = MAT_ELEM(matrix, 1, 3);
    z = MAT_ELEM(matrix, 2, 3);

    rowOut.setValue(MDL_ANGLE_ROT, rot);
    rowOut.setValue(MDL_ANGLE_TILT, tilt);
    rowOut.setValue(MDL_ANGLE_PSI, psi);
    rowOut.setValue(MDL_SHIFT_X, x);
    rowOut.setValue(MDL_SHIFT_Y, y);
    rowOut.setValue(MDL_SHIFT_Z, z);
    rowOut.setValue(MDL_CORRELATION_IDX, score);
}
