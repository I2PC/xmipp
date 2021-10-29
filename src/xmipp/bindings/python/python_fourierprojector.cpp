/***************************************************************************
 *
 * Authors:     Airen Zaldivar (azaldivar@cnb.csic.es)
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

#include "python_fourierprojector.h"
#include "python_image.h"
#include "data/fourier_projection.h"

/***************************************************************/
/*                            FourierProjector                         */
/**************************************************************/

 /* FourierProjector methods */
 PyMethodDef FourierProjector_methods[] =
 {
    { "projectVolume", (PyCFunction) FourierProjector_projectVolume,
      METH_VARARGS, "projects Volume" },
    { nullptr } /* Sentinel */
 };//FourierProjector_methods

 /*FourierProjector Type */
 PyTypeObject FourierProjectorType =
 {
     PyObject_HEAD_INIT(nullptr)
     "xmipp.FourierProjector", /*tp_name*/
     sizeof(FourierProjectorObject), /*tp_basicsize*/
     0, /*tp_itemsize*/
     (destructor)FourierProjector_dealloc, /*tp_dealloc*/
     nullptr, /*tp_print*/
     nullptr, /*tp_getattr*/
     nullptr, /*tp_setattr*/
     nullptr, /*tp_compare*/
     nullptr, /*tp_repr*/
     nullptr, /*tp_as_number*/
     nullptr, /*tp_as_sequence*/
     nullptr, /*tp_as_mapping*/
     nullptr, /*tp_hash */
     nullptr, /*tp_call*/
     nullptr, /*tp_str*/
     nullptr, /*tp_getattro*/
     nullptr, /*tp_setattro*/
     nullptr, /*tp_as_buffer*/
     Py_TPFLAGS_DEFAULT, /*tp_flags*/
     "Python wrapper to Xmipp FourierProjector class",/* tp_doc */
     nullptr, /* tp_traverse */
     nullptr, /* tp_clear */
     nullptr, /* tp_richcompare */
     nullptr, /* tp_weaklistoffset */
     nullptr, /* tp_iter */
     nullptr, /* tp_iternext */
     FourierProjector_methods, /* tp_methods */
     nullptr, /* tp_members */
     nullptr, /* tp_getset */
     nullptr, /* tp_base */
     nullptr, /* tp_dict */
     nullptr, /* tp_descr_get */
     nullptr, /* tp_descr_set */
     nullptr, /* tp_dictoffset */
     nullptr, /* tp_init */
     nullptr, /* tp_alloc */
     FourierProjector_new, /* tp_new */
 };//FourierProjectorType


 /* Constructor */
 PyObject *
 FourierProjector_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
 {

     FourierProjectorObject *self = (FourierProjectorObject*)type->tp_alloc(type, 0);
     XMIPP_TRY

     if (self != nullptr)
     {
         PyObject *image = nullptr;
         double padding_factor, max_freq, spline_degree;
         padding_factor = 2;
         max_freq = 0.5;
         spline_degree = 3;
         if (PyArg_ParseTuple(args, "O|ddd", &image, &padding_factor, &max_freq, &spline_degree))
         {
        	 Image_Value(image).getDimensions(self->dims);
        	 MultidimArray<double> *pdata;
        	 Image_Value(image).data->getMultidimArrayPointer(pdata);
        	 pdata->setXmippOrigin();
        	 self->fourier_projector = new FourierProjector(*(pdata), padding_factor, max_freq, spline_degree);

         }
     }
     XMIPP_CATCH

     return (PyObject *)self;
 }

 /* Destructor */
  void FourierProjector_dealloc(FourierProjectorObject* self)
 {
     delete self->fourier_projector;
     //delete self->dims;
     Py_TYPE(self)->tp_free((PyObject*)self);
 }

/* projectVolume */

PyObject * FourierProjector_projectVolume(PyObject * obj, PyObject *args, PyObject *kwargs)
{
      FourierProjectorObject *self = (FourierProjectorObject*) obj;
      double rot, tilt, psi;
      PyObject *projection_image = nullptr;
      if (self != nullptr && PyArg_ParseTuple(args, "O|ddd", &projection_image, &rot, &tilt, &psi))
      {
          try
          {
        	  Projection P;
              projectVolume(FourierProjector_Value(self), P, self->dims.xdim, self->dims.ydim, rot, tilt, psi);
              Image_Value(projection_image).data->setImage(MULTIDIM_ARRAY(P));
          }
          catch (XmippError &xe)
          {
              PyErr_SetString(PyXmippError, xe.msg.c_str());
          }
      }
      Py_RETURN_NONE;
}






