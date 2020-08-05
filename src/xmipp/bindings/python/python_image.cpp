/***************************************************************************
 *
 * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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

#include "xmippmodule.h"
#include <data/ctf.h>
#include <data/filters.h>
#include "data/dimensions.h"
#include "reconstruction/psd_estimator.h"

/***************************************************************/
/*                            Image                         */
/**************************************************************/

/* Destructor */
void Image_dealloc(ImageObject* self)
{

    delete self->image;
    Py_TYPE(self)->tp_free((PyObject*)self);
}//function Image_dealloc


/* Image methods that behave like numbers */
PyNumberMethods Image_NumberMethods =
    {
        Image_add, //binaryfunc  nb_add;
        Image_subtract, //binaryfunc  nb_subtract;
        Image_multiply, //binaryfunc  nb_multiply;
        // TODO: division no longer works,  I do not know why
        // may be related with the fact that division in python3
        // does not work as it use to do in python 2
        Image_true_divide, //binaryfunc  nb_divide;
        0, //binaryfunc  nb_remainder;
        0, //binaryfunc  nb_divmod;
        0, //ternaryfunc nb_power;
        0, //unaryfunc nb_negative;
        0, //unaryfunc nb_positive;
        0, //unaryfunc nb_absolute;
        0, //inquiry nb_nonzero;       /* Used by PyObject_IsTrue */
        /* Added in release 2.0 */
        Image_iadd, //binaryfunc  nb_inplace_add;
        Image_isubtract, //binaryfunc  nb_inplace_subtract;
        Image_imultiply, //binaryfunc  nb_inplace_multiply;
        Image_idivide, //binaryfunc  nb_inplace_divide;
    };//Image_NumberMethods

/* Image methods */
PyMethodDef Image_methods[] =
    {
        { "applyGeo", (PyCFunction) Image_applyGeo, METH_VARARGS,
          "Apply geometry in referring metadata to an image" },
        { "read", (PyCFunction) Image_read, METH_VARARGS,
          "Read image from disk" },
        { "readPreview", (PyCFunction) Image_readPreview, METH_VARARGS,
          "Read image preview" },
        { "readPreviewSmooth", (PyCFunction) Image_readPreviewSmooth, METH_VARARGS,
          "Read image preview, downsample in Fourier space" },
        { "equal", (PyCFunction) Image_equal, METH_VARARGS,
          "return true if both images are equal up to precision" },
        { "convertPSD", (PyCFunction) Image_convertPSD, METH_VARARGS,
          "Convert to PSD: center FFT and use logarithm" },
        { "readApplyGeo", (PyCFunction) Image_readApplyGeo, METH_VARARGS,
          "Read image from disk applying geometry in referring metadata" },
        { "write", (PyCFunction) Image_write, METH_VARARGS,
          "Write image to disk" },
        { "getData", (PyCFunction) Image_getData, METH_VARARGS,
          "Return NumPy array from image data" },

        { "setData", (PyCFunction) Image_setData, METH_VARARGS,
          "Copy NumPy array to image data" },
        { "getPixel", (PyCFunction) Image_getPixel, METH_VARARGS,
          "Return a pixel value" },
        { "initConstant", (PyCFunction) Image_initConstant, METH_VARARGS,
          "Initialize to value" },
        { "mirrorY", (PyCFunction) Image_mirrorY, METH_VARARGS,
          "mirror image so up goes down" },
        { "applyTransforMatScipion", (PyCFunction) Image_applyTransforMatScipion, METH_VARARGS,
          "apply transformationMatrix as defined by EMX and used by Scipion" },
		{ "applyWarpAffine", (PyCFunction) Image_warpAffine, METH_VARARGS,
		  "apply a warp affine transformation equivalent to cv2.warpaffine and used by Scipion" },
        { "initRandom", (PyCFunction) Image_initRandom, METH_VARARGS,
          "Initialize to random value" },
        { "resize", (PyCFunction) Image_resize, METH_VARARGS,
          "Resize the image dimensions" },
        { "scale", (PyCFunction) Image_scale, METH_VARARGS,
          "Scale the image" },
        { "reslice", (PyCFunction) Image_reslice, METH_VARARGS,
            "Change the slices order in a volume" },
		{ "writeSlices", (PyCFunction) Image_writeSlices, METH_VARARGS,
		  "Write slices as separate files (rootname, extension, axis). Axis is 'X', 'Y' or 'Z'" },
        { "patch", (PyCFunction) Image_patch, METH_VARARGS,
          "Make a patch with other image" },
        { "getDataType", (PyCFunction) Image_getDataType, METH_VARARGS,
          "get DataType for Image" },
        { "setDataType", (PyCFunction) Image_setDataType, METH_VARARGS,
          "set DataType for Image" },
        { "convert2DataType", (PyCFunction) Image_convert2DataType, METH_VARARGS,
          "Convert datatype of Image" },
        { "setPixel", (PyCFunction) Image_setPixel, METH_VARARGS,
          "Set the value of some pixel" },
        { "getDimensions", (PyCFunction) Image_getDimensions, METH_VARARGS,
          "Return image dimensions as a tuple" },
        { "resetOrigin", (PyCFunction) Image_resetOrigin, METH_VARARGS,
          "set origin to 0 0 0" },
        { "getEulerAngles", (PyCFunction) Image_getEulerAngles, METH_VARARGS,
          "Return euler angles as a tuple" },
        { "getMainHeaderValue", (PyCFunction) Image_getMainHeaderValue, METH_VARARGS,
          "Return value from MainHeader" },
        { "setMainHeaderValue", (PyCFunction) Image_setMainHeaderValue, METH_VARARGS,
          "Set value to MainHeader" },
        { "getHeaderValue", (PyCFunction) Image_getHeaderValue, METH_VARARGS,
          "Return value from Header" },
        { "setHeaderValue", (PyCFunction) Image_setHeaderValue, METH_VARARGS,
          "Set value to Header" },
        { "computeStats", (PyCFunction) Image_computeStats, METH_VARARGS,
          "Compute image statistics, return mean, dev, min and max" },
        { "computePSD", (PyCFunction) Image_computePSD, METH_VARARGS,
            "Compute PSD for current image, returns the PSD image" },
        { "adjustAndSubtract", (PyCFunction) Image_adjustAndSubtract, METH_VARARGS,
          "I1=I1-adjusted(I2)" },
		{ "correlation", (PyCFunction) Image_correlation, METH_VARARGS,
		  "correlation(I1,I2)" },
        { "inplaceAdd", (PyCFunction) Image_inplaceAdd, METH_VARARGS,
          "Add another image to self (does not create another Image instance)" },
        { "inplaceSubtract", (PyCFunction) Image_inplaceSubtract, METH_VARARGS,
          "Subtract another image from self (does not create another Image instance)" },
        { "inplaceMultiply", (PyCFunction) Image_inplaceMultiply, METH_VARARGS,
          "Multiply image by a constant or another image (does not create a new Image instance)" },
        { "inplaceDivide", (PyCFunction) Image_inplaceDivide, METH_VARARGS,
          "Divide image by a constant (does not create another Image instance)" },
        { "applyWarpAffine", (PyCFunction) Image_warpAffine, METH_VARARGS,
          "apply a warp affine transformation equivalent to cv2.warpaffine and used by Scipion" },


        { NULL } /* Sentinel */
    };//Image_methods


/*Image Type */
PyTypeObject ImageType = {
                             PyObject_HEAD_INIT(NULL)
                             "xmipp.Image", /*tp_name*/
                             sizeof(ImageObject), /*tp_basicsize*/
                             0, /*tp_itemsize*/
                             (destructor)Image_dealloc, /*tp_dealloc*/
                             0, /*tp_print*/
                             0, /*tp_getattr*/
                             0, /*tp_setattr*/
                             0, /*tp_compare*/
                             Image_repr, /*tp_repr*/
                             &Image_NumberMethods, /*tp_as_number*/
                             0, /*tp_as_sequence*/
                             0, /*tp_as_mapping*/
                             0, /*tp_hash */
                             0, /*tp_call*/
                             0, /*tp_str*/
                             0, /*tp_getattro*/
                             0, /*tp_setattro*/
                             0, /*tp_as_buffer*/
                             Py_TPFLAGS_DEFAULT, /*tp_flags*/
                             "Python wrapper to Xmipp Image class",/* tp_doc */
                             0, /* tp_traverse */
                             0, /* tp_clear */
                             Image_RichCompareBool, /* tp_richcompare */
                             0, /* tp_weaklistoffset */
                             0, /* tp_iter */
                             0, /* tp_iternext */
                             Image_methods, /* tp_methods */
                             0, /* tp_members */
                             0, /* tp_getset */
                             0, /* tp_base */
                             0, /* tp_dict */
                             0, /* tp_descr_get */
                             0, /* tp_descr_set */
                             0, /* tp_dictoffset */
                             0, /* tp_init */
                             0, /* tp_alloc */
                             Image_new, /* tp_new */
                         };//ImageType

/* Constructor */
PyObject *
Image_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        PyObject *input = NULL;

        if (PyArg_ParseTuple(args, "|O", &input))
        {
            if (input != NULL)
            {
                try
                {
                    PyObject *pyStr;
                    // If the input object is a tuple, consider it (index, filename)
                    if (PyTuple_Check(input))
                    {
                      // Get the index and filename from the Python tuple object
                      size_t index = PyLong_AsSsize_t(PyTuple_GetItem(input, 0));
                      PyObject* repr = PyObject_Str(PyTuple_GetItem(input, 1));
                      const char * filename = PyUnicode_AsUTF8(repr);
                      // Now read using both of index and filename
                      self->image = new ImageGeneric();
                      self->image->read(filename, DATA, index);

                    }
                    else if ((pyStr = PyObject_Str(input)) != NULL)
                    {
                        self->image = new ImageGeneric(PyUnicode_AsUTF8(pyStr));
                        //todo: add copy constructor
                    }
                    else
                    {
                        PyErr_SetString(PyExc_TypeError,
                                        "Image_new: Expected string, FileName or tuple as first argument");
                        return NULL;
                    }
                }
                catch (XmippError &xe)
                {
                    PyErr_SetString(PyXmippError, xe.msg.c_str());
                    return NULL;
                }
            }
            else
                self->image = new ImageGeneric();
        }
        else
            return NULL;
    }
    return (PyObject *)self;
}//function Image_new

/* Image string representation */
PyObject *
Image_repr(PyObject * obj)
{
    ImageObject *self = (ImageObject*) obj;
    String s;
    self->image->toString(s);
    return PyUnicode_FromString(s.c_str());
}//function Image_repr

/* Image compare function */
PyObject*
Image_RichCompareBool(PyObject * obj, PyObject * obj2, int opid)
{
    if (obj != NULL && obj2 != NULL)
    {
        try
        {
            if (opid == Py_EQ)
                {
                    if (Image_Value(obj) == Image_Value(obj2))
                        Py_RETURN_TRUE;
                    else
                       Py_RETURN_FALSE;
                }
            else if  (opid == Py_NE)
                    {
                        if (Image_Value(obj) == Image_Value(obj2))
                            Py_RETURN_FALSE;
                        else
                           Py_RETURN_TRUE;
                    }
            else
                return Py_NotImplemented;

        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_compare

/* Compare two images up to a precision */
PyObject *
Image_equal(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    XMIPP_TRY
    ImageObject *self = (ImageObject*) obj;
    if (self != NULL)
    {
        double precision = 1.e-3;
        PyObject *image2 = NULL;
        if (PyArg_ParseTuple(args, "O|d", &image2, &precision))
        {
            try
            {
                if (Image_Value(obj).equal(Image_Value(image2), precision))
                    Py_RETURN_TRUE;
                else
                    Py_RETURN_FALSE;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.msg.c_str());
            }
        }
    }
    XMIPP_CATCH
    return NULL;
}//function Image_equal

/* write */
PyObject *
Image_write(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    if (self != NULL)
    {
        PyObject *input = NULL;
        if (PyArg_ParseTuple(args, "O", &input))
        {
            try
            {
              PyObject *pyStr;
              // If the input object is a tuple, consider it (index, filename)
              if (PyTuple_Check(input))
              {
                // Get the index and filename from the Python tuple object
                size_t index = PyLong_AsSsize_t(PyTuple_GetItem(input, 0));
                PyObject* repr = PyObject_Str(PyTuple_GetItem(input, 1));
                const char * filename = PyUnicode_AsUTF8(repr);
                // Now read using both of index and filename
                bool isStack = (index > 0);
                WriteMode writeMode = isStack ? WRITE_REPLACE : WRITE_OVERWRITE;
                self->image->write(filename, index, isStack, writeMode);

                Py_RETURN_NONE;
              }
              if ((pyStr = PyObject_Str(input)) != NULL)
              {
                  const char * filename = PyUnicode_AsUTF8(pyStr);
                  self->image->write(filename);
                  Py_RETURN_NONE;
              }
              else
              {
                  PyErr_SetString(PyExc_TypeError,
                                  "Image_write: Expected an string or FileName as first argument");
              }
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.msg.c_str());
            }
        }
    }
    return NULL;
}//function Image_write

/* read */
PyObject *
Image_read(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    if (self != NULL)
    {
        int datamode = DATA;
        PyObject *input = NULL;
        if (PyArg_ParseTuple(args, "O|i", &input, &datamode))
        {

            try
            {
              PyObject *pyStr, *pyStr1;
              // If the input object is a tuple, consider it (index, filename)
              if (PyTuple_Check(input))
              {
                // Get the index and filename from the Python tuple object
                size_t index = PyLong_AsSsize_t(PyTuple_GetItem(input, 0));
                PyObject* repr = PyObject_Str(PyTuple_GetItem(input, 1));
                const char *filename = PyUnicode_AsUTF8(repr);
                // Now read using both of index and filename
                self->image->read(filename,(DataMode)datamode, index);
                Py_RETURN_NONE;
              }
              else if ((pyStr = PyObject_Str(input)) != NULL)
              {
                  self->image->read(PyUnicode_AsUTF8(pyStr),(DataMode)datamode);
                  Py_RETURN_NONE;
              }
              else
              {
                  PyErr_SetString(PyExc_TypeError,
                                  "Image_write: Expected an string or FileName as first argument");
              }
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.msg.c_str());
            }
        }
    }
    return NULL;
}//function Image_read

void readImagePreview(ImageGeneric *ig, FileName fn, size_t xdim, int slice)
{
    ig->read(fn, HEADER);
    ImageInfo ii;
    ig->getInfo(ii);
    if (xdim > 0 || xdim != ii.adim.xdim)
      ig->readPreview(fn, xdim, 0, slice);
    else
      ig->read(fn);
}

/* read preview*/
PyObject *
Image_readPreview(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    if (self != NULL)
    {
        PyObject *input = NULL;
        int x = 0;
        int slice = CENTRAL_SLICE;

        if (PyArg_ParseTuple(args, "O|ii", &input, &x, &slice))
        {
            try
            {
              PyObject *pyStr;
              if ((pyStr = PyObject_Str(input)) != NULL)
              {
                  readImagePreview(self->image, PyUnicode_AsUTF8(pyStr), x, slice);
                  Py_RETURN_NONE;
              }
              else
              {
                  PyErr_SetString(PyExc_TypeError,
                                  "Image_readPreview: Expected string or FileName as first argument");
              }
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.msg.c_str());
            }
        }
        else
          PyErr_SetString(PyXmippError,"Error with input arguments, expected filename and optional size");
    }
    return NULL;
}//function Image_readPreview

/* read preview, downsampling in Fourier space*/
PyObject *
Image_readPreviewSmooth(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    if (self != NULL)
    {
        PyObject *input = NULL;
        int x;
        if (PyArg_ParseTuple(args, "Oi", &input, &x))
        {
            try
            {

              PyObject *pyStr;
              if ((pyStr = PyObject_Str(input)) != NULL)
              {
                self->image->readPreviewSmooth(PyUnicode_AsUTF8(pyStr), x);
                Py_RETURN_NONE;
              }
              else
              {
                  PyErr_SetString(PyExc_TypeError,
                                  "Image_readPreviewSmooth: Expected string or FileName as first argument");
              }
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.msg.c_str());
            }
        }
    }
    return NULL;
}//function Image_readPreviewSmooth


NPY_TYPES datatype2NpyType(DataType dt)
{
    switch (dt)
    {
    case DT_Float:
        return NPY_FLOAT;
    case DT_Double:
        return NPY_DOUBLE;
    case DT_Int:
        return NPY_INT;
    case DT_UInt:
        return NPY_UINT;
    case DT_Short:
        return NPY_SHORT;
    case DT_UShort:
        return NPY_USHORT;
    case DT_SChar:
        return NPY_BYTE;
    case DT_UChar:
        return NPY_UBYTE;
    case DT_Bool:
        return NPY_BOOL;
    case DT_CFloat:
        return NPY_CFLOAT;
    case DT_CDouble:
        return NPY_CDOUBLE;
    default:
        return NPY_NOTYPE;
    }
}

DataType npyType2Datatype(int npy)
{
    switch (npy)
    {
    case NPY_FLOAT:
        return DT_Float;
    case NPY_DOUBLE:
        return DT_Double;
    case NPY_INT:
        return DT_Int;
    case NPY_UINT:
        return DT_UInt;
    case NPY_SHORT:
        return DT_Short;
    case NPY_USHORT:
        return DT_UShort;
    case NPY_BYTE:
        return DT_SChar;
    case NPY_UBYTE:
        return DT_UChar;
    case NPY_BOOL:
        return DT_Bool;
    case NPY_CFLOAT:
        return DT_CFloat;
    case NPY_CDOUBLE:
        return DT_CDouble;
    default:
        return DT_Unknown;
    }
}

/** Just to statically call the function import_array
 * required to work with NumPy arrays
 */
class NumpyStaticImport
{
public:
    NumpyStaticImport()
    {
        _import_array();
    }
}
;//class NumpyStaticImport

//Declare a variable to call the constructor
static NumpyStaticImport _npyImport;

/* getData */
PyObject *
Image_getData(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    if (self != NULL)
    {
        try
        {
            ArrayDim adim;
            ImageGeneric & image = Image_Value(self);
            DataType dt = image.getDatatype();
            int nd = image.image->mdaBase->getDim();
            MULTIDIM_ARRAY_GENERIC(image).getDimensions(adim);
            npy_intp dims[4];
            dims[0] = adim.ndim;
            dims[1] = adim.zdim;
            dims[2] = adim.ydim;
            dims[3] = adim.xdim;
            //Get the pointer to data
            void *mymem = image().getArrayPointer();
            NPY_TYPES type = datatype2NpyType(dt);
            //dims pointer is shifted if ndim or zdim are 1
            PyArrayObject * arr = (PyArrayObject*) PyArray_SimpleNew(nd, dims+4-nd, type);
            void * data = PyArray_DATA(arr);
            memcpy(data, mymem, adim.nzyxdim * gettypesize(dt));

            return (PyObject*)arr;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_getData



/* setData */
PyObject *
Image_setData(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    PyArrayObject * arr = NULL;

    if (self != NULL && PyArg_ParseTuple(args, "O", &arr))
    {
        try
        {
            ImageGeneric & image = Image_Value(self);
            DataType dt = npyType2Datatype(PyArray_TYPE(arr));
            int nd = PyArray_NDIM(arr);
            //Setup of image
            image.setDatatype(dt);
            ArrayDim adim;
            adim.ndim = (nd == 4 ) ? PyArray_DIM(arr, 0) : 1;
            adim.zdim = (nd > 2 ) ? PyArray_DIM(arr, nd - 3) : 1;
            adim.ydim = PyArray_DIM(arr, nd - 2);
            adim.xdim = PyArray_DIM(arr, nd - 1);

            MULTIDIM_ARRAY_GENERIC(image).resize(adim, false);
            void *mymem = image().getArrayPointer();
            void * data = PyArray_DATA(arr);
            memcpy(mymem, data, adim.nzyxdim * gettypesize(dt));
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_setData

/* getPixel */
PyObject *
Image_getPixel(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    int i, j, k, n;
    double value;

    if (self != NULL && PyArg_ParseTuple(args, "iiii", &n, &k, &i, &j ))
    {
        try
        {
            self->image->data->im->resetOrigin();
            if (n==0 && k==0)
                 value = self->image->getPixel(i, j);
            else
                 value = self->image->getPixel(n, k , i, j);
            return PyFloat_FromDouble(value);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_getPixel

/* setPixel */
PyObject *
Image_setPixel(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    int i, j, k, n;
    double value = -1;

    if (self != NULL && PyArg_ParseTuple(args, "iiiid", &n, &k, &i, &j, &value))
    {
        try
        {
            if (n==0 && k==0)
                self->image->setPixel(i, j, value);
            else
                self->image->setPixel(n, k, i, j, value);

            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_setPixel

/* initConstant */
PyObject *
Image_initConstant(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    double value = -1;

    if (self != NULL && PyArg_ParseTuple(args, "d", &value))
    {
        try
        {
            self->image->initConstant(value);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_initConstant

/* initConstant */
PyObject *
Image_mirrorY(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    try
    {
        self->image->mirrorY();
        Py_RETURN_NONE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return NULL;
}//function Image_initConstant

/* initRandom */
PyObject *
Image_initRandom(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    double op1 = 0;
    double op2 = 1;
    RandomMode mode = RND_UNIFORM;
    int pyMode = -1;

    if (self != NULL && PyArg_ParseTuple(args, "|ddi", &op1, &op2, &pyMode))
    {
        try
        {
            if (pyMode != -1)
                mode = (RandomMode) pyMode;

            self->image->initRandom(op1, op2, mode);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_initRandom

/* Resize Image */
PyObject *
Image_resize(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    int xDim = 0, yDim = 0, zDim = 1;
    size_t nDim = 1;

    if (self != NULL && PyArg_ParseTuple(args, "ii|in", &xDim, &yDim, &zDim, &nDim))
    {
        try
        {
            self->image->resize(xDim, yDim, zDim, nDim, false); // TODO: Take care of copy mode if needed
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_resize

/* Scale Image */
PyObject *
Image_scale(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    int xDim = 0, yDim = 0, zDim = 1;
    int forceVolume=0;
    if (self != NULL && PyArg_ParseTuple(args, "ii|ii", &xDim, &yDim, &zDim, &forceVolume))
    {
        try
        {
            MultidimArrayGeneric& I=MULTIDIM_ARRAY_GENERIC(Image_Value(self));
            I.setXmippOrigin();
            size_t xdim, ydim, zdim, ndim;
            I.getDimensions(xdim, ydim, zdim, ndim);
            if (forceVolume && zdim==1 && ndim>1)
               I.setDimensions(xdim,ydim,ndim,1);
            selfScaleToSize(LINEAR, I, xDim, yDim, zDim);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_scale

/* Change the slices order in a volume */
PyObject *
Image_reslice(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    int axis = VIEW_Z_NEG;

    if (self != NULL && PyArg_ParseTuple(args, "i", &axis))
    {
        try
        {
            self->image->reslice((AxisView) axis);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_reslice

/* Change the slices order in a volume */
PyObject *
Image_writeSlices(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    if (self != NULL)
    {
        PyObject *oRootname = NULL;
        PyObject *oExt = NULL;
        PyObject *oAxis = NULL;
        if (PyArg_ParseTuple(args, "OOO", &oRootname, &oExt, &oAxis))
        {
            try
            {
                // Get the index and filename from the Python tuple object
                const char * rootname = PyUnicode_AsUTF8(PyObject_Str(oRootname));
                const char * ext = PyUnicode_AsUTF8(PyObject_Str(oExt));
                const char * axis = PyUnicode_AsUTF8(PyObject_Str(oAxis));
                // Now read using both of index and filename

                ImageGeneric Iout;
                Iout.setDatatype(self->image->getDatatype());

				size_t xdim, ydim, zdim, ndim;
				MULTIDIM_ARRAY_GENERIC(*(self->image)).getDimensions(xdim, ydim, zdim, ndim);
				size_t N=zdim;
				char caxis=axis[0];
                if (caxis=='Y')
                	N=ydim;
                else if (caxis=='X')
                	N=xdim;
                for (size_t n=0; n<N; n++)
                {
                	MULTIDIM_ARRAY_GENERIC(*(self->image)).getSlice(n,&MULTIDIM_ARRAY_GENERIC(Iout),caxis);
                	Iout.write(formatString("%s_%04d.%s",rootname,n,ext));
                }
                Py_RETURN_NONE;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.msg.c_str());
            }
        }
    }
    return NULL;

}//function writeSlices

/* Patch Image */
PyObject *
Image_patch(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    PyObject *patch;
    int x = 0, y = 0;

    if (self != NULL && PyArg_ParseTuple(args, "Oii", &patch, &x, &y))
    {
        try
        {
            MULTIDIM_ARRAY_GENERIC(Image_Value(self)).patch(MULTIDIM_ARRAY_GENERIC(Image_Value(patch)), x, y);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_patch

/* Get Data Type */
PyObject *
Image_getDataType(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    if (self != NULL)
    {
        try
        {
            ImageGeneric & image = Image_Value(self);
            DataType dt = image.getDatatype();
            return  Py_BuildValue("i", dt);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_getDataType

/* Set Data Type */
PyObject *
Image_setDataType(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    int datatype;

    if (self != NULL && PyArg_ParseTuple(args, "i", &datatype))
    {
        try
        {
            self->image->setDatatype((DataType)datatype);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_setDataType

/* Set Data Type */
PyObject *
Image_convert2DataType(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    int datatype;
    int castMode=CW_CONVERT;

    if (self != NULL && PyArg_ParseTuple(args, "i|i", &datatype, &castMode))
    {
        try
        {
            self->image->convert2Datatype((DataType)datatype, (CastWriteMode)castMode);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_setDataType

/* Return image dimensions as a tuple */
PyObject *
Image_getDimensions(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    if (self != NULL)
    {
        try
        {
            size_t xdim, ydim, zdim, ndim;
            MULTIDIM_ARRAY_GENERIC(*self->image).getDimensions(xdim, ydim, zdim, ndim);
            return Py_BuildValue("iiik", xdim, ydim, zdim, ndim);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_getDimensions

/* reset origin */
PyObject *
Image_resetOrigin(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    if (self != NULL)
    {
        try
        {
            self->image->data->im->resetOrigin();
            Py_RETURN_NONE; 
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_getDimensions

/* Return image dimensions as a tuple */
PyObject *
Image_getEulerAngles(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    if (self != NULL)
    {
        try
        {
            double rot, tilt, psi;
            self->image->getEulerAngles(rot, tilt, psi);
            return Py_BuildValue("fff", rot, tilt, psi);

        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_getEulerAngles


/* Return value from MainHeader*/
PyObject *
Image_getMainHeaderValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    PyObject *pyValue;
    int label;

    if (self != NULL && PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            MDRow &mainHeader = self->image->image->MDMainHeader;
            if (mainHeader.containsLabel((MDLabel)label))
            {
                MDObject * object = mainHeader.getObject((MDLabel) label);
                pyValue = getMDObjectValue(object);
                return pyValue;
            }
            else
                Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}

/* Set value to MainHeader*/
PyObject *
Image_setMainHeaderValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    int label;
    PyObject *pyValue; //Only used to skip label and value

    if (self != NULL && PyArg_ParseTuple(args, "iO", &label, &pyValue))
    {
        try
        {
            MDRow &mainHeader = self->image->image->MDMainHeader;

            MDObject * object = createMDObject(label, pyValue);
            if (!object)
                return NULL;
            mainHeader.setValue(*object);
            delete object;
            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}

/* Return value from Header, now using only the first image*/
PyObject *
Image_getHeaderValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    PyObject *pyValue;
    int label;

    if (self != NULL && PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            MDRow &mainHeader = self->image->image->MD[0];
            if (mainHeader.containsLabel((MDLabel)label))
            {
                MDObject * object = mainHeader.getObject((MDLabel) label);
                pyValue = getMDObjectValue(object);
                return pyValue;
            }
            else
                Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}
/* Set value to Header, now using only the first image*/
PyObject *
Image_setHeaderValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    int label;
    PyObject *pyValue; //Only used to skip label and value

    if (self != NULL && PyArg_ParseTuple(args, "iO", &label, &pyValue))
    {
        try
        {
            MDRow &mainHeader = self->image->image->MD[0];

            MDObject * object = createMDObject(label, pyValue);
            if (!object)
                return NULL;
            mainHeader.setValue(*object);
            delete object;
            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}

/* Return image dimensions as a tuple */
PyObject *
Image_computeStats(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    if (self != NULL)
    {
        try
        {
            double mean, dev, min, max;
            self->image->data->computeStats(mean, dev, min, max);


            return Py_BuildValue("ffff", mean, dev, min, max);

        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_computeStats

PyObject *
Image_computePSD(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    if (nullptr == self) return nullptr;
    try {
        // keep default values consistent with the python
        float overlap = 0.4f;
        int dimX = 384;
        int dimY = 384;
        unsigned threads = 1;
        ImageObject *result = PyObject_New(ImageObject, &ImageType);
        if (PyArg_ParseTuple(args, "|fIIb", &overlap, &dimX, &dimY, &threads)
                && (nullptr != result)) {
            // prepare dims
            auto dims = Dimensions(dimX, dimY);
            // prepare input image
            ImageGeneric *image = self->image;
            image->convert2Datatype(DT_Double);
            MultidimArray<double> *in;
            MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(in);
            // prepare output image
            result->image = new ImageGeneric(DT_Double);
            MultidimArray<double> *out;
            MULTIDIM_ARRAY_GENERIC(*result->image).getMultidimArrayPointer(out);
            // call the estimation
            PSDEstimator<double>::estimatePSD(*in, overlap, dims, *out, threads);
        } else {
            PyErr_SetString(PyXmippError, "Unknown error while allocating data for output or parsing data");
        }
        return (PyObject *)result;
    } catch (XmippError &xe) {
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
}

/* Return image dimensions as a tuple */
PyObject *
Image_adjustAndSubtract(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    PyObject *pimg2 = NULL;
    ImageObject * result = PyObject_New(ImageObject, &ImageType);
    if (self != NULL)
    {
        try
        {
            if (PyArg_ParseTuple(args, "O", &pimg2))
            {
                ImageObject *img2=(ImageObject *)pimg2;
                result->image = new ImageGeneric(Image_Value(img2));
                MULTIDIM_ARRAY_GENERIC(*result->image).rangeAdjust(MULTIDIM_ARRAY_GENERIC(*self->image));
                MULTIDIM_ARRAY_GENERIC(*result->image) *=-1;
                MULTIDIM_ARRAY_GENERIC(*result->image) += MULTIDIM_ARRAY_GENERIC(*self->image);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return (PyObject *)result;
}//function Image_adjustAndSubtract

/* Return image dimensions as a tuple */
PyObject *
Image_correlation(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;
    if (self != NULL)
    {
        try
        {
            PyObject *pimg2 = NULL;
            if (PyArg_ParseTuple(args, "O", &pimg2))
            {
	            ImageGeneric *image = self->image;
	            image->convert2Datatype(DT_Double);
	            MultidimArray<double> * pImage=NULL;
	            MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(pImage);

	            ImageObject *img2=(ImageObject *)pimg2;
	            ImageGeneric *image2 = img2->image;
	            image2->convert2Datatype(DT_Double);
	            MultidimArray<double> * pImage2=NULL;
	            MULTIDIM_ARRAY_GENERIC(*image2).getMultidimArrayPointer(pImage2);

	            double corr=correlationIndex(*pImage,*pImage2);
                return Py_BuildValue("f", corr);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_correlation


/* Add two images, operator + */
PyObject *
Image_add(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = PyObject_New(ImageObject, &ImageType);
    if (result != NULL)
    {
        try
        {
            result->image = new ImageGeneric(Image_Value(obj1));
            Image_Value(result).add(Image_Value(obj2));
        }
        catch (XmippError &xe)
        {
        	result=NULL;
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return (PyObject *)result;
}//operator +

/** Image inplace add, equivalent to += operator */
PyObject *
Image_iadd(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = NULL;
    try
    {
        Image_Value(obj1).add(Image_Value(obj2));
        if ((result = PyObject_New(ImageObject, &ImageType)))
            result->image = new ImageGeneric(Image_Value(obj1));
        //return obj1;
    }
    catch (XmippError &xe)
    {
    	result=NULL;
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return (PyObject *)result;
}//operator +=

/** Image inplace add, equivalent to *= operator
 * but this not return a new instance of image, mainly
 * for efficiency reasons.
 * */
PyObject *
Image_inplaceAdd(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try
    {
      PyObject *other = NULL;
      if (PyArg_ParseTuple(args, "O", &other) &&
          Image_Check(other))
      {
		  STARTINGX(MULTIDIM_ARRAY_BASE(Image_Value(self)))=
				  STARTINGX(MULTIDIM_ARRAY_BASE(Image_Value(other)));
    	  STARTINGY(MULTIDIM_ARRAY_BASE(Image_Value(self)))=
    			  STARTINGY(MULTIDIM_ARRAY_BASE(Image_Value(other)));
    	  STARTINGZ(MULTIDIM_ARRAY_BASE(Image_Value(self)))=
    			  STARTINGZ(MULTIDIM_ARRAY_BASE(Image_Value(other)));

        Image_Value(self).add(Image_Value(other));
        Py_RETURN_NONE;
      }
      else
        PyErr_SetString(PyXmippError, "Expecting Image as second argument");
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return NULL;
}// similar to -=


/* Subtract two images, operator - */
PyObject *
Image_subtract(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = PyObject_New(ImageObject, &ImageType);
    if (result != NULL)
    {
        try
        {
            result->image = new ImageGeneric(Image_Value(obj1));
            Image_Value(result).subtract(Image_Value(obj2));
        }
        catch (XmippError &xe)
        {
            result = NULL;
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return (PyObject *)result;
}//operator -

/** Image inplace subtraction, equivalent to -= operator */
PyObject *
Image_isubtract(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = NULL;
    try
    {
        Image_Value(obj1).subtract(Image_Value(obj2));
        if ((result = PyObject_New(ImageObject, &ImageType)))
            result->image = new ImageGeneric(Image_Value(obj1));
    }
    catch (XmippError &xe)
    {
        result = NULL;
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return (PyObject *)result;
}//operator -=

/** Image inplace subtract, equivalent to *= operator
 * but this not return a new instance of image, mainly
 * for efficiency reasons.
 * */
PyObject *
Image_inplaceSubtract(PyObject *self, PyObject *args, PyObject *kwargs)
{
  try
  {
    PyObject *other = NULL;
    if (PyArg_ParseTuple(args, "O", &other) &&
        Image_Check(other))
    {
      Image_Value(self).subtract(Image_Value(other));
      Py_RETURN_NONE;
    }
    else
      PyErr_SetString(PyXmippError, "Expecting Image as second argument");
  }
  catch (XmippError &xe)
  {
      PyErr_SetString(PyXmippError, xe.msg.c_str());
  }
  return NULL;
}

/* Multiply image and constant, operator * */
PyObject *
Image_multiply(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = PyObject_New(ImageObject, &ImageType);
    if (result != NULL)
    {
        try
        {
            result->image = new ImageGeneric(Image_Value(obj1));
            double value = PyFloat_AsDouble(obj2);
            Image_Value(result).multiply(value);
        }
        catch (XmippError &xe)
        {
            result = NULL;
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return (PyObject *)result;
}//operator *


PyObject *
Image_imultiply(PyObject *obj1, PyObject *obj2)
{
    try
    {
        ImageObject * result = NULL;
        if ((result = PyObject_New(ImageObject, &ImageType)))
            result->image = new ImageGeneric(Image_Value(obj1));
        double value = PyFloat_AsDouble(obj2);
        Image_Value(result).multiply(value);
        return (PyObject*) result;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return NULL;
}//operator *=

/** Image inplace multiply, equivalent to *= operator
 * but this not return a new instance of image, mainly
 * for efficiency reasons.
 * */
PyObject *
Image_inplaceMultiply(PyObject *self, PyObject *args, PyObject *kwargs)
{
  try
  {
    PyObject *other = NULL;
    if (PyArg_ParseTuple(args, "O", &other))
    {
      if (Image_Check(other))
      {
        Image_Value(self).multiply(Image_Value(other));
      }
      else
      {
        Image_Value(self).multiply(PyFloat_AsDouble(other));
      }
      Py_RETURN_NONE;
    }
    else
      PyErr_SetString(PyXmippError, "Expecting Number or Image as second argument");
  }
  catch (XmippError &xe)
  {
      PyErr_SetString(PyXmippError, xe.msg.c_str());
  }
  return NULL;
}// similar to *=

/* Divide image and constant, operator * */
PyObject *
Image_true_divide(PyObject *obj1, PyObject *obj2)
{
    std::cerr << "TODO: not working Image_divide_____________________________" << std::endl;
    ImageObject * result = PyObject_New(ImageObject, &ImageType);
    if (result != NULL)
    {
        try
        {
            result->image = new ImageGeneric(Image_Value(obj1));
            double value = PyFloat_AsDouble(obj2);
            Image_Value(result).divide(value);
        }
        catch (XmippError &xe)
        {
            result = NULL;
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return (PyObject *)result;
}//operator /

/** Image division
 * NOTE (JM): For efficiency reasons, we break the Python convention
 * and return None instead of a new reference of Image
 * just to avoid creating a new Image object.
 * */
PyObject *
Image_idivide(PyObject *obj1, PyObject *obj2)
{
    std::cerr << "TODO: not working Image_idivide_____________________________" << std::endl;
    try
    {
      ImageObject * result = NULL;
      if ((result = PyObject_New(ImageObject, &ImageType)))
          result->image = new ImageGeneric(Image_Value(obj1));
      double value = PyFloat_AsDouble(obj2);
      Image_Value(result).divide(value);
      return (PyObject*) result;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return NULL;
}//operator /=

/** Image inplace divide, equivalent to /= operator
 * but this not return a new instance of image, mainly
 * for efficiency reasons.
 * */
PyObject *
Image_inplaceDivide(PyObject *self, PyObject *args, PyObject *kwargs)
{
  try
  {
    PyObject *other = NULL;
    if (PyArg_ParseTuple(args, "O", &other))
    {
      if (Image_Check(other))
      {
        Image_Value(self).divide(Image_Value(other));
      }
      else
      {
        Image_Value(self).divide(PyFloat_AsDouble(other));
      }
      Py_RETURN_NONE;
    }
    else
      PyErr_SetString(PyXmippError, "Expecting Number or Image as second argument");
  }
  catch (XmippError &xe)
  {
      PyErr_SetString(PyXmippError, xe.msg.c_str());
  }
  return NULL;

}// similar to /=

/** Image inplace subtraction, equivalent to -= operator */
PyObject *
Image_applyTransforMatScipion(PyObject *obj, PyObject *args, PyObject *kwargs)
{

	PyObject * list = NULL;
    PyObject * item = NULL;
    ImageObject *self = (ImageObject*) obj;
    ImageBase * img;
    PyObject *only_apply_shifts = Py_False;
    PyObject *wrap = (WRAP ? Py_True : Py_False);
    img = self->image->image;
    bool boolOnly_apply_shifts = false;
    bool boolWrap = WRAP;

    try
    {
        PyArg_ParseTuple(args, "O|OO", &list, &only_apply_shifts, &wrap);
        if (PyList_Check(list))
        {
            if (PyBool_Check(only_apply_shifts))
                boolOnly_apply_shifts = (only_apply_shifts == Py_True);
            else
                PyErr_SetString(PyExc_TypeError, "ImageGeneric::applyGeo: Expecting boolean value");
            if (PyBool_Check(wrap))
                boolWrap = (wrap == Py_True);
            else
                PyErr_SetString(PyExc_TypeError, "ImageGeneric::applyGeo: Expecting boolean value");


            size_t size = PyList_Size(list);
            Matrix2D<double> A;
            A.initIdentity(4);
            for (size_t i = 0; i < size; ++i)
            {
                item  = PyList_GetItem(list, i);
                MAT_ELEM(A,i/4,i%4 ) = PyFloat_AsDouble(item);
            }
            double scale, shiftX, shiftY, rot,tilt, psi;
            bool flip;
            transformationMatrix2Parameters2D(A, flip, scale, shiftX, shiftY, psi);//, shiftZ, rot,tilt, psi);
            double _rot;
            if (tilt==0.)
                _rot=rot + psi;
            else
                _rot=psi;
            img->setEulerAngles(0,0.,_rot);
            img->setShifts(shiftX,shiftY);
            img->setScale(scale);
            img->setFlip(flip);
            img->selfApplyGeometry(LINEAR, boolWrap, boolOnly_apply_shifts);//wrap, onlyShifts
            Py_RETURN_NONE;
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "ImageGeneric::applyTransforMatScipion: Expecting a list");

        }
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return NULL;
}//operator +=


PyObject *
Image_readApplyGeo(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    if (self != NULL)
    {
        PyObject *md = NULL;
        PyObject *only_apply_shifts = Py_False;
        PyObject *wrap = (WRAP ? Py_True : Py_False);
        size_t objectId = BAD_OBJID;
        bool boolOnly_apply_shifts = false;
        bool boolWrap = WRAP;
        int datamode = DATA;
        size_t select_img = ALL_IMAGES;

        if (PyArg_ParseTuple(args, "Ok|OikO", &md, &objectId, &only_apply_shifts, &datamode, &select_img, &wrap))
        {
            try
            {
                if (PyBool_Check(only_apply_shifts))
                    boolOnly_apply_shifts = (only_apply_shifts == Py_True);
                else
                    PyErr_SetString(PyExc_TypeError, "ImageGeneric::readApplyGeo: Expecting boolean value");
                if (PyBool_Check(wrap))
                    boolWrap = (wrap == Py_True);
                else
                    PyErr_SetString(PyExc_TypeError, "ImageGeneric::readApplyGeo: Expecting boolean value");
                ApplyGeoParams params;
                params.only_apply_shifts = boolOnly_apply_shifts;
                params.datamode = (DataMode)datamode;
                params.select_img = select_img;
                params.wrap = boolWrap;
                self->image->readApplyGeo(MetaData_Value(md), objectId, params);
                Py_RETURN_NONE;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.msg.c_str());
            }
        }
    }
    return NULL;
}

/** Image inplace subtraction, equivalent to -= operator */
PyObject *
Image_warpAffine(PyObject *obj, PyObject *args, PyObject *kwargs)
{

	PyObject * list = NULL;
    PyObject * item = NULL;
    PyObject * dsize = NULL;
    PyObject * wrap = Py_True;
    PyObject * border_value = NULL;

    ImageObject *self = (ImageObject*) obj;
    ImageGeneric * image = self->image;
    double doubleBorder_value = 1.0;
    size_t Xdim, Ydim, Zdim;
    bool doWrap = true;
    image->getDimensions(Xdim, Ydim, Zdim);
    try
    {
        PyArg_ParseTuple(args, "O|OOO", &list, &dsize, &wrap, &border_value);
        if (PyList_Check(list))
        {
        	if (nullptr != dsize)
        	{
        		Ydim = PyLong_AsSsize_t(PyTuple_GetItem(dsize, 0));
        		Xdim = PyLong_AsSsize_t(PyTuple_GetItem(dsize, 1));
        	}
            if (PyBool_Check(wrap))
                doWrap = (wrap == Py_True);
            else
                PyErr_SetString(PyExc_TypeError, "ImageGeneric::warpAffine: Expecting boolean value for wrapping");
            if (border_value!=NULL)
           	    doubleBorder_value = PyFloat_AsDouble(border_value);

            size_t size = PyList_Size(list);
            Matrix2D<double> A;
            A.initIdentity(3);
            for (size_t i = 0; i < size; ++i)
            {
                item  = PyList_GetItem(list, i);
                MAT_ELEM(A,i/3,i%3 ) = PyFloat_AsDouble(item);
            }

            image->convert2Datatype(DT_Double);
            MultidimArray<double> *in;
            MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(in);
            in->setXmippOrigin();

            ImageObject *result = PyObject_New(ImageObject, &ImageType);
            result->image = new ImageGeneric(DT_Double);
            MultidimArray<double> *out;
            MULTIDIM_ARRAY_GENERIC(*result->image).getMultidimArrayPointer(out);
            out->resize(Ydim,Xdim);
            out->initConstant(doubleBorder_value);
            out->setXmippOrigin();

            applyGeometry(3, *out, *in, A, false, doWrap, doubleBorder_value);
            return (PyObject *)result;
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "ImageGeneric::warpAffine: Expecting a list");

        }
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return NULL;
}


PyObject *
Image_applyGeo(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    if (self != NULL)
    {
        PyObject *md = NULL;
        PyObject *only_apply_shifts = Py_False;
        PyObject *wrap = (WRAP ? Py_True : Py_False);
        size_t objectId = BAD_OBJID;
        bool boolOnly_apply_shifts = false;
        bool boolWrap = WRAP;

        if (PyArg_ParseTuple(args, "Ok|OO", &md, &objectId, &only_apply_shifts, &wrap))
        {
            try
            {
                if (PyBool_Check(only_apply_shifts))
                    boolOnly_apply_shifts = (only_apply_shifts == Py_True);
                else
                    PyErr_SetString(PyExc_TypeError, "ImageGeneric::applyGeo: Expecting boolean value");
                if (PyBool_Check(wrap))
                    boolWrap = (wrap == Py_True);
                else
                    PyErr_SetString(PyExc_TypeError, "ImageGeneric::applyGeo: Expecting boolean value");
                ApplyGeoParams params;
                params.only_apply_shifts = boolOnly_apply_shifts;
                params.wrap = boolWrap;
                self->image->applyGeo(MetaData_Value(md), objectId, params);
                Py_RETURN_NONE;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.msg.c_str());
            }
        }
    }
    return NULL;
}
