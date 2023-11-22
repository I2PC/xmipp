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

#include "python_image.h"
#include "python_metadata.h"
#include "core/transformations.h"
#include "data/dimensions.h"
#include "data/filters.h"
#include "reconstruction/psd_estimator.h"

/***************************************************************/
/*                            Image                         */
/**************************************************************/



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
        nullptr, //binaryfunc  nb_remainder;
        nullptr, //binaryfunc  nb_divmod;
        nullptr, //ternaryfunc nb_power;
        nullptr, //unaryfunc nb_negative;
        nullptr, //unaryfunc nb_positive;
        nullptr, //unaryfunc nb_absolute;
        nullptr, //inquiry nb_nonzero;       /* Used by PyObject_IsTrue */
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
		{ "correlationAfterAlignment", (PyCFunction) Image_correlationAfterAlignment, METH_VARARGS,
		  "correlationAfterAlignment(I1,I2). I2 is aligned to I1 (including mirrors), and then the correlation is returned" },
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
        { "window2D", (PyCFunction) Image_window2D, METH_VARARGS,
          "Return a window of the input image. imageOut = imageIn.window(x0,y0,xF,yF)" },
		{ "radialAverageAxis", (PyCFunction) Image_radialAvgAxis, METH_VARARGS,
		  "compute radial average around an axis" },
        { "centerOfMass", (PyCFunction) Image_centerOfMass, METH_VARARGS,
          "Return image center of mass as a tuple" },

        { nullptr } /* Sentinel */
    };//Image_methods


/*Image Type */
PyTypeObject ImageType = {
                             PyObject_HEAD_INIT(0)
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

/* Destructor */
void Image_dealloc(ImageObject* self)
{
    self->~ImageObject(); // Call the destructor
    Py_TYPE(self)->tp_free((PyObject*)self);
}//function Image_dealloc

/* Constructor */
PyObject *
Image_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*)type->tp_alloc(type, 0);

    if (self != nullptr)
    {
        PyObject *input = nullptr;

        if (PyArg_ParseTuple(args, "|O", &input))
        {
            if (input != nullptr)
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
                      self->image = std::make_unique<ImageGeneric>();
                      self->image->read(filename, DATA, index);

                    }
                    else if ((pyStr = PyObject_Str(input)) != nullptr)
                    {
                        self->image = std::make_unique<ImageGeneric>(PyUnicode_AsUTF8(pyStr));
                        //todo: add copy constructor
                    }
                    else
                    {
                        PyErr_SetString(PyExc_TypeError,
                                        "Image_new: Expected string, FileName or tuple as first argument");
                        return nullptr;
                    }
                }
                catch (XmippError &xe)
                {
                    PyErr_SetString(PyXmippError, xe.what());
                    return nullptr;
                }
            }
            else
                self->image = std::make_unique<ImageGeneric>();
        }
        else
            return nullptr;
    }
    return (PyObject *)self;
}//function Image_new

/* Image string representation */
PyObject *
Image_repr(PyObject * obj)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    String s;
    self->image->toString(s);
    return PyUnicode_FromString(s.c_str());
}//function Image_repr

/* Image compare function */
PyObject*
Image_RichCompareBool(PyObject * obj, PyObject * obj2, int opid)
{
    if (obj != nullptr && obj2 != nullptr)
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
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_compare

/* Compare two images up to a precision */
PyObject *
Image_equal(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        double precision = 1.e-3;
        PyObject *image2 = nullptr;
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
                PyErr_SetString(PyXmippError, xe.what());
            }
        }
    }
    return nullptr;
}//function Image_equal

/* write */
PyObject *
Image_write(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        PyObject *input = nullptr;
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
              if ((pyStr = PyObject_Str(input)) != nullptr)
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
                PyErr_SetString(PyXmippError, xe.what());
            }
        }
    }
    return nullptr;
}//function Image_write

/* read */
PyObject *
Image_read(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    if (self != nullptr)
    {
        int datamode = DATA;
        PyObject *input = nullptr;
        if (PyArg_ParseTuple(args, "O|i", &input, &datamode))
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
                const char *filename = PyUnicode_AsUTF8(repr);
                // Now read using both of index and filename
                self->image->read(filename,(DataMode)datamode, index);
                Py_RETURN_NONE;
              }
              else if ((pyStr = PyObject_Str(input)) != nullptr)
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
                PyErr_SetString(PyXmippError, xe.what());
            }
        }
    }
    return nullptr;
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
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    if (self != nullptr)
    {
        PyObject *input = nullptr;
        int x = 0;
        int slice = CENTRAL_SLICE;

        if (PyArg_ParseTuple(args, "O|ii", &input, &x, &slice))
        {
            try
            {
              PyObject *pyStr;
              if ((pyStr = PyObject_Str(input)) != nullptr)
              {
                  readImagePreview(self->image.get(), PyUnicode_AsUTF8(pyStr), x, slice);
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
                PyErr_SetString(PyXmippError, xe.what());
            }
        }
        else
          PyErr_SetString(PyXmippError,"Error with input arguments, expected filename and optional size");
    }
    return nullptr;
}//function Image_readPreview

/* read preview, downsampling in Fourier space*/
PyObject *
Image_readPreviewSmooth(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    if (self != nullptr)
    {
        PyObject *input = nullptr;
        int x;
        if (PyArg_ParseTuple(args, "Oi", &input, &x))
        {
            try
            {

              PyObject *pyStr;
              if ((pyStr = PyObject_Str(input)) != nullptr)
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
                PyErr_SetString(PyXmippError, xe.what());
            }
        }
    }
    return nullptr;
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
    case DT_HalfFloat:
        return NPY_HALF;
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
    case NPY_HALF:
        return DT_HalfFloat;
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
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    if (self != nullptr)
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
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_getData



/* setData */
PyObject *
Image_setData(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    PyArrayObject * arr = nullptr;

    if (self != nullptr && PyArg_ParseTuple(args, "O", &arr))
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
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_setData

/* getPixel */
PyObject *
Image_getPixel(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    int i;
    int j;
    int k;
    int n;
    double value;

    if (self != nullptr && PyArg_ParseTuple(args, "iiii", &n, &k, &i, &j ))
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
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_getPixel

/* setPixel */
PyObject *
Image_setPixel(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    int i;
    int j;
    int k;
    int n;
    double value = -1;

    if (self != nullptr && PyArg_ParseTuple(args, "iiiid", &n, &k, &i, &j, &value))
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
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_setPixel

/* initConstant */
PyObject *
Image_initConstant(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    double value = -1;

    if (self != nullptr && PyArg_ParseTuple(args, "d", &value))
    {
        try
        {
            self->image->initConstant(value);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_initConstant

/* initConstant */
PyObject *
Image_mirrorY(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    try
    {
        self->image->mirrorY();
        Py_RETURN_NONE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}//function Image_initConstant

/* initRandom */
PyObject *
Image_initRandom(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    double op1 = 0;
    double op2 = 1;
    RandomMode mode = RND_UNIFORM;
    int pyMode = -1;

    if (self != nullptr && PyArg_ParseTuple(args, "|ddi", &op1, &op2, &pyMode))
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
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_initRandom

/* Resize Image */
PyObject *
Image_resize(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    int xDim = 0;
    int yDim = 0;
    int zDim = 1;
    size_t nDim = 1;

    if (self != nullptr && PyArg_ParseTuple(args, "ii|in", &xDim, &yDim, &zDim, &nDim))
    {
        try
        {
            self->image->resize(xDim, yDim, zDim, nDim, false); // TODO: Take care of copy mode if needed
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_resize

/* Scale Image */
PyObject *
Image_scale(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    int xDim = 0;
    int yDim = 0;
    int zDim = 1;
    int forceVolume=0;
    if (self != nullptr && PyArg_ParseTuple(args, "ii|ii", &xDim, &yDim, &zDim, &forceVolume))
    {
        try
        {
            MultidimArrayGeneric& I=MULTIDIM_ARRAY_GENERIC(Image_Value(self));
            I.setXmippOrigin();
            size_t xdim, ydim, zdim, ndim;
            I.getDimensions(xdim, ydim, zdim, ndim);
            if (forceVolume && zdim==1 && ndim>1)
               I.setDimensions(xdim,ydim,ndim,1);
            selfScaleToSize(xmipp_transformation::LINEAR, I, xDim, yDim, zDim);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_scale

/* Change the slices order in a volume */
PyObject *
Image_reslice(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    int axis = VIEW_Z_NEG;

    if (self != nullptr && PyArg_ParseTuple(args, "i", &axis))
    {
        try
        {
            self->image->reslice((AxisView) axis);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_reslice

/* Change the slices order in a volume */
PyObject *
Image_writeSlices(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        PyObject *oRootname = nullptr;
        PyObject *oExt = nullptr;
        PyObject *oAxis = nullptr;
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

				size_t xdim;
                size_t ydim;
                size_t zdim;
                size_t ndim;
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
                PyErr_SetString(PyXmippError, xe.what());
            }
        }
    }
    return nullptr;

}//function writeSlices

/* Patch Image */
PyObject *
Image_patch(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    PyObject *patch;
    int x = 0;
    int y = 0;

    if (self != nullptr && PyArg_ParseTuple(args, "Oii", &patch, &x, &y))
    {
        try
        {
            MULTIDIM_ARRAY_GENERIC(Image_Value(self)).patch(MULTIDIM_ARRAY_GENERIC(Image_Value(patch)), x, y);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_patch

/* Get Data Type */
PyObject *
Image_getDataType(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    if (self != nullptr)
    {
        try
        {
            ImageGeneric & image = Image_Value(self);
            DataType dt = image.getDatatype();
            return  Py_BuildValue("i", dt);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_getDataType

/* Set Data Type */
PyObject *
Image_setDataType(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    int datatype;

    if (self != nullptr && PyArg_ParseTuple(args, "i", &datatype))
    {
        try
        {
            self->image->setDatatype((DataType)datatype);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_setDataType

/* Set Data Type */
PyObject *
Image_convert2DataType(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    int datatype;
    int castMode=CW_CONVERT;

    if (self != nullptr && PyArg_ParseTuple(args, "i|i", &datatype, &castMode))
    {
        try
        {
            self->image->convert2Datatype((DataType)datatype, (CastWriteMode)castMode);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_setDataType

/* Return image dimensions as a tuple */
PyObject *
Image_getDimensions(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        try
        {
            size_t xdim;
            size_t ydim;
            size_t zdim;
            size_t ndim;
            MULTIDIM_ARRAY_GENERIC(*self->image).getDimensions(xdim, ydim, zdim, ndim);
            return Py_BuildValue("iiik", xdim, ydim, zdim, ndim);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_getDimensions

/* reset origin */
PyObject *
Image_resetOrigin(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        try
        {
            self->image->data->im->resetOrigin();
            Py_RETURN_NONE; 
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_getDimensions

/* Return image dimensions as a tuple */
PyObject *
Image_getEulerAngles(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        try
        {
            double rot;
            double tilt;
            double psi;
            self->image->getEulerAngles(rot, tilt, psi);
            return Py_BuildValue("fff", rot, tilt, psi);

        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_getEulerAngles


/* Return value from MainHeader*/
PyObject *
Image_getMainHeaderValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    PyObject *pyValue;
    int label;

    if (self != nullptr && PyArg_ParseTuple(args, "i", &label))
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
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* Set value to MainHeader*/
PyObject *
Image_setMainHeaderValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    int label;
    PyObject *pyValue; //Only used to skip label and value

    if (self != nullptr && PyArg_ParseTuple(args, "iO", &label, &pyValue))
    {
        try
        {
            MDRow &mainHeader = self->image->image->MDMainHeader;

            auto object = createMDObject(label, pyValue);
            if (!object)
                return nullptr;
            mainHeader.setValue(*object);
            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* Return value from Header, now using only the first image*/
PyObject *
Image_getHeaderValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    PyObject *pyValue;
    int label;

    if (self != nullptr && PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            MDRow &mainHeader = *(self->image->image->MD[0]);
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
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}
/* Set value to Header, now using only the first image*/
PyObject *
Image_setHeaderValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    int label;
    PyObject *pyValue; //Only used to skip label and value

    if (self != nullptr && PyArg_ParseTuple(args, "iO", &label, &pyValue))
    {
        try
        {
            MDRow &mainHeader = *(self->image->image->MD[0]);

            auto object = createMDObject(label, pyValue);
            if (!object)
                return nullptr;
            mainHeader.setValue(*object);
            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* Compute statistics */
PyObject *
Image_computeStats(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        try
        {
            double mean=0.0;
            double dev=0.0;
            double min=0.0;
            double max=0.0;
            self->image->data->computeStats(mean, dev, min, max);
            return Py_BuildValue("ffff", mean, dev, min, max);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_computeStats

PyObject *
Image_computePSD(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (nullptr == self) return nullptr;
    try {
        // keep default values consistent with the python
        float overlap = 0.4f;
        int dimX = 384;
        int dimY = 384;
        unsigned threads = 1;
        ImageObject *result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
        if (PyArg_ParseTuple(args, "|fIIb", &overlap, &dimX, &dimY, &threads)
                && (nullptr != result)) {
            // prepare dims
            auto dims = Dimensions(dimX, dimY);
            // prepare input image
            auto &image = self->image;
            image->convert2Datatype(DT_Double);
            MultidimArray<double> *in;
            MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(in);
            // prepare output image
            result->image = std::make_unique<ImageGeneric>(DT_Double);
            MultidimArray<double> *out;
            MULTIDIM_ARRAY_GENERIC(*result->image).getMultidimArrayPointer(out);
            // call the estimation
            PSDEstimator<double>::estimatePSD(*in, overlap, dims, *out, threads, false);
        } else {
            PyErr_SetString(PyXmippError, "Unknown error while allocating data for output or parsing data");
        }
        return (PyObject *)result;
    } catch (XmippError &xe) {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return Py_BuildValue("");
}

/* Adjust and subtract */
PyObject *
Image_adjustAndSubtract(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    PyObject *pimg2 = nullptr;
    ImageObject * result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
    if (self != nullptr)
    {
        try
        {
            if (PyArg_ParseTuple(args, "O", &pimg2))
            {
                ImageObject *img2=(ImageObject *)pimg2;
                result->image = std::make_unique<ImageGeneric>(Image_Value(img2));
                MULTIDIM_ARRAY_GENERIC(*result->image).rangeAdjust(MULTIDIM_ARRAY_GENERIC(*self->image));
                MULTIDIM_ARRAY_GENERIC(*result->image) *=-1;
                MULTIDIM_ARRAY_GENERIC(*result->image) += MULTIDIM_ARRAY_GENERIC(*self->image);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return (PyObject *)result;
}//function Image_adjustAndSubtract

/* Correlation */
PyObject *
Image_correlation(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        try
        {
            PyObject *pimg2 = nullptr;
            if (PyArg_ParseTuple(args, "O", &pimg2))
            {
	            auto &image = self->image;
	            image->convert2Datatype(DT_Double);
	            MultidimArray<double> * pImage=nullptr;
	            MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(pImage);

	            ImageObject *img2=(ImageObject *)pimg2;
	            auto &image2 = img2->image;
	            image2->convert2Datatype(DT_Double);
	            MultidimArray<double> * pImage2=nullptr;
	            MULTIDIM_ARRAY_GENERIC(*image2).getMultidimArrayPointer(pImage2);

	            double corr=correlationIndex(*pImage,*pImage2);
                return Py_BuildValue("f", corr);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_correlation

/* Correlation after alignment of I2 to I1 */
PyObject *
Image_correlationAfterAlignment(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    if (self != nullptr)
    {
        try
        {
            PyObject *pimg2 = nullptr;
            if (PyArg_ParseTuple(args, "O", &pimg2))
            {
	            auto &image = self->image;
	            image->convert2Datatype(DT_Double);
	            MultidimArray<double> * pImage=nullptr;
	            MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(pImage);

	            ImageObject *img2=(ImageObject *)pimg2;
	            auto &image2 = img2->image;
	            image2->convert2Datatype(DT_Double);
	            MultidimArray<double> * pImage2=nullptr;
	            MULTIDIM_ARRAY_GENERIC(*image2).getMultidimArrayPointer(pImage2);
	            pImage->setXmippOrigin();
	            pImage2->setXmippOrigin();

	            Matrix2D<double> M;
	            MultidimArray<double> I2Copy=*pImage2;

	            double corr=alignImagesConsideringMirrors(*pImage, I2Copy, M, xmipp_transformation::WRAP);
                return Py_BuildValue("f", corr);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* Add two images, operator + */
PyObject *
Image_add(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
    if (result != nullptr)
    {
        try
        {
            result->image = std::make_unique<ImageGeneric>(Image_Value(obj1));
            Image_Value(result).add(Image_Value(obj2));
        }
        catch (XmippError &xe)
        {
        	result=nullptr;
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return (PyObject *)result;
}//operator +

/** Image inplace add, equivalent to += operator */
PyObject *
Image_iadd(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = nullptr;
    try
    {
        Image_Value(obj1).add(Image_Value(obj2));
        if ((result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "")))
            result->image = std::make_unique<ImageGeneric>(Image_Value(obj1));
        //return obj1;
    }
    catch (XmippError &xe)
    {
    	result=nullptr;
        PyErr_SetString(PyXmippError, xe.what());
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
      PyObject *other = nullptr;
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
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}// similar to -=


/* Subtract two images, operator - */
PyObject *
Image_subtract(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
    if (result != nullptr)
    {
        try
        {
            result->image = std::make_unique<ImageGeneric>(Image_Value(obj1));
            Image_Value(result).subtract(Image_Value(obj2));
        }
        catch (XmippError &xe)
        {
            result = nullptr;
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return (PyObject *)result;
}//operator -

/** Image inplace subtraction, equivalent to -= operator */
PyObject *
Image_isubtract(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = nullptr;
    try
    {
        Image_Value(obj1).subtract(Image_Value(obj2));
        if ((result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "")))
            result->image = std::make_unique<ImageGeneric>(Image_Value(obj1));
    }
    catch (XmippError &xe)
    {
        result = nullptr;
        PyErr_SetString(PyXmippError, xe.what());
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
    PyObject *other = nullptr;
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
      PyErr_SetString(PyXmippError, xe.what());
  }
  return nullptr;
}

/* Multiply image and constant, operator * */
PyObject *
Image_multiply(PyObject *obj1, PyObject *obj2)
{
    ImageObject * result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
    if (result != nullptr)
    {
        try
        {
            result->image = std::make_unique<ImageGeneric>(Image_Value(obj1));
            double value = PyFloat_AsDouble(obj2);
            Image_Value(result).multiply(value);
        }
        catch (XmippError &xe)
        {
            result = nullptr;
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return (PyObject *)result;
}//operator *


PyObject *
Image_imultiply(PyObject *obj1, PyObject *obj2)
{
    try
    {
        ImageObject * result = nullptr;
        if ((result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "")))
            result->image = std::make_unique<ImageGeneric>(Image_Value(obj1));
        double value = PyFloat_AsDouble(obj2);
        Image_Value(result).multiply(value);
        return (PyObject*) result;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
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
    PyObject *other = nullptr;
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
      PyErr_SetString(PyXmippError, xe.what());
  }
  return nullptr;
}// similar to *=

/* Divide image and constant, operator * */
PyObject *
Image_true_divide(PyObject *obj1, PyObject *obj2)
{
    std::cerr << "TODO: not working Image_divide_____________________________" << std::endl;
    ImageObject * result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
    if (result != nullptr)
    {
        try
        {
            result->image = std::make_unique<ImageGeneric>(Image_Value(obj1));
            double value = PyFloat_AsDouble(obj2);
            Image_Value(result).divide(value);
        }
        catch (XmippError &xe)
        {
            result = nullptr;
            PyErr_SetString(PyXmippError, xe.what());
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
      ImageObject * result = nullptr;
      if ((result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "")))
          result->image = std::make_unique<ImageGeneric>(Image_Value(obj1));
      double value = PyFloat_AsDouble(obj2);
      Image_Value(result).divide(value);
      return (PyObject*) result;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
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
    PyObject *other = nullptr;
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
      PyErr_SetString(PyXmippError, xe.what());
  }
  return nullptr;

}// similar to /=

/** Image inplace subtraction, equivalent to -= operator */
PyObject *
Image_applyTransforMatScipion(PyObject *obj, PyObject *args, PyObject *kwargs)
{

	PyObject * list = nullptr;
    PyObject * item = nullptr;
    const auto *self = reinterpret_cast<ImageObject*>(obj);
    ImageBase * img;
    PyObject *only_apply_shifts = Py_False;
    PyObject *wrap = (xmipp_transformation::WRAP ? Py_True : Py_False);
    img = self->image->image;
    bool boolOnly_apply_shifts = false;
    bool boolWrap = xmipp_transformation::WRAP;

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
            double scale;
            double shiftX;
            double shiftY;
            double rot;
            double tilt;
            double psi;
            tilt = 0.0;
            rot = 0.0;
            bool flip;
            transformationMatrix2Parameters2D(A, flip, scale, shiftX, shiftY, psi);//, shiftZ, rot,tilt, psi);
            img->setEulerAngles(rot,tilt,psi);
            img->setShifts(shiftX,shiftY);
            img->setScale(scale);
            img->setFlip(flip);
            img->selfApplyGeometry(xmipp_transformation::LINEAR, boolWrap, boolOnly_apply_shifts);//wrap, onlyShifts
            Py_RETURN_NONE;
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "ImageGeneric::applyTransforMatScipion: Expecting a list");

        }
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}//operator +=


PyObject *
Image_readApplyGeo(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    if (self != nullptr)
    {
        PyObject *md = nullptr;
        PyObject *only_apply_shifts = Py_False;
        PyObject *wrap = (xmipp_transformation::WRAP ? Py_True : Py_False);
        size_t objectId = BAD_OBJID;
        bool boolOnly_apply_shifts = false;
        bool boolWrap = xmipp_transformation::WRAP;
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
                PyErr_SetString(PyXmippError, xe.what());
            }
        }
    }
    return nullptr;
}

/** Image inplace subtraction, equivalent to -= operator */
PyObject *
Image_warpAffine(PyObject *obj, PyObject *args, PyObject *kwargs)
{

	PyObject * list = nullptr;
    PyObject * item = nullptr;
    PyObject * dsize = nullptr;
    PyObject * wrap = Py_True;
    PyObject * border_value = nullptr;

    const auto *self = reinterpret_cast<ImageObject*>(obj);
    auto &image = self->image;
    double doubleBorder_value = 1.0;
    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
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
            if (border_value!=nullptr)
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

            ImageObject *result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
            result->image = std::make_unique<ImageGeneric>(DT_Double);
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
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}


PyObject *
Image_applyGeo(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<ImageObject*>(obj);

    if (self != nullptr)
    {
        PyObject *md = nullptr;
        PyObject *only_apply_shifts = Py_False;
        PyObject *wrap = (xmipp_transformation::WRAP ? Py_True : Py_False);
        size_t objectId = BAD_OBJID;
        bool boolOnly_apply_shifts = false;
        bool boolWrap = xmipp_transformation::WRAP;

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
                PyErr_SetString(PyXmippError, xe.what());
            }
        }
    }
    return nullptr;
}


PyObject *
Image_window2D(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    ImageObject *self = (ImageObject*) obj;

    if (self != nullptr)
    {
        try {
            int x0;
            int y0;
            int xF;
            int yF;
            ImageObject *result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");

            if (PyArg_ParseTuple(args, "|IIII", &x0, &y0, &xF, &yF)
                    && (nullptr != result)) {
                // prepare input image
                const auto& image = self->image;
                image->convert2Datatype(DT_Double);
                MultidimArray<double> *pImage_in;
                MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(pImage_in);
                // prepare output image
                result->image = std::make_unique<ImageGeneric>(DT_Double);
                MultidimArray<double> *pImage_out;
                MULTIDIM_ARRAY_GENERIC(*result->image).getMultidimArrayPointer(pImage_out);
                // call the estimation
                window2D(*pImage_in, *pImage_out, (size_t)y0, (size_t)x0, (size_t)yF, (size_t)xF);

            } else {
                PyErr_SetString(PyXmippError, "Unknown error while allocating data for output or parsing data");
            }
            return (PyObject *)result;
        } catch (XmippError &xe) {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}


/* Compute radial average around an axis, operator * */
PyObject *
Image_radialAvgAxis(PyObject *obj, PyObject *args, PyObject *kwargs)
{
	const auto *self = reinterpret_cast<ImageObject*>(obj);
	    if (nullptr == self) return nullptr;
	    try {
	    	char axis = 'z';
	        ImageObject *result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
	        if (PyArg_ParseTuple(args, "|c", &axis)
	                && (nullptr != result)) {
	            // prepare input image
	            auto &volume = self->image;
	            volume->convert2Datatype(DT_Double);
	            MultidimArray<double> *in;
	            MULTIDIM_ARRAY_GENERIC(*volume).getMultidimArrayPointer(in);
	            // prepare output image
	            result->image = std::make_unique<ImageGeneric>(DT_Double);
	            MultidimArray<double> *out;
	            MULTIDIM_ARRAY_GENERIC(*result->image).getMultidimArrayPointer(out);
	            // call the estimation
	            radialAverageAxis(*in, axis, *out);
	        } else {
	            PyErr_SetString(PyXmippError, "Unknown error while allocating data for output or parsing data");
	        }
	        return (PyObject *)result;
	    } catch (XmippError &xe) {
	        PyErr_SetString(PyXmippError, xe.what());
	    }
    return Py_BuildValue("");
}

/* Return center of mass as a tuple */
PyObject *
Image_centerOfMass(PyObject *obj)
{
    auto *self = (ImageObject*) obj;
    if (self != nullptr)
    {
        try
        {
            Matrix1D< double > center;
            self->image->convert2Datatype(DT_Double);
            MultidimArray<double> *in;
            MULTIDIM_ARRAY_GENERIC(*self->image).getMultidimArrayPointer(in);
            in->centerOfMass(center);
            return Py_BuildValue("fff", XX(center), YY(center), ZZ(center));
        }
        catch (const XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}
