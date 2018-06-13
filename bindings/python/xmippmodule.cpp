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
#include <bindings/python/xmippmoduleCore.h>
#include <bindings/python/python_image.h>
#include <reconstruction/ctf_enhance_psd.h>

PyObject * PyXmippError;


/***************************************************************/
/*                   Some specific utility functions           */
/***************************************************************/
/* calculate enhanced psd and return preview*/
PyObject *
xmipp_fastEstimateEnhancedPSD(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn, *pyImage;
    //ImageObject *pyImage;
    double downsampling;
    int dim, Nthreads;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdii", &pyImage, &pyStrFn, &downsampling, &dim, &Nthreads))
    {
        try
        {
            if (validateInputImageString(pyImage, pyStrFn, fn))
            {
                MultidimArray<double> data;
                fastEstimateEnhancedPSD(fn, downsampling, data, Nthreads);
                selfScaleToSize(LINEAR, data, dim, dim);
                Image_Value(pyImage).setDatatype(DT_Double);
                Image_Value(pyImage).data->setImage(data);
                Py_RETURN_NONE;
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}
/** Some helper macros repeated in filter functions*/
#define FILTER_TRY()\
try {\
if (validateInputImageString(pyImage, pyStrFn, fn)) {\
Image<double> img;\
img.read(fn);\
MultidimArray<double> &data = MULTIDIM_ARRAY(img);\
ArrayDim idim;\
data.getDimensions(idim);

#define FILTER_CATCH()\
size_t w = dim, h = dim, &x = idim.xdim, &y = idim.ydim;\
if (x > y) h = y * (dim/x);\
else if (y > x)\
  w = x * (dim/y);\
selfScaleToSize(LINEAR, data, w, h);\
Image_Value(pyImage).setDatatype(DT_Double);\
data.resetOrigin();\
MULTIDIM_ARRAY_GENERIC(Image_Value(pyImage)).setImage(data);\
Py_RETURN_NONE;\
}} catch (XmippError &xe)\
{ PyErr_SetString(PyXmippError, xe.msg.c_str());}\


/* calculate enhanced psd and return preview
* used for protocol preprocess_particles*/
PyObject *
xmipp_bandPassFilter(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn, *pyImage;
    double w1, w2, raised_w;
    int dim;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdddi", &pyImage, &pyStrFn, &w1, &w2, &raised_w, &dim))
    {
        FILTER_TRY()
        bandpassFilter(data, w1, w2, raised_w);
        FILTER_CATCH()
    }
    return NULL;
}

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_gaussianFilter(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn, *pyImage;
    double freqSigma;
    int dim;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdi", &pyImage, &pyStrFn, &freqSigma, &dim))
    {
        FILTER_TRY()
        gaussianFilter(data, freqSigma);
        FILTER_CATCH()
    }
    return NULL;
}

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_realGaussianFilter(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn, *pyImage;
    double realSigma;
    int dim;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdi", &pyImage, &pyStrFn, &realSigma, &dim))
    {
        FILTER_TRY()
        realGaussianFilter(data, realSigma);
        FILTER_CATCH()
    }
    return NULL;
}

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_badPixelFilter(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn, *pyImage;
    double factor;
    int dim;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdi", &pyImage, &pyStrFn, &factor, &dim))
    {
        FILTER_TRY()
        BadPixelFilter filter;
        filter.type = BadPixelFilter::OUTLIER;
        filter.factor = factor;
        filter.apply(data);
        FILTER_CATCH()
    }
    return NULL;
}

PyObject *
xmipp_errorBetween2CTFs(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd1, *pyMd2;
    double minFreq=0.05,maxFreq=0.25;
    size_t dim=256;

    if (PyArg_ParseTuple(args, "OO|idd"
                         ,&pyMd1, &pyMd2
                         ,&dim,&minFreq,&maxFreq))
    {
        try
        {
            if (!MetaData_Check(pyMd1))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as first argument");
            else if (!MetaData_Check(pyMd2))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as second argument");
            else
            {
                double error = errorBetween2CTFs(MetaData_Value(pyMd1),
                                                 MetaData_Value(pyMd2),
                                                 dim,
                                                 minFreq,maxFreq);
                return Py_BuildValue("f", error);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;

}

PyObject *
xmipp_errorMaxFreqCTFs(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd1;
    double phaseDiffRad;

    if (PyArg_ParseTuple(args, "Od", &pyMd1, &phaseDiffRad))
    {
        try
        {
            if (!MetaData_Check(pyMd1))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as first argument");
            else
            {
                double resolutionA = errorMaxFreqCTFs(MetaData_Value(pyMd1),
                                                      phaseDiffRad
                                                     );
                return Py_BuildValue("f", resolutionA);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;

}

PyObject *
xmipp_errorMaxFreqCTFs2D(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd1, *pyMd2;

    if (PyArg_ParseTuple(args, "OO", &pyMd1, &pyMd2))
    {
        try
        {
            if (!MetaData_Check(pyMd1))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as first argument");
            else if (!MetaData_Check(pyMd2))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as second argument");
            else
            {
                double resolutionA = errorMaxFreqCTFs2D(MetaData_Value(pyMd1),
                                                        MetaData_Value(pyMd2)
                                                       );
                return Py_BuildValue("f", resolutionA);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;

}

/* convert to psd */
PyObject *
Image_convertPSD(PyObject *obj, PyObject *args, PyObject *kwargs)
{
	PyObject *pyImage;
    if (PyArg_ParseTuple(args, "O", &pyImage))
    {
        ImageObject *img=(ImageObject *)pyImage;
        try
        {
            ImageGeneric *image = img->image;
            image->convert2Datatype(DT_Double);
            MultidimArray<double> *in;
            MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(in);
            xmipp2PSD(*in, *in, true);

            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_convertPSD

/* I2aligned=align(I1,I2) */
PyObject *
Image_align(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pimg1 = NULL;
    PyObject *pimg2 = NULL;
    ImageObject * result = PyObject_New(ImageObject, &ImageType);
	try
	{
		if (PyArg_ParseTuple(args, "OO", &pimg1, &pimg2))
		{
			ImageObject *img1=(ImageObject *)pimg1;
			ImageObject *img2=(ImageObject *)pimg2;

			result->image = new ImageGeneric(Image_Value(img2));
			*result->image = *img2->image;

			result->image->convert2Datatype(DT_Double);
			MultidimArray<double> *mimgResult;
			MULTIDIM_ARRAY_GENERIC(*result->image).getMultidimArrayPointer(mimgResult);

			img1->image->convert2Datatype(DT_Double);
			MultidimArray<double> *mimg1;
			MULTIDIM_ARRAY_GENERIC(*img1->image).getMultidimArrayPointer(mimg1);

			Matrix2D<double> M;
			alignImagesConsideringMirrors(*mimg1, *mimgResult, M, true);
		}
	}
	catch (XmippError &xe)
	{
		PyErr_SetString(PyXmippError, xe.msg.c_str());
	}
    return (PyObject *)result;
}//function Image_align

/* Apply CTF to this image */
PyObject *
Image_applyCTF(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pimg = NULL;
    PyObject *input = NULL;
    double Ts=1.0;
    size_t rowId;
    PyObject *pyReplace = Py_False;
    bool absPhase = false;

    try
    {
        PyArg_ParseTuple(args, "OOd|kO", &pimg, &input,&Ts,&rowId,&pyReplace);
        if (pimg!=NULL && input != NULL)
        {
			if(PyBool_Check(pyReplace))
				absPhase = pyReplace == Py_True;

			PyObject *pyStr;
			if (PyString_Check(input) || MetaData_Check(input))
			{
				ImageObject *img = (ImageObject*) pimg;
				ImageGeneric *image = img->image;
				image->convert2Datatype(DT_Double);
				MultidimArray<double> * mImage=NULL;
				MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(mImage);

				// COSS: This is redundant? image->data->getMultidimArrayPointer(mImage);

				CTFDescription ctf;
				ctf.enable_CTF=true;
				ctf.enable_CTFnoise=false;
				if (MetaData_Check(input))
					ctf.readFromMetadataRow(MetaData_Value(input), rowId );
				else
			   {
				   pyStr = PyObject_Str(input);
				   FileName fnCTF = PyString_AsString(pyStr);
				   ctf.read(fnCTF);
			   }
				ctf.produceSideInfo();
				ctf.applyCTF(*mImage,Ts,absPhase);
				Py_RETURN_NONE;
			}
		}
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.msg.c_str());
    }
    return NULL;
}

/* projectVolumeDouble */
PyObject *
Image_projectVolumeDouble(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pvol = NULL;
    ImageObject * result = NULL;
    double rot, tilt, psi;

    if (PyArg_ParseTuple(args, "Oddd", &pvol, &rot,&tilt,&psi))
    {
        try
        {
            // We use the following macro to release the Python Interpreter Lock (GIL)
            // while running this C extension code and allows threads to run concurrently.
            // See: https://docs.python.org/2.7/c-api/init.html for details.
            Py_BEGIN_ALLOW_THREADS
            Projection P;
			ImageObject *vol = (ImageObject*) pvol;
            MultidimArray<double> * mVolume;
            vol->image->data->getMultidimArrayPointer(mVolume);
            ArrayDim aDim;
            mVolume->getDimensions(aDim);
            mVolume->setXmippOrigin();
            projectVolume(*mVolume, P, aDim.xdim, aDim.ydim,rot, tilt, psi);
            result = PyObject_New(ImageObject, &ImageType);
            Image <double> I;
            result->image = new ImageGeneric();
            result->image->setDatatype(DT_Double);
            result->image->data->setImage(MULTIDIM_ARRAY(P));
            Py_END_ALLOW_THREADS
            return (PyObject *)result;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.msg.c_str());
        }
    }
    return NULL;
}//function Image_projectVolumeDouble


static PyMethodDef
xmipp_methods[] =
    {
        { "fastEstimateEnhancedPSD", (PyCFunction) xmipp_fastEstimateEnhancedPSD, METH_VARARGS,
          "Utility function to calculate PSD preview" },
        { "bandPassFilter", (PyCFunction) xmipp_bandPassFilter, METH_VARARGS,
          "Utility function to apply bandpass filter" },
        { "gaussianFilter", (PyCFunction) xmipp_gaussianFilter, METH_VARARGS,
          "Utility function to apply gaussian filter in Fourier space" },
        { "realGaussianFilter", (PyCFunction) xmipp_realGaussianFilter, METH_VARARGS,
          "Utility function to apply gaussian filter in Real space" },
        { "badPixelFilter", (PyCFunction) xmipp_badPixelFilter, METH_VARARGS,
          "Bad pixel filter" },
        { "errorBetween2CTFs", (PyCFunction) xmipp_errorBetween2CTFs,
          METH_VARARGS, "difference between two metadatas" },
        { "errorMaxFreqCTFs", (PyCFunction) xmipp_errorMaxFreqCTFs,
          METH_VARARGS, "resolution at which CTFs phase differs more than 90 degrees" },
        { "errorMaxFreqCTFs2D", (PyCFunction) xmipp_errorMaxFreqCTFs2D,
          METH_VARARGS, "resolution at which CTFs phase differs more than 90 degrees, 2D case" },
		{ "convertPSD", (PyCFunction) Image_convertPSD, METH_VARARGS,
		  "Convert to PSD: center FFT and use logarithm" },
		{ "image_align", (PyCFunction) Image_align, METH_VARARGS,
		  "I2aligned=image_align(I1,I2), align I2 to resemble I1." },
		{ "applyCTF", (PyCFunction) Image_applyCTF, METH_VARARGS,
		  "Apply CTF to this image. Ts is the sampling rate of the image." },
		{ "projectVolumeDouble", (PyCFunction) Image_projectVolumeDouble, METH_VARARGS,
		  "project a volume using Euler angles" },
        { NULL } /* Sentinel */
    };//xmipp_methods

#define INIT_TYPE(type) if (PyType_Ready(&type##Type) < 0) return; Py_INCREF(&type##Type);\
    PyModule_AddObject(module, #type, (PyObject *) &type##Type);

PyMODINIT_FUNC initxmipp(void)
{
    //Initialize module variable
    PyObject* module;
    module = Py_InitModule3("xmipp", xmipp_methods,
                            "Xmipp module as a Python extension.");
    import_array();

    //Check types and add to module
    INIT_TYPE(FourierProjector);

    //Add PyXmippError
    char message[32]="xmipp.XmippError";
    PyXmippError = PyErr_NewException(message, NULL, NULL);
    Py_INCREF(PyXmippError);
    PyModule_AddObject(module, "XmippError", PyXmippError);

    //Add MDLabel constants
    PyObject * dict = PyModule_GetDict(module);
    addLabels(dict);
}
