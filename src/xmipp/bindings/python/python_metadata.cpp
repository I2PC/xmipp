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

#include <sstream>
#include "python_metadata.h"
#include "python_filename.h"
#include "core/metadata_sql.h"

/***************************************************************/
/*                            MDQuery                          */
/***************************************************************/

PyTypeObject MDQueryType =
    {
        PyObject_HEAD_INIT(0)
        "xmipp.MDQuery", /*tp_name*/
        sizeof(MDQueryObject), /*tp_basicsize*/
        0, /*tp_itemsize*/
        (destructor)MDQuery_dealloc, /*tp_dealloc*/
        0, /*tp_print*/
        0, /*tp_getattr*/
        0, /*tp_setattr*/
        0, /*tp_compare*/
        MDQuery_repr, /*tp_repr*/
        0, /*tp_as_number*/
        0, /*tp_as_sequence*/
        0, /*tp_as_mapping*/
        0, /*tp_hash */
        0, /*tp_call*/
        0, /*tp_str*/
        0, /*tp_getattro*/
        0, /*tp_setattro*/
        0, /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT, /*tp_flags*/
        "Python wrapper to Xmipp MDQuery class",/* tp_doc */
        0, /* tp_traverse */
        0, /* tp_clear */
        0, /* tp_richcompare */
        0, /* tp_weaklistoffset */
        0, /* tp_iter */
        0, /* tp_iternext */
        MDQuery_methods, /* tp_methods */
        0, /* tp_members */
        0, /* tp_getset */
        0, /* tp_base */
        0, /* tp_dict */
        0, /* tp_descr_get */
        0, /* tp_descr_set */
        0, /* tp_dictoffset */
        0, /* tp_init */
        0, /* tp_alloc */
        MDQuery_new /* tp_new */
    }; //MDQueryType

PyMethodDef MDQuery_methods[] = { { nullptr } /* Sentinel */
                                };

/* Destructor */
void MDQuery_dealloc(MDQueryObject* self)
{
    self->~MDQueryObject(); // Call the destructor
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Constructor */
PyObject *
MDQuery_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    auto *self = (MDQueryObject*)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/* String representation */
PyObject *
MDQuery_repr(PyObject * obj)
{
    auto *self = (MDQueryObject*) obj;
    if (self->query)
    {
        String s = self->query->whereString() + self->query->limitString()
                   + self->query->orderByString();
        return PyUnicode_FromString(s.c_str());
    }
    else
        return PyUnicode_FromString("");
}

/* Methods for constructing concrete queries */

/* Helper function to create relational queries */
PyObject *
createMDValueRelational(PyObject *args, int op)
{
    int label;
    int limit = -1;
    int offset = 0;
    auto orderLabel = (int) MDL_OBJID;
    PyObject *pyValue; //Only used to skip label and value

    if ((op == -1 && PyArg_ParseTuple(args, "iO|iiii", &label, &pyValue, &op,
                                      &limit, &offset, &orderLabel)) || PyArg_ParseTuple(args, "iO|iii",
                                              &label, &pyValue, &limit, &offset, &orderLabel))
    {
        auto object = createMDObject(label, pyValue);
        if (!object)
            return nullptr;
        auto *pyQuery = (MDQueryObject*)PyObject_CallFunction((PyObject*)&MDQueryType, "");
        pyQuery->query = std::make_unique<MDValueRelational>(*object, (RelationalOp) op,
                                               limit, offset, (MDLabel) orderLabel);
        return (PyObject *) pyQuery;
    }
    return nullptr;
}
/* MDValue Relational */
PyObject *
xmipp_MDValueRelational(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    return createMDValueRelational(args, -1);
}
/* MDValueEQ */
PyObject *
xmipp_MDValueEQ(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    return createMDValueRelational(args, EQ);
}
/* MDValueEQ */
PyObject *
xmipp_MDValueNE(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    return createMDValueRelational(args, NE);
}
/* MDValueLT */
PyObject *
xmipp_MDValueLT(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    return createMDValueRelational(args, LT);
}
/* MDValueLE */
PyObject *
xmipp_MDValueLE(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    return createMDValueRelational(args, LE);
}
/* MDValueLT */
PyObject *
xmipp_MDValueGT(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    return createMDValueRelational(args, GT);
}
/* MDValueLE */
PyObject *
xmipp_MDValueGE(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    return createMDValueRelational(args, GE);
}
/* MDValueRange */
PyObject *
xmipp_MDValueRange(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    int limit = -1;
    int offset = 0;
    auto orderLabel = (int) MDL_OBJID;
    PyObject *pyValue1;
    PyObject *pyValue2; //Only used to skip label and value

    if (PyArg_ParseTuple(args, "iOO|iii", &label, &pyValue1, &pyValue2, &limit,
                         &offset, &orderLabel))
    {
        auto object1 = createMDObject(label, pyValue1);
        auto object2 = createMDObject(label, pyValue2);
        if (!object1 || !object2)
            return nullptr;
        auto *pyQuery = (MDQueryObject*)PyObject_CallFunction((PyObject*)&MDQueryType, "");
        pyQuery->query = std::make_unique<MDValueRange>(*object1, *object2, limit, offset,
                                          (MDLabel) orderLabel);
        return (PyObject *) pyQuery;
    }
    return nullptr;
}
/* add alias for label in run time */
PyObject *
xmipp_addLabelAlias(PyObject *obj, PyObject *args, PyObject *kwargs)
{


    int label;
    int type;
    PyObject *input = nullptr;
    PyObject *pyStr = nullptr;
    PyObject *pyStr1 = nullptr;
    auto *pyReplace = Py_False;
    const char *str =nullptr;
    bool replace = true;


    if (PyArg_ParseTuple(args, "iO|Oi", &label, &input, &pyReplace, &type))
    {
        try
        {
            if ((pyStr = PyObject_Str(input)) != nullptr &&
                (PyBool_Check(pyReplace)))
            {
                replace = pyReplace == Py_True;
                str = PyUnicode_AsUTF8(pyStr);
                MDL::addLabelAlias((MDLabel)label,(String)str, replace, (MDLabelType)type);
                Py_RETURN_NONE;
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }

    return nullptr;
}

/* add labels in run time */
PyObject *
xmipp_getNewAlias(PyObject *obj, PyObject *args, PyObject *kwargs)
{


    int type;
    PyObject *input = nullptr;
    PyObject *pyStr = nullptr;
    PyObject *pyStr1 = nullptr;
    const char *str = nullptr;


    if (PyArg_ParseTuple(args, "Oi", &input, &type))
    {
        try
        {
            if ((pyStr = PyObject_Str(input)) != nullptr )
            {
                str = PyUnicode_AsUTF8(pyStr);
                return PyLong_FromLong(MDL::getNewAlias((String)str, (MDLabelType)type));
                
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }

    return nullptr;
}
/***************************************************************/
/*                            MetaData                         */
/**************************************************************/
PyMethodDef MetaData_methods[] =
    {
        { "read", (PyCFunction) MetaData_read, METH_VARARGS,
          "Read data from file" },
        { "write", (PyCFunction) MetaData_write, METH_VARARGS,
          "Write MetaData content to disk" },
        { "readBlock", (PyCFunction) MetaData_readBlock,
          METH_VARARGS, "Read block from a metadata file" },
        { "append", (PyCFunction) MetaData_append,
          METH_VARARGS, "Append MetaData content to disk" },
        { "addObject", (PyCFunction) MetaData_addObject,
          METH_NOARGS,
          "Add a new object and return its id" },
        { "firstObject", (PyCFunction) MetaData_firstObject, // FIXME: change to firstRowId
          METH_NOARGS,
          "Goto first metadata object, return its object id" },
        { "lastObject", (PyCFunction) MetaData_lastObject, // FIXME: change to lastRowId
          METH_NOARGS,
          "Goto last metadata object, return its object id" },
        { "size", (PyCFunction) MetaData_size, METH_NOARGS,
          "Return number of objects in MetaData" },
        { "getParsedLines", (PyCFunction) MetaData_getParsedLines, METH_NOARGS,
          "Return number of objects in MetaData file, even if read with maxRows > 0" },
        { "isEmpty", (PyCFunction) MetaData_isEmpty,
          METH_NOARGS,
          "Check whether the MetaData is empty" },
        { "clear", (PyCFunction) MetaData_clear, METH_NOARGS,
          "Clear MetaData" },
        { "getColumnFormat",
          (PyCFunction) MetaData_getColumnFormat,
          METH_NOARGS, "Get column format info" },
        { "setColumnFormat",
          (PyCFunction) MetaData_setColumnFormat,
          METH_VARARGS, "Set column format info" },
        { "setValue", (PyCFunction) MetaData_setValue,
          METH_VARARGS,
          "Set the value for column(label) for a given object" },
        { "setValueCol", (PyCFunction) MetaData_setValueCol,
          METH_VARARGS,
          "Set the same value for column(label) for all objects" },
        { "removeLabel", (PyCFunction) MetaData_removeLabel,
          METH_VARARGS,
          "Remove a label if exists. The values are still in the table." },
        { "getValue", (PyCFunction) MetaData_getValue,
          METH_VARARGS, "Get the value for column(label)" },
        { "getRow", (PyCFunction) MetaData_getRow,
          METH_VARARGS, "Get a complete row (as dict)" },
        { "setRow", (PyCFunction) MetaData_setRow,
          METH_VARARGS, "Set a complete row (dict)" },
        { "getColumnValues", (PyCFunction) MetaData_getColumnValues,
          METH_VARARGS, "Get all values value from column(label)" },
        { "setColumnValues", (PyCFunction) MetaData_setColumnValues,
          METH_VARARGS, "Set all values value from column(label)" },
        { "getActiveLabels",
          (PyCFunction) MetaData_getActiveLabels,
          METH_VARARGS,
          "Return a list with the labels of the Metadata" },
        { "getMaxStringLength",
          (PyCFunction) MetaData_getMaxStringLength,
          METH_VARARGS,
          "Return the maximun length of a value on this column(label)" },
        { "containsLabel",
          (PyCFunction) MetaData_containsLabel,
          METH_VARARGS,
          "True if this metadata contains this label" },
        { "addLabel", (PyCFunction) MetaData_addLabel,
          METH_VARARGS, "Add a new label to MetaData" },
        { "addItemId", (PyCFunction) MetaData_addItemId,
          METH_VARARGS, "Add a item id to the MetaData" },
        { "fillConstant", (PyCFunction) MetaData_fillConstant,
          METH_VARARGS, "Fill a column with constant value" },
        { "fillRandom", (PyCFunction) MetaData_fillRandom,
          METH_VARARGS, "Fill a column with random value" },
        { "fillExpand", (PyCFunction) MetaData_fillExpand,
          METH_VARARGS, "Fill several columns expanding a values from a column with a path to a row metadata" },
        { "copyColumn", (PyCFunction) MetaData_copyColumn,
          METH_VARARGS, "Copy the values of one column to another" },
        { "copyColumnTo", (PyCFunction) MetaData_copyColumnTo,
          METH_VARARGS, "Copy the values of one column to another in other md" },
        { "makeAbsPath", (PyCFunction) MetaData_makeAbsPath,
          METH_VARARGS,
          "Make filenames with absolute paths" },
        { "importObjects",
          (PyCFunction) MetaData_importObjects,
          METH_VARARGS,
          "Import objects from another metadata" },
        { "removeObjects",
          (PyCFunction) MetaData_removeObjects,
          METH_VARARGS, "Remove objects from metadata" },
        { "removeDisabled",
          (PyCFunction) MetaData_removeDisabled,
          METH_VARARGS, "Remove disabled objects from metadata" },
        { "aggregateSingle",
          (PyCFunction) MetaData_aggregateSingle,
          METH_VARARGS,
          "Aggregate operation in metadata (double single value result)" },
        { "aggregateSingleInt",
          (PyCFunction) MetaData_aggregateSingleInt,
          METH_VARARGS,
          "Aggregate operation in metadata (int single value result)" },
        { "aggregate", (PyCFunction) MetaData_aggregate,
          METH_VARARGS,
          "Aggregate operation in metadata. The results is stored in self." },
        { "aggregateMdGroupBy", (PyCFunction) MetaData_aggregateMdGroupBy,
          METH_VARARGS,
          "Aggregate operation in metadata, may use several MDL for aggregation."
          " The results is stored in self." },
        { "unionAll", (PyCFunction) MetaData_unionAll,
          METH_VARARGS,
          "Union of two metadatas. The results is stored in self." },
        { "merge", (PyCFunction) MetaData_merge, METH_VARARGS,
          "Merge columns of two metadatas. The results is stored in self." },
          {
                "join1",
                (PyCFunction) MetaData_join1,
                METH_VARARGS,
                "join between two metadatas using label as common attribute. The results is stored in self." },
          {
                "join2",
                (PyCFunction) MetaData_join2,
                METH_VARARGS,
                "join between two metadatas, md1.label1=md2.label2. The results is stored in self." },
        {
              "joinNatural",
              (PyCFunction) MetaData_joinNatural,
              METH_VARARGS,
              "natural join between two metadatas. The results is stored in self." },
        {
            "addIndex",
            (PyCFunction) MetaData_addIndex,
            METH_VARARGS,
            "Create index so search is faster." },
        { "readPlain", (PyCFunction) MetaData_readPlain,
          METH_VARARGS,
          "Import metadata from a plain text file." },
          { "addPlain", (PyCFunction) MetaData_addPlain,
          METH_VARARGS,
          "Add imported metadata from a plain text file." },
        {"intersection",
         (PyCFunction) MetaData_intersection,
         METH_VARARGS,
         "Intersection of two metadatas using a common label. The results is stored in self." },
        { "setComment", (PyCFunction) MetaData_setComment,
          METH_VARARGS, "Set comment in Metadata." },
        { "getComment", (PyCFunction) MetaData_getComment,
          METH_VARARGS, "Get comment in Metadata." },
        { "operate", (PyCFunction) MetaData_operate,
          METH_VARARGS, "Replace values in some column." },
        { "replace", (PyCFunction) MetaData_replace,
          METH_VARARGS, "Basic operations on columns data." },
        {"randomize",
         (PyCFunction) MetaData_randomize,
         METH_VARARGS,
         "Randomize another metadata and keep in self." },
        {"selectPart",
         (PyCFunction) MetaData_selectPart,
         METH_VARARGS,
         "select a part of another metadata starting from start and with a number of objects" },
        { "removeDuplicates", (PyCFunction) MetaData_removeDuplicates,
          METH_VARARGS, "Remove duplicate rows" },
        { "renameColumn", (PyCFunction) MetaData_renameColumn,
          METH_VARARGS, "Rename one column" },
        {
            "sort", (PyCFunction) MetaData_sort,
            METH_VARARGS,
            "Sort metadata according to a label" },
        { nullptr } /* Sentinel */
    };//MetaData_methods


PyTypeObject MetaDataType =
    {
        PyObject_HEAD_INIT(0)
        "xmipp.MetaData", /*tp_name*/
        sizeof(MetaDataObject), /*tp_basicsize*/
        0, /*tp_itemsize*/
        (destructor)MetaData_dealloc, /*tp_dealloc*/
        0, /*tp_vectorcall_offset*/
        /*MetaData_print, /*tp_print*/
        0, /*tp_getattr*/
        0, /*tp_setattr*/
        0, /* tp_as_async */
        MetaData_repr, /*tp_repr*/
        0, /*tp_as_number*/
        0, /*tp_as_sequence*/
        0, /*tp_as_mapping*/
        0, /*tp_hash */
        0, /*tp_call*/
        0, /*tp_str*/
        0, /*tp_getattro*/
        0, /*tp_setattro*/
        0, /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT, /*Py_TPFLAGS_HAVE_ITER, */ /*tp_flags*/
        "Python wrapper to Xmipp MetaData class",/* tp_doc */
        0, /* tp_traverse */
        0, /* tp_clear */
        MetaData_RichCompareBool, /* tp_richcompare */
        0, /* tp_weaklistoffset */
        MetaData_iter, /* tp_iter */
        MetaData_iternext, /* tp_iternext */
        MetaData_methods, /* tp_methods */
        0, /* tp_members */
        0, /* tp_getset */
        0, /* tp_base */
        0, /* tp_dict */
        0, /* tp_descr_get */
        0, /* tp_descr_set */
        0, /* tp_dictoffset */
        0, /* tp_init */
        0, /* tp_alloc */
        MetaData_new, /* tp_new */
    };//MetaDataType

/* Destructor */
void MetaData_dealloc(MetaDataObject* self)
{
    self->~MetaDataObject(); // Call the destructor
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Constructor */
PyObject *
MetaData_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    auto *self = (MetaDataObject*)type->tp_alloc(type, 0);

    if (self != nullptr)
    {
        PyObject *input = nullptr;
        PyObject *pyStr = nullptr;
        PyArg_ParseTuple(args, "|O", &input);
        if (input != nullptr)
        {
            try
            {
                if (MetaData_Check(input))
                    self->metadata = std::make_unique<MetaDataDb>(MetaData_Value(input));
                else if ((pyStr = PyObject_Str(input)) != nullptr)
                {
                    const char *str = PyUnicode_AsUTF8(pyStr);
                    self->metadata = std::make_unique<MetaDataDb>(str);
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError,
                                    "MetaData_new: Bad string value for reading metadata");
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
        {
            self->metadata = std::make_unique<MetaDataDb>();
        }
    }
    return (PyObject *)self;
}

int MetaData_print(PyObject *obj, FILE *fp, int flags)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        std::stringstream ss;
        self->metadata->write(ss);
        fprintf(fp, "%s", ss.str().c_str());
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
        return -1;
    }
    return 0;
}

/* String representation */
PyObject *
MetaData_repr(PyObject * obj)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);
    return PyUnicode_FromString(
               (self->metadata->getFilename() + "(MetaData)").c_str());
}

/* MetaData compare function */
PyObject*
MetaData_RichCompareBool(PyObject * obj, PyObject * obj2, int opid)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);
    auto *md2 = (MetaDataObject*) obj2;
    int result = -1;

    if (self != nullptr && md2 != nullptr)
    {
        try
        {
            if (opid == Py_EQ)
                {
                    if (*(self->metadata) == *(md2->metadata))
                        Py_RETURN_TRUE;
                    else
                       Py_RETURN_FALSE;
                }
            else if  (opid == Py_NE)
                    {
                        if (*(self->metadata) == *(md2->metadata))
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
}

/* read */
PyObject *
MetaData_read(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);

    if (self != nullptr)
    {
        PyObject *list = nullptr; //list can be a list of labels or maxRows
        PyObject *input = nullptr;
        PyObject *pyStr = nullptr;
        const char *str = nullptr;
        if (PyArg_ParseTuple(args, "O|O", &input,  &list))
        {
            try
            {
                if ((pyStr = PyObject_Str(input)) != nullptr)
                {
                    str = PyUnicode_AsUTF8(pyStr);
                    if (list != nullptr)
                    {
                        if (PyList_Check(list))
                        {
                            size_t size = PyList_Size(list);
                            PyObject * item = nullptr;
                            int iValue = 0;
                            MDLabelVector vValue(size);
                            for (size_t i = 0; i < size; ++i)
                            {
                                item = PyList_GetItem(list, i);
                                if (!PyLong_Check(item))
                                {
                                    PyErr_SetString(PyExc_TypeError,
                                                    "MDL labels must be integers (MDLABEL)");
                                    return nullptr;
                                }
                                iValue = PyLong_AsLong(item);
                                vValue[i] = (MDLabel)iValue;
                            }
                            self->metadata->read(str,&vValue);
                        }
                        else if (PyLong_Check(list)){
                          auto maxRows = (size_t) PyLong_AsLong(list);
                          self->metadata->setMaxRows(maxRows);
                          self->metadata->read(str);
                        }
                    }
                    else
                        self->metadata->read(str);
                    Py_RETURN_NONE;
                }
                else
                    return nullptr;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.what());
                return nullptr;
            }
        }
    }
    return nullptr;
}

/* read */
PyObject *
MetaData_readPlain(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);

    if (self != NULL)
    {
        PyObject *input = NULL;
        PyObject *input2 = NULL;
        PyObject *pyStr = NULL;
        PyObject *pyLabels = NULL;
        PyObject *pySep = NULL;
        const char *str = NULL;
        const char *labels = NULL;

        if (PyArg_ParseTuple(args, "OO|O", &input, &input2, &pySep))
        {
            try
            {
                pyStr = PyObject_Str(input);
                pyLabels = PyObject_Str(input2);

                if ((NULL != pyStr) && (NULL != pyLabels))
                {
                    str = PyUnicode_AsUTF8(pyStr);
                    labels = PyUnicode_AsUTF8(pyLabels);

                    self->metadata->readPlain(str, labels);
                    Py_RETURN_NONE;
                }
                else
                    return NULL;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.what());
                return NULL;
            }
        }
    }
    return NULL;
}

/* addPlain */
PyObject *
MetaData_addPlain(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);

    if (self != NULL)
    {
        PyObject *input = NULL;
        PyObject *input2 = NULL;
        PyObject *pyStr = NULL;
        PyObject *pyLabels = NULL;
        PyObject *pySep = NULL;
        const char *str = NULL;
        const char *labels = NULL;

        if (PyArg_ParseTuple(args, "OO|O", &input, &input2, &pySep))
        {
            try
            {
                pyStr = PyObject_Str(input);
                pyLabels = PyObject_Str(input2);
                if ((NULL != pyStr) && (NULL != pyLabels))
                {
                    str = PyUnicode_AsUTF8(pyStr);
                    labels = PyUnicode_AsUTF8(pyLabels);
                    self->metadata->addPlain(str, labels);
                    Py_RETURN_NONE;
                }
                else
                    return NULL;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.what());
                return NULL;
            }
        }
    }
    return NULL;
}

/* read block */
PyObject *
MetaData_readBlock(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);

    if (self != NULL)
    {
        PyObject *input = NULL;
        PyObject *blockName = NULL;
        PyObject *pyStr = NULL;
        PyObject *pyStrBlock = NULL;
        const char *str = NULL;
        const char *strBlock = NULL;
        if (PyArg_ParseTuple(args, "OO", &input, &blockName))
        {
            try
            {
                pyStr = PyObject_Str(input);
                pyStrBlock = PyObject_Str(blockName);
                if ((NULL != pyStr) && (NULL != pyStrBlock))
                {
                    str = PyUnicode_AsUTF8(pyStr);
                    strBlock = PyUnicode_AsUTF8(pyStrBlock);
                    self->metadata->read((std::string) strBlock + "@" + str,
                                         NULL);
                    Py_RETURN_NONE;
                }
                else
                    return NULL;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.what());
                return NULL;
            }
        }
    }
    return NULL;
}

/* write */
PyObject *
MetaData_write(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);
    WriteModeMetaData wmd;
    wmd = MD_OVERWRITE;
    if (self != NULL)
    {
        PyObject *input = nullptr;
        PyObject *pyStr = nullptr;
        const char *str = NULL;
        if (PyArg_ParseTuple(args, "O|i", &input, &wmd))
        {
            try
            {
                if ((pyStr = PyObject_Str(input)) != NULL)
                {
                    str = PyUnicode_AsUTF8(pyStr);
                    self->metadata->write(str, (WriteModeMetaData) wmd);
                    Py_RETURN_NONE;
                }
                else
                    return NULL;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.what());
                return NULL;
            }
        }
    }
    return NULL;
}

/* append */
PyObject *
MetaData_append(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);

    if (self != nullptr)
    {
        PyObject *input = nullptr;
        if (PyArg_ParseTuple(args, "O", &input))
        {
            try
            {
                if (PyUnicode_Check(input))
                   {
                      const char *str = PyUnicode_AsUTF8(input);
                      self->metadata->append(str);
                   }
                else if (FileName_Check(input))
                    self->metadata->append(FileName_Value(input));
                else
                    return nullptr;
                Py_RETURN_NONE;
            }
            catch (XmippError &xe)
            {
                PyErr_SetString(PyXmippError, xe.what());
                return nullptr;
            }
        }
    }
    return nullptr;
}
/* addObject */
PyObject *
MetaData_addObject(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);
    return PyLong_FromUnsignedLong(self->metadata->addObject());
}
/* firstObject */
PyObject *
MetaData_firstObject(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);
    return PyLong_FromUnsignedLong(self->metadata->firstRowId());
}
/* lastObject */
PyObject *
MetaData_lastObject(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);
    return PyLong_FromUnsignedLong(self->metadata->lastRowId());
}
/* size */
PyObject *
MetaData_size(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        return PyLong_FromUnsignedLong(self->metadata->size());
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* getParsedLines */
PyObject *
MetaData_getParsedLines(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        return PyLong_FromUnsignedLong(self->metadata->getParsedLines());
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* isEmpty */
PyObject *
MetaData_isEmpty(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        if (self->metadata->isEmpty())
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}
/* getColumnFormat */
PyObject *
MetaData_getColumnFormat(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        if (self->metadata->isColumnFormat())
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}
/* setColumnFormat */
PyObject *
MetaData_setColumnFormat(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *input = nullptr;
    if (PyArg_ParseTuple(args, "O", &input))
    {
        try
        {
            if (PyBool_Check(input))
            {
                const auto *self = reinterpret_cast<MetaDataObject*>(obj);
                self->metadata->setColumnFormat(input == Py_True);
                Py_RETURN_NONE;
            }
            else
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::setColumnFormat: Expecting boolean value");
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* setValue */
PyObject *
MetaData_setValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    size_t objectId = BAD_OBJID;
    PyObject *pyValue; //Only used to skip label and value

    if (PyArg_ParseTuple(args, "iOk", &label, &pyValue, &objectId))
    {
        try
        {
            auto object = createMDObject(label, pyValue);
            if (!object)
                return nullptr;
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->setValue(*object, objectId);
            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

PyObject *
MetaData_setRow(PyObject *obj, PyObject *args, PyObject *)
{
    PyObject *dict;
    if (size_t objectId = BAD_OBJID; PyArg_ParseTuple(args, "Ok", &dict, &objectId)) {
        try
        {
            PyObject *key;
            PyObject *value;
            Py_ssize_t pos = 0;
            MDRowVec row;
            while (PyDict_Next(dict, &pos, &key, &value)) {
                auto label = PyLong_AsLong(key);
                auto object = createMDObject(static_cast<int>(label), value);
                row.setValue(*object);
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->setRow(row, objectId);
            Py_RETURN_TRUE;
        }
        catch (const XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* setValueCol */
PyObject *
MetaData_setValueCol(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    PyObject *pyValue; //Only used to skip label and value

    if (PyArg_ParseTuple(args, "iO", &label, &pyValue))
    {
        try
        {
            auto object = createMDObject(label, pyValue);
            if (!object)
                return nullptr;
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->setValueCol(*object);
            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* drop column */
PyObject *
MetaData_removeLabel(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    if (PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            if (self->metadata->removeLabel((MDLabel) label))
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* getValue */
PyObject *
MetaData_getValue(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    size_t objectId = BAD_OBJID;
    PyObject *pyValue;

    if (PyArg_ParseTuple(args, "ik", &label, &objectId))
    {
        try
        {
            auto object = MDObject((MDLabel) label);
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            if (self->metadata->getValue(object, objectId))
            {
                pyValue = getMDObjectValue(&object);
                return pyValue;
            }
            else
            {
                Py_RETURN_NONE;
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

PyObject *
MetaData_getRow(PyObject *obj, PyObject *args, PyObject *)
{
    if (size_t objectId = BAD_OBJID; PyArg_ParseTuple(args, "k", &objectId))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            const auto row = self->metadata->getRowSql(objectId);
            PyObject *d = PyODict_New();
            for (const auto *o : row) {
                PyODict_SetItem(d, PyLong_FromLong(o->label), getMDObjectValue(o));
            }
            return d;
        }
        catch (const XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}



PyObject *
MetaData_getColumnValues(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    if (PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);

            std::vector<MDObject> v;
            self->metadata->getColumnValues((MDLabel) label,v);

            size_t size=v.size();
            PyObject * list = PyList_New(size);

            for (size_t i = 0; i < size; ++i)
                PyList_SetItem(list, i, getMDObjectValue(&(v[i])));

            return list;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* setValue */
PyObject *
MetaData_setColumnValues(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    PyObject *list = nullptr;
    if (PyArg_ParseTuple(args, "iO", &label, &list))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            size_t size=PyList_Size(list);
            bool addObjects=(self->metadata->size()==0);
            MDObject object((MDLabel) label);
            if (addObjects)
            {
                for (size_t i=0; i<size; ++i)
                {
                    size_t id=self->metadata->addObject();
                    setMDObjectValue(&object,PyList_GetItem(list,i));
                    self->metadata->setValue(object,id);
                }
            }
            else
            {
                if (self->metadata->size()!=size)
                    PyErr_SetString(PyXmippError, "Metadata size different from list size");
                size_t i=0;
                for (size_t objId : self->metadata->ids())
                {
                    setMDObjectValue(&object,PyList_GetItem(list,i++));
                    self->metadata->setValue(object, objId);
                }
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
            return nullptr;
        }
    }
    Py_RETURN_NONE;
}

/* containsLabel */
PyObject *
MetaData_getActiveLabels(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        std::vector<MDLabel> labels = self->metadata->getActiveLabels();
        int size = labels.size();
        PyObject * list = PyList_New(size);

        for (int i = 0; i < size; ++i)
            PyList_SetItem(list, i, PyLong_FromLong(labels.at(i)));

        return list;

    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}
/* containsLabel */
PyObject *
xmipp_getBlocksInMetaDataFile(PyObject *obj, PyObject *args)
{
    PyObject *input;
    const char *fileName = NULL;
    FileName fn;
    StringVector blocks;

    try
    {
        if (PyArg_ParseTuple(args, "O", &input))
        {

            if (PyUnicode_Check(input)){
                 fileName = PyUnicode_AsUTF8(input);
                 fn = fileName;
            }
            else if (FileName_Check(input))
                fn = FileName_Value(input);
            else
                return NULL;
            getBlocksInMetaDataFile(fn, blocks);
            int size = blocks.size();
            PyObject * list = PyList_New(size);

            for (int i = 0; i < size; ++i)
            {
                PyList_SetItem(list, i, PyUnicode_FromString(blocks[i].c_str()));
            }
            return list;
        }
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return NULL;
}

/* containsLabel */
PyObject *
MetaData_getMaxStringLength(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    if (PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            int length = self->metadata->getMaxStringLength((MDLabel) label);

            return PyLong_FromLong(length);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* containsLabel */
PyObject *
MetaData_containsLabel(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    if (PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            if (self->metadata->containsLabel((MDLabel) label))
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* addLabel */
PyObject *
MetaData_addLabel(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    int pos = -1;
    if (PyArg_ParseTuple(args, "i|i", &label, &pos))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->addLabel((MDLabel) label, pos);
            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* addLabel */
PyObject *
MetaData_addItemId(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        self->metadata->addItemId();
        Py_RETURN_NONE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* fillConstant */
PyObject *
MetaData_fillConstant(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    PyObject *pyValue = nullptr;
    PyObject *pyStr = nullptr;
    if (PyArg_ParseTuple(args, "i|O", &label, &pyValue))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            if ((pyStr = PyObject_Str(pyValue)) != nullptr)
            {
                const char * str = PyUnicode_AsUTF8(pyStr);
                if (str != nullptr)
                {
                    self->metadata->fillConstant((MDLabel) label, str);
                    Py_RETURN_TRUE;
                }
            }
            PyErr_SetString(PyXmippError, "MetaData.fillConstant: couldn't convert second argument to string");
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* fillRandom */
PyObject *
MetaData_fillExpand(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;

    if (PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->fillExpand((MDLabel) label);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* fillConstant */
PyObject *
MetaData_fillRandom(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    double op1 = 0.;
    double op2 = 0.;
    double op3 = 0.;
    PyObject *pyValue = nullptr;
    PyObject *pyStr = nullptr;

    if (PyArg_ParseTuple(args, "iOdd|d", &label, &pyValue, &op1, &op2, &op3))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            if ((pyStr = PyObject_Str(pyValue)) != nullptr)
            {
                const char * str = PyUnicode_AsUTF8(pyStr);
                if (str != nullptr)
                {
                    self->metadata->fillRandom((MDLabel) label, str, op1, op2, op3);
                    Py_RETURN_TRUE;
                }
            }
            PyErr_SetString(PyXmippError, "MetaData.fillRandom: couldn't convert second argument to string");
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* copyColumn */
PyObject *
MetaData_copyColumn(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int labelDst;
    int labelSrc;
    if (PyArg_ParseTuple(args, "ii", &labelDst, &labelSrc))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->copyColumn((MDLabel)labelDst, (MDLabel)labelSrc);
            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* copyColumn */
PyObject *
MetaData_renameColumn(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject * oldLabel = nullptr;
    PyObject * newLabel = nullptr;
    if (PyArg_ParseTuple(args, "OO", &oldLabel, &newLabel))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            if(PyLong_Check ( oldLabel ) && PyLong_Check ( newLabel ))
            {
                self->metadata->renameColumn((MDLabel) PyLong_AsLong (oldLabel),
                                             (MDLabel) PyLong_AsLong (newLabel));
            }
            else if (PyList_Check(oldLabel)&& PyList_Check ( newLabel ))
            {
                size_t size = PyList_Size(oldLabel);
                PyObject * itemOld = nullptr;
                PyObject * itemNew = nullptr;
                int iOldValue = 0;
                int iNewValue = 0;
                std::vector<MDLabel> vOldValue(size);
                std::vector<MDLabel> vNewValue(size);
                for (size_t i = 0; i < size; ++i)
                {
                    itemOld = PyList_GetItem(oldLabel, i);
                    itemNew = PyList_GetItem(newLabel, i);
                    if (!PyLong_Check(itemOld) || !PyLong_Check(itemNew))
                    {
                        PyErr_SetString(PyExc_TypeError,
                                        "MDL labels must be integers (MDLABEL)");
                        return nullptr;
                    }
                    iOldValue = PyLong_AsLong(itemOld);
                    iNewValue = PyLong_AsLong(itemNew);
                    vOldValue[i] = (MDLabel)iOldValue;
                    vNewValue[i] = (MDLabel)iNewValue;
                }
                //self->metadata->read(str,&vValue);
                self->metadata->renameColumn(vOldValue,vNewValue);
            }

            Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* copyColumnTo */
PyObject *
MetaData_copyColumnTo(PyObject *obj, PyObject *args, PyObject *kwargs);

/* removeObjects */
PyObject *
MetaData_removeObjects(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyQuery = nullptr;

    if (PyArg_ParseTuple(args, "O", &pyQuery))
    {
        try
        {
            if (!MDQuery_Check(pyQuery))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::removeObjects: Expecting MDQuery as second arguments");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->removeObjects(MDQuery_Value(pyQuery));
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* removeObjects */
PyObject *
MetaData_removeDisabled(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        self->metadata->removeDisabled();
        Py_RETURN_NONE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* Make absolute path */
PyObject *
MetaData_makeAbsPath(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label = (MDLabel) MDL_IMAGE;
    if (PyArg_ParseTuple(args, "|i", &label))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->makeAbsPath((MDLabel) label);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* clear */
PyObject *
MetaData_clear(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        self->metadata->clear();
        Py_RETURN_NONE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
        return nullptr;
    }
}

/** Iteration functions */
PyObject *
MetaData_iter(PyObject *obj)
{
    try
    {
        auto *self = reinterpret_cast<MetaDataObject*>(obj);
        self->iter = std::make_unique<MetaDataDb::id_iterator>(self->metadata->ids().begin());
        self->iter_end = std::make_unique<MetaDataDb::id_iterator>(self->metadata->ids().end());
        Py_INCREF(self);
        return (PyObject *) self;
        //return Py_BuildValue("l", self->metadata->iteratorBegin());
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}
PyObject *
MetaData_iternext(PyObject *obj)
{
    try
    {
        const auto *self = reinterpret_cast<MetaDataObject*>(obj);
        if (*(self->iter) == *(self->iter_end))
            return nullptr;
        size_t objId = **(self->iter);
        ++(*self->iter);
        //type format should be "n" instead of "i" but I put i since python 2.4 does not support n
        return Py_BuildValue("i", objId);
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}
/** Sort Metadata */
PyObject *
MetaData_sort(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label = MDL_IMAGE;
    auto *ascPy = Py_True;
    bool asc=true;
    int limit   = -1;
    int offset  =  0;
    if (PyArg_ParseTuple(args, "|iOii", &label,&ascPy,&limit,&offset))
    {
        try
        {
            if (PyBool_Check(ascPy))
                asc = (ascPy == Py_True);
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            MetaDataDb MDaux = *(self->metadata);
            self->metadata->clear();
            self->metadata->sort(MDaux, (MDLabel) label,asc,limit,offset);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/** remove Duplicate rows Metadata */
PyObject *
MetaData_removeDuplicates(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label = MDL_UNDEFINED;

    if (PyArg_ParseTuple(args, "|i", &label))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            MetaDataDb MDaux = *(self->metadata);
            self->metadata->clear();
            self->metadata->removeDuplicates(MDaux, (MDLabel)label);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* importObjects */
PyObject *
MetaData_importObjects(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd = nullptr;
    PyObject *pyQuery = nullptr;

    if (PyArg_ParseTuple(args, "OO", &pyMd, &pyQuery))
    {
        try
        {
            if (!MetaData_Check(pyMd))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::importObjects: Expecting MetaData as first argument");
                return nullptr;
            }
            if (!MDQuery_Check(pyQuery))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::importObjects: Expecting MDQuery as second argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->importObjects(MetaData_Value(pyMd),
                                          MDQuery_Value(pyQuery));
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* aggregateSingle */
PyObject *
MetaData_aggregateSingle(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    AggregateOperation op;
    MDLabel label;
    PyObject *pyValue;

    if (PyArg_ParseTuple(args, "ii", &op, &label))
    {
        try
        {
            auto object = MDObject(label);
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->aggregateSingle(object, op, label);
            pyValue = getMDObjectValue(&object);
            return pyValue;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* aggregateSingleInt */
PyObject *
MetaData_aggregateSingleInt(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    AggregateOperation op;
    MDLabel label;
    PyObject *pyValue;

    if (PyArg_ParseTuple(args, "ii", &op, &label))
    {
        try
        {
            auto object = MDObject(label);
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->aggregateSingleInt(object, op, label);
            pyValue = getMDObjectValue(&object);
            return pyValue;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/*aggregate*/
PyObject *
MetaData_aggregate(PyObject *obj, PyObject *args, PyObject *kwargs)
{

    AggregateOperation op;
    MDLabel aggregateLabel;
    MDLabel operateLabel;
    MDLabel resultLabel;
    PyObject *pyMd = nullptr;

    if (PyArg_ParseTuple(args, "Oiiii", &pyMd, &op, &aggregateLabel,
                         &operateLabel, &resultLabel))
    {
        try
        {
            if (!MetaData_Check(pyMd))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::aggregate: Expecting MetaData as first argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->aggregate(MetaData_Value(pyMd),
                                      op, aggregateLabel,
                                      operateLabel, resultLabel);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/*aggregate*/
PyObject *
MetaData_aggregateMdGroupBy(PyObject *obj, PyObject *args, PyObject *kwargs)
{

    AggregateOperation op;
    PyObject *aggregateLabel= nullptr;
    MDLabel operateLabel;
    MDLabel resultLabel;
    PyObject *pyMd = nullptr;

    if (PyArg_ParseTuple(args, "OiOii", &pyMd, &op, &aggregateLabel,
                         &operateLabel, &resultLabel))
    {
        try
        {
            if (!MetaData_Check(pyMd))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::aggregateMdGroupBy: Expecting MetaData as first argument");
                return nullptr;
            }
            if (!PyList_Check(aggregateLabel))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::aggregateMdGroupBy: Input must be a mdl List not a mdl label");
                return nullptr;

            }
            size_t size = PyList_Size(aggregateLabel);
            std::vector<MDLabel> vaggregateLabel(size);
            PyObject * item = nullptr;
            int iValue = 0;
            for (size_t i = 0; i < size; ++i)
            {
                item = PyList_GetItem(aggregateLabel, i);
                if (!PyLong_Check(item))
                {
                    PyErr_SetString(PyExc_TypeError,
                                    "MDL labels must be integers (MDLABEL)");
                    return nullptr;
                }
                iValue = PyLong_AsLong(item);
                vaggregateLabel[i] = (MDLabel)iValue;
            }

            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->aggregateGroupBy(MetaData_Value(pyMd),
                                             (AggregateOperation) op, vaggregateLabel,
                                             operateLabel, resultLabel);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* UnionAll */
PyObject *
MetaData_unionAll(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd = nullptr;

    if (PyArg_ParseTuple(args, "O", &pyMd))
    {
        try
        {
            if (!MetaData_Check(pyMd))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::unionAll: Expecting MetaData as first argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->unionAll(MetaData_Value(pyMd));
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* merge */
PyObject *
MetaData_merge(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd = nullptr;

    if (PyArg_ParseTuple(args, "O", &pyMd))
    {
        try
        {
            if (!MetaData_Check(pyMd))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::merge: Expecting MetaData as first argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->merge(MetaData_Value(pyMd));
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* setComment */
PyObject *
MetaData_setComment(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    char * str = nullptr;
    if (PyArg_ParseTuple(args, "s", &str))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->setComment(str);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* getComment */
PyObject *
MetaData_getComment(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    const auto *self = reinterpret_cast<MetaDataObject*>(obj);
    return PyUnicode_FromString(self->metadata->getComment().c_str());
}

/* addIndex MetaData_join*/
PyObject *
MetaData_addIndex(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    if (PyArg_ParseTuple(args, "i", &label))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->addIndex((MDLabel)label);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}
/* join */
PyObject *
MetaData_join1(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int labelLeft;
    PyObject *pyMdLeft = nullptr;
    PyObject *pyMdright = nullptr;
    JoinType jt=LEFT;

    if (PyArg_ParseTuple(args, "OOi|i", &pyMdLeft, &pyMdright, &labelLeft, &jt))
    {
        try
        {
            if (!MetaData_Check(pyMdLeft))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::join: Expecting MetaData as first argument");
                return nullptr;
            }
            if (!MetaData_Check(pyMdright))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::join: Expecting MetaData as second argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->join1(MetaData_Value(pyMdLeft),
                                      MetaData_Value(pyMdright),
                                      (MDLabel) labelLeft,
                                      (JoinType) jt);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* join */
PyObject *
MetaData_join2(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int labelLeft;
    int labelRight;
    PyObject *pyMdLeft = nullptr;
    PyObject *pyMdright = nullptr;
    JoinType jt=LEFT;

    if (PyArg_ParseTuple(args, "OOii|i", &pyMdLeft, &pyMdright, &labelLeft,&labelRight, &jt))
    {
        try
        {
            if (!MetaData_Check(pyMdLeft))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::join: Expecting MetaData as first argument");
                return nullptr;
            }
            if (!MetaData_Check(pyMdright))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::join: Expecting MetaData as second argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->join2(MetaData_Value(pyMdLeft),
                                  MetaData_Value(pyMdright),
                                  (MDLabel) labelLeft,
                                  (MDLabel) labelRight, (JoinType) jt);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* join */
PyObject *
MetaData_joinNatural(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMdLeft = nullptr;
    PyObject *pyMdright = nullptr;

    if (PyArg_ParseTuple(args, "OO", &pyMdLeft, &pyMdright))
    {
        try
        {
            if (!MetaData_Check(pyMdLeft))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::joinNatural: Expecting MetaData as first argument");
                return nullptr;
            }
            if (!MetaData_Check(pyMdright))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::joinNatural: Expecting MetaData as second argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->joinNatural(MetaData_Value(pyMdLeft),
                                  MetaData_Value(pyMdright));
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* Intersection */
PyObject *
MetaData_intersection(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    PyObject *pyMd = nullptr;

    if (PyArg_ParseTuple(args, "Oi", &pyMd, &label))
    {
        try
        {
            if (!MetaData_Check(pyMd))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::intersection: Expecting MetaData as first argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->intersection(MetaData_Value(pyMd), (MDLabel) label);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* Basic operations on columns data. */
PyObject *
MetaData_operate(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    char * str = nullptr;

    if (PyArg_ParseTuple(args, "s", &str))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->operate(str);
            //free(str);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    // free(str);
    return nullptr;
}

/*Replace string values in some label. */
PyObject *
MetaData_replace(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int label;
    char * oldStr = nullptr;
    char * newStr = nullptr;
    if (PyArg_ParseTuple(args, "iss", &label, &oldStr, &newStr))
    {
        try
        {
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->replace((MDLabel)label, oldStr, newStr);
            //free(oldStr);
            //free(newStr);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    //free(oldStr);
    //free(newStr);
    return nullptr;
}

/* Randomize */
PyObject *
MetaData_randomize(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd = nullptr;

    if (PyArg_ParseTuple(args, "O", &pyMd))
    {
        try
        {
            if (!MetaData_Check(pyMd))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::randomize: Expecting MetaData as first argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->randomize(MetaData_Value(pyMd));
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* Select Part */
PyObject *
MetaData_selectPart(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    auto label=(int)MDL_OBJID;
    size_t start;
    size_t numberOfObjects;
    PyObject *pyMd = nullptr;

    if (PyArg_ParseTuple(args, "Okk|i", &pyMd, &start, &numberOfObjects, &label))
    {
        try
        {
            if (!MetaData_Check(pyMd))
            {
                PyErr_SetString(PyExc_TypeError,
                                "MetaData::selectPart: Expecting MetaData as first argument");
                return nullptr;
            }
            const auto *self = reinterpret_cast<MetaDataObject*>(obj);
            self->metadata->selectPart(MetaData_Value(pyMd), start, numberOfObjects, (MDLabel) label);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

std::unique_ptr<MDObject>
createMDObject(int label, PyObject *pyValue)
{
    try
    {
        if (PyBool_Check(pyValue))
        {
            bool bValue = (pyValue == Py_True);
            RETURN_MDOBJECT(bValue);
        }
        if (PyLong_Check(pyValue) && MDL::isInt((MDLabel)label))
        {
            int iValue = PyLong_AS_LONG(pyValue);
            RETURN_MDOBJECT(iValue);
        }
        if (PyLong_Check(pyValue) && MDL::isLong((MDLabel)label))
        {
            size_t value = PyLong_AsUnsignedLong(pyValue);
            RETURN_MDOBJECT(value);
        }
        if (PyUnicode_Check(pyValue))
        {
            const char * str = PyUnicode_AsUTF8(PyObject_Str(pyValue));
            RETURN_MDOBJECT(std::string(str));
        }
        if (FileName_Check(pyValue))
        {
            RETURN_MDOBJECT(*((FileNameObject*)pyValue)->filename);
        }
        if (PyFloat_Check(pyValue))
        {
            double dValue = PyFloat_AS_DOUBLE(pyValue);
            RETURN_MDOBJECT(double(dValue));
        }
        if (PyList_Check(pyValue))
        {
            size_t size = PyList_Size(pyValue);
            PyObject * item = nullptr;
            double dValue = 0.;
            std::vector<double> vValue(size);
            for (size_t i = 0; i < size; ++i)
            {
                item = PyList_GetItem(pyValue, i);
                if (!PyFloat_Check(item))
                {
                    PyErr_SetString(PyExc_TypeError,
                                    "Vectors are only supported for double");
                    return nullptr;
                }
                dValue = PyFloat_AS_DOUBLE(item);
                vValue[i] = dValue;
            }
            RETURN_MDOBJECT(vValue);
        }
        PyErr_SetString(PyExc_TypeError, "Unrecognized type to create MDObject");
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* copyColumnTo */
PyObject *
MetaData_copyColumnTo(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int labelDst;
    int labelSrc;
    PyObject * pyMd;
    if (PyArg_ParseTuple(args, "Oii", &pyMd, &labelDst, &labelSrc))
    {
        try
        {
            if (MetaData_Check(pyMd))
            {
                MetaData_Value(obj).copyColumnTo(MetaData_Value(pyMd), (MDLabel)labelDst, (MDLabel)labelSrc);
                Py_RETURN_TRUE;
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

void setMDObjectValue(MDObject *obj, PyObject *pyValue)
{
    try
    {
        if (PyLong_Check(pyValue))
            obj->setValue((int)PyLong_AS_LONG(pyValue));
        else if (PyLong_Check(pyValue))
            obj->setValue((size_t)PyLong_AsUnsignedLong(pyValue));
        else if (PyUnicode_Check(pyValue))
            {
              const char * str = PyUnicode_AsUTF8(PyObject_Str(pyValue));
              obj->setValue(std::string(str));
            }
        else if (FileName_Check(pyValue))
            obj->setValue((*((FileNameObject*)pyValue)->filename));
        else if (PyFloat_Check(pyValue))
            obj->setValue(PyFloat_AS_DOUBLE(pyValue));
        else if (PyBool_Check(pyValue))
            obj->setValue((pyValue == Py_True));
        else if (PyList_Check(pyValue))
        {
            size_t size = PyList_Size(pyValue);
            PyObject * item = nullptr;
            double dValue = 0.;
            std::vector<double> vValue(size);
            for (size_t i = 0; i < size; ++i)
            {
                item = PyList_GetItem(pyValue, i);
                if (!PyFloat_Check(item))
                {
                    PyErr_SetString(PyExc_TypeError,
                                    "Vectors are only supported for double");
                }
                dValue = PyFloat_AS_DOUBLE(item);
                vValue[i] = dValue;
            }
            obj->setValue(vValue);
        }
        else
            PyErr_SetString(PyExc_TypeError, "Unrecognized type to create MDObject");
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
}

PyObject *
getMDObjectValue(const MDObject * obj)
{
    if (obj->label == MDL_UNDEFINED) //if undefine label, store as a literal string
        return nullptr;
    switch (MDL::labelType(obj->label))
    {
    case LABEL_BOOL: //bools are int in sqlite3
        if (obj->data.boolValue)
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    case LABEL_INT:
        return PyLong_FromLong(obj->data.intValue);
    case LABEL_SIZET:
        return PyLong_FromLong(obj->data.longintValue);
    case LABEL_DOUBLE:
        return PyFloat_FromDouble(obj->data.doubleValue);
    case LABEL_STRING:
        return PyUnicode_FromString(obj->data.stringValue->c_str());
    case LABEL_VECTOR_DOUBLE:
        {
            std::vector<double> & vector = *(obj->data.vectorValue);
            int size = vector.size();
            PyObject * list = PyList_New(size);
            for (int i = 0; i < size; ++i)
                PyList_SetItem(list, i, PyFloat_FromDouble(vector[i]));
            return list;
        }
    default:
            return nullptr;
    }//close switch
    return nullptr;
}

