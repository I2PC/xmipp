/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
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

#include "frm.h"
#include <core/geometry.h>
#include <core/transformations.h>

//#define DEBUG
#ifdef DEBUG
#include <core/xmipp_image.h>
#endif
void alignVolumesFRM(PyObject *pFunc, const MultidimArray<double> &Iref, MultidimArray<double> &I,
		PyObject *Imask,
		double &rot, double &tilt, double &psi, double &x, double &y, double &z, double &score,
		Matrix2D<double> &A,
		int maxshift, double maxFreq, const MultidimArray<int> *mask)
{
	PyObject *pyIref = Python::convertToNumpy(Iref);
	PyObject *pyI = Python::convertToNumpy(I);
	PyObject *pyMask=Py_None;
	if (mask!=NULL)
		pyMask=Python::convertToNumpy(*mask);

//#define DEBUG
#ifdef DEBUG
	Image<double> save;
	save()=I;
	save.write("PPPI.vol");
	save()=Iref;
	save.write("PPPIref.vol");
	std::cout << "Imask=" << Imask << std::endl;
	std::cout << "Press any key\n";
	char c; std::cin >> c;
#endif

	// Call frm
	int bandWidthSphericalHarmonics0=4;
	int bandWidthSphericalHarmonicsF=64;
	int frequencyPixels=(int)(XSIZE(Iref)*maxFreq);
	PyObject *arglistfrm = Py_BuildValue("OOOO(ii)iiO", pyI, Imask, pyIref, Py_None,
			bandWidthSphericalHarmonics0, bandWidthSphericalHarmonicsF, frequencyPixels, maxshift, pyMask);
	PyObject *resultfrm = PyObject_CallObject(pFunc, arglistfrm);
	Py_DECREF(arglistfrm);
	Py_DECREF(pyIref);
	Py_DECREF(pyI);
	if (mask!=NULL)
		Py_DECREF(pyMask);
	if (resultfrm!=NULL)
	{
		PyObject *shift=PyTuple_GetItem(resultfrm,0);
		PyObject *euler=PyTuple_GetItem(resultfrm,1);
		score=PyFloat_AsDouble(PyTuple_GetItem(resultfrm,2));
		double xfrm=PyFloat_AsDouble(PyList_GetItem(shift,0))-XSIZE(Iref)/2;
		double yfrm=PyFloat_AsDouble(PyList_GetItem(shift,1))-YSIZE(Iref)/2;
		double zfrm=PyFloat_AsDouble(PyList_GetItem(shift,2))-ZSIZE(Iref)/2;
		double angz1=PyFloat_AsDouble(PyList_GetItem(euler,0));
		double angz2=PyFloat_AsDouble(PyList_GetItem(euler,1));
		double angx=PyFloat_AsDouble(PyList_GetItem(euler,2));
		Matrix2D<double> Efrm, E(3,3);
		Euler_anglesZXZ2matrix(-angz1, -angx, -angz2, Efrm); // -angles because FRM rotation definition
		                                                     // is the opposite of Xmipp

		// Reorganize the matrix because the Z and X axes in the coordinate system are reversed with
		// respect to Xmipp
		// Exmipp=[0 0 1; 0 1 0; 1 0 0]*Efrm*[0 0 1; 0 1 0; 1 0 0]
		MAT_ELEM(E,0,0)=MAT_ELEM(Efrm, 2, 2); MAT_ELEM(E,0,1)=MAT_ELEM(Efrm, 2, 1); MAT_ELEM(E,0,2)=MAT_ELEM(Efrm, 2, 0);
		MAT_ELEM(E,1,0)=MAT_ELEM(Efrm, 1, 2); MAT_ELEM(E,1,1)=MAT_ELEM(Efrm, 1, 1); MAT_ELEM(E,1,2)=MAT_ELEM(Efrm, 1, 0);
		MAT_ELEM(E,2,0)=MAT_ELEM(Efrm, 0, 2); MAT_ELEM(E,2,1)=MAT_ELEM(Efrm, 0, 1); MAT_ELEM(E,2,2)=MAT_ELEM(Efrm, 0, 0);
		E=E.inv();
		Euler_matrix2angles(E,rot,tilt,psi);
		Py_DECREF(resultfrm);

		x=-zfrm;
		y=-yfrm;
		z=-xfrm;

		// Apply
        Euler_angles2matrix(rot, tilt, psi, A, true);
        Matrix2D<double> Aaux;
        Matrix1D<double> r(3);
        XX(r)=x;
        YY(r)=y;
        ZZ(r)=z;
        translation3DMatrix(r,Aaux);
        A=A*Aaux;
#ifdef DEBUG
		std::cout << "Result: " << rot << " " << tilt << " " << psi << " " << x << " " << y << " " << z << " -> " << score << std::endl;
#endif
	}
	else
	{
		x=y=z=rot=tilt=psi=score=0;
		A.initIdentity(4);

		std::cout << "No result\n";
		std::cout << Python::print_traceback();
	}
}
#undef DEBUG
