/***************************************************************************
 * Authors:     AUTHOR_NAME (jvargas@cnb.csic.es)
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

#include "image_gray_adjust_lsq.h"

#include "core/matrix1d.h"
#include "core/matrix2d.h"
#include "core/alglib/solvers.h"

// Read arguments ==========================================================
void ProgImageGrayAdjustLsq::readParams()
{
    XmippMetadataProgram::readParams();
	referenceFilename = getParam("-r");
}

// Define parameters ==========================================================
void ProgImageGrayAdjustLsq::defineParams()
{
    addUsageLine("Adjust image grayscales according to a reference");
    each_image_produces_an_output = true;
    XmippMetadataProgram::defineParams();
    addParamsLine("   -r       : Reference volume");
}

void ProgImageGrayAdjustLsq::preProcess()
{
	m_volume.read(referenceFilename);
	m_projector = std::make_unique<FourierProjector>(m_volume(), 2.0, 1.0, 3);

}

void ProgImageGrayAdjustLsq::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	m_image.read(fnImg);

	double rot;
	rowIn.getValue(MDL_ANGLE_ROT, rot);
	double tilt;
	rowIn.getValue(MDL_ANGLE_TILT, tilt);
	double psi;
	rowIn.getValue(MDL_ANGLE_PSI, psi);
	double shiftX;
	rowIn.getValue(MDL_SHIFT_X, shiftX);
	double shiftY;	
	rowIn.getValue(MDL_SHIFT_Y, shiftY);

	m_projector->project(rot, tilt, psi, shiftX, shiftY);
	
	double a;
	double b;
	fitLeastSquares(m_projector->projection(), m_image(), a, b);	

	if (a < 0.0)
	{
		rowOut.setValue(MDL_ENABLED, false);
	}
	else
	{
		m_image() *= a;
		m_image() += b;
	}

	rowOut.setValue(MDL_SUBTRACTION_BETA1, a);
	rowOut.setValue(MDL_SUBTRACTION_BETA0, b);

	m_image.write(fnImgOut);
}


bool ProgImageGrayAdjustLsq::fitLeastSquares(const MultidimArray<double> &ref, 
											 const MultidimArray<double> &exp, 
											 double &a, double &b )
{
	Matrix1D<double> x;
	Matrix1D<double> y;
	exp.getAliasAsRowVector(x);
	ref.getAliasAsRowVector(y);

	// Construct desing matrix
	alglib::real_2d_array desing;
	desing.setlength(VEC_XSIZE(x), 2);
	for (std::size_t i = 0; i < VEC_XSIZE(x); ++i)
	{
		desing(i, 0) = VEC_ELEM(x, i);
		desing(i, 1) = 1.0;
	}

	// Construct the right hand size
	alglib::real_1d_array right;
	right.setcontent(VEC_XSIZE(y), y.vdata);

	alglib::real_1d_array solution;
	alglib::ae_int_t info;
	alglib::densesolverlsreport report;
	alglib::rmatrixsolvels(
		desing, VEC_XSIZE(x), 2, 
		right, 
		0.0,
		info,
		report,
		solution
	);

	bool result = info == 1;
	if (result)
	{
		a = 1.0 / solution[0];
		b = -solution[1] * a;
	}

	return result;
}
