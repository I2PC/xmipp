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

#include "ctf_correct_wiener2d.h"


// Read arguments ==========================================================
void ProgCorrectWiener2D::readParams()
{
    XmippMetadataProgram::readParams();
    phase_flipped = checkParam("--phase_flipped");
    pad = XMIPP_MAX(1.,getDoubleParam("--pad"));
    isIsotropic = checkParam("--isIsotropic");
    wiener_constant  = getDoubleParam("--wc");
    correct_envelope = checkParam("--correct_envelope");
	ctf.readParams(this);
}

// Define parameters ==========================================================
void ProgCorrectWiener2D::defineParams()
{
    addUsageLine("Perform CTF correction to 2D projection images with estimated ctfs using a Wiener filter.");
    each_image_produces_an_output = true;
    XmippMetadataProgram::defineParams();
    addKeywords("correct CTF by Wiener filtering");
    addParamsLine("   [--phase_flipped]       : Is the data already phase-flipped?");
    addParamsLine("   [--isIsotropic]         : Must be considered the defocus isotropic?");
    addParamsLine("   [--wc <float=-1>]       : Wiener-filter constant (if < 0: use FREALIGN default)");
    addParamsLine("   [--pad <factor=2.> ]    : Padding factor for Wiener correction");
    addParamsLine("   [--correct_envelope]     : Correct the CTF envelope");
	ctf.defineParams(this);
}

// Define parameters ==========================================================
void ProgCorrectWiener2D::postProcess()
{
	MetaData &ptrMdOut = getOutputMd();

	ptrMdOut.removeLabel(MDL_CTF_DEFOCUSA);
	ptrMdOut.removeLabel(MDL_CTF_DEFOCUSU);
	ptrMdOut.removeLabel(MDL_CTF_DEFOCUS_ANGLE);
	ptrMdOut.removeLabel(MDL_CTF_DEFOCUSV);
	ptrMdOut.removeLabel(MDL_CTF_BG_BASELINE);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN2_ANGLE);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN2_CU);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN2_CV);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN2_K);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN2_SIGMAU);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN2_SIGMAV);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN_ANGLE);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN_CU);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN_CV);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN_K);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN_SIGMAU);
	ptrMdOut.removeLabel(MDL_CTF_BG_GAUSSIAN_SIGMAV);
	ptrMdOut.removeLabel(MDL_CTF_BG_SQRT_ANGLE);
	ptrMdOut.removeLabel(MDL_CTF_BG_SQRT_K);
	ptrMdOut.removeLabel(MDL_CTF_BG_SQRT_U);
	ptrMdOut.removeLabel(MDL_CTF_BG_SQRT_V);
	ptrMdOut.removeLabel(MDL_CTF_CA);
	ptrMdOut.removeLabel(MDL_CTF_CONVERGENCE_CONE);
	ptrMdOut.removeLabel(MDL_CTF_ENERGY_LOSS);
	ptrMdOut.removeLabel(MDL_CTF_ENVELOPE);
	ptrMdOut.removeLabel(MDL_CTF_LENS_STABILITY);
	ptrMdOut.removeLabel(MDL_CTF_TRANSVERSAL_DISPLACEMENT);
	ptrMdOut.removeLabel(MDL_CTF_LONGITUDINAL_DISPLACEMENT);
	ptrMdOut.removeLabel(MDL_CTF_K);

	ptrMdOut.write(fn_out.replaceExtension("xmd"));

}


void ProgCorrectWiener2D::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	WF.pad = pad;
	WF.correct_envelope = correct_envelope;
	WF.wiener_constant = wiener_constant;
	WF.isIsotropic = isIsotropic;
	WF.phase_flipped = phase_flipped;
	WF.applyWienerFilter(fnImg, fnImgOut, rowIn, rowOut);

}

