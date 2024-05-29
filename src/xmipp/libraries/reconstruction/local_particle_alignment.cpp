/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez	federico.pdeisidro@astx.com (2024)
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


#include "local_particle_alignment.h"


// --------------------- IN/OUT FUNCTIONS -----------------------------

void ProgLocalParticleAlignment::readParams()
{
	fnIn = getParam("--inputParticles");
	fnOut = getParam("--outputParticles");

	fnOutMetatada = fnOut.removeAllExtensions() + ".xmd";
	fnOutParticles = fnOut.removeAllExtensions() + ".stk";

	String centerProjectionStr = getParam("--proyectionCenter");

	/*
		This regex matches the format x, y, z for 3D coordinate input.
		Allows:
			- White spaces an the begining and end of the string
			- White spaces between numbers and commands
			- Decimal digits in any of the numbers
	*/
    std::regex pattern(R"(^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*,\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*,\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$)");
	std::smatch matches;

    if (!std::regex_match(centerProjectionStr, matches, pattern)) {
        REPORT_ERROR(ERR_IO_NOWRITE, "Invalid coordinate formatting, expected format is x, y, z");
    }

    double x = std::stod(matches[1].str());
    double y = std::stod(matches[2].str());
    double z = std::stod(matches[3].str());

	alignmentCenter.initZeros(4);
	XX(alignmentCenter) = x;
	YY(alignmentCenter) = y;
	ZZ(alignmentCenter) = z;
}

void ProgLocalParticleAlignment::defineParams()
{
	addUsageLine("This program refine the alignment of particles focalized in a volume region.");
	addParamsLine("  --inputParticles	<xmd_file=\"\">     : File path to input particle with alignments.");
	addParamsLine("  --outputParticles	<output=\"\">       : File path to save output particles and metadata (.stk and .xmd).");

	addParamsLine("  --proyectionCenter	<pc=\"\">       	: Proyection center (\"x,y,z\" format) referenced from the center of the volume.");
}

void ProgLocalParticleAlignment::saveMetadata()
{
	MetaDataVec mdIn;
	MetaDataVec mdOut;

	FileName fn;

	mdIn.read(fnIn);

	size_t idx = 1;

	for (auto& row : mdIn)
	{
		fn.compose(idx, fnOutParticles);
		row.setValue(MDL_IMAGE, fn, false);
		size_t id = mdOut.addRow(row);

		idx++;
	}

	mdOut.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Output metada saved at: " << fnOutMetatada << std::endl;
	#endif
}


// ---------------------- MAIN FUNCTIONS -----------------------------

void ProgLocalParticleAlignment::recenterParticles()
{
	MetaDataVec md;
	md.read(fnIn);

	size_t nDim = md.size();
	getParticleSize();

	shifedParticles.initZeros(nDim, zDim, yDim, xDim);

	Image<double> particleImg;
	auto &particle = particleImg();

	MultidimArray<double> shiftParticle;

	Matrix2D<double> eulerMat;
	Matrix2D<double> shiftMat;

	FileName fn;

	size_t idx = 0;

	for (const auto& row : md)
	{
		row.getValue(MDL_IMAGE, fn);
		particleImg.read(fn);
		particle.setXmippOrigin();

		eulerMat.initIdentity(4);
		geo2TransformationMatrix(row, eulerMat);

		calculateShiftDisplacement(eulerMat, shiftMat);

		shiftParticle.resizeNoCopy(particle);

		applyGeometry(xmipp_transformation::BSPLINE3, 
					  shiftParticle, 
					  particle, 
					  shiftMat, 
					  xmipp_transformation::IS_NOT_INV, 
					  true, 
					  0.);


		for (size_t i = 0; i < yDim; i++)
		{
			for (size_t j = 0; i < xDim; i++)
			{
				DIRECT_NZYX_ELEM(shifedParticles, idx, 1, i, j) = DIRECT_A2D_ELEM(shiftParticle, i, j);
			}
		}

		idx++;
	}

	Image<double> shifedParticlesImg;
	shifedParticlesImg() = shifedParticles;
	shifedParticlesImg.write(fnOutParticles);
}


// --------------------------- MAIN ----------------------------------

void ProgLocalParticleAlignment::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();

	recenterParticles();
	saveMetadata();

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

	#ifdef VERBOSE_OUTPUT
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
	#endif
}


// --------------------------- UTILS FUNCTIONS ----------------------------

void ProgLocalParticleAlignment::getParticleSize()
{
	MetaDataVec md;

	FileName fn;

	Image<double> particleImg;
	auto &particle = particleImg();

	md.read(fnIn);
	md.getValue(MDL_IMAGE, fn, 1);

	particleImg.read(fn);
	particle.setXmippOrigin();

	xDim = XSIZE(particle);
	yDim = YSIZE(particle);
}

void ProgLocalParticleAlignment::calculateShiftDisplacement(Matrix2D<double> particleAlignment, Matrix2D<double> &shifts)
{
	Matrix1D<double> projectedCenter = particleAlignment * alignmentCenter;

	shifts.initIdentity(3);
	MAT_ELEM(shifts, 0, 2) = XX(projectedCenter);
	MAT_ELEM(shifts, 1, 2) = YY(projectedCenter);
}

