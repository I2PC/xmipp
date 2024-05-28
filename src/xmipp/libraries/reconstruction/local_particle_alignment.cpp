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
	fnIn = getParam("--inputPaticles");

	
}


void ProgLocalParticleAlignment::defineParams()
{
	addUsageLine("This program refine the alignment of particles focalized in a volume region.");
	addParamsLine("  --inputPaticles       	: File path to input particle with alignments.");

}


// ---------------------- MAIN FUNCTIONS -----------------------------

void ProgLocalParticleAlignment::recenterParticle()
{
	MetaDataVec md;
	md.read(fnIn);

	size_t nDim = md.size();
	getParticleSize();

	particles.initZeros(nDim, yDim, xDim);

	Image<double> particleImg;
	auto &particle = particleImg();

	MultidimArray<double> particleShifted;

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

		particleShifted.resizeNoCopy(particle);

		applyGeometry(xmipp_transformation::BSPLINE3, 
					  particleShifted, 
					  particle, 
					  shiftMat, 
					  xmipp_transformation::IS_NOT_INV, 
					  true, 
					  0.);


		for (size_t i = 0; i < yDim; i++)
		{
			for (size_t j = 0; i < xDim; i++)
			{
				DIRECT_A3D_ELEM(particles, idx, i, j) = DIRECT_A2D_ELEM(particleShifted, idx, i, j);
			}
		}

		idx++;
	}
}


// --------------------------- MAIN ----------------------------------

void ProgLocalParticleAlignment::run()
{

}


// --------------------------- UTILS FUNCTIONS ----------------------------

void ProgLocalParticleAlignment::getParticleSize()
{
	MetaDataVec md;

	FileName fn;

	Image<double> particleImg;
	// auto &particle = particleImg();

	md.read(fnIn);

	const auto& row = md.firstObject();
	row.getValue(MDL_IMAGE, fn);

	particleImg.read(fn);
	particle.setXmippOrigin();

	MultidimArray<double> particle;

	particle.getDimensions(xDim, yDim);
}


void ProgLocalParticleAlignment::calculateShiftDisplacement(Matrix2D<double> particleAlignment, Matrix2D<double> shifts)
{
	Matrix1D<double> projectedCenter = particleAlignment * alignemntCenter;
	
	shifts.initIdentity(4);
	MAT_ELEM(shifts, 0, 4) = MAT_ELEM(projectedCenter, 0, 4);
	MAT_ELEM(shifts, 1, 4) = MAT_ELEM(projectedCenter, 1, 4);
}

