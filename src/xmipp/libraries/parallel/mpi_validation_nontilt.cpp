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

#include "mpi_validation_nontilt.h"

void MpiProgValidationNonTilt::read(int argc, char** argv)
{
    node = std::make_unique<MpiNode>(argc, argv);
   	rank = node->rank;
   	Nprocessors = node->size;
   	ProgValidationNonTilt::read(argc, (const char **)argv);
}

void MpiProgValidationNonTilt::synchronize()
{
	node->barrierWait();
}

void MpiProgValidationNonTilt::gatherClusterability()
{
	/*
	// Share all Ps and image index
	MultidimArray<double> aux;
	if (rank==0)
		aux.resizeNoCopy(cc);
	MPI_Reduce(MULTIDIM_ARRAY(weight), MULTIDIM_ARRAY(aux), MULTIDIM_SIZE(weight), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank==0)
		weight=aux;
	MPI_Reduce(MULTIDIM_ARRAY(cc), MULTIDIM_ARRAY(aux), MULTIDIM_SIZE(cc), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank==0)
		cc=aux;
*/
	// Write all metadatas
	if (rank!=0)
	{
		FileName fnPartial=formatString("%s/partial_node%03d.xmd",fnDir.c_str(),(int)rank);
		if (mdPartial.size()>0)
			mdPartial.write(fnPartial);
	}

	synchronize();

	// Now the master takes all of them
	if (rank==0)
	{
		MetaDataDb MDAux;
		for (size_t otherRank=1; otherRank<Nprocessors; ++otherRank)
		{
				FileName fnP = formatString("%s/partial_node%03d.xmd",fnDir.c_str(),(int)otherRank);

				if (fnP.exists())
				{
					MDAux.read(fnP);
					mdPartial.unionAll(MDAux);
					deleteFile(fnP);
				}
		}
	}

	synchronize();

}

