/***************************************************************************
 * Authors:     Javier Vargas (jvargas@cnb.csic.es)
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

#include "mpi_multireference_aligneability.h"

MpiMultireferenceAligneability::~MpiMultireferenceAligneability()
{
	delete node;
}

void MpiMultireferenceAligneability::read(int argc, char** argv)
{
    node = new MpiNode(argc, argv);
   	rank = node->rank;
   	Nprocessors = node->size;
   	MultireferenceAligneability::read(argc, (const char **)argv);
}

void MpiMultireferenceAligneability::synchronize()
{
	node->barrierWait();
}

void MpiMultireferenceAligneability::gatherResults()
{
	// Write all metadatas
	if (rank!=0)
	{
		FileName fnPartial=formatString("%s/partial_node%03d.xmd",fnDir.c_str(),(int)rank);
		if (mdPartialParticles.size()>0)
			mdPartialParticles.write(fnPartial);
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
					mdPartialParticles.unionAll(MDAux);
					deleteFile(fnP);
				}
		}
	}

	synchronize();

}
