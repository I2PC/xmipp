/***************************************************************************
 *
 * Authors:     Jeison Méndez García (jmendez@utp.edu.co)
 *
 * Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas -- IIMAS
 * Universidad Nacional Autónoma de México -UNAM
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

#include "mpi_angular_assignment_mag.h"

void MpiProgAngularAssignmentMag::defineParams()
{
	ProgAngularAssignmentMag::defineParams();
	MpiMetadataProgram::defineParams();
}
void MpiProgAngularAssignmentMag::readParams()
{
	MpiMetadataProgram::readParams();
	ProgAngularAssignmentMag::readParams();
	Nsimul = getIntParam("--Nsimultaneous");
}
void MpiProgAngularAssignmentMag::read(int argc, char **argv)
{
	MpiMetadataProgram::read(argc,argv);
}
void MpiProgAngularAssignmentMag::preProcess()
{
	rank = node->rank;
	Nprocessors = node->size;

	auto Nturns = (int)ceil((double)node->size/Nsimul);
	auto myTurn = (int)floor((double)node->rank/Nsimul);
	for (int turn=0; turn<=Nturns; turn++)
	{
		if (turn==myTurn)
			ProgAngularAssignmentMag::preProcess();
		node->barrierWait();
	}
	MetaData &mdIn = *getInputMd();
	mdIn.addLabel(MDL_GATHER_ID);
	mdIn.fillLinear(MDL_GATHER_ID,1,1);
	createTaskDistributor(mdIn, blockSize);
}
void MpiProgAngularAssignmentMag::startProcessing()
{
	if (node->rank==1)
	{
		verbose=1;
		ProgAngularAssignmentMag::startProcessing();
	}
	node->barrierWait();
}
void MpiProgAngularAssignmentMag::showProgress()
{
	if (node->rank==1)
	{
		time_bar_done=first+1;
		ProgAngularAssignmentMag::showProgress();
	}
}
bool MpiProgAngularAssignmentMag::getImageToProcess(size_t &objId, size_t &objIndex)
{
	return getTaskToProcess(objId, objIndex);
}
void MpiProgAngularAssignmentMag::finishProcessing()
{
	node->gatherMetadatas(getOutputMd(), fn_out);
	MetaDataVec MDaux;
	MDaux.sort(getOutputMd(), MDL_GATHER_ID);
	MDaux.removeLabel(MDL_GATHER_ID);
	getOutputMd()=MDaux;
	if (node->isMaster())
		ProgAngularAssignmentMag::finishProcessing();
}
void MpiProgAngularAssignmentMag::wait()
{
	distributor->wait();
}

void MpiProgAngularAssignmentMag::synchronize(){
	node->barrierWait();
}
