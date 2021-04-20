/***************************************************************************
 *
 * Authors:  Mohamad Harastani mohamad.harastani@upmc.fr
 *	         Slavica Jonic slavica.jonic@upmc.fr
 *           Carlos Oscar Sanchez Sorzano coss.eps@ceu.es
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 * Lab. de Bioingenieria, Univ. San Pablo CEU
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/
#include "mpi_volumeset_align.h"
#include "core/metadata.h"

// Redefine read to initialize MPI environment =======================
void MpiProgVolumeSetAlign::read(int argc, char **argv, bool )
{
	node = std::make_shared<MpiNode>(argc, argv);
	if (!node->isMaster())
		verbose=0;
	fileMutex = std::make_shared<MpiFileMutex>(node.get());
	ProgVolumeSetAlign::read(argc, argv);
}


// Main body ==========================================================
void MpiProgVolumeSetAlign::createWorkFiles()
{
	//Master node should prepare some stuff before start working
	MetaData &mdIn = *getInputMd(); //get a reference to input metadata
	const char* list_of_volumes = "/WorkingOnThese.xmd";
	if (node->isMaster()){
		ProgVolumeSetAlign::createWorkFiles();
		mdIn.write(fnOutDir + list_of_volumes);
	}
	node->barrierWait();//Sync all before start working
	mdIn.read(fnOutDir + list_of_volumes);
	mdIn.findObjects(imgsId);//get objects ids
	distributor = std::make_shared<MpiTaskDistributor>(mdIn.size(), 1, node.get());
}


// Only master do starting progress bar stuff
void MpiProgVolumeSetAlign::startProcessing()
{
	if (node->isMaster())
		ProgVolumeSetAlign::startProcessing();
}

//Only master show progress
void MpiProgVolumeSetAlign::showProgress()
{
	if (node->isMaster())
		ProgVolumeSetAlign::showProgress();
}

// Now use the distributor to grasp images
bool MpiProgVolumeSetAlign::getImageToProcess(size_t &objId, size_t &objIndex)
{
	size_t first;
	size_t last;
	bool moreTasks = distributor->getTasks(first, last);

	if (moreTasks){
		time_bar_done = first + 1;
		objIndex = first;
		objId = imgsId[first];
		return true;
	}
	time_bar_done = getInputMd()->size();
	objId = BAD_OBJID;
	objIndex = BAD_INDEX;
	return false;
}

void MpiProgVolumeSetAlign::finishProcessing()
{
	distributor->wait();
	//All nodes wait for each other
	node->barrierWait();
	if (node->isMaster())
		ProgVolumeSetAlign::finishProcessing();
	node->barrierWait();
}

void MpiProgVolumeSetAlign:: writeVolumeParameters(const FileName &fnImg)
{
	fileMutex->lock();
	ProgVolumeSetAlign::writeVolumeParameters(fnImg);
	fileMutex->unlock();
}
