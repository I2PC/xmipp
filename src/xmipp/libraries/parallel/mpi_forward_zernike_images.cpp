/***************************************************************************
 *
 * Authors:  David Herreros Calero dherreros@cnb.csic.es
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

#include <mpi.h>
#include <parallel/xmipp_mpi.h>
#include <reconstruction/forward_zernike_images.h>


class MpiProgForwardZernikeImages: public ProgForwardZernikeImages, public MpiMetadataProgram
{

//AJ new
private:
	MpiFileMutex *fileMutex;
    MPI_Comm slaves;
//END AJ

public:
    void defineParams()
    {
        ProgForwardZernikeImages::defineParams();
        MpiMetadataProgram::defineParams();
    }
    void readParams()
    {
        MPI_Comm_split(MPI_COMM_WORLD,(node->rank != 0),node->rank,&slaves);
        MpiMetadataProgram::readParams();
        ProgForwardZernikeImages::readParams();
        blockSize = 1;
    }
    void read(int argc, char **argv, bool reportErrors = true)
    {
    	//AJ new
    	fileMutex = new MpiFileMutex(node);
    	//END AJ
        MpiMetadataProgram::read(argc,argv);
    }
    void showProgress()
    {
        if (node->rank==1)
        {
            time_bar_done=first+1;
            ProgForwardZernikeImages::showProgress();
        }
    }

    void createWorkFiles()
    {
        //Master node should prepare some stuff before start working
        MetaData &mdIn = *getInputMd(); //get a reference to input metadata

        if (node->isMaster())
        { 
        	ProgForwardZernikeImages::createWorkFiles();
            mdIn.write(fnOutDir + "/sphTodo.xmd");
        }
        node->barrierWait();//Sync all before start working
        mdIn.read(fnOutDir + "/sphTodo.xmd");
        mdIn.findObjects(imgsId);//get objects ids
        distributor = new MpiTaskDistributor(mdIn.size(), 1, node);
    }
    void startProcessing()
    {
        if (node->rank==1)
        {
        	verbose=1;
        	ProgForwardZernikeImages::startProcessing();
        }
        node->barrierWait();
    }
    virtual bool getImageToProcess(size_t &objId, size_t &objIndex) override
    {
        //return getTaskToProcess(objId, objIndex);
        size_t first, last;
        bool moreTasks = distributor->getTasks(first, last);

        if (moreTasks)
        {
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

    void finishProcessing()
    {
    	distributor->wait();

        //All nodes wait for each other
        node->barrierWait();
        if (node->isMaster())
        	ProgForwardZernikeImages::finishProcessing();
        node->barrierWait();
    }
    //END AJ
    void wait()
    {
		distributor->wait();
    }

    void gatherMetadatas()
    {
        FileName fn = formatString("%s_node%d.xmd", fnDone.removeExtension("xmd").c_str(), node->rank);
        getOutputMd().write(fn);
    }

    void checkPoint()
    {    
        gatherMetadatas();
        // slaves --> Definition of a new MPI rule that waits only for the slaves (splitting of the nodes)
        MPI_Barrier(slaves);
        // One (randomly chosen) slave node is reponsible of gathering all the partial results
        if (node->rank==1)
        {
            MetaDataDb MDAux;
            MetaDataDb MDTotal, MDTotalSorted;
            for (size_t otherRank=1; otherRank<nProcs; ++otherRank)
            {
                FileName fnP = formatString("%s_node%d.xmd", fnDone.removeExtension("xmd").c_str(), otherRank);

                // Gather partial metadata from nodes
                MDAux.read(fnP);
                MDTotal.unionAll(MDAux);
                deleteFile(fnP);
            }
            MDTotalSorted.sort(MDTotal, MDL_IMAGE);
            // MDTotalSorted.append(fnDone);  FIXME: Do a working append
            MDTotalSorted.write(fnDone);
        }
        MPI_Barrier(slaves);
    }

};