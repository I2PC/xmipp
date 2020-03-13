/***************************************************************************
 *
 * Authors:  Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
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
#include <reconstruction/angular_sph_alignment.h>


class MpiProgAngularSphAlignment: public ProgAngularSphAlignment, public MpiMetadataProgram
{

//AJ new
private:
	MpiFileMutex *fileMutex;
//END AJ

public:
    void defineParams()
    {
        ProgAngularSphAlignment::defineParams();
        MpiMetadataProgram::defineParams();
    }
    void readParams()
    {
        MpiMetadataProgram::readParams();
        ProgAngularSphAlignment::readParams();
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
            ProgAngularSphAlignment::showProgress();
        }
    }
    /*void preProcess()
    {
    	ProgAngularSphAlignment::preProcess();
        MetaData &mdIn = *getInputMd();
        mdIn.addLabel(MDL_GATHER_ID);
        mdIn.fillLinear(MDL_GATHER_ID,1,1);
        createTaskDistributor(mdIn, blockSize);
    }*/
    void createWorkFiles()
    {
        //Master node should prepare some stuff before start working
        MetaData &mdIn = *getInputMd(); //get a reference to input metadata

        if (node->isMaster())
        {
        	ProgAngularSphAlignment::createWorkFiles();
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
        	ProgAngularSphAlignment::startProcessing();
        }
        node->barrierWait();
    }
    bool getImageToProcess(size_t &objId, size_t &objIndex)
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
    /*void gatherMetadatas()
    {
        node->gatherMetadatas(*getOutputMd(), fn_out);
    	MetaData MDaux;
    	MDaux.sort(*getOutputMd(), MDL_GATHER_ID);
        MDaux.removeLabel(MDL_GATHER_ID);
        *getOutputMd()=MDaux;
    }
    void finishProcessing()
    {
    	gatherMetadatas();
        if (node->isMaster())
            ProgAngularSphAlignment::finishProcessing();
    }*/
    void finishProcessing()
    {
    	distributor->wait();

        //All nodes wait for each other
        node->barrierWait();
        if (node->isMaster())
        	ProgAngularSphAlignment::finishProcessing();
        node->barrierWait();
    }
    //AJ new
    void writeImageParameters(const FileName &fnImg)
    {
        fileMutex->lock();
        ProgAngularSphAlignment::writeImageParameters(fnImg);
        fileMutex->unlock();
    }
    //END AJ
    void wait()
    {
		distributor->wait();
    }
};
