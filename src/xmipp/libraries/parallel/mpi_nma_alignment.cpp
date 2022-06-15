/***************************************************************************
 *
 * Authors:  Slavica Jonic slavica.jonic@impmc.jussieu.fr
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
//#include <mpi.h>
#include <parallel/xmipp_mpi.h>
#include <reconstruction/nma_alignment.h>


/** Class to perfom the NMA Alignment with  MPI parallelization */
class MpiProgNMA: public ProgNmaAlignment
{
private:
    std::shared_ptr<MpiNode> node;
    std::unique_ptr<MpiTaskDistributor> distributor;
    std::vector<size_t> imgsId;
    std::unique_ptr<MpiFileMutex> fileMutex;

public:
    /** Redefine read to initialize MPI environment */
    void read(int argc, char **argv)
    {
    	node = std::make_shared<MpiNode>(argc, argv);
        if (!node->isMaster())
        	verbose=0;
        fileMutex = std::make_unique<MpiFileMutex>(node);
        ProgNmaAlignment::read(argc, argv);
    }
    /** main body */
    void createWorkFiles() override
    {
        //Master node should prepare some stuff before start working
        MetaData &mdIn = *getInputMd(); //get a reference to input metadata

        if (node->isMaster())
        {
            ProgNmaAlignment::createWorkFiles();
            mdIn.write(fnOutDir + "/nmaTodo.xmd");
        }
        node->barrierWait();//Sync all before start working
        mdIn.read(fnOutDir + "/nmaTodo.xmd");
        mdIn.findObjects(imgsId);//get objects ids
        rangen = node->rank;
        distributor = std::make_unique<MpiTaskDistributor>(mdIn.size(), 1, node);
    }
    //Only master do starting progress bar stuff
    void startProcessing()
    {
        if (node->isMaster())
            ProgNmaAlignment::startProcessing();
    }
    //Only master show progress
    void showProgress()
    {
        if (node->isMaster())
            ProgNmaAlignment::showProgress();
    }
    //Now use the distributor to grasp images
    virtual bool getImageToProcess(size_t &objId, size_t &objIndex) override
    {
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
            ProgNmaAlignment::finishProcessing();
        node->barrierWait();
    }

    void writeImageParameters(const FileName &fnImg)
    {
        fileMutex->lock();
        ProgNmaAlignment::writeImageParameters(fnImg);
        fileMutex->unlock();
    }
}
;//end of class MpiProgNMA
