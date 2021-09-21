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
#include <reconstruction/art_zernike3d.h>


class MpiProgArtZernike3D: public ProgArtZernike3D, public MpiMetadataProgram
{

//AJ new
private:
	MpiFileMutex *fileMutex;
    MPI_Comm slaves;
//END AJ

public:
    void defineParams()
    {
        ProgArtZernike3D::defineParams();
        MpiMetadataProgram::defineParams();
    }
    void readParams()
    {
        MPI_Comm_split(MPI_COMM_WORLD,(node->rank != 0),node->rank,&slaves);
        MpiMetadataProgram::readParams();
        ProgArtZernike3D::readParams();
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
            ProgArtZernike3D::showProgress();
        }
    }

    void createWorkFiles()
    {
        //Master node should prepare some stuff before start working
        MetaData &mdIn = *getInputMd(); //get a reference to input metadata

        if (node->isMaster())
        { 
        	ProgArtZernike3D::createWorkFiles();
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
        	ProgArtZernike3D::startProcessing();
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
        	ProgArtZernike3D::finishProcessing();
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

    void gatherVolumes()
    {
        FileName fn = formatString("%s_node%d.vol", fnVolO.removeExtension("vol").c_str(), node->rank);
        Vrefined.write(fn);
    }

    void checkPoint()
    {    
        gatherMetadatas();
        gatherVolumes();
        // slaves --> Definition of a new MPI rule that waits only for the slaves (splitting of the nodes)
        MPI_Barrier(slaves);
        // One (randomly chosen) slave node is reponsible of gathering all the partial results
        if (node->rank==1)
        {
            MetaDataDb MDAux;
            MetaDataDb MDTotal, MDTotalSorted;
            // Image<double> Vi, Vk, Vnode;
            // Vi.read(fnVolR);
            // if (fnVolO.exists())
            //     Vk.read(fnVolO);
            // else
            //     Vk = Vi;
            Image<double> Vk, Vnode;
            Vk.read(fnVolR);
            for (size_t otherRank=1; otherRank<nProcs; ++otherRank)
            {
                FileName fnP = formatString("%s_node%d.xmd", fnDone.removeExtension("xmd").c_str(), otherRank);
                FileName fnC = formatString("%s_node%d.vol", fnVolO.removeExtension("vol").c_str(), otherRank);

                // Gather partial metadata from nodes
                MDAux.read(fnP);
                MDTotal.unionAll(MDAux);
                deleteFile(fnP);

                // Accumulate all current corrections from nodes in the initial
                Vnode.read(fnC);
                // FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Vk())
                //     DIRECT_A3D_ELEM(Vk(),k,i,j) += DIRECT_A3D_ELEM(Vnode(),k,i,j) - DIRECT_A3D_ELEM(Vi(),k,i,j);
                // FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Vk())
                //     DIRECT_A3D_ELEM(Vk(),k,i,j) += DIRECT_A3D_ELEM(Vnode(),k,i,j) - DIRECT_A3D_ELEM(Vk(),k,i,j);
                FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Vk()){
                    if (DIRECT_A3D_ELEM(Vmask,k,i,j) == 1) {
                        DIRECT_A3D_ELEM(Vk(),k,i,j) += DIRECT_A3D_ELEM(Vnode(),k,i,j);
                    }
                }
                deleteFile(fnC);
            }
            MDTotalSorted.sort(MDTotal, MDL_IMAGE);
            MDTotalSorted.write(fnDone);
            Vk.write(fnVolO);
        }
        MPI_Barrier(slaves);
    }

};