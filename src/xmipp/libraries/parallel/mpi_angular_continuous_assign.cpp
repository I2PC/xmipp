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
#include <reconstruction/angular_continuous_assign2.h>


class MpiProgAngularContinuousAssign2: public ProgAngularContinuousAssign2, public MpiMetadataProgram
{
public:

    void defineParams()
    {
        ProgAngularContinuousAssign2::defineParams();
        MpiMetadataProgram::defineParams();
    }
    void readParams()
    {
        MpiMetadataProgram::readParams();
        ProgAngularContinuousAssign2::readParams();
    }
    void read(int argc, char **argv, bool reportErrors = true)
    {
        MpiMetadataProgram::read(argc,argv);
    }
    void preProcess()
    {
    	rank=node->rank;
        ProgAngularContinuousAssign2::preProcess();
   		node->barrierWait();

   		// Get the volume padded size from rank 0
   		int realSize, origin;
   		if (node->rank==0)
   		{
   			realSize = XSIZE(projector->VfourierRealCoefs);
   			origin = STARTINGX(projector->VfourierRealCoefs);
   		}
        MPI_Bcast(&realSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&origin, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(projector->volumePaddedSize), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(projector->volumeSize), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank!=0)
        {
        	projector->VfourierRealCoefs.resizeNoCopy(realSize,realSize,realSize);
        	projector->VfourierImagCoefs.resizeNoCopy(realSize,realSize,realSize);
        	STARTINGX(projector->VfourierRealCoefs)=STARTINGY(projector->VfourierRealCoefs)=STARTINGZ(projector->VfourierRealCoefs)=origin;
        	STARTINGX(projector->VfourierImagCoefs)=STARTINGY(projector->VfourierImagCoefs)=STARTINGZ(projector->VfourierImagCoefs)=origin;
        }
        MPI_Bcast(MULTIDIM_ARRAY(projector->VfourierRealCoefs), MULTIDIM_SIZE(projector->VfourierRealCoefs), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(MULTIDIM_ARRAY(projector->VfourierImagCoefs), MULTIDIM_SIZE(projector->VfourierImagCoefs), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    	if (rank!=0)
        	projector->produceSideInfoProjection();

        MetaData &mdIn = *getInputMd();
        mdIn.addLabel(MDL_GATHER_ID);
        mdIn.fillLinear(MDL_GATHER_ID,1,1);
        createTaskDistributor(mdIn, blockSize);
    }
    void startProcessing()
    {
        if (node->rank==1)
        {
        	verbose=1;
            ProgAngularContinuousAssign2::startProcessing();
        }
        node->barrierWait();
    }
    void showProgress()
    {
        if (node->rank==1)
        {
            time_bar_done=first+1;
            ProgAngularContinuousAssign2::showProgress();
        }
    }
    bool getImageToProcess(size_t &objId, size_t &objIndex)
    {
        return getTaskToProcess(objId, objIndex);
    }
    void finishProcessing()
    {
        node->gatherMetadatas(*getOutputMd(), fn_out);
    	MetaData MDaux;
    	MDaux.sort(*getOutputMd(), MDL_GATHER_ID);
        MDaux.removeLabel(MDL_GATHER_ID);
        *getOutputMd()=MDaux;
        if (node->isMaster())
            ProgAngularContinuousAssign2::finishProcessing();
    }
    void wait()
    {
		distributor->wait();
    }
};
