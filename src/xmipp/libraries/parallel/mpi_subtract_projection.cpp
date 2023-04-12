/***************************************************************************
 *
 * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

#include "mpi_subtract_projection.h"

void MpiProgSubtractProjection::defineParams()
{
    ProgSubtractProjection::defineParams();
    MpiMetadataProgram::defineParams();
}
void MpiProgSubtractProjection::readParams()
{
    MpiMetadataProgram::readParams();
    ProgSubtractProjection::readParams();
}
void MpiProgSubtractProjection::read(int argc, char **argv, bool reportErrors)
{
    MpiMetadataProgram::read(argc, argv);
}
void MpiProgSubtractProjection::preProcess()
{
    rank = node->rank;
    std::cout << "MPI rank=" << rank << std::endl;
    ProgSubtractProjection::preProcess();

    // Get the volume padded size from rank 0
    int realSize, origin, realSizeMask, originMask;
    if (node->rank == 0)
    {
        realSize = XSIZE(projector->VfourierRealCoefs);
        origin = STARTINGX(projector->VfourierRealCoefs);
        realSizeMask = XSIZE(projectorMask->VfourierRealCoefs);
        originMask = STARTINGX(projectorMask->VfourierRealCoefs);
    }
    MPI_Bcast(&realSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(projector/*->volumePaddedSize*/), 1, MPI_INT, 0, MPI_COMM_WORLD); //aqui no llegan los rank !0
    //MPI_Bcast(&(projector->volumeSize), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&realSizeMask, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&originMask, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(projectorMask/*->volumePaddedSize*/), 1, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&(projectorMask->volumeSize), 1, MPI_INT, 0, MPI_COMM_WORLD);
    //std::cout << " rank =" << rank << "-------10-------" << std::endl;  

    // if (rank != 0)
    // {
    //     //projector/*->VfourierRealCoefs*/.resizeNoCopy(realSize,realSize,realSize);
    //     std::cout << " rank =" << rank << "-------11-------" << std::endl;
    //     //projector->VfourierImagCoefs.resizeNoCopy(realSize,realSize,realSize);
    //     std::cout << " rank =" << rank << "-------12-------" << std::endl;
    //     //STARTINGX(projector/*->VfourierRealCoefs*/)=STARTINGY(projector->VfourierRealCoefs)=STARTINGZ(projector->VfourierRealCoefs)=origin;
    //     std::cout << " rank =" << rank << "-------13-------" << std::endl;
    //     //STARTINGX(projector->VfourierImagCoefs)=STARTINGY(projector->VfourierImagCoefs)=STARTINGZ(projector->VfourierImagCoefs)=origin;
    //     std::cout << " rank =" << rank << "-------14-------" << std::endl;

    //     //projectorMask/*->VfourierRealCoefs*/.resizeNoCopy(realSizeMask,realSizeMask,realSizeMask);
    //     std::cout << " rank =" << rank << "-------15-------" << std::endl;
    //     //projectorMask->VfourierImagCoefs.resizeNoCopy(realSizeMask,realSizeMask,realSizeMask);
    //     std::cout << " rank =" << rank << "-------16-------" << std::endl;
    //     //STARTINGX(projectorMask/*->VfourierRealCoefs*/)=STARTINGY(projectorMask->VfourierRealCoefs)=STARTINGZ(projectorMask->VfourierRealCoefs)=originMask;
    //     std::cout << " rank =" << rank << "-------17-------" << std::endl;
    //     //STARTINGX(projectorMask->VfourierImagCoefs)=STARTINGY(projectorMask->VfourierImagCoefs)=STARTINGZ(projectorMask->VfourierImagCoefs)=originMask;
    //     std::cout << " rank =" << rank << "-------18-------" << std::endl;
    // }

    // //MPI_Bcast(MULTIDIM_ARRAY(projector/*->VfourierRealCoefs*/), MULTIDIM_SIZE(projector->VfourierRealCoefs), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // std::cout << " rank =" << rank << "-------19-------" << std::endl;
    // //MPI_Bcast(MULTIDIM_ARRAY(projector->VfourierImagCoefs), MULTIDIM_SIZE(projector->VfourierImagCoefs), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // std::cout << " rank =" << rank << "-------20-------" << std::endl;
    // //MPI_Bcast(MULTIDIM_ARRAY(projectorMask-/*>VfourierRealCoefs*/), MULTIDIM_SIZE(projectorMask->VfourierRealCoefs), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // std::cout << " rank =" << rank << "-------21-------" << std::endl;
    // //MPI_Bcast(MULTIDIM_ARRAY(projectorMask->VfourierImagCoefs), MULTIDIM_SIZE(projectorMask->VfourierImagCoefs), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // std::cout << " rank =" << rank << "-------22-------" << std::endl;

    // if (rank != 0)
    // {
    //     std::cout << " rank =" << rank << "-------10-------" << std::endl;
    //     projector->produceSideInfoProjection();
    //     std::cout << " rank =" << rank << "-------11-------" << std::endl;
    //     projectorMask->produceSideInfoProjection();
    //     std::cout << " rank =" << rank << "-------12-------" << std::endl;
    // }

    MetaData &mdIn = *getInputMd();
    mdIn.addLabel(MDL_GATHER_ID);
    mdIn.fillLinear(MDL_GATHER_ID, 1, 1);
    createTaskDistributor(mdIn, blockSize);
    std::cout << "------preProcess done------" << std::endl;
}
void MpiProgSubtractProjection::startProcessing()
{
    if (node->rank == 1)
    {
        verbose = 1;
        ProgSubtractProjection::startProcessing();
    }
    node->barrierWait();
    std::cout <<  " rank =" << rank << "------startProcessing done------" << std::endl;
}
void MpiProgSubtractProjection::showProgress() // NO ENTRA AQUI
{
    std::cout <<  " rank =" << rank << "------1------" << std::endl;
    if (node->rank == 1)
    {
        std::cout <<  " rank =" << rank << "------2------" << std::endl;
        time_bar_done = first + 1;
        std::cout <<  " rank =" << rank << "------3------" << std::endl;
        ProgSubtractProjection::showProgress();
    }
    std::cout << "------showProgress done------" << std::endl;
}


void MpiProgSubtractProjection::finishProcessing()
{
    node->gatherMetadatas(getOutputMd(), fn_out);
    MetaDataVec MDaux;
    MDaux.sort(getOutputMd(), MDL_GATHER_ID);
    MDaux.removeLabel(MDL_GATHER_ID);
    getOutputMd() = MDaux;
    if (node->isMaster())
        ProgSubtractProjection::finishProcessing();
    std::cout << "------finishProcessing done------" << std::endl;
}
void MpiProgSubtractProjection::wait()
{
    distributor->wait();
}
