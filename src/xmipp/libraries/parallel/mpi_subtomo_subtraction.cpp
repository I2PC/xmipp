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

#include "mpi_subtomo_subtraction.h"

void MpiProgSubtomoSubtraction::defineParams()
{
    ProgSubtomoSubtraction::defineParams();
    MpiMetadataProgram::defineParams();
}
void MpiProgSubtomoSubtraction::readParams()
{
    MpiMetadataProgram::readParams();
    ProgSubtomoSubtraction::readParams();
}
void MpiProgSubtomoSubtraction::read(int argc, char **argv, bool reportErrors)
{
    MpiMetadataProgram::read(argc, argv);
}
void MpiProgSubtomoSubtraction::preProcess()
{
    rank = (int)node->rank;
    ProgSubtomoSubtraction::preProcess();
    MetaData &mdIn = *getInputMd();
    mdIn.addLabel(MDL_GATHER_ID);
    mdIn.fillLinear(MDL_GATHER_ID, 1, 1);
    createTaskDistributor(mdIn, blockSize);
}
void MpiProgSubtomoSubtraction::startProcessing()
{
    if (node->rank == 1)
    {
        verbose = 1;
        ProgSubtomoSubtraction::startProcessing();
    }
    node->barrierWait();
}
void MpiProgSubtomoSubtraction::showProgress() 
{
    if (node->rank == 1)
    {
        time_bar_done = first + 1;
        ProgSubtomoSubtraction::showProgress();
    }
}

void MpiProgSubtomoSubtraction::finishProcessing()
{
    node->gatherMetadatas(getOutputMd(), fn_out);
    MetaDataVec MDaux;
    MDaux.sort(getOutputMd(), MDL_GATHER_ID);
    MDaux.removeLabel(MDL_GATHER_ID);
    getOutputMd() = MDaux;
    if (node->isMaster())
        ProgSubtomoSubtraction::finishProcessing();
}
void MpiProgSubtomoSubtraction::wait()
{
    distributor->wait();
}
