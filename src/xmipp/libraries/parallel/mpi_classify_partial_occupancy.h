/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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
#include <reconstruction/classify_partial_occupancy.h>


class MpiProgClassifyPartialOccupancy: public ProgClassifyPartialOccupancy, public MpiMetadataProgram
{
public:

    void defineParams() override;
    void readParams() override;
    void read(int argc, char **argv, bool reportErrors = true) override;
    void preProcess() override;
    void startProcessing() override;
    void showProgress() override;
    bool getImageToProcess(size_t &objId, size_t &objIndex) override
    {
        return getTaskToProcess(objId, objIndex);
    }
    void finishProcessing() override;
    
    void wait() override;
    
};
