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

#include <parallel/xmipp_mpi.h>
#include <reconstruction/nma_alignment_vol.h>
#include <memory>

/** Class to perform the NMA Alignment for volumes with  MPI parallelization */
class MpiProgNMAVol: public ProgNmaAlignmentVol
{
private:
    std::shared_ptr<MpiNode> node;
    std::shared_ptr<MpiTaskDistributor> distributor;
    std::shared_ptr<MpiFileMutex> fileMutex;
    std::vector<size_t> imgsId;

    // main body
    void createWorkFiles();
    
    //Only master do starting progress bar stuff
    void startProcessing();
    
    //Only master show progress
    void showProgress();
    
    //Now use the distributor to grasp volumes
    bool getImageToProcess(size_t &objId, size_t &objIndex);

    void finishProcessing();

    void writeVolumeParameters(const FileName &fnImg);

public:
    // Redefine read to initialize MPI environment
    void read(int argc, char **argv);
}
;//end of class MpiProgNMAVol
