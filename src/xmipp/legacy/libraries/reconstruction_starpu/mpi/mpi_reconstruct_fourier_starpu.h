/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
 *              Roberto Marabini (roberto@cnb.csic.es)
 *              Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *              Jose Roman Bilbao-Castro (jrbcast@ace.ual.es)
 *              Vahid Abrishami (vabrishami@cnb.csic.es)
 *              David Strelak (davidstrelak@gmail.com)
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
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef __RECONSTRUCT_FOURIER_GPU_MPI_H
#define __RECONSTRUCT_FOURIER_GPU_MPI_H

// Uses its transitive dependencies
#include <reconstruction_starpu/reconstruct_fourier_starpu.h>
#include <mpi.h>
#include <deque>

/**
 * MPI extension of ProgRecFourierStarPU.
 * MPI should already be initialized externally. (This is done by runReconstructFourierMpiStarPU function)
 *
 * At least two program instances are needed - one master and at least one worker (though any real speedup is expected
 * only with multiple workers). It is expected, but not required, that the master will run on the same machine as one
 * of the workers, as it uses minimal processing power.
 *
 * All program instances, no matter their role, should have the access to identical input files (through network storage,
 * manual copy, etc.).
 *
 * First worker (i.e. second global rank) will write out the result.
 */
class ProgRecFourierMpiStarPU : public ProgRecFourierStarPU {

	bool mpiInitialized = false;

	/** 0..1, how many batches should be distributed by default?
	* Larger values lead to less overhead in homogeneous environment,
	* while smaller values can distribute batches better in more heterogeneous environment.
	* At least one batch is always given to workers, no matter what. */
	float percentOfJobsDistributedByDefault = 0.3f;
	/** Amount of batches that should always be in the pipeline, if possible.
	* Similar tradeoff like percentOfJobsDistributedByDefault,
	* but some number is always needed so that the pipeline does not stall. */
	uint32_t preferredBatchesInPipeline = 5;

public:

	/** Specify supported command line arguments */
	void defineParams() override;

	/** Read arguments from command line */
	void readParams() override;

	/** Read with MPI Initialization */
	void read(int argc, char **argv, bool reportErrors = true) override;

	/** try with MPI abort */
	int tryRun() override;

	void run() override;

	/** MPI shutdown */
	~ProgRecFourierMpiStarPU() override;

private:

	/** How many and which batches should each worker get by default.
	 * @param batchCount how many batches are there to distribute
	 * @param workerCount between how many workers should the work be distributed
	 * @param selfWorkerId if called from worker id, myBatches will be filled with the numbers of batches to be processed
	 * by default by this worker. The worker can begin to work on these right away.
	 * @param myBatches will be filled with numbers of batches on which this worker should work on, may be nullptr
	 * @return amount of batches distributed by default, equal to the next batch to distribute. If this value is equal to batchCount,
	 * it means that all batches have been distributed by default and dynamic batch distribution is not needed. */
	uint32_t defaultBatchDistribution(uint32_t batchCount, int workerCount, int selfWorkerId, std::deque<uint32_t>* myBatches) const;

	/** Distributes the work to workers. Does not do any work itself.
	 * @param workerCount amount of workers */
	void runMaster(int workerCount);

	/** Takes orders from the master and processes given batches.
	 * @param workComm shared communicator for all workers
	 * @param workerCount amount of workers in workComm
	 * @param workerRank of this worker. Rank 0 is primary and the one which gathers results,
	 *                   post-processes them and saves the final output. */
	void runWorker(MPI_Comm workComm, int workerCount, int workerRank);

	struct DistributedBatchProvider;
};

#endif // __RECONSTRUCT_FOURIER_GPU_MPI_H
