/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
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

#include <cstdint>
#include <vector>
#include <deque>
#include <semaphore.h>

#define STARPU_DONT_INCLUDE_CUDA_HEADERS
#include <starpu.h>

#include "../reconstruct_fourier_util.h"
#include "../reconstruct_fourier_starpu_util.h"

#include "mpi_reconstruct_fourier_starpu.h"

/** DEVELOPMENT: Enable to display logs from this file */
const bool mpiLogging = false;

#define LOG_MPI(COUT) do { if (mpiLogging) {\
int logRank = -1, logIsMpiInitialized = 0;\
MPI_Initialized(&logIsMpiInitialized);\
if (logIsMpiInitialized) MPI_Comm_rank(MPI_COMM_WORLD, &logRank);\
std::cout << "[mpi_rec_fou] " << logRank << ": " << COUT << '\n'; } } while (0)

void ProgRecFourierMpiStarPU::defineParams() {
	ProgRecFourierStarPU::defineParams();
	addParamsLine("  [--mpiDistribute <pc=0.3>]     : Ratio [0,1] describing how many batches should be initially distributed.");
	addParamsLine("                                 : Use less for more heterogeneous group of computers, more for more homogeneous group.");
	addParamsLine("  [--mpiPrefetch <batches=25>]    : How many batches should be in the processing pipeline at all times.");
	addParamsLine("                                 : Higher numbers could be faster with faster and/or more homogeneous computers.");
	addParamsLine("                                 : Very low numbers will have severe performance implications.");
}

void ProgRecFourierMpiStarPU::readParams() {
	ProgRecFourierStarPU::readParams();

	percentOfJobsDistributedByDefault = static_cast<float>(getDoubleParam("--mpiDistribute"));
	preferredBatchesInPipeline = static_cast<uint32_t>(getIntParam("--mpiPrefetch"));
}

void ProgRecFourierMpiStarPU::run() {
	int rank, size;
	CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

	{// Implementation of tip 6 from https://www.open-mpi.org/faq/?category=debugging
		// Set this env variable to the rank which is to be debugged
		char* rankStr = getenv("XMIPP_MPI_DEBUG_RANK");
		int rankToBlock = rankStr == nullptr ? -1 : atoi(rankStr);

		if (rankToBlock == rank) {
			volatile /* so it isn't optimized out */ int i = 0;
			char hostname[256];
			gethostname(hostname, sizeof(hostname));
			printf("PID %d on %s ready for attach\n", getpid(), hostname);
			fflush(stdout);

			// After attaching with debugger, set i to arbitrary non-zero value to continue
			// gdb> set var i = 7
			while (0 == i)
				sleep(1);
		}
	}

	if (size < 2) {
		if (rank == 0) {
			REPORT_ERROR(ERR_ARG_DEPENDENCE, "This MPI implementation needs at least two instances");
		} else {
			// no need to report error from all nodes at the same time
			return;
		}
	}

	MPI_Comm workComm;
	// Rest of the code in this file assumes, that world rank 0 is master and other ranks are simply shifted by 1 to form workComm.
	// This code does that, but any modification must take this requirement into account.
	CHECK_MPI(MPI_Comm_split(MPI_COMM_WORLD, rank == 0 ? MPI_UNDEFINED : 0, 0, &workComm));

	const int workerCount = size - 1;

	if (workComm == MPI_COMM_NULL) {
		runMaster(workerCount);
	} else {
		int workerRank;
		CHECK_MPI(MPI_Comm_rank(workComm, &workerRank));
		runWorker(workComm, workerCount, workerRank);
	}
}

/** How many and which batches should each worker get by default.
 * @param batchCount how many batches are there to distribute
 * @param workerCount between how many workers should the work be distributed
 * @param selfWorkerId if called from worker id, myBatches will be filled with the numbers of batches to be processed
 * by default by this worker. The worker can begin to work on these right away.
 * @param myBatches will be filled with numbers of batches on which this worker should work on, may be nullptr
 * @return amount of batches distributed by default, equal to the next batch to distribute. If this value is equal to batchCount,
 * it means that all batches have been distributed by default and dynamic batch distribution is not needed. */
uint32_t ProgRecFourierMpiStarPU::defaultBatchDistribution(const uint32_t batchCount, const int workerCount, const int selfWorkerId, std::deque<uint32_t>* myBatches) const {
	uint32_t distributedByDefault = static_cast<uint32_t>(percentOfJobsDistributedByDefault * batchCount);
	// Always distribute at least preferredBatchesInPipeline batches to each worker
	distributedByDefault = XMIPP_MAX(distributedByDefault, workerCount * preferredBatchesInPipeline);
	// But when there is less batches than workers, that is not possible
	distributedByDefault = XMIPP_MIN(distributedByDefault, batchCount);

	uint32_t nextRoundFirstBatch = 0;
	while (nextRoundFirstBatch < distributedByDefault) {
		// There is still something to distribute
		uint32_t roundSize = XMIPP_MIN(distributedByDefault - nextRoundFirstBatch, workerCount);
		if (myBatches != nullptr && selfWorkerId < roundSize) {
			myBatches->push_back(nextRoundFirstBatch + selfWorkerId);
		}
		nextRoundFirstBatch += roundSize;
	}

	return nextRoundFirstBatch;
}

const int TAG_WORKER_TO_MASTER = 1;
const int TAG_MASTER_TO_WORKER = 2;

void ProgRecFourierMpiStarPU::runMaster(int workerCount) {
	if (verbose) {
		show();
	}

	// mpiLogging completely breaks the progress bar
	const bool progressBar = !mpiLogging && static_cast<bool>(verbose) && false /* Progress bar on MPI processes is broken anyway. */;

	prepareMetaData(fn_in, SF);
	const uint32_t batchCount = computeBatchCount(params, SF);

	if (progressBar) {
		init_progress_bar(batchCount);
	}

	uint32_t nextBatch = defaultBatchDistribution(batchCount, workerCount, workerCount, nullptr);
	assert(nextBatch <= batchCount);

	LOG_MPI("default distributed " << nextBatch << " out of " << batchCount << " batches");

	if (nextBatch == batchCount) {
		// Everything has been given out by default, nobody should ask for anything, master is done
		LOG_MPI("skipping batch distribution loop");
		return;
	}

	// Workers will send how many batches they have processed uint32_t[1]
	// and master will respond with the next batch they should process int32_t[1], or -1 if they should not process any more batches

	struct WorkerInfo {
		/** Amount of batches the worker reports as processed. */
		uint32_t processedBatches;
		/** MPI ISend buffer */
		int32_t sendBuffer;
		/** Last request which used the sendBuffer. */
		MPI_Request lastSendRequest = MPI_REQUEST_NULL;
	};
	std::vector<WorkerInfo> workerInfos(workerCount);
	int workersComplete = 0;
	uint32_t batchesComplete = 0;

	while (workersComplete < workerCount) {
		LOG_MPI("waiting for next batch query");
		MPI_Status recvStatus;
		uint32_t complete;
		CHECK_MPI(MPI_Recv(&complete, 1, MPI_UINT32_T, MPI_ANY_SOURCE, TAG_WORKER_TO_MASTER, MPI_COMM_WORLD, &recvStatus));
		// Somebody needs another batch to process

		LOG_MPI("batch query received from " << recvStatus.MPI_SOURCE << " (" << complete << " completed)");

		// NOTE: This assumes, that ranks of workers are 1..workerCount! This is currently true as it is how MPI_Comm_split does things.
		WorkerInfo& workerInfo = workerInfos[recvStatus.MPI_SOURCE - 1];
		CHECK_MPI(MPI_Wait(&workerInfo.lastSendRequest, MPI_STATUS_IGNORE)); // Should be no-op

		// Send new batch or notification that there are no more batches
		if (nextBatch >= batchCount) {
			LOG_MPI("replying with no more batches");
			workerInfo.sendBuffer = -1;
			workersComplete++;
		} else {
			LOG_MPI("replying with batch " << nextBatch);
			workerInfo.sendBuffer = nextBatch++;
		}
		CHECK_MPI(MPI_Isend(&workerInfo.sendBuffer, 1, MPI_INT32_T, recvStatus.MPI_SOURCE, TAG_MASTER_TO_WORKER, MPI_COMM_WORLD, &workerInfo.lastSendRequest));

		// Request carried information about how many batches has that worker completed in total.
		// Update internal counters with that.
		uint32_t newCompleteBatches = workerInfo.processedBatches - complete;
		workerInfo.processedBatches = complete;
		batchesComplete += newCompleteBatches;

		if (progressBar) {
			progress_bar(batchesComplete);
		}
	}

	LOG_MPI("batch loop complete, waiting for any pending messages");

	// Every batch has been sent, make sure it has reached its destination
	for (WorkerInfo& workerInfo : workerInfos) {
		CHECK_MPI(MPI_Wait(&workerInfo.lastSendRequest, MPI_STATUS_IGNORE));
	}

	if (progressBar) {
		// Just to make sure that it is really done
		progress_bar(batchCount);
	}

	LOG_MPI("done");
	// Done
}

struct ProgRecFourierMpiStarPU::DistributedBatchProvider : ProgRecFourierStarPU::BatchProvider {
private:
	const ProgRecFourierMpiStarPU& parent;

	/** Queue of batches to be processed */
	std::deque<uint32_t> defaultBatchQueue;

	/** The maximum possible amount of batches this provider will ever provide */
	uint32_t maxBatchCount;

	enum class State {
		/** There are some items in default queue and after that runs out, check the pool */
		QUEUE_AND_POOL,
		/** All items are in the default queue, do not check the pool */
		QUEUE_ONLY,
		/** Default queue is empty, next item should be retrieved by waiting for the MPI request */
		POOL_REQUEST_PENDING,
		/** There are no more batches anywhere, distribution task is done */
		NO_MORE_BATCHES
	} state;

	/** Amount of batches that were marked as completed by batchCompleted() */
	uint32_t completedBatches = 0;
	/** Semaphore controlling the amount of batches that were retrieved by nextBatch(), but not yet marked as completed.
	 * This exists to never have more than preferredBatchesInPipeline batches in the pipeline. */
	sem_t batchesInPipeline;

	MPI_Request mpiRequests[2];
	uint32_t mpiSendBuf;
	int32_t mpiRecvBuf;

	void doRequestFromPool() {
		// See runMaster() for description of sent data
		LOG_MPI("doRequestFromPool");
		mpiSendBuf = completedBatches;
		CHECK_MPI(MPI_Isend(&mpiSendBuf, 1, MPI_UINT32_T, 0, TAG_WORKER_TO_MASTER, MPI_COMM_WORLD, &mpiRequests[0]));
		CHECK_MPI(MPI_Irecv(&mpiRecvBuf, 1, MPI_INT32_T, 0, TAG_MASTER_TO_WORKER, MPI_COMM_WORLD, &mpiRequests[1]));
	}

	int32_t doCompleteRequestFromPool() {
		// Wait for the request!
		LOG_MPI("doCompleteRequestFromPool");
		CHECK_MPI(MPI_Waitall(2, mpiRequests, MPI_STATUSES_IGNORE));
		return mpiRecvBuf;
	}

public:
	DistributedBatchProvider(const ProgRecFourierMpiStarPU& parent, uint32_t batchCount, int workerCount, int workerRank)
	: parent(parent) {
		uint32_t nextBatch = parent.defaultBatchDistribution(batchCount, workerCount, workerRank, &defaultBatchQueue);
		maxBatchCount = static_cast<uint32_t>(defaultBatchQueue.size() + batchCount - nextBatch);

		if (defaultBatchQueue.empty()) {
			assert(nextBatch >= batchCount);
			state = State::NO_MORE_BATCHES;
		} else if (nextBatch >= batchCount) {
			state = State::QUEUE_ONLY;
		} else {
			state = State::QUEUE_AND_POOL;
		}

		sem_init(&batchesInPipeline, 0, parent.preferredBatchesInPipeline);
	}

	~DistributedBatchProvider() {
		sem_destroy(&batchesInPipeline);
	}

	uint32_t maxBatches() override {
		return maxBatchCount;
	}

	int32_t nextBatch() override {
		switch (state) {
			case State::QUEUE_AND_POOL: {
				uint32_t fromQueue = defaultBatchQueue.front();
				defaultBatchQueue.pop_front();
				if (defaultBatchQueue.empty()) {
					// Send request for more
					doRequestFromPool();
					state = State::POOL_REQUEST_PENDING;
				}
				sem_wait(&batchesInPipeline);
				return fromQueue;
			}
			case State::QUEUE_ONLY: {
				uint32_t fromQueue = defaultBatchQueue.front();
				defaultBatchQueue.pop_front();
				if (defaultBatchQueue.empty()) {
					// That's it
					state = State::NO_MORE_BATCHES;
				}
				sem_wait(&batchesInPipeline);
				return fromQueue;
			}
			case State::POOL_REQUEST_PENDING: {
				int32_t fromPool = doCompleteRequestFromPool();
				if (fromPool == -1) {
					// That's it
					state = State::NO_MORE_BATCHES;
					return -1;
				} else {
					assert(fromPool >= 0);
					// Got something meaningful, query for more!
					doRequestFromPool();
					sem_wait(&batchesInPipeline);
					return fromPool;
				}
			}
			case State::NO_MORE_BATCHES: {
				return -1;
			}
		}
		assert(false);
	}

	void batchCompleted() override {
		completedBatches++;
		sem_post(&batchesInPipeline);
	}
};

void ProgRecFourierMpiStarPU::runWorker(MPI_Comm workComm, int workerCount, int workerRank) {
	prepareMetaData(fn_in, SF);
	const uint32_t batchCount = computeBatchCount(params, SF);

	prepareConstants(params, SF, fn_sym, computeConstants);

	initStarPU();

	LOG_MPI("runWorker - starpu start");
	ProgRecFourierMpiStarPU::DistributedBatchProvider batchSource(*this, batchCount, workerCount, workerRank);
	ComputeStarPUResult result = computeStarPU(params, SF, computeConstants, batchSource, (bool) verbose);
	LOG_MPI("runWorker - starpu end");

	// We won't need StarPU anymore
	shutdownStarPU();

	const bool primary = workerRank == 0;
	{ // Reduce result into the primary worker
		int reduceSize = (computeConstants.maxVolumeIndex + 1) * (computeConstants.maxVolumeIndex + 1) *
		                 (computeConstants.maxVolumeIndex + 1);
		MPI_Request requests[2];
		LOG_MPI("runWorker - reduce submit");
		CHECK_MPI(MPI_Ireduce(primary ? MPI_IN_PLACE : result.volumeData, result.volumeData,
		                      reduceSize * 2, MPI_FLOAT,
		                      MPI_SUM, 0, workComm, &requests[0]));
		CHECK_MPI(MPI_Ireduce(primary ? MPI_IN_PLACE : result.weightsData, result.weightsData,
		                      reduceSize, MPI_FLOAT,
		                      MPI_SUM, 0, workComm, &requests[1]));

		LOG_MPI("runWorker - reduce wait");
		CHECK_MPI(MPI_Waitall(2, requests, MPI_STATUSES_IGNORE));
		LOG_MPI("runWorker - reduce done");
	}

	if (!primary) {
		// Only primary worker continues past this point, other workers may end now
		result.destroy();
		LOG_MPI("runWorker - non primary quitting");
		return;
	}

	// Convert flat volume and weight arrays into multidimensional arrays and destroy originals
	std::complex<float>*** tempVolume = result.createXmippStyleVolume(computeConstants.maxVolumeIndex);
	float*** tempWeights = result.createXmippStyleWeights(computeConstants.maxVolumeIndex);
	result.destroy();

	// Adjust and save the resulting volume
	postProcessAndSave(params, computeConstants, fn_out, tempVolume, tempWeights);
}

void ProgRecFourierMpiStarPU::read(int argc, char **argv, bool reportErrors) {
	// Not using XmippMpiProgram, because it is unnecessarily complicated for this use-case
	// and does not initialize MPI with thread support, which is required
	LOG_MPI("Initializing MPI");
	int providedThreadSupport;
	CHECK_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &providedThreadSupport));
	mpiInitialized = true;
	LOG_MPI("MPI initialized with thread level " << providedThreadSupport);

	if (providedThreadSupport < MPI_THREAD_SERIALIZED) {
		std::cerr << "MPI implementation does not support MPI_THREAD_SERIALIZED concurrency level, which is required. Provided level is " << providedThreadSupport << '\n';
		REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Could not initialize MPI");
	}

	XmippProgram::read(argc, argv, reportErrors);
}

int ProgRecFourierMpiStarPU::tryRun() {
	// Based on XmippMpiProgram::tryRun()
	try {
		if (doRun)
			this->run();
	} catch (XmippError &xe) {
		std::cerr << xe;
		errorCode = xe.__errno;
		MPI_Abort(MPI_COMM_WORLD, xe.__errno);
	}
	return errorCode;
}

ProgRecFourierMpiStarPU::~ProgRecFourierMpiStarPU() {
	if (mpiInitialized) {
		LOG_MPI("Finalizing MPI");
		CHECK_MPI(MPI_Finalize());
	}
}
