/***************************************************************************
 *
 * Authors:    Jan Polák (456647@mail.muni.cz)
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

#ifndef XMIPP_RECONSTRUCT_FOURIER_SCHEDULER_H
#define XMIPP_RECONSTRUCT_FOURIER_SCHEDULER_H

#include <starpu.h>
#include <bitset>
#include <deque>
#include <mutex>
#include <atomic>

#include "reconstruct_fourier_timing.h"
#include "util/queue_bag.h"

#ifndef STARPU_NMAXWORKERS
// FOR EDITING ONLY (should not be needed when compiling)
#define STARPU_NMAXWORKERS 10
#endif

#define RFS_LOGGING 1

struct Schedulers {

	/** Specialized scheduler for fourier reconstruction. */
	starpu_sched_policy reconstruct_fourier;

	/** Initializes the schedulers. */
	Schedulers() noexcept;
};

extern Schedulers schedulers;

struct task_data {
	/** The task this relates to. */
	starpu_task* task;

	/** Which implementation is best for given worker. -1 if no such implementation. */
	int8_t best_implementation_by_worker[STARPU_NMAXWORKERS] = {};
	/** Time it will take to execute this task on this worker given the best implementation. */
	uint32_t best_implementation_time_by_worker[STARPU_NMAXWORKERS] = {};

	explicit task_data(starpu_task * task):task(task) {}

	void recompute_metrics(const std::bitset<STARPU_NMAXWORKERS>& available_workers, unsigned sched_ctx_id);

	int find_best_worker(const std::bitset<STARPU_NMAXWORKERS>& available_workers);
};

struct worker_data {

	/** Timing device key, used for finding out nextSimilarWorker */
	timing_device_key deviceKey;

	/** Index of the next worker, with the same deviceKey. May be own deviceId -> following these should be cyclic.
	 * Used for queue sharing (basically a lightweight version of stealing). */
	int nextSimilarWorker;

	/** If this worker is a captain of a combined worker, this is an worker id fo that combined worker. */
	int combinedWorkerCaptain = -1;

	/** Flag checking whether the worker is available for any combined worker fun. */
	std::atomic_bool busy;

	/** Lock this before manipulating queued_load or queue */
	std::mutex workerMutex;

	/** Amount of time in queue. */
	uint64_t queued_load = 0;

	/** Queue of tasks to be completed by this worker */
	rfs::queue_bag<task_data> queue;

	// Following fields may be modified only by the worker itself!
	/** Load of the task that is being evaluated right now. */
	uint32_t current_load = 0;
	/** When did the task execution start. For custom time measurements, because StarPU won't expose this info to us. */
	double execStartTimeUs = 0;

	bool isQueuedLoadCorrect(unsigned workerId) {
		uint64_t load = 0;
		rfs::forEach(queue, [&load, workerId](const task_data& task) {
			uint64_t loadBefore = load;
			load += task.best_implementation_time_by_worker[workerId];
			assert(loadBefore < load);
			return true;
		});
		if (load != queued_load) {
			fprintf(stderr, "Expected load %llu, got %llu\n", (unsigned long long) queued_load,
			        (unsigned long long) load);
			return false;
		}
		return true;
	}
};

struct rfs_data {

	/** Information about which workers are enabled. */
	std::bitset<STARPU_NMAXWORKERS> workers_mask;
	worker_data workers[STARPU_NMAXWORKERS];

	timing timing;

#if RFS_LOGGING
	FILE* log = nullptr;
#endif
};

#endif //XMIPP_RECONSTRUCT_FOURIER_SCHEDULER_H
