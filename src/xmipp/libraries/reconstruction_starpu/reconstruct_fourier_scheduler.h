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

#define RFS_LOGGING 0

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

	int32_t picked_worker = -1;

	explicit task_data(starpu_task * task):task(task) {}

	int32_t best_worker(const std::bitset<STARPU_NMAXWORKERS>& available_workers, unsigned sched_ctx_id, bool includeTransferTime) {
		if (!task->cl) {
			return -1;
		}
		recompute_metrics(available_workers, sched_ctx_id, false);
		return find_best_worker(available_workers);
	}

	/*
	 * Debug tracking of the task state.
	 * This makes sure that the assertions on the task lifecycle hold.
	 *
	 * There are two assumed paths:
	 * 1. Regular
	 * SUBMITTED -> QUEUED (-> stealing happens here) -> POPPED -> EXECUTING -> DONE
	 * 2. Outside of the scheduler
	 * SUBMITTED -> QUEUED_OUTSIDE -> EXECUTING -> DONE
	 *
	 * Since the StarPU documentation does not really specify this, these methods assert that it holds.
	 */

	void stateOnPush() {
		switchState(SUBMITTED, QUEUED);
	}
	void stateOnPushNotify() {
		switchState(SUBMITTED, QUEUED_OUTSIDE);
	}
	void stateOnPop() {
		switchState(QUEUED, POPPED);
	}
	void stateOnPreExec() {
		switchState(POPPED, QUEUED_OUTSIDE, EXECUTING);
	}
	void stateOnPostExec() {
		switchState(EXECUTING, DONE);
	}

private:

	enum task_data_state {
		SUBMITTED,
		QUEUED,
		QUEUED_OUTSIDE,
		POPPED,
		EXECUTING,
		DONE
	};

#if RFS_LOGGING
	std::atomic<task_data_state> state { SUBMITTED };
#endif

	void switchState(task_data_state from, task_data_state to) {
#if RFS_LOGGING
		if (!state.compare_exchange_strong(from, to)) {
			fprintf(stderr, "Failed to transition to %d (from: %d)\n", to, from);
			assert(false);
		}
#endif
	}

	void switchState(task_data_state from1, task_data_state from2, task_data_state to) {
#if RFS_LOGGING
		if (!state.compare_exchange_strong(from1, to) && !state.compare_exchange_strong(from2, to)) {
			fprintf(stderr, "Failed to transition to %d (from1: %d, from2: %d)\n", to, from1, from2);
			assert(false);
		}
#endif
	}

	void recompute_metrics(const std::bitset<STARPU_NMAXWORKERS>& available_workers, unsigned sched_ctx_id, bool includeTransferTime);

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

	/** Total amount of time in tasks that were submitted (not yet in the queue), those in queue and those that are being executed right now. */
	uint64_t total_load = 0;

	/** Queue of tasks to be completed by this worker */
	rfs::queue_bag<task_data*> queue;

	// Following fields may be modified only by the worker itself!
	/** Load of the task that is being evaluated right now. */
	uint32_t current_load = 0;
	/** When did the task execution start. For custom time measurements, because StarPU won't expose this info to us. */
	double execStartTimeUs = 0;
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
