#include "reconstruct_fourier_timing.h"
#include <algorithm>
#include <inttypes.h>

void timing::record(starpu_codelet *codelet, unsigned impl, starpu_worker_archtype archtype, int devid, size_t size, double microseconds) {
	if (codelet == nullptr) {
		return;
	}

	std::lock_guard<std::mutex> lg(lock);
	//fprintf(stderr, "Recording that %s(%d) on %d:%d (%zd bytes) did take %" PRIu64 " us\n", codelet->name, impl, archtype, devid, size, (uint64_t)microseconds);
	timings[timing_key{codelet->name, size, (uint32_t)impl, archtype, (uint16_t) devid}].add(microseconds);
}

static uint64_t estimateForNewSize(size_t originalSize, uint64_t originalEstimate, size_t forSize) {
	const double overhead = 0.5;
	return (uint64_t) (originalEstimate * overhead + (originalEstimate * (1.0 - overhead)) * (double)forSize / (double)originalSize);
}

double timing::estimate(starpu_codelet *codelet, unsigned impl, starpu_worker_archtype archtype, int devid, size_t size) {
	if (codelet == nullptr) {
		return MIN_TIME;
	}

	std::lock_guard<std::mutex> lg(lock);
	if (timings.empty()) {
		return MIN_TIME;
	}

	auto key = timing_key{codelet->name, size, (uint32_t) impl, archtype, (uint16_t) devid};
	auto found = timings.lower_bound(key);

	if (found != timings.end() && found->first == key) {
		uint64_t result = found->second.average(true);
		if (result != MIN_TIME) {
			// This key exists and provides valid estimates, use them
			return result;
		}
	}

	// We need to check around for key with good estimates
	const size_t maxDistance = std::numeric_limits<size_t>::max();
	// Check up
	size_t upDistance = maxDistance;
	uint64_t upEstimate = 0;
	{
		auto up = found;
		while (up != timings.end() && up->first.equalsExceptSize(key)) {
			uint64_t result = up->second.average(true);
			if (result != MIN_TIME) {
				// We could use it!
				upDistance = up->first.size - size;
				upEstimate = estimateForNewSize(up->first.size, result, size);
				break;
			}
			up = std::next(up);
		}
	}

	size_t downDistance = maxDistance;
	uint64_t downEstimate = 0;
	{
		auto down = found;
		while (down != timings.begin()) {
			down = std::prev(down);
			if (!down->first.equalsExceptSize(key)) {
				break;
			}
			uint64_t result = down->second.average(true);
			if (result != MIN_TIME) {
				downDistance = size - down->first.size;
				downEstimate = estimateForNewSize(down->first.size, result, size);
				break;
			}
		}
	}

	if (upDistance == maxDistance && downDistance == maxDistance) {
		return MIN_TIME;
	}

	if (downDistance < upDistance) {
		//fprintf(stderr, "Using close down estimate for %s(%d) on %d:%d (%zd bytes, distance %zd) will take %" PRIu64 " us\n", codelet->name, impl, archtype, devid, size, downDistance, downEstimate);
		return downEstimate;
	} else {
		//fprintf(stderr, "Using close up estimate for %s(%d) on %d:%d (%zd bytes, distance %zd) will take %" PRIu64 " us\n", codelet->name, impl, archtype, devid, size, upDistance, upEstimate);
		return upEstimate;
	}
}

void timing::load(const char *name) noexcept {
	FILE* file = fopen(name, "r");
	if (file == nullptr) {
		return;
	}

	std::lock_guard<std::mutex> lg(lock);
	timings.clear();

	int line = 0;
	while (!feof(file)) {
		line++;
		char codeletName[128];
		size_t size;
		unsigned impl, archtype, devid;
		uint32_t timing;

		int read = fscanf(file, "%127s %zd %u %u %u %" PRIu32 "\n", codeletName, &size, &impl, &archtype, &devid, &timing);
		if (read <= 0) {
			// ok, normal failure
			continue;
		}
		if (read != 6) {
			// WEIRD
			fprintf(stderr, "Invalid format on line %d\n", line);
			continue;
		}

		auto key = timing_key { strdup(codeletName), size, (uint32_t) impl, (starpu_worker_archtype) archtype, (uint16_t) devid };
		timings[key].fill(timing);
	}

	fclose(file);
}

void timing::save(const char *name) {
	FILE* file = fopen(name, "w");
	if (file == nullptr) {
		perror("Failed to save timings");
		return;
	}

	std::lock_guard<std::mutex> lg(lock);
	for (auto it: timings) {
		uint64_t average = it.second.average(false);
		if (average <= MIN_TIME) {
			// Don't store anecdotal evidence
			continue;
		}
		fprintf(file, "%s %zd %u %u %u %" PRIu64 "\n", it.first.codelet_name, it.first.size, it.first.impl, it.first.device.device_type, it.first.device.device_id, average);
	}
	fclose(file);
}

void timing_data::add(double us) {
	timings[next] = sanitize_time(us);
	if (next + 1 > count) {
		count = next + 1;
	}
	next = (next + 1u) & ((1u << TIMING_COUNT_POW) - 1);
	dirty = true;
}

void timing_data::fill(uint32_t us) {
	// Fill half of the timings only
	count = 1u << (TIMING_COUNT_POW - 1u);
	for (int i = 0; i < count; ++i) {
		timings[i] = us;
	}
	next = count;
	dirty = false;
}

uint64_t timing_data::average(bool onlyProven) {
	if (count <= 0) {
		// No info
		return MIN_TIME;
	}

	if (onlyProven && count <= 2) {
		// Can't be sure yet
		return MIN_TIME;
	}

	if (dirty) {
		dirty = false;
		std::sort(std::begin(timings), std::begin(timings) + count);
	}

	// Actually just a mean
	return timings[count / 2];
}
