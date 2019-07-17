#ifndef XMIPP_RECONSTRUCT_FOURIER_TIMING_H
#define XMIPP_RECONSTRUCT_FOURIER_TIMING_H

#include <cstdint>
#include <limits>
#include <cmath>
#include <map>
#include <mutex>

#define STARPU_DONT_INCLUDE_CUDA_HEADERS
#include <starpu.h>
#undef STARPU_DONT_INCLUDE_CUDA_HEADERS

/** A large amount of time (in microseconds) to prevent overflows for ridiculously slow tasks. */
static const uint32_t MAX_TIME = std::numeric_limits<uint32_t>::max();// little over 1 hour
static const uint32_t MIN_TIME = 1;

static inline uint64_t sanitize_time(double microseconds) {
	if (microseconds < MIN_TIME) {
		return MIN_TIME;
	} else if (microseconds > MAX_TIME) {
		return MAX_TIME;
	} else if (isnan(microseconds)) {
		// Not calibrated yet: We need a small value to force the calibration to run,
		// but not zero, because that will cause it to be always used.
		return MIN_TIME;
	} else {
		return (uint64_t) microseconds;
	}
}

struct timing_data {
	static const uint8_t TIMING_COUNT_POW = 6;

	uint32_t timings[1u << TIMING_COUNT_POW] = {0};
	uint8_t count = 0;
	uint8_t next = 0;
	bool dirty = true;

	void add(double us);

	void fill(uint32_t us);

	uint64_t average(bool onlyProven);
};

struct timing_device_key {
	uint16_t device_type;
	uint16_t device_id;

	timing_device_key():device_type(0), device_id(0) {}

	timing_device_key(starpu_worker_archtype device_type, uint16_t device_id)
	: device_type(device_type), device_id(device_type == starpu_worker_archtype::STARPU_CPU_WORKER ? -1 : device_id) {}

	uint32_t key() const {
		return ((uint32_t)device_type << 16u) | device_id;
	}
};

inline bool operator<(const timing_device_key& a, const timing_device_key& b) {
	return a.key() < b.key();
}
inline bool operator==(const timing_device_key& a, const timing_device_key& b) {
	return a.key() == b.key();
}
inline bool operator!=(const timing_device_key& a, const timing_device_key& b) {
	return a.key() != b.key();
}

struct timing_key {
	const char* const codelet_name;
	const uint32_t impl;
	const timing_device_key device;
	const size_t size;

	timing_key(const char* codelet_name, size_t size, uint32_t impl, starpu_worker_archtype device_type, uint16_t device_id)
	: codelet_name(codelet_name), size(size), impl(impl), device(device_type, device_id) {}

	uint64_t key() const {
		return ((uint64_t)impl << 32u) | (uint64_t)device.key();
	}

	bool equalsExceptSize(const timing_key& other) const {
		return strcmp(codelet_name, other.codelet_name) == 0 && key() == other.key();
	}
};
inline bool operator<(const timing_key& a, const timing_key& b) {
	int name = strcmp(a.codelet_name, b.codelet_name);
	return name < 0
	|| (name == 0 && a.key() < b.key())
	|| (name == 0 && a.key() == b.key() && a.size < b.size);
}
inline bool operator==(const timing_key& a, const timing_key& b) {
	return strcmp(a.codelet_name, b.codelet_name) == 0 && a.size == b.size && a.key() == b.key();
}

struct timing {
	std::map<timing_key, timing_data> timings;
	std::mutex lock;

	void record(starpu_codelet* codelet, unsigned impl, starpu_worker_archtype archtype, int devid, size_t size, double microseconds);

	double estimate(starpu_codelet* codelet, unsigned impl, starpu_worker_archtype archtype, int devid, size_t size);

	void load(const char* name) noexcept;

	void save(const char* name);
};

#endif //XMIPP_RECONSTRUCT_FOURIER_TIMING_H
