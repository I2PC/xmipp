#ifndef _PROG_VOLUME_RESTORATION_DENOISE
#define _PROG_VOLUME_RESTORATION_DENOISE

template< typename T >
class VolumeRestorationDenoise {
	const size_t iterations;


public:

	VolumeRestorationDenoise(size_t iterations)
	: iterations(iterations) {}
};

#endif