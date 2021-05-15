#ifndef CUDA_VOLUME_DEFORM_SPH_DEFINES_H
#define CUDA_VOLUME_DEFORM_SPH_DEFINES_H

// Kernel block sizes
#define BLOCK_X_DIM 16
#define BLOCK_Y_DIM 8
#define BLOCK_Z_DIM 1

// Tuning parameters

#ifndef L1
#define L1 5
#endif
#ifndef L2
#define L2 5
#endif

#endif// CUDA_VOLUME_DEFORM_SPH_DEFINES_H
