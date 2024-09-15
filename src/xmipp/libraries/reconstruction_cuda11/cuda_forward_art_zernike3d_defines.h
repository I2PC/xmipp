#ifndef CUDA_FORWARD_ART_ZERNIKE3D_DEFINES_H
#define CUDA_FORWARD_ART_ZERNIKE3D_DEFINES_H


#define NONE 0
#define MAXWELL 4
#define PASCAL 5
#define TURING 6
#define AMPERE 7

#define BLOCK_SIZE 256

// Degrees of the basis
// (Setting exact degrees you use in the run may increase performance,
// especially when the degrees are low)
#ifndef L1
#define L1 15
#endif
#ifndef L2
#define L2 12
#endif

// Universal block sizes
// (cannot be done fully automatic because we are using BLOCK_*_DIM macros in host code)

// Set ARCH to the architecture of your GPU for better performance
#define ARCH PASCAL

#if ARCH == MAXWELL
#define BLOCK_X_DIM 16
#define BLOCK_Y_DIM 4
#define BLOCK_Z_DIM 2
#elif ARCH == PASCAL
#define BLOCK_X_DIM 16
#define BLOCK_Y_DIM 8
#define BLOCK_Z_DIM 1
#elif ARCH == TURING
#define BLOCK_X_DIM 16
#define BLOCK_Y_DIM 8
#define BLOCK_Z_DIM 1
#elif ARCH == AMPERE
#define BLOCK_X_DIM 32
#define BLOCK_Y_DIM 1
#define BLOCK_Z_DIM 4
#else
#define BLOCK_X_DIM 8
#define BLOCK_Y_DIM 4
#define BLOCK_Z_DIM 4
#endif
// Tuning parameters

#endif	// CUDA_FORWARD_ART_ZERNIKE3D_DEFINES_H
