/*
 * cuda_compatibility.h
 *
 *  Created on: Dec 13, 2018
 *      Author: david
 */

#ifndef LIBRARIES_DATA_CUDA_COMPATIBILITY_H_
#define LIBRARIES_DATA_CUDA_COMPATIBILITY_H_


#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

#ifdef __CUDACC__
#define CUDA_H __host__
#else
#define CUDA_H
#endif


#endif /* LIBRARIES_DATA_CUDA_COMPATIBILITY_H_ */
