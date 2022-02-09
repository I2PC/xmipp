#ifndef CUDA_VEC2
#define CUDA_VEC2

#include <cuda_runtime_api.h>

/*
 * Vec2 is used for templated version of float2/double2
*/
template< typename T > struct Vec2;

template <> struct Vec2<float> {
	using type = float2;
};

template <> struct Vec2<double> {
	using type = double2;
};

template< typename T >
using vec2_type = typename Vec2<T>::type;

#endif // CUDA_VEC2