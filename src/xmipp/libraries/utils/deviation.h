#ifndef DEVIATION_H
#define DEVIATION_H

#include <random>
#include <cmath>

/**
 * Implementation of some new distribution that can be
 * supplied to RandUtils
 * Fun fact: it should've been normal distribution,
 * but something went wrong
*/
template< typename T >
class SomeNewDistribution {
public:

	SomeNewDistribution( T mean=0, T std=1 )
	: mean( mean )
	, std( std ) {}

	template< typename generator >
	T operator()( generator& gen ) {
		T u, v, x, y, q;
		do {
			u = uniform( gen );
			v = 1.7156 * ( uniform( gen ) - 0.5 );
			x = u - 0.449871;
			y = abs( v ) + 0.386595;
			q = x * x + y * ( 0.19600 * y - 0.25472 * x );
		} while( q > 0.27597 && ( q > 0.27846 || v * v > -4. * std::log( u ) * u * u ) );
		return mean + std * v / u;
	}

private:

	T mean;
	T std;

	std::uniform_real_distribution<T> uniform;
};

#endif //DEVIATION_H
