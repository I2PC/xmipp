#include <iostream>
#include <random>

#include "prng.h"
#include "deviation.h"

/**
 * Example program using new RandUtils
*/

int main() {
	/**
	 * You can supply own RNG generator
	*/
	RandUtils<double, std::exponential_distribution<>, std::mt19937_64> exp_rand( 2.0 );

	/**
	 * Some distributions have predefined generators
	*/
	ExponentialGenerator< double > eg( 2.0 );
	GaussGenerator< double > gg( 2.0, 0.5 );
	/**
	* You can use RandUtils with own deviation class
	*/
	RandUtils< double, SomeNewDistribution< double > > exp_rand2;

	/**
	* reseed with own seed if determinism is needed
	*/
	exp_rand.reseed( 1 );

	double x = 0;

	for ( int i = 0; i < 100000; ++i ) {
		x += gg.rand();
	}

	/**
	 * if the generator is used only once, then just call static function
	*/
	x += ExponentialGenerator< double >::rand_once( 2.0 );

	std::cout << "x = " << x << std::endl;
}