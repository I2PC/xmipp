#ifndef PRNG_H
#define PRNG_H

#include <random>

/**
 `distribution` must comply to https://en.cppreference.com/w/cpp/named_req/RandomNumberDistribution

  standard library distributions are also on that page
*/
template< typename T, typename distribution, typename generator=std::mt19937 >
class RandUtils {
public:

	template< typename... Args >
	RandUtils( Args... args )
	: distr( args... ) {
		gen.seed( rd() );
	}

	/**
	 * Reseeds generator with `seed`
	*/
	void reseed( uint64_t seed ) { gen.seed( seed ); }
	/**
	 * Uses random device to reseed generator
	*/
	void reseed() { gen.seed( rd() ); }

	T rand() {
		return distr( gen );
	}

	/**
	 * Don't use this for generation of large amount of values !!
	*/
	template< typename... Args >
	static T rand_once( Args... args ) {
		std::random_device rd;
		std::mt19937 gen( rd() );

		return distribution( args... )( gen );
	}

private:
	std::random_device rd;
	generator gen;
	distribution distr;
};

template< typename T >
using GaussGenerator = RandUtils< T, std::normal_distribution<> >;

template< typename T >
using ExponentialGenerator = RandUtils< T, std::exponential_distribution<> >;

#endif // PRNG_H