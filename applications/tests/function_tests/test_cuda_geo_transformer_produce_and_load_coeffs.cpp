#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"

template< typename T >
class GeoTransformerTest : public GeoTransformer< T > {
public:

    template< typename IN >
    void produceAndLoadCoeffsTest( const MultidimArray<IN>& input ) {
        GeoTransformer<T>::produceAndLoadCoeffs( input );
    }

    std::unique_ptr<T[]> copy_out_d_inTest( size_t size ) const {
        return GeoTransformer<T>::copy_out_d_in( size );
    }
};

template< typename T >
class GeoTransformerProduceAndLoadCoeffsTest : public ::testing::Test {

public:

    void compare_results( T* true_values, T* approx_values, size_t size ) {
        for ( size_t i = 0; i < size; ++i ) {
            ASSERT_NEAR( true_values[i], approx_values[i], static_cast< T >( 0.00001 ) ) << "at index=" << i;
        }
    }

    void allocate_arrays() {
        size = x * y;
        in = std::unique_ptr<T[]>( new T[size]() );
        out = std::unique_ptr<T[]>( new T[size]() );
    }

    void run_test() {
        cpu_reference( in.get(), out.get(), x, y );

        MultidimArray< T > tmpIn( y, x );
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY( tmpIn ) {
            DIRECT_MULTIDIM_ELEM( tmpIn, n ) = in[n];
        }

        GeoTransformerTest< T > gt;
        gt.initLazyForBSpline( x, y, 1, 1, 1, 1 );
        gt.produceAndLoadCoeffsTest( tmpIn );

        auto gpu_out = gt.copy_out_d_inTest( size );

        compare_results( out.get(), gpu_out.get(), size );
    }

    std::pair< size_t, size_t > random_size(int seed) {
        gen.seed( seed );
        std::uniform_int_distribution<> dis( 128, 8192 );

        return { dis( gen ), dis( gen ) };
    }

    void randomly_initialize(T* array, int size, int seed) {
        gen.seed( seed );
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for ( int i = 0; i < size; ++i ) {
            array[i] = dis(gen);
        }
    }

    void cpu_reference(const T *in, T *out, int x, int y) {
        const T gain = 6.0;
        const T z = sqrt(3.0) - 2.0;
        T z1;

        // process lines
        for (int line = 0; line < y; line++) {
            T* myLine = out + (line * x);

            // copy input data
            for (int i = 0; i < x; i++) {
                myLine[i] = in[i + (line * x)] * gain;
            }

            // compute 'sum'
            T sum = (myLine[0] + pow(z, x)
                * myLine[x - 1]) * (1.0 + z) / z;
            z1 = z;
            T z2 = pow(z, 2 * x - 2);
            T iz = 1.0 / z;
            for (int j = 1; j < (x - 1); ++j) {
                sum += (z2 + z1) * myLine[j];
                z1 *= z;
                z2 *= iz;
            }

            // iterate back and forth
            myLine[0] = sum * z / (1.0 - pow(z, 2 * x));
            for (int j = 1; j < x; ++j) {
                myLine[j] += z * myLine[j - 1];
            }
            myLine[x - 1] *= z / (z - 1.f);
            for (int j = x - 2; 0 <= j; --j) {
                myLine[j] = z * (myLine[j + 1] - myLine[j]);
            }
        }

        // process columns
        for (int col = 0; col < x; col++) {
            T* myCol = out + col;

            // multiply by gain (input data are already copied)
            for (int i = 0; i < y; i++) {
                myCol[i*x] *= gain;
            }

            // compute 'sum'
            T sum = (myCol[0*x] + pow(z, y)
                * myCol[(y - 1)*x]) * (1.0 + z) / z;
            z1 = z;
            T z2 = pow(z, 2 * y - 2);
            T iz = 1.0 / z;
            for (int j = 1; j < (y - 1); ++j) {
                sum += (z2 + z1) * myCol[j*x];
                z1 *= z;
                z2 *= iz;
            }

            // iterate back and forth
            myCol[0*x] = sum * z / (1.0 - pow(z, 2 * y));
            for (int j = 1; j < y; ++j) {
                myCol[j*x] += z * myCol[(j - 1)*x];
            }
            myCol[(y - 1)*x] *= z / (z - 1.0);
            for (int j = y - 2; 0 <= j; --j) {
                myCol[j*x] = z * (myCol[(j + 1)*x] - myCol[j*x]);
            }
        }
    }

    size_t x = 0;
    size_t y = 0;
    size_t size;

    std::unique_ptr< T[] > in;
    std::unique_ptr< T[] > out;
    std::unique_ptr< T[] > gpu_out;

    std::mt19937 gen;
};

TYPED_TEST_CASE_P(GeoTransformerProduceAndLoadCoeffsTest);

TYPED_TEST_P(GeoTransformerProduceAndLoadCoeffsTest, RandomSizeZeroInput) {
    std::tie( this->x, this->y ) = this->random_size( 41 );
    this->allocate_arrays();

    this->run_test();
}

TYPED_TEST_P(GeoTransformerProduceAndLoadCoeffsTest, PaddedSizeRandomInput) {
    this->x = 2048 + 128;
    this->y = 1024;
    this->allocate_arrays();

    this->randomly_initialize( this->in.get(), this->size, 53 );

    this->run_test();
}

TYPED_TEST_P(GeoTransformerProduceAndLoadCoeffsTest, PaddedSquareSizeRandomInput) {
    this->x = 1024;
    this->y = this->x;
    this->allocate_arrays();

    this->randomly_initialize( this->in.get(), this->size, 67 );

    this->run_test();
}

TYPED_TEST_P(GeoTransformerProduceAndLoadCoeffsTest, SquareSizeRandomInput) {
    this->x = 259;
    this->y = this->x;
    this->allocate_arrays();

    this->randomly_initialize( this->in.get(), this->size, 89 );

    this->run_test();
}

TYPED_TEST_P(GeoTransformerProduceAndLoadCoeffsTest, RandomSizeRandomInput) {
    std::tie( this->x, this->y ) = this->random_size( 39 );
    this->allocate_arrays();

    this->randomly_initialize( this->in.get(), this->size, 19 );

    this->run_test();
}

TYPED_TEST_P(GeoTransformerProduceAndLoadCoeffsTest, OddXEvenYRandomInput) {
    this->x = 479;
    this->y = 342;

    this->allocate_arrays();
    this->randomly_initialize( this->in.get(), this->size, 33 );

    this->run_test();
}

REGISTER_TYPED_TEST_CASE_P(GeoTransformerProduceAndLoadCoeffsTest,
    RandomSizeZeroInput,
    PaddedSizeRandomInput,
    PaddedSquareSizeRandomInput,
    SquareSizeRandomInput,
    RandomSizeRandomInput,
    OddXEvenYRandomInput
);

using ScalarTypes = ::testing::Types< float, double >;
INSTANTIATE_TYPED_TEST_CASE_P(ScalarTypesInstantiation, GeoTransformerProduceAndLoadCoeffsTest, ScalarTypes);


GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}