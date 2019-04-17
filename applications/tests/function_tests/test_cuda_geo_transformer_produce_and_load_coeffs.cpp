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
};

class GeoTransformerProduceAndLoadCoeffsTest : public ::testing::Test {

public:

    void compare_results( float* true_values, float* approx_values, size_t size ) {
        for ( int i = 0; i < size; ++i ) {
            ASSERT_NEAR( true_values[i], approx_values[i], 0.00001f );
        }
    }

    void allocate_arrays() {
        size = x * y;
        in = std::unique_ptr<float[]>( new float[size] );
        out = std::unique_ptr<float[]>( new float[size] );
    }

    void run_test() {
        cpu_reference( in.get(), out.get(), x, y );

        MultidimArray< float > tmpIn( y, x );
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY( tmpIn ) {
            DIRECT_MULTIDIM_ELEM( tmpIn, n ) = in[n];
        }

        GeoTransformerTest< float > gt;
        gt.initLazyForBSpline( x, y, 1, 1, 1, 1 );
        gt.produceAndLoadCoeffsTest( tmpIn );

        auto gpu_out = gt.copy_out_d_in( size );

        compare_results( out.get(), gpu_out.get(), size );
    }

    std::pair< size_t, size_t > random_size() {
        std::random_device rd;
        std::mt19937 gen( rd() );
        std::uniform_int_distribution<> dis( 128, 8192 );

        return { dis( gen ), dis( gen ) };
    }

    void randomly_initialize(float* array, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for ( int i = 0; i < size; ++i ) {
        array[i] = dis(gen);
    }
}

void cpu_reference(const float *in, float *out, int x, int y) {
    const float gain = 6.0f;
    const float z = sqrtf(3.f) - 2.f;
    float z1;

    // process lines
    for (int line = 0; line < y; line++) {
        float* myLine = out + (line * x);

        // copy input data
        for (int i = 0; i < x; i++) {
            myLine[i] = in[i + (line * x)] * gain;
        }

        // compute 'sum'
        float sum = (myLine[0] + powf(z, x)
            * myLine[x - 1]) * (1.f + z) / z;
        z1 = z;
        float z2 = powf(z, 2 * x - 2);
        float iz = 1.f / z;
        for (int j = 1; j < (x - 1); ++j) {
            sum += (z2 + z1) * myLine[j];
            z1 *= z;
            z2 *= iz;
        }

        // iterate back and forth
        myLine[0] = sum * z / (1.f - powf(z, 2 * x));
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
        float* myCol = out + col;

        // multiply by gain (input data are already copied)
        for (int i = 0; i < y; i++) {
            myCol[i*x] *= gain;
        }

        // compute 'sum'
        float sum = (myCol[0*x] + powf(z, y)
            * myCol[(y - 1)*x]) * (1.f + z) / z;
        z1 = z;
        float z2 = powf(z, 2 * y - 2);
        float iz = 1.f / z;
        for (int j = 1; j < (y - 1); ++j) {
            sum += (z2 + z1) * myCol[j*x];
            z1 *= z;
            z2 *= iz;
        }

        // iterate back and forth
        myCol[0*x] = sum * z / (1.f - powf(z, 2 * y));
        for (int j = 1; j < y; ++j) {
            myCol[j*x] += z * myCol[(j - 1)*x];
        }
        myCol[(y - 1)*x] *= z / (z - 1.f);
        for (int j = y - 2; 0 <= j; --j) {
            myCol[j*x] = z * (myCol[(j + 1)*x] - myCol[j*x]);
        }
    }
}

    size_t x;
    size_t y;
    size_t size;

    std::unique_ptr< float[] > in;
    std::unique_ptr< float[] > out;
    std::unique_ptr< float[] > gpu_out;
};

TEST_F(GeoTransformerProduceAndLoadCoeffsTest, RandomSizeZeroInput) {
    std::tie( x, y ) = random_size();
    allocate_arrays();

    run_test();
}

TEST_F(GeoTransformerProduceAndLoadCoeffsTest, PaddedSizeRandomInput) {
    x = 2048 + 128;
    y = 1024;
    allocate_arrays();

    randomly_initialize( in.get(), size );

    run_test();
}

TEST_F(GeoTransformerProduceAndLoadCoeffsTest, PaddedSquareSizeRandomInput) {
    x = 1024;
    y = x;
    allocate_arrays();

    randomly_initialize( in.get(), size );

    run_test();
}

TEST_F(GeoTransformerProduceAndLoadCoeffsTest, SquareSizeRandomInput) {
    x = 259;
    y = x;
    allocate_arrays();

    randomly_initialize( in.get(), size );

    run_test();
}

TEST_F(GeoTransformerProduceAndLoadCoeffsTest, RandomSizeRandomInput) {
    std::tie( x, y ) = random_size();

    allocate_arrays();

    randomly_initialize( in.get(), size );

    run_test();

}

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}