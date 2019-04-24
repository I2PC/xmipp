#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"

template< typename T >
class GeoTransformerTest : public GeoTransformer< T > {

};

class GeoTransformerApplyBSplineTransformTest : public ::testing::Test {

public:

    void compare_results( float* true_values, float* approx_values, size_t size ) {
        for ( int i = 0; i < size; ++i ) {
            ASSERT_NEAR( true_values[i], approx_values[i], 0.00001f );
        }
    }

    void allocate_arrays() {
        size = x * y;

        in.resize( y, x );
        out.resize( y, x );

        coeffsX.resize( splineX * splineY * splineN );
        coeffsY.resize( splineX * splineY * splineN );
    }

    void run_transformation() {
        GeoTransformerTest< float > gt;
        gt.initLazyForBSpline( x, y, 1, splineX, splineY, splineN );
        gt.applyBSplineTransform( 3, out, in, { coeffsX, coeffsY }, imageIdx, outside );
    }

    std::pair< size_t, size_t > random_size() {
        std::random_device rd;
        std::mt19937 gen( rd() );
        std::uniform_int_distribution<> dis( 128, 8192 );

        return { dis( gen ), dis( gen ) };
    }

    void randomly_initialize( MultidimArray< float >& array ) {
        std::random_device rd;
        std::mt19937 gen( rd() );
        std::uniform_real_distribution<> dis( -1.0, 1.0 );

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY( array ) {
            DIRECT_MULTIDIM_ELEM( array, n ) = dis( gen );
        }
    }

    void set_to_value( MultidimArray< float >& array, float value ) {
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY( array ) {
            DIRECT_MULTIDIM_ELEM( array, n ) = value;
        }
    }

    void randomly_initialize(Matrix1D< float >& array) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        FOR_ALL_ELEMENTS_IN_MATRIX1D( array ) {
            MATRIX1D_ARRAY( array )[i] = dis(gen);
        }

    }

    size_t x;
    size_t y;
    size_t size;
    size_t splineX = 1;
    size_t splineY = 1;
    size_t splineN = 1;
    size_t imageIdx = 0;

    float outside = 0;

    Matrix1D< float > coeffsX;
    Matrix1D< float > coeffsY;

    MultidimArray< float > in;
    MultidimArray< float > out;
};

TEST_F(GeoTransformerApplyBSplineTransformTest, NoChangeIfCoeffsAreZero) {
    x = 259;
    y = 311;
    allocate_arrays();

    randomly_initialize( in );
    in.write( "test_NoChangeIfCoeffsAreZero.input" );

    run_transformation();

    out.write( "test_NoChangeIfCoeffsAreZero.output" );

    compare_results( in.data, out.data, size );

}

TEST_F(GeoTransformerApplyBSplineTransformTest, ZeroInputWithNonzeroCoeffsIsZeroOutput) {
    x = 147;
    y = 147;
    splineX = 6;
    splineY = 5;
    splineN = 4;
    allocate_arrays();
    in.write( "test_ZeroInputWithNonzeroCoeffsIsZeroOutput.input" );

    randomly_initialize( coeffsX );
    randomly_initialize( coeffsY );
    coeffsX.write( "test_ZeroInputWithNonzeroCoeffsIsZeroOutput.coeffX" );
    coeffsY.write( "test_ZeroInputWithNonzeroCoeffsIsZeroOutput.coeffY" );

    run_transformation();

    out.write( "test_ZeroInputWithNonzeroCoeffsIsZeroOutput.output" );

    compare_results( in.data, out.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, RandomInputWithNonzeroCoeffs) {
    x = 256;
    y = 128;
    splineX = 6;
    splineY = 5;
    splineN = 4;
    allocate_arrays();
    randomly_initialize( in );
    in.write( "test_RandomInputWithNonzeroCoeffs.input" );

    randomly_initialize( coeffsX );
    randomly_initialize( coeffsY );
    coeffsX.write( "test_RandomInputWithNonzeroCoeffs.coeffX" );
    coeffsY.write( "test_RandomInputWithNonzeroCoeffs.coeffY" );

    run_transformation();

    out.write( "test_RandomInputWithNonzeroCoeffs.output" );

    // compare_results( in.data, out.data, size );
}

// TEST_F(GeoTransformerApplyBSplineTransformTest, PaddedSizeRandomInput) {
//     x = 2048 + 128;
//     y = 1024;
//     allocate_arrays();

//     randomly_initialize( in.get(), size );

//     run_test();
// }

// TEST_F(GeoTransformerApplyBSplineTransformTest, PaddedSquareSizeRandomInput) {
//     x = 1024;
//     y = x;
//     allocate_arrays();

//     randomly_initialize( in.get(), size );

//     run_test();
// }

// TEST_F(GeoTransformerApplyBSplineTransformTest, SquareSizeRandomInput) {
//     x = 259;
//     y = x;
//     allocate_arrays();

//     randomly_initialize( in.get(), size );

//     run_test();
// }

// TEST_F(GeoTransformerApplyBSplineTransformTest, RandomSizeRandomInput) {
//     std::tie( x, y ) = random_size();

//     allocate_arrays();

//     randomly_initialize( in.get(), size );

//     run_test();

// }

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}