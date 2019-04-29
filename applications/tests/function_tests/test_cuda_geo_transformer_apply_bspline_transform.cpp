#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"

template< typename T >
class GeoTransformerTest : public GeoTransformer< T > {

};

class GeoTransformerApplyBSplineTransformTest : public ::testing::Test {

public:

    /*
    Version that prints out number of errors
    */
    // void compare_results( float* true_values, float* approx_values, size_t size ) {
    //     testing::internal::CaptureStdout();

    //     int errors = 0;
    //     for ( int i = 0; i < size; ++i ) {
    //         EXPECT_NEAR( true_values[i], approx_values[i], 0.0001f ) << ( errors++ < 10 ? "at index: " + std::to_string( i ) : "" );
    //         if ( errors == 10 ) {
    //             std::cout << "Omitting next errors\n";
    //         }
    //     }

    //     std::string output = testing::internal::GetCapturedStdout();
    //     std::cout << output.substr(0, 100) << std::endl;

    //     if ( errors > 0 ) {
    //         std::cout << "Total errors: " << errors << std::endl;
    //     }
    // }

    /*

    */
    void compare_results( float* true_values, float* approx_values, size_t size ) {
        for ( int i = 0; i < size; ++i ) {
            EXPECT_NEAR( true_values[i], approx_values[i], 0.0001f ) << "at index:" << i << ", x=" << i % x << ", y=" << i / x;
        }
    }


    // void compare_results( float* true_values, float* approx_values, size_t size ) {
    //     for ( int i = 0; i < size; ++i ) {
    //         ASSERT_NEAR( true_values[i], approx_values[i], 0.0001f ) << "at index:" << i;
    //     }
    // }

    void allocate_arrays() {
        size = x * y;

        in.resize( y, x );
        out.resize( y, x );
        out_ref.resize( y, x );

        coeffsX.resize( splineX * splineY * splineN );
        coeffsY.resize( splineX * splineY * splineN );
    }

    void run_transformation() {
        GeoTransformerTest< float > gt;
        gt.initLazyForBSpline( x, y, 1, splineX, splineY, splineN );
        gt.applyBSplineTransformNew( 3, out, in, { coeffsX, coeffsY }, imageIdx, outside );
    }

    void compute_reference_result() {
        GeoTransformerTest< float > gt;
        gt.initLazyForBSpline( x, y, 1, splineX, splineY, splineN );
        gt.applyBSplineTransform( 3, out_ref, in, { coeffsX, coeffsY }, imageIdx, outside );
    }

    std::pair< size_t, size_t > random_size( int seed ) {
        gen.seed( seed );
        std::uniform_int_distribution<> dis( 128, 8192 );

        return { dis( gen ), dis( gen ) };
    }

    void randomly_initialize( MultidimArray< float >& array, int seed ) {
        gen.seed( seed );
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

    void randomly_initialize(Matrix1D< float >& array, int seed) {
        gen.seed( seed );
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        FOR_ALL_ELEMENTS_IN_MATRIX1D( array ) {
            MATRIX1D_ARRAY( array )[i] = dis(gen);
        }

    }

    void save_array( const MultidimArray< float >& array, const std::string& filename ) {
    	std::ofstream out( filename );

    	out << std::setprecision( 7 );
    	out << array;
    }

    void save_coeffs( const Matrix1D< float >& array, const std::string& filename ) {
    	std::ofstream out( filename );

    	out << std::setprecision( 7 );
    	out << array;
    }

    MultidimArray< float > load_array( const std::string& filename ) {
    	std::ifstream input( filename );

    	MultidimArray< float > array;
    	array.resize( y, x );

    	size_t index = 0;

    	while ( true ) {
    		float value;
    		input >> value;
    		if ( input.eof() ) break;
    		DIRECT_MULTIDIM_ELEM( array, index++ ) = value;
    	}
    	assert( index == y * x );

    	return array;
    }

    Matrix1D< float > load_coeffs( const std::string& filename ) {
    	std::ifstream input( filename );

    	Matrix1D< float > coeffs;
    	coeffs.resize( splineX * splineY * splineN );

    	size_t index = 0;

    	while ( true ) {
    		float value;
    		input >> value;
    		if ( input.eof() ) break;
    			MATRIX1D_ARRAY( coeffs )[index++] = value;
    	}
    	assert( index == splineX * splineY * splineN );

    	return coeffs;
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
    MultidimArray< float > out_ref;

    std::mt19937 gen;
};

TEST_F(GeoTransformerApplyBSplineTransformTest, ZeroStaysZero) {
    x = 256;
    y = 256;
    allocate_arrays();

    run_transformation();

    compare_results( in.data, out.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, NoChangeIfCoeffsAreZeroWithZeroCoeffs) {
    x = 259;
    y = 311;
    // x = 256;
    // y = 256;
    allocate_arrays();

    randomly_initialize( in, 13 );

    run_transformation();

    compare_results( in.data, out.data, size );

}

TEST_F(GeoTransformerApplyBSplineTransformTest, ZeroInputWithNonzeroCoeffsIsZeroOutput) {
    x = 147;
    y = 147;
    splineX = 6;
    splineY = 5;
    splineN = 4;
    allocate_arrays();

    randomly_initialize( coeffsX, 19 );
    randomly_initialize( coeffsY, 17 );

    run_transformation();

    compare_results( in.data, out.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, RandomInputWithNonzeroCoeffs) {
    x = 256;
    y = 128;
    splineX = 6;
    splineY = 5;
    splineN = 4;
    allocate_arrays();
    randomly_initialize( in, 23 );
    randomly_initialize( coeffsX, 31 );
    randomly_initialize( coeffsY, 35 );

    run_transformation();
    compute_reference_result();

    compare_results( out.data, out_ref.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, RandomInputWithNonzeroDifferentDimCoeffs) {
    x = 256;
    y = 128;
    splineX = 3;
    splineY = 8;
    splineN = 3;
    allocate_arrays();
    randomly_initialize( in, 81 );
    randomly_initialize( coeffsX, 73 );
    randomly_initialize( coeffsY, 7 );

    run_transformation();
    compute_reference_result();

    compare_results( out.data, out_ref.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, EvenButNotPaddedInput) {
    x = 322;
    y = 344;
    splineX = 6;
    splineY = 5;
    splineN = 4;
    allocate_arrays();
    randomly_initialize( in, 24 );
    randomly_initialize( coeffsX, 47 );
    randomly_initialize( coeffsY, 19 );

    run_transformation();
    compute_reference_result();

    compare_results( out.data, out_ref.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, OddEvenSizedInput) {
    x = 311;
    y = 134;
    splineX = 9;
    splineY = 3;
    splineN = 6;
    allocate_arrays();
    randomly_initialize( in, 63 );
    randomly_initialize( coeffsX, 21 );
    randomly_initialize( coeffsY, 3 );

    run_transformation();
    compute_reference_result();

    compare_results( out.data, out_ref.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, EvenOddSizedInput) {
    x = 260;
    y = 201;
    splineX = 7;
    splineY = 7;
    splineN = 6;
    allocate_arrays();
    randomly_initialize( in, 10 );
    randomly_initialize( coeffsX, 31 );
    randomly_initialize( coeffsY, 97 );

    run_transformation();
    compute_reference_result();

    compare_results( out.data, out_ref.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, BiggerSize4K) {
    x = 3840;
    y = 2160;
    splineX = 4;
    splineY = 8;
    splineN = 6;
    allocate_arrays();
    randomly_initialize( in, 11 );
    randomly_initialize( coeffsX, 33 );
    randomly_initialize( coeffsY, 98 );

    run_transformation();
    compute_reference_result();

    compare_results( out.data, out_ref.data, size );
}

// TEST_F(GeoTransformerApplyBSplineTransformTest, BiggerSizeInOneDimension) {
//     x = 3840;
//     y = 256;
//     splineX = 4;
//     splineY = 8;
//     splineN = 6;
//     allocate_arrays();
//     randomly_initialize( in, 81 );
//     randomly_initialize( coeffsX, 7 );
//     randomly_initialize( coeffsY, 43 );

//     run_transformation();
//     compute_reference_result();

//     compare_results( out.data, out_ref.data, size );
// }

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}