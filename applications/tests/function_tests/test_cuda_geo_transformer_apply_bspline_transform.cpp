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
            ASSERT_NEAR( true_values[i], approx_values[i], 0.0001f );
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
};

TEST_F(GeoTransformerApplyBSplineTransformTest, NoChangeIfCoeffsAreZero) {
    x = 259;
    y = 311;
    allocate_arrays();

    randomly_initialize( in );

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

    randomly_initialize( coeffsX );
    randomly_initialize( coeffsY );

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
    in = load_array( "test_RandomInputWithNonzeroCoeffs.input" );
    coeffsX = load_coeffs( "test_RandomInputWithNonzeroCoeffs.coeffX" );
    coeffsY = load_coeffs( "test_RandomInputWithNonzeroCoeffs.coeffY" );

    run_transformation();

    MultidimArray< float > true_result = load_array( "test_RandomInputWithNonzeroCoeffs.output" );

    compare_results( out.data, true_result.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, OddEvenSizedInput) {
    x = 311;
    y = 134;
    splineX = 9;
    splineY = 2;
    splineN = 6;
    allocate_arrays();
    in = load_array( "test_OddEvenSizedInput.input" );
    coeffsX = load_coeffs( "test_OddEvenSizedInput.coeffX" );
    coeffsY = load_coeffs( "test_OddEvenSizedInput.coeffY" );

    run_transformation();

    MultidimArray< float > true_result = load_array( "test_OddEvenSizedInput.output" );

    compare_results( out.data, true_result.data, size );
}

TEST_F(GeoTransformerApplyBSplineTransformTest, EvenOddSizedInput) {
    x = 260;
    y = 201;
    splineX = 7;
    splineY = 7;
    splineN = 6;
    allocate_arrays();
    in = load_array( "test_EvenOddSizedInput.input" );
    coeffsX = load_coeffs( "test_EvenOddSizedInput.coeffX" );
    coeffsY = load_coeffs( "test_EvenOddSizedInput.coeffY" );

    run_transformation();

    MultidimArray< float > true_result = load_array( "test_EvenOddSizedInput.output" );

    compare_results( out.data, true_result.data, size );
}

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}