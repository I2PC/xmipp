#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"

template< typename T >
class GeoTransformerTest : public GeoTransformer< T > {

};

template< typename T >
class GeoTransformerApplyBSplineTransformTest : public ::testing::Test {

public:

    /*
    Version that prints out number of errors
    */
    void compare_results( T* true_values, T* approx_values, size_t size ) {
        testing::internal::CaptureStdout();

        int errors = 0;
        for ( int i = 0; i < size; ++i ) {
            EXPECT_NEAR( true_values[i], approx_values[i], static_cast< T >( 0.0001 ) ) << ( errors++ < 10 ? "at index: " + std::to_string( i ) : "" );
            if ( errors == 10 ) {
                std::cout << "Omitting next errors\n";
            }
        }

        std::string output = testing::internal::GetCapturedStdout();
        std::cout << output.substr(0, 100) << std::endl;

        if ( errors > 0 ) {
            std::cout << "Total errors: " << errors << std::endl;
        }
    }

    /*

    */
    // void compare_results( T* true_values, T* approx_values, size_t size ) {
        // for ( int i = 0; i < size; ++i ) {
            // EXPECT_NEAR( true_values[i], approx_values[i], static_cast< T >( 0.00001 ) ) << "at index:" << i << ", x=" << i % x << ", y=" << i / x;
        // }
    // }


    // void compare_results( T* true_values, T* approx_values, size_t size ) {
    //     for ( int i = 0; i < size; ++i ) {
    //         ASSERT_NEAR( true_values[i], approx_values[i], static_cast< T >( 0.0001 ) ) << "at index:" << i;
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
        GeoTransformerTest< T > gt;
        gt.initLazyForBSpline( x, y, 1, splineX, splineY, splineN );
        gt.applyBSplineTransformNew( 3, out, in, { coeffsX, coeffsY }, imageIdx, outside );
    }

    void compute_reference_result() {
        GeoTransformerTest< T > gt;
        gt.initLazyForBSpline( x, y, 1, splineX, splineY, splineN );
        gt.applyBSplineTransform( 3, out_ref, in, { coeffsX, coeffsY }, imageIdx, outside );
    }

    std::pair< size_t, size_t > random_size( int seed ) {
        gen.seed( seed );
        std::uniform_int_distribution<> dis( 128, 8192 );

        return { dis( gen ), dis( gen ) };
    }

    void randomly_initialize( MultidimArray< T >& array, int seed ) {
        gen.seed( seed );
        std::uniform_real_distribution<> dis( -1.0, 1.0 );

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY( array ) {
            DIRECT_MULTIDIM_ELEM( array, n ) = dis( gen );
        }
    }

    void set_to_value( MultidimArray< T >& array, T value ) {
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY( array ) {
            DIRECT_MULTIDIM_ELEM( array, n ) = value;
        }
    }

    void randomly_initialize(Matrix1D< T >& array, int seed) {
        gen.seed( seed );
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        FOR_ALL_ELEMENTS_IN_MATRIX1D( array ) {
            MATRIX1D_ARRAY( array )[i] = dis(gen);
        }

    }

    void save_array( const MultidimArray< T >& array, const std::string& filename ) {
    	std::ofstream out( filename );

    	// out << std::setprecision( 7 );
    	out << array;
    }

    void save_coeffs( const Matrix1D< T >& array, const std::string& filename ) {
    	std::ofstream out( filename );

    	// out << std::setprecision( 7 );
    	out << array;
    }

    MultidimArray< T > load_array( const std::string& filename ) {
    	std::ifstream input( filename );

    	MultidimArray< T > array;
    	array.resize( y, x );

    	size_t index = 0;

    	while ( true ) {
    		T value;
    		input >> value;
    		if ( input.eof() ) break;
    		DIRECT_MULTIDIM_ELEM( array, index++ ) = value;
    	}
    	assert( index == y * x );

    	return array;
    }

    Matrix1D< T > load_coeffs( const std::string& filename ) {
    	std::ifstream input( filename );

    	Matrix1D< T > coeffs;
    	coeffs.resize( splineX * splineY * splineN );

    	size_t index = 0;

    	while ( true ) {
    		T value;
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

    T outside = 0;

    Matrix1D< T > coeffsX;
    Matrix1D< T > coeffsY;

    MultidimArray< T > in;
    MultidimArray< T > out;
    MultidimArray< T > out_ref;

    std::mt19937 gen;
};

TYPED_TEST_CASE_P(GeoTransformerApplyBSplineTransformTest);

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, ZeroStaysZero) {
    this->x = 256;
    this->y = 256;
    this->allocate_arrays();

    this->run_transformation();

    this->compare_results( this->in.data, this->out.data, this->size );
}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, NoChangeIfCoeffsAreZeroWithZeroCoeffs) {
    this->x = 259;
    this->y = 311;
    this->allocate_arrays();

    this->randomly_initialize( this->in, 13 );

    this->run_transformation();

    this->compare_results( this->in.data, this->out.data, this->size );

}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, ZeroInputWithNonzeroCoeffsIsZeroOutput) {
    this->x = 147;
    this->y = 147;
    this->splineX = 6;
    this->splineY = 5;
    this->splineN = 4;
    this->allocate_arrays();

    this->randomly_initialize( this->coeffsX, 19 );
    this->randomly_initialize( this->coeffsY, 17 );

    this->run_transformation();

    this->compare_results( this->in.data, this->out.data, this->size );
}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, RandomInputWithNonzeroCoeffs) {
    this->x = 256;
    this->y = 128;
    this->splineX = 6;
    this->splineY = 5;
    this->splineN = 4;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 23 );
    this->randomly_initialize( this->coeffsX, 31 );
    this->randomly_initialize( this->coeffsY, 35 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data, this->size );
}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, RandomInputWithNonzeroDifferentDimCoeffs) {
    this->x = 256;
    this->y = 128;
    this->splineX = 3;
    this->splineY = 8;
    this->splineN = 3;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 81 );
    this->randomly_initialize( this->coeffsX, 73 );
    this->randomly_initialize( this->coeffsY, 7 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data, this->size );
}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, EvenButNotPaddedInput) {
    this->x = 322;
    this->y = 344;
    this->splineX = 6;
    this->splineY = 5;
    this->splineN = 4;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 24 );
    this->randomly_initialize( this->coeffsX, 47 );
    this->randomly_initialize( this->coeffsY, 19 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data, this->size );
}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, OddEvenSizedInput) {
    this->x = 311;
    this->y = 134;
    this->splineX = 9;
    this->splineY = 3;
    this->splineN = 6;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 63 );
    this->randomly_initialize( this->coeffsX, 21 );
    this->randomly_initialize( this->coeffsY, 3 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data, this->size );
}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, EvenOddSizedInput) {
    this->x = 260;
    this->y = 201;
    this->splineX = 7;
    this->splineY = 7;
    this->splineN = 6;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 10 );
    this->randomly_initialize( this->coeffsX, 31 );
    this->randomly_initialize( this->coeffsY, 97 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data, this->size );
}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, BiggerSize4K) {
    this->x = 3840;
    this->y = 2160;
    this->splineX = 4;
    this->splineY = 8;
    this->splineN = 6;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 11 );
    this->randomly_initialize( this->coeffsX, 33 );
    this->randomly_initialize( this->coeffsY, 98 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data, this->size );
}

TYPED_TEST_P(GeoTransformerApplyBSplineTransformTest, BiggerSizeInOneDimension) {
    this->x = 3840;
    this->y = 256;
    this->splineX = 4;
    this->splineY = 8;
    this->splineN = 6;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 81 );
    this->randomly_initialize( this->coeffsX, 7 );
    this->randomly_initialize( this->coeffsY, 43 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data, this->size );
}

REGISTER_TYPED_TEST_CASE_P(GeoTransformerApplyBSplineTransformTest,
    ZeroStaysZero,
    NoChangeIfCoeffsAreZeroWithZeroCoeffs,
    ZeroInputWithNonzeroCoeffsIsZeroOutput,
    RandomInputWithNonzeroCoeffs,
    RandomInputWithNonzeroDifferentDimCoeffs,
    EvenButNotPaddedInput,
    OddEvenSizedInput,
    EvenOddSizedInput,
    BiggerSize4K,
    BiggerSizeInOneDimension
);

using ScalarTypes = ::testing::Types< float, double >;
INSTANTIATE_TYPED_TEST_CASE_P(ScalarTypesInstantiation, GeoTransformerApplyBSplineTransformTest, ScalarTypes);

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}