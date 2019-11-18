#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>

#include "reconstruction_cuda/cuda_volume_halves_restorator.h"
#include "reconstruction/volume_halves_restoration.h"

template< typename T >
class CudaVolumeHalvesRestorationTest : public ::testing::Test {

	using Builder = typename VolumeHalvesRestorator<T>::Builder;

public:
	unsigned denoisingIters = 0;
	/*
	 * Parameters for deconvolution
	*/
	unsigned deconvolutionIters = 0;
	T sigma = 0.2;
	T lambda = 0.001;

	/*
	 * Parameters for difference
	*/
	unsigned differenceIters = 0;
	T Kdiff = 1.5;

	/*
	 * Parameters for filter bank
	*/
	T bankStep = 0;
	T bankOverlap = 0.5;
	unsigned weightFun = 1;
	T weightPower = 3;

	const std::string dir_path = std::string{ "/tmp/" };
	const std::string input_file1 = dir_path + std::string{ "test_input1.vol" };
	const std::string input_file2 = dir_path + std::string{ "test_input2.vol" };
	const std::string output_root = dir_path + std::string{ "test" };

	const double double_epsilon = 1e-7;

	void compare_results(const double* true_values, const double* approx_values, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            ASSERT_NEAR(true_values[i], approx_values[i], double_epsilon) << "at index:" << i;
        }
    }

	VolumeHalvesRestorator<T> createCudaRestorator() const {
		return Builder().setFilterBank(bankStep, bankOverlap, weightFun, weightPower)
					  .setDenoising(denoisingIters)
					  .setDeconvolution(deconvolutionIters, sigma, lambda)
					  .setDifference(differenceIters, Kdiff)
					  // .setVerbosity(1)
					  .build();
	}

	ProgVolumeHalvesRestoration createReferenceRestorator() const {
		ProgVolumeHalvesRestoration prog;
		prog.NiterReal = denoisingIters;
		prog.NiterFourier = deconvolutionIters;
		prog.bankStep = bankStep;
		prog.bankOverlap = bankOverlap;
		prog.sigma0 = sigma;
		prog.NiterDiff = differenceIters;
		prog.Kdiff = Kdiff;
		prog.lambda = lambda;
		prog.weightFun = weightFun;
		prog.weightPower = weightPower;
		prog.fnV1 = input_file1;
		prog.fnV2 = input_file2;
		prog.fnRoot = output_root;
		prog.verbose = 0;

		return prog;
	}

	void randomly_initialize(MultidimArray<T>& array, int seed) {
		gen.seed(seed);
		std::uniform_real_distribution<> dis(-0.4, 0.4);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(array) {
            DIRECT_MULTIDIM_ELEM(array, n) = dis(gen);
        }
	}

	void compare_restored1(const VolumeHalvesRestorator<T>& restorator) {
		Image<T> ref;
		ref.read(this->output_root + std::string{ "_restored1.vol" });
		auto gpu_res = restorator.getReconstructedVolume1();
		compare_results(ref().data, gpu_res.data, MULTIDIM_SIZE(gpu_res));
	}

	void compare_restored2(const VolumeHalvesRestorator<T>& restorator) {
		Image<T> ref;
		ref.read(this->output_root + std::string{ "_restored2.vol" });
		auto gpu_res = restorator.getReconstructedVolume2();
		compare_results(ref().data, gpu_res.data, MULTIDIM_SIZE(gpu_res));
	}

	void compare_filter_bank(const VolumeHalvesRestorator<T>& restorator) {
		Image<T> ref;
		ref.read(this->output_root + std::string{ "_filterBank.vol" });
		auto gpu_res = restorator.getFilterBankVolume();
		compare_results(ref().data, gpu_res.data, MULTIDIM_SIZE(gpu_res));
	}

	void compare_deconvolved(const VolumeHalvesRestorator<T>& restorator) {
		Image<T> ref;
		ref.read(this->output_root + std::string{ "_deconvolved.vol" });
		auto gpu_res = restorator.getDeconvolvedS();
		compare_results(ref().data, gpu_res.data, MULTIDIM_SIZE(gpu_res));
	}

	void compare_convolved(const VolumeHalvesRestorator<T>& restorator) {
		Image<T> ref;
		ref.read(this->output_root + std::string{ "_convolved.vol" });
		auto gpu_res = restorator.getConvolvedS();
		compare_results(ref().data, gpu_res.data, MULTIDIM_SIZE(gpu_res));
	}

	void compare_diff(const VolumeHalvesRestorator<T>& restorator) {
		Image<T> ref;
		ref.read(this->output_root + std::string{ "_avgDiff.vol" });
		auto gpu_res = restorator.getAverageDifference();
		compare_results(ref().data, gpu_res.data, MULTIDIM_SIZE(gpu_res));
	}

private:

	std::mt19937 gen;
};

TYPED_TEST_CASE_P(CudaVolumeHalvesRestorationTest);

TYPED_TEST_P(CudaVolumeHalvesRestorationTest, FilterBankTest) {
	this->bankStep = 0.07;
	this->bankOverlap = 0.6;
	this->weightFun = 2;

	TypeParam type = 0;
	auto cuda_restorator = this->createCudaRestorator();
	auto ref_program = this->createReferenceRestorator();

	int dim = 99;
	int volume_size = dim * dim * dim;

	MultidimArray<double> input1{ dim, dim, dim };
	MultidimArray<double> input2{ dim, dim, dim };
	this->randomly_initialize(input1, 5);
	this->randomly_initialize(input2, 7);

	Image<TypeParam> image1{ input1 };
	Image<TypeParam> image2{ input2 };

	image1.write(this->input_file1);
	image2.write(this->input_file2);

	ref_program.run();
	cuda_restorator.apply(input1, input2, nullptr);

	this->compare_restored1(cuda_restorator);
	this->compare_restored2(cuda_restorator);
	this->compare_filter_bank(cuda_restorator);
}

TYPED_TEST_P(CudaVolumeHalvesRestorationTest, DenoisingTest) {
	this->denoisingIters = 5;
	this->bankStep = 0.05;

	TypeParam type = 0;
	auto cuda_restorator = this->createCudaRestorator();
	auto ref_program = this->createReferenceRestorator();

	int dim = 99;
	int volume_size = dim * dim * dim;

	MultidimArray<double> input1{ dim, dim, dim };
	MultidimArray<double> input2{ dim, dim, dim };
	this->randomly_initialize(input1, 9);
	this->randomly_initialize(input2, 11);

	Image<TypeParam> image1{ input1 };
	Image<TypeParam> image2{ input2 };

	image1.write(this->input_file1);
	image2.write(this->input_file2);

	ref_program.run();
	cuda_restorator.apply(input1, input2, nullptr);

	this->compare_restored1(cuda_restorator);
	this->compare_restored2(cuda_restorator);
	this->compare_filter_bank(cuda_restorator);
}

TYPED_TEST_P(CudaVolumeHalvesRestorationTest, DeconvolutionTest) {
	this->deconvolutionIters = 2;
	this->sigma = 0.25;
	this->lambda = 0.002;
	this->bankStep = 0.05;

	TypeParam type = 0;
	auto cuda_restorator = this->createCudaRestorator();
	auto ref_program = this->createReferenceRestorator();

	int xdim = 99;
	int ydim = 80;
	int zdim = 75;
	int volume_size = zdim * ydim * xdim;

	MultidimArray<double> input1{ zdim, ydim, xdim };
	MultidimArray<double> input2{ zdim, ydim, xdim };
	this->randomly_initialize(input1, 13);
	this->randomly_initialize(input2, 15);

	Image<TypeParam> image1{ input1 };
	Image<TypeParam> image2{ input2 };

	image1.write(this->input_file1);
	image2.write(this->input_file2);

	ref_program.run();
	cuda_restorator.apply(input1, input2, nullptr);

	this->compare_restored1(cuda_restorator);
	this->compare_restored2(cuda_restorator);
	this->compare_deconvolved(cuda_restorator);
	this->compare_convolved(cuda_restorator);
}

TYPED_TEST_P(CudaVolumeHalvesRestorationTest, DifferenceTest) {
	this->differenceIters = 2;
	this->Kdiff = 1.7;
	this->bankStep = 0.05;

	auto cuda_restorator = this->createCudaRestorator();
	auto ref_program = this->createReferenceRestorator();

	const int dim = 199;
	const int volume_size = dim * dim * dim;

	MultidimArray<double> input1{ dim, dim, dim };
	MultidimArray<double> input2{ dim, dim, dim };
	this->randomly_initialize(input1, 17);
	this->randomly_initialize(input2, 19);

	Image<TypeParam> image1{ input1 };
	Image<TypeParam> image2{ input2 };

	image1.write(this->input_file1);
	image2.write(this->input_file2);

	ref_program.run();
	cuda_restorator.apply(input1, input2, nullptr);

	this->compare_restored1(cuda_restorator);
	this->compare_restored2(cuda_restorator);
	this->compare_diff(cuda_restorator);
}

REGISTER_TYPED_TEST_CASE_P(CudaVolumeHalvesRestorationTest,
	FilterBankTest,
	DenoisingTest,
	DeconvolutionTest,
	DifferenceTest
	);

using ScalarTypes = ::testing::Types< double >;
INSTANTIATE_TYPED_TEST_CASE_P(ScalarTypesInstantiation, CudaVolumeHalvesRestorationTest, ScalarTypes);



GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}