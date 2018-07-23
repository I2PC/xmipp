#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "advisor.h"
#include "benchmarkResult.h"
#include "inputParser.h"

int printHelp() {
  std::cout << "DESCRIPTION:" << std::endl;
  std::cout << "\tcuFFTAdvisor is a program and a library to help you to "
               "select optimal lenght of the signal for Cuda FFT (cuFFT) "
               "library"
            << std::endl;
  std::cout << "Usage:" << std::endl;
  std::cout << "\t-benchmark -x X [OPTIONS] - to benchmark given FFT of size X"
            << std::endl;
  std::cout << "\t-recommend COUNT -x X [OPTIONS] [LIMITS] - to recommend "
               "COUNT best settings for given FFT of size X (no performance "
               "check, just estimation)"
            << std::endl;
  std::cout << "\t-find COUNT -x X [OPTIONS] [LIMITS] - to find COUNT best "
               "settings for given FFT of size X (with performance check)"
            << std::endl;
  std::cout << "OPTIONS:" << std::endl;
  std::cout << "\t-y Y : Y dimension, default = 1" << std::endl;
  std::cout << "\t-z Z : Z dimension, default = 1" << std::endl;
  std::cout << "\t-n N : batch size, default = 1" << std::endl;
  std::cout << "\t-device ID : ID of the device to use, default = 0"
            << std::endl;
  std::cout << "\t--realOnly|complexOnly : consider only real|complex input. "
               "Both by default"
            << std::endl;
  std::cout << "\t--floatOnly|doubleOnly : consider only float|double input. "
               "Both by default"
            << std::endl;
  std::cout << "\t--forwardOnly|inverseOnly : consider only forward|inverse "
               "transformation. Both by default"
            << std::endl;
  std::cout << "\t--inPlaceOnly|outOfPlaceOnly : consider only "
               "in-place|out-of-place data layout. Both by default"
            << std::endl;
  std::cout << "\t--batchOnly|noBatch : consider only batched|single input. "
               "Both by default"
            << std::endl;
  std::cout << "\t--allowTransposition : consider also transposed input "
               "(swapping dimensions). Prohibited by default. Valid for "
               "'-find' only."
            << std::endl;
  std::cout << "\t--SquareOnly : consider only square shapes "
               "(X dimension size will be used as a starting point). "
               "Incompatible with --allowTransposition."
            << std::endl;
  std::cout << "LIMITS:" << std::endl;
  std::cout
      << "\t--maxSignalInc VAL : maximal percentual increase of the signal"
      << std::endl;
  std::cout << "\t--maxMem MB : max memory (in MB) that transformation can "
               "use, default = device limit"
            << std::endl;
  return -1;
}

int parseBenchmark(int argc, char **argv) {
  try {
    cuFFTAdvisor::InputParser parser(
        argc - 2, argv + 2);  // skip program and 'benchmark' arg
    if (parser.reportUnparsed(stderr)) {
      printHelp();
      return EXIT_FAILURE;
    }

    std::vector<cuFFTAdvisor::BenchmarkResult const *> *results =
        cuFFTAdvisor::Advisor::benchmark(parser.device, parser.x, parser.y,
                                         parser.z, parser.n, parser.isBatched,
                                         parser.isFloat, parser.isForward,
                                         parser.isInPlace, parser.isReal);

    cuFFTAdvisor::BenchmarkResult::printHeader(stdout);
    std::cout << std::endl;
    for (auto& r : *results) {
	  r->print(stdout);
	  printf("\n");
	  delete r;
	}
    delete results;
  } catch (std::logic_error &e) {
    std::cout << e.what();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int parseRecommend(int argc, char **argv, int howMany) {
  try {
    cuFFTAdvisor::InputParser parser(
        argc - 3, argv + 3);  // skip program and 'recommend X' args
    if (parser.reportUnparsed(stderr)) {
      printHelp();
      return EXIT_FAILURE;
    }

    std::vector<const cuFFTAdvisor::Transform *> *results =
        cuFFTAdvisor::Advisor::recommend(
            howMany, parser.device, parser.x, parser.y, parser.z, parser.n,
            parser.isBatched, parser.isFloat, parser.isForward,
            parser.isInPlace, parser.isReal, parser.maxSignalInc,
            parser.maxMemMB, parser.allowTransposition, parser.squareOnly);

	for (auto& r : *results) {
		r->print(stdout);
		printf("\n");
		delete r;
	}
    delete results;

  } catch (std::logic_error &e) {
    std::cout << e.what();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int parseFind(int argc, char **argv, int howMany) {
  try {
    cuFFTAdvisor::InputParser parser(
        argc - 3, argv + 3);  // skip program and 'recommend X' args
    if (parser.reportUnparsed(stderr)) {
      printHelp();
      return EXIT_FAILURE;
    }
    std::vector<cuFFTAdvisor::BenchmarkResult const *> *results =
        cuFFTAdvisor::Advisor::find(howMany, parser.device, parser.x, parser.y,
                                    parser.z, parser.n, parser.isBatched,
                                    parser.isFloat, parser.isForward,
                                    parser.isInPlace, parser.isReal,
                                    parser.maxSignalInc, parser.maxMemMB,
                                    parser.allowTransposition, parser.squareOnly);

    cuFFTAdvisor::BenchmarkResult::printHeader(stdout);
    std::cout << std::endl;
    for (auto& r : *results) {
      r->print(stdout);
	  printf("\n");
      delete r;
    }
    delete results;
  } catch (std::logic_error &e) {
    std::cout << e.what();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  for (int i = 0; i < (argc - 1); i++) {
    if (0 == strcmp(argv[i], "-benchmark")) {
      return parseBenchmark(argc, argv);
    }
    if (0 == strcmp(argv[i], "-recommend")) {
      return parseRecommend(argc, argv, atoi(argv[i + 1]));
    }
    if (0 == strcmp(argv[i], "-find")) {
      return parseFind(argc, argv, atoi(argv[i + 1]));
    }
  }
  printHelp();
  return EXIT_FAILURE;
}
