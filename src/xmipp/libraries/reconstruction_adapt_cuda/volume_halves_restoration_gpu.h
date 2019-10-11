#ifndef _PROG_VOLUME_HALVES_RESTORATION_GPU
#define _PROG_VOLUME_HALVES_RESTORATION_GPU

#include <iostream>

#include <core/xmipp_program.h>
#include <data/mask.h>

#include "reconstruction_cuda/cuda_volume_halves_restorator.h"

/*
 * This is GPU implementation of reconstruction/volume_halves_restoration.h
 * It checks validity of input and delegates the computation to VolumeHalvesRestorator
*/
template< typename T >
class ProgVolumeHalvesRestorationGpu : public XmippProgram {
    static_assert(std::is_floating_point<T>::value, "Only float and double are allowed as template parameters");

    using Complex = std::complex<T>;
    static constexpr size_t type_size = sizeof(T);
    static constexpr size_t complex_size = sizeof(Complex);

    /** Filename of the two halves and the output root */
    FileName fnV1, fnV2, fnRoot;

    Image<T> V1, V2;
    Mask mask;
    int* maskData = nullptr;

    typename VolumeHalvesRestorator<T>::Builder builder;

public:
    /*
    * Extract parameters from command line and check their values
    */
    void readParams() override;

    /*
     * Defines command line arguments for the program
     */
    void defineParams() override;

    /*
     * Runs the volume halves restoration algorithm
     */
    void run() override;

private:

    /*
    * Prints information about program settings to standard output
    */
    void show(const VolumeHalvesRestorator<T>& restorator);

    /*
     * helper methods for readParams
    */
    void readFilenames();
    void readDenoisingParams();
    void readDeconvolutionParams();
    void readFilterBankParams();
    void readDifferenceParams();
    void readMaskParams();

    /*
     * Loads V1, V2 and mask
    */
    void readData();
    void checkInputDimensions();

    void saveResults(const VolumeHalvesRestorator<T>& restorator);
    void saveImage(Image<T>& image, std::string&& filename);
};

#endif
