/***************************************************************************
 *
 * Authors:    Martin Horacek (horacek1martin@gmail.com)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
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

    /** Filename of the two halves and the output root */
    FileName fnV1;
    FileName fnV2;
    FileName fnRoot;

    Image<T> V1;
    Image<T> V2;
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
