/***************************************************************************
 *
 * Authors:     David Strelak (davidstrelak@gmail.com)
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

#include "core/xmipp_program.h"
#include "core/xmipp_filename.h"
#include "core/xmipp_image.h"
#include "reconstruction/psd_estimator.h"

class PSDEstimatorProgram final : public XmippProgram
{
private:
    void defineParams() override
    {
        addParamsLine("-i <input_file>             : Micrograph to be analyzed");
        addParamsLine("-o <output_file>            : PSD to be stored");
        addParamsLine("[--overlap <o=0.4>]         : overlap of the patches");
        addParamsLine("[--patches <x=384> <y=384>] :  size of the patches");
        addParamsLine("[--threads <t=4>]           : for FFT");
        addParamsLine("[--skipNormalization]       : if not present, FFT will be centered, and log_10 applied");
    }

    void readParams() override
    {
        mIn = getParam("-i");
        mOut = getParam("-o");
        mOverlap = getFloatParam("--overlap");
        auto x = getIntParam("--patches", 0);
        auto y = getIntParam("--patches", 1);
        mDims = Dimensions(x, y);
        mThreads = getIntParam("--threads");
        mNormalize = !checkParam("--skipNormalization");
    }

    void run()
    {
        auto micrograph = Image<float>();
        auto PSD = Image<float>();
        micrograph.read(mIn);
        PSDEstimator<float>::estimatePSD(micrograph(), mOverlap, mDims, PSD(), mThreads, mNormalize);
        PSD.write(mOut);
    }

    unsigned mThreads;
    Dimensions mDims = Dimensions(0);
    float mOverlap;
    FileName mIn;
    FileName mOut;
    bool mNormalize;
};

RUN_XMIPP_PROGRAM(PSDEstimatorProgram)