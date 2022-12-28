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

#include <core/xmipp_program.h>
#include <reconstruction/phantom_movie.h>

class PhantomMovieProgram final : public XmippProgram
{
private:
    typedef PhantomMovie<double>::Content Content;
    typedef PhantomMovie<double>::DisplacementParams DisplacementParams;
    typedef PhantomMovie<double>::Params Params;
    typedef PhantomMovie<double>::Options Options;
    void defineParams() override
    {
        addParamsLine(Content::size_param + std::string(" <x=4096> <y=4096> <n=40>                :"
                                                        " Movie size"));
        addParamsLine(Content::step_param + std::string(" <x=50> <y=50>                           :"
                                                        " Distance between the lines/rows of the grid (before the transform is applied)"));
        addParamsLine("[--thickness <t=5>]                                   :"
                      " Thickness of the grid lines");
        addParamsLine("[--signal <t=0.15>]                                   :"
                      " Value of the grid pixels, either noiseless or mean for the Poisson distribution");
        addParamsLine(std::string("[") + DisplacementParams::shift_param + " <a1=-0.039> <a2=0.002> <b1=-0.02> <b2=0.002>]:"
                                                                           " Parameters of the shift. To see the result, we encourage you to use script attached with source files!");
        addParamsLine(std::string("[") + DisplacementParams::barrel_param + " <k1_start=0.01> <k1_end=0.015> <k2_start=0.01> <k2_end=0.015>]:"
                                                                            " Parameters of the barrel / pincushion transformation.");
        addParamsLine("-o <output_file>                                      :"
                      " resulting movie");
        addParamsLine("[--skipBarrel]                                        :"
                      " skip applying the barrel deformation");
        addParamsLine("[--skipShift]                                         :"
                      " skip applying shift on each frame");
        addParamsLine("[--shiftAfterBarrel]                                  :"
                      " if set, shift will be applied after barrel deformation (if present)");
        addParamsLine("[--skipDose]                                          :"
                      " generate phantom without Poisson noise");
        addParamsLine("[--skipIce]                                           :"
                      " generate phantom without ice (background)");
        addParamsLine("[--seed <s=42>]                                       :"
                      " seed used to generate the noise");
        addParamsLine("[--ice <avg=1.0> <stddev=1.0> <min=0.0> <max=2.0>]    :"
                      " Ice properties (simulated via Gaussian noise) + range adjustment");
        addParamsLine("[--low <w1=0.05> <raisedW=0.02>]                      :"
                      " Ice low-pass filter properties");
        addParamsLine("[--dose <mean=1>]                                     :"
                      " Mean of the Poisson noise");

        addUsageLine("Create phantom movie with grid, using shift and barrel / pincushion transform.");
        addUsageLine("Bear in mind that the following function of the shift is applied in 'backward'"
                     " fashion,");
        addUsageLine(" as it's original form produces biggest shift towards the end"
                     " as opposed to real movies (which has biggest shift in first frames).");
        addUsageLine(DisplacementParams::doc);
        addUsageLine("If noisy movie is generated, we first generate ice blurred via low-pass filter.");
        addUsageLine("After that, the reference frame is normalized. ");
        addUsageLine("On top of this, we add signal in form of the grid. ");
        addUsageLine("Finally, each frame is generated using poisson distribution.");

        addExampleLine("xmipp_phantom_movie -size 4096 4096 60 -step 50 50 --skipBarrel -o phantom_movie.stk");
    }

    void readParams() override
    {
        auto x = getIntParam(Content::size_param, 0);
        auto y = getIntParam(Content::size_param, 1);
        auto n = getIntParam(Content::size_param, 2);
        params.req_size = Dimensions(x, y, 1, n);

        content.xstep = getIntParam(Content::step_param, 0);
        content.ystep = getIntParam(Content::step_param, 1);
        content.thickness = getIntParam("--thickness");
        content.signal_val = getDoubleParam("--signal", 0);

        dispParams.a1 = getDoubleParam(DisplacementParams::shift_param, 0);
        dispParams.a2 = getDoubleParam(DisplacementParams::shift_param, 1);
        dispParams.b1 = getDoubleParam(DisplacementParams::shift_param, 2);
        dispParams.b2 = getDoubleParam(DisplacementParams::shift_param, 3);

        dispParams.k1_start = getDoubleParam(DisplacementParams::barrel_param, 0);
        dispParams.k1_end = getDoubleParam(DisplacementParams::barrel_param, 1);
        dispParams.k2_start = getDoubleParam(DisplacementParams::barrel_param, 2);
        dispParams.k2_end = getDoubleParam(DisplacementParams::barrel_param, 3);

        options.skipBarrel = checkParam("--skipBarrel");
        options.skipShift = checkParam("--skipShift");
        options.shiftAfterBarrel = checkParam("--shiftAfterBarrel");
        options.skipDose = checkParam("--skipDose");
        options.skipIce = checkParam("--skipIce");

        params.fn_out = getParam("-o");

        content.seed = getIntParam("--seed");
        content.ice_avg = getDoubleParam("--ice", 0);
        content.ice_stddev = getDoubleParam("--ice", 1);
        content.ice_min = getDoubleParam("--ice", 2);
        content.ice_max = getDoubleParam("--ice", 3);
        content.dose = getDoubleParam("--dose");

        content.low_w1 = getDoubleParam("--low", 0);
        content.low_raised_w = getDoubleParam("--low", 1);
    }

    void run()
    {
        auto instance = PhantomMovie<double>(dispParams, options, content, params);
        instance.run();
    }

    DisplacementParams dispParams;
    Options options;
    Content content;
    Params params;
};

RUN_XMIPP_PROGRAM(PhantomMovieProgram)