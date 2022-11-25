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

#pragma once

#include "core/xmipp_filename.h"
#include "data/dimensions.h"

template <typename T>
class MultidimArray;

/**@defgroup PhantomMovie Phantom Movie
   @ingroup ReconsLibrary */
//@{
template <typename T>
class PhantomMovie final
{
public:
    struct DisplacementParams
    {
        float a1;
        float a2;
        float b1;
        float b2;
        float k1_start;
        float k1_end;
        float k2_start;
        float k2_end;
        static constexpr auto doc = "x(t) = a1*t + a2*t*t + cos(t)/10\n"
                                    "y(t) = b1*t + b2*t*t + sin(t*t)/5\n"
                                    "The barrel/pincushion transform params are linearly interpolated between first and last frame.\n"
                                    "For normalized coordinates ([-1..1]) its distance is given by:\n"
                                    "r_out = r_in(1 + k1*(r_in)^2 + k2*(r_in)^4";
        static constexpr auto shift_param = "--shift";
        static constexpr auto barrel_param = "--barrel";
    };

    struct Options
    {
        bool skipBarrel;
        bool skipShift;
        bool shiftAfterBarrel;
        bool skipDose;
        bool skipIce;
    };

    struct Content
    {
        size_t xstep;
        size_t ystep;
        size_t thickness;
        float signal_val;
        int seed;
        float ice_avg;
        float ice_stddev;
        float ice_min;
        float ice_max;
        float dose;
        float low_w1;
        float low_raised_w;
        static constexpr auto size_param = "-size";
        static constexpr auto step_param = "-step";
    };

    struct Params
    {
        Dimensions req_size = Dimensions(1);
        Dimensions work_size = Dimensions(1);
        FileName fn_out;
    };

    PhantomMovie(DisplacementParams dp, Options o, Content c, Params p) : params(p), dispParams(dp), options(o), content(c) {}

    void run() const;

private:
    void addGrid(MultidimArray<T> &movie) const;
    T bilinearInterpolation(const MultidimArray<T> &src, float x, float y) const;
    auto shiftX(size_t t) const;
    auto shiftY(size_t t) const;
    void generateIce(MultidimArray<T> &frame) const;
    template <bool SKIP_DOSE>
    void generateMovie(const MultidimArray<T> &refFrame) const;
    void applyLowPass(MultidimArray<T> &frame) const;
    void displace(float &x, float &y, size_t n) const;
    MultidimArray<T> findWorkSize() const;

    const Params params;
    const DisplacementParams dispParams;
    const Options options;
    const Content content;
};
