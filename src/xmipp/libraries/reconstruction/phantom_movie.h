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
        bool simple;
        static constexpr auto doc = "x(t) = a1*t + a2*t*t + cos(t)/10\n"
                                    "y(t) = b1*t + b2*t*t + sin(t*t)/5\n"
                                    "The barrel/pincushion transform params are linearly interpolated between first and last frame.\n"
                                    "For normalized coordinates ([-1..1]) its distance is given by:\n"
                                    "r_out = r_in(1 + k1*(r_in)^2 + k2*(r_in)^4"
                                    "X and Y shift can be overriden, then only a1 and b1 will be used";
        static constexpr auto shift_param = "--shift";
        static constexpr auto barrel_param = "--barrel";
        static constexpr auto simple_param = "--simple";
    };

    struct Options
    {
        bool skipBarrel;
        bool skipShift;
        bool shiftAfterBarrel;
        bool skipDose;
        bool skipIce;
    };

    struct Ice
    {
        int seed;
        float avg;
        float stddev;
        float min;
        float max;
        float low_w1;
        float low_raised_w;
    };

    struct Params
    {
        Dimensions req_size = Dimensions(1);
        Dimensions work_size = Dimensions(1);
        FileName fn_out;
        FileName fn_gain;
        FileName fn_dark;
    };

    enum class PhantomType 
    {
        grid,
        particleCross,
        particleCircle
    };

    struct Content
    {
        PhantomType type;
        size_t xstep; // of the grid
        size_t ystep; // of the grid
        size_t thickness; // of the line
        size_t minSize; // of the circle / cross
        size_t maxSize; // of the circle / cross
        size_t count; // of the circle / cross
        float signal_val;
        int seed;
        float dose;
    };

    PhantomMovie(DisplacementParams dp, Options o, Ice i, Content c, Params p) : params(p), dispParams(dp), options(o), ice(i), content(c) {}

    void run() const;

private:
    void addContent(MultidimArray<T> &movie) const;
    void addGrid(MultidimArray<T> &frame) const;
    void addCircles(MultidimArray<T> &frame) const;
    void addCrosses(MultidimArray<T> &frame) const;
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
    const Ice ice;
    const Content content;
};
