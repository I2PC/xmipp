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

#ifndef PHANTOM_MOVIE_H_
#define PHANTOM_MOVIE_H_

#include "core/xmipp_program.h"
#include "core/multidim_array.h"
#include "core/xmipp_filename.h"
#include "data/dimensions.h"

/**@defgroup PhantomMovie Phantom Movie
   @ingroup ReconsLibrary */
//@{
template <typename T>
class PhantomMovie final : public XmippProgram
{
public:
    /** Read parameters. */
    void readParams();

    /** Define parameters */
    void defineParams();

    /** Run */
    void run();

private:
    void addGrid(MultidimArray<T> &movie);
    T bilinearInterpolation(const MultidimArray<T> &src, float x, float y) const;
    auto shiftX(size_t t) const { return a1 * t + a2 * t * t + std::cos(t / 10.f) / 10.f; };
    auto shiftY(size_t t) const { return b1 * t + b2 * t * t + (std::sin(t * t)) / 5.f; };

    void generateIce(MultidimArray<T> &frame) const;
    template<bool SKIP_DOSE>
    void generateMovie(const MultidimArray<T> &refFrame) const;
    void applyLowPass(MultidimArray<T> &frame) const;

    void displace(float &x, float &y, size_t n) const;
    MultidimArray<T> findWorkSize();
    Dimensions req_size = Dimensions(1);
    Dimensions work_size = Dimensions(1);

    // displacement params
    float a1, a2, b1, b2;
    float k1_start, k1_end;
    float k2_start, k2_end;
    // grid properties
    size_t xstep;
    size_t ystep;
    size_t thickness;
    float signal_val;
    // other options
    bool skipBarrel;
    bool skipShift;
    bool shiftAfterBarrel;
    bool skipDose;
    bool skipIce;
    // content properties
    int seed;
    float ice_avg;
    float ice_stddev;
    float ice_min;
    float ice_max;
    float dose;
    float low_w1;
    float low_raised_w;

    const std::string size_param = std::string("-size");
    const std::string step_param = std::string("-step");
    const std::string shift_param = std::string("--shift");
    const std::string barrel_param = std::string("--barrel");
    FileName fn_out;
};

//@}
#endif /* PHANTOM_MOVIE_H_ */
