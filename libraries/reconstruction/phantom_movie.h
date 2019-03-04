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

#include <core/xmipp_program.h>
#include <core/xmipp_image.h>

template<typename T>
class PhantomMovie: public XmippProgram {
public:
    /** Read parameters. */
    void readParams();

    /** Define parameters */
    void defineParams();

    /** Run */
    void run();
private:
    void generateGrid();
    void addShiftBarrelDeformation();
    void addShift();
    T bilinearInterpolation(Image<T>& src, T x, T y);
    bool inRangeX(T x) { return (x >= 0) && (x < xdim); };
    bool inRangeY(T y) { return (y >= 0) && (y < ydim); };
    bool inRange(T x, T y) { return inRangeX(x) && inRangeY(y); };
    inline T getValue(Image<T>& src, T x, T y);
    T shiftX(T t) { return a1*t + a2*t*t + std::cos(t/T(10))/(T)10; };
    T shiftY(T t) { return b1*t + b2*t*t + (std::sin(t*t))/(T)5; };
protected:
    size_t xdim;
    size_t ydim;
    size_t ndim;

    size_t xstep;
    size_t ystep;

    T a1, a2, b1, b2;
    T k1_start, k1_end;
    T k2_start, k2_end;

    size_t thickness;

    bool skipBarrel;
    bool skipShift;
    bool shiftAfterBarrel;

    const std::string size_param = std::string("-size");
    const std::string step_param = std::string("-step");
    const std::string shift_param = std::string("--shift");
    const std::string barrel_param = std::string("--barrel");
    FileName fn_out;

    Image<T> movie;
};

#endif /* PHANTOM_MOVIE_H_ */
