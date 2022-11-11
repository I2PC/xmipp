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

#include "phantom_movie.h"
#include <random>
#include "reconstruction/fftwT.h"
#include "data/fourier_filter.h"
#include <filesystem>

template <typename T>
void PhantomMovie<T>::defineParams()
{
    addParamsLine(size_param + " <x=4096> <y=4096> <n=40>                :"
                               " Movie size");
    addParamsLine(step_param + " <x=50> <y=50>                           :"
                               " Distance between the lines/rows of the grid (before the transform is applied)");
    addParamsLine("[--thickness <t=5>]                                   :"
                  " Thickness of the grid lines");
    addParamsLine("[--gridVal <t=1.0>]                                   :"
                  " Value of the grid pixels");
    addParamsLine("[" + shift_param + " <a1=-0.039> <a2=0.002> <b1=-0.02> <b2=0.002>]:"
                                      " Parameters of the shift. To see the result, we encourage you to use script attached with source files!");
    addParamsLine("[" + barrel_param + " <k1_start=0.04> <k1_end=0.05> <k2_start=0.02> <k2_end=0.025>]:"
                                       " Parameters of the barrel / pincushion transformation.");
    addParamsLine("-o <output_file>                                      :"
                  " resulting movie");
    addParamsLine("[--skipBarrel]                                        :"
                  " skip applying the barrel deformation");
    addParamsLine("[--skipShift]                                         :"
                  " skip applying shift on each frame");
    addParamsLine("[--shiftAfterBarrel]                                  :"
                  " if set, shift will be applied after barrel deformation (if present)");
    addParamsLine("[--skipNoise]                                         :"
                  " generate phantom without noise");
    addParamsLine("[--seed <s=42>]                                       :"
                  " seed used to generate the noise");
    addParamsLine("[--norm <avg=0.0> <stddev=1.0>]                       :"
                  " Normalization values");
    addParamsLine("[--gauss <avg=0.0> <stddev=1.0>]                      :"
                  " Gauss noise properties");
    addParamsLine("[--poisson <mean=0.0>]                                :"
                  " Poisson noise properties");
    addParamsLine("[--low <w1=0.05> <raisedW=0.02>]                      :"
                  " Low-pass filter properties");

    addUsageLine("Create phantom movie with grid, using shift and barrel / pincushion transform.");
    addUsageLine("Bear in mind that the following function of the shift is applied in 'backward'"
                 " fashion,");
    addUsageLine(" as it's original form produces biggest shift towards the end"
                 " as opposed to real movies (which has biggest shift in first frames).");
    addUsageLine("x(t) = a1*t + a2*t*t + cos(t)/10");
    addUsageLine("y(t) = b1*t + b2*t*t + sin(t*t)/5");
    addUsageLine("The barrel/pincushion transform params are linearly interpolated between first and last frame.");
    addUsageLine("For normalized coordinates ([-1..1]) its distance is given by:");
    addUsageLine("r_out = r_in(1 + k1*(r_in)^2 + k2*(r_in)^4");
    addUsageLine("If noisy movie is generated, we first generate gaussian noise (ice) blurred via low-pass filter.");
    addUsageLine("After that, the reference frame is normalized. ");
    addUsageLine("On top of this, we add signal in form of the grid. ");
    addUsageLine("Finally, each frame is generated using poisson distribution.");

    addExampleLine("xmipp_phantom_movie -size 4096 4096 60 -step 50 50 --skipBarrel -o phantom_movie.stk");
}

template <typename T>
void PhantomMovie<T>::readParams()
{
    const char *size_param_ch = size_param.c_str();
    auto x = getIntParam(size_param_ch, 0);
    auto y = getIntParam(size_param_ch, 1);
    auto n = getIntParam(size_param_ch, 2);
    req_size = Dimensions(x, y, 1, n);

    const char *step_param_ch = step_param.c_str();
    xstep = getIntParam(step_param_ch, 0);
    ystep = getIntParam(step_param_ch, 1);
    thickness = getIntParam("--thickness");
    gridVal = getDoubleParam("--gridVal");

    const char *shift_param_ch = shift_param.c_str();
    a1 = getDoubleParam(shift_param_ch, 0);
    a2 = getDoubleParam(shift_param_ch, 1);
    b1 = getDoubleParam(shift_param_ch, 2);
    b2 = getDoubleParam(shift_param_ch, 3);

    const char *barrel_param_ch = barrel_param.c_str();
    k1_start = getDoubleParam(barrel_param_ch, 0);
    k1_end = getDoubleParam(barrel_param_ch, 1);
    k2_start = getDoubleParam(barrel_param_ch, 2);
    k2_end = getDoubleParam(barrel_param_ch, 3);

    skipBarrel = checkParam("--skipBarrel");
    skipShift = checkParam("--skipShift");
    shiftAfterBarrel = checkParam("--shiftAfterBarrel");
    skipNoise = checkParam("--skipNoise");

    fn_out = getParam("-o");

    seed = getIntParam("--seed");
    norm_avg = getDoubleParam("--norm", 0);
    norm_stddev = getDoubleParam("--norm", 1);
    gauss_avg = getDoubleParam("--gauss", 0);
    gauss_stddev = getDoubleParam("--gauss", 1);
    poisson_mean = getDoubleParam("--poisson", 0);
    low_w1 = getDoubleParam("--low", 0);
    low_raised_w = getDoubleParam("--low", 1);
}

template <typename T>
T PhantomMovie<T>::bilinearInterpolation(const MultidimArray<T> &src, float x, float y)
{
    float x_center = src.xdim / (T)2;
    float y_center = src.ydim / (T)2;
    x += x_center;
    y += y_center;
    size_t xf = std::floor(x);
    size_t xc = std::ceil(x);
    size_t yf = std::floor(y);
    size_t yc = std::ceil(y);
    float xw = x - xf;
    float yw = y - yf;
    T vff = src.data[src.xdim * yf + xf];
    T vfc = src.data[src.xdim * yc + xf];
    T vcf = src.data[src.xdim * yf + xc];
    T vcc = src.data[src.xdim * yc + xc];
    return vff * (1.f - xw) * (1.f - yw) + vcf * xw * (1.f - yw) + vfc * (1.f - xw) * yw + vcc * xw * yw;
}

template <typename T>
void PhantomMovie<T>::displace(float &x, float &y, size_t n)
{
    float x_shift = skipShift ? 0 : shiftX(req_size.n() - n - 1); // 'reverse' the order (see doc)
    float y_shift = skipShift ? 0 : shiftY(req_size.n() - n - 1); // 'reverse' the order (see doc)
    if (skipBarrel)
    {
        x += x_shift;
        y += y_shift;
    }
    else
    {
        float x_center = req_size.x() / (T)2;
        float y_center = req_size.y() / (T)2;
        float k1 = k1_start + n * (k1_end - k1_start) / (req_size.n() - 1);
        float k2 = k2_start + n * (k2_end - k2_start) / (req_size.n() - 1);
        float y_norm = (y - y_center + (shiftAfterBarrel ? 0 : y_shift)) / y_center;
        float x_norm = (x - x_center + (shiftAfterBarrel ? 0 : x_shift)) / x_center;
        float r_out = sqrt(x_norm * x_norm + y_norm * y_norm);
        float r_out_2 = r_out * r_out;
        float r_out_4 = r_out_2 * r_out_2;
        float scale = (1 + k1 * r_out_2 + k2 * r_out_4);
        x = (x_norm * scale * x_center) + x_center + (shiftAfterBarrel ? x_shift : 0);
        y = (y_norm * scale * y_center) + y_center + (shiftAfterBarrel ? y_shift : 0);
    }
}

template <typename T>
void PhantomMovie<T>::addGrid(MultidimArray<T> &frame)
{
    std::cout << "Generating grid" << std::endl;
    // add rows
    for (size_t y = ystep; y < frame.ydim; y += ystep)
    {
        for (int t = -(thickness / 2); t < std::ceil(thickness / 2.0); ++t)
        {
            if ((y >= thickness / 2) && (y + t < frame.ydim))
            {
                size_t y_offset = (y + t) * frame.xdim;
                for (size_t x = 0; x < frame.xdim; ++x)
                {
                    size_t index = y_offset + x;
                    frame.data[index] += gridVal;
                }
            }
        }
    }
    // add columns
    for (size_t x = xstep; x < frame.xdim; x += xstep)
    {
        for (int t = -(thickness / 2); t < std::ceil(thickness / 2.0); ++t)
        {
            if ((x >= thickness / 2) && (x + t < frame.xdim))
            {
                size_t x_offset = (x + t);
                for (size_t y = 0; y < frame.ydim; ++y)
                {
                    size_t index = x_offset + y * frame.xdim;
                    frame.data[index] += gridVal;
                }
            }
        }
    }
}

template <typename T>
MultidimArray<T> PhantomMovie<T>::findWorkSize()
{
    float x_max_shift = 0;
    float y_max_shift = 0;
    for (size_t n = 0; n < req_size.n(); ++n)
    {
        float x_0 = 0;
        float y_0 = 0;
        float x_n = req_size.x() - 1;
        float y_n = req_size.y() - 1;
        displace(x_0, y_0, n); // top left corner
        displace(x_n, y_n, n); // bottom right corner
        // barrel deformation moves to center, so we need to read outside of the edge - [0,0] will be negative and [n-1,n-1] will be bigger than that
        auto x_shift = std::max(std::abs(x_0), x_n - req_size.x() - 1);
        auto y_shift = std::max(std::abs(y_0), y_n - req_size.y() - 1);
        x_max_shift = std::max(x_max_shift, x_shift);
        y_max_shift = std::max(y_max_shift, y_shift);
    }
    // new size must incorporate 'gaps' on both sides, min values should be negative, max values positive
    // the shift might not be uniform, so the safer solution is to have the bigger gap on both sides
    size_t x_new = req_size.x() + 2 * std::ceil(x_max_shift);
    size_t y_new = req_size.y() + 2 * std::ceil(y_max_shift);
    printf("Due to displacement, working with frames of size [%lu, %lu]\n", x_new, y_new);
    return MultidimArray<T>(1, 1, y_new, x_new);
}

template <typename T>
void PhantomMovie<T>::applyLowPass(MultidimArray<T> &frame)
{
    std::cout << "Applying low-pass filter\n";
    auto filter = FourierFilter();
    filter.w1 = low_w1;
    filter.raised_w = low_raised_w;
    filter.FilterBand = LOWPASS;
    filter.FilterShape = RAISED_COSINE;
    filter.apply(frame);
}

template <typename T>
void PhantomMovie<T>::generateIce(MultidimArray<T> &frame)
{
    std::cout << "Generating ice\n";
    std::mt19937 gen(seed);
    std::normal_distribution<> d(gauss_avg, gauss_stddev);
    for (size_t i = 0; i < frame.nzyxdim; ++i)
    {
        frame[i] = d(gen);
    }
}

template <typename T>
void PhantomMovie<T>::generateMovie(const MultidimArray<T> &refFrame)
{
    MultidimArray<T> frame(1, 1, req_size.y(), req_size.x());
    float x_center = frame.xdim / (T)2;
    float y_center = frame.ydim / (T)2;
    fn_out.deleteFile();
    for (size_t n = 0; n < req_size.n(); ++n)
    {
        std::cout << "Processing frame " << n << std::endl;
        for (size_t y = 0; y < req_size.y(); ++y)
        {
            for (size_t x = 0; x < req_size.x(); ++x)
            {
                float x_tmp = x;
                float y_tmp = y;
                displace(x_tmp, y_tmp, n);
                // move coordinate system to center - [0, 0] will be in the center of the frame
                auto val = bilinearInterpolation(refFrame, x_tmp - x_center, y_tmp - y_center);
                frame.data[y * frame.xdim + x] = val; // I'm not sure how to use the poisson distribution, as call to dist(gen) expects a generator and returns random non-negative integer values, where we want floating number
            }
        }
        Image<T> tmp(frame);
        tmp.write(fn_out, n + 1, true, WRITE_REPLACE);
    }
}

template <typename T>
void PhantomMovie<T>::run()
{
    auto refFrame = findWorkSize();
    if (!skipNoise)
    {
        generateIce(refFrame);
        applyLowPass(refFrame);
        refFrame.statisticsAdjust(norm_avg, norm_stddev);
    }
    addGrid(refFrame);
    generateMovie(refFrame);
}

// template class PhantomMovie<float>; // can't be used because the fourier filter doesn't support it
template class PhantomMovie<double>;
