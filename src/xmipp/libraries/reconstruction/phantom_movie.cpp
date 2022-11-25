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
    addParamsLine("[--signal <t=1.0>]                                    :"
                  " Value of the grid pixels, either noiseless or mean for the Poisson distribution");
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
    addParamsLine("[--skipDose]                                          :"
                  " generate phantom without Poisson noise");
    addParamsLine("[--skipIce]                                           :"
                  " generate phantom without ice (background)");
    addParamsLine("[--seed <s=42>]                                       :"
                  " seed used to generate the noise");
    addParamsLine("[--ice <avg=0.0> <stddev=1.0> <min=0.1> <max=1.0>]    :"
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
    addUsageLine("x(t) = a1*t + a2*t*t + cos(t)/10");
    addUsageLine("y(t) = b1*t + b2*t*t + sin(t*t)/5");
    addUsageLine("The barrel/pincushion transform params are linearly interpolated between first and last frame.");
    addUsageLine("For normalized coordinates ([-1..1]) its distance is given by:");
    addUsageLine("r_out = r_in(1 + k1*(r_in)^2 + k2*(r_in)^4");
    addUsageLine("If noisy movie is generated, we first generate ice blurred via low-pass filter.");
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
    signal_val = getDoubleParam("--signal", 0);

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
    skipDose = checkParam("--skipDose");
    skipIce = checkParam("--skipIce");

    fn_out = getParam("-o");

    seed = getIntParam("--seed");
    ice_avg = getDoubleParam("--ice", 0);
    ice_stddev = getDoubleParam("--ice", 1);
    ice_min = getDoubleParam("--ice", 2);
    ice_max = getDoubleParam("--ice", 3);
    dose = getDoubleParam("--dose");

    low_w1 = getDoubleParam("--low", 0);
    low_raised_w = getDoubleParam("--low", 1);
}

template <typename T>
T PhantomMovie<T>::bilinearInterpolation(const MultidimArray<T> &src, float x, float y) const
{
    auto x_center = static_cast<float>(src.xdim) / 2.f;
    auto y_center = static_cast<float>(src.ydim) / 2.f;
    const auto xdim = static_cast<float>(src.xdim);
    x += x_center;
    y += y_center;
    auto xf = std::floor(x);
    auto xc = std::ceil(x);
    auto yf = std::floor(y);
    auto yc = std::ceil(y);
    auto xw = x - xf;
    auto yw = y - yf;
    auto vff = src.data[static_cast<size_t>(xdim * yf + xf)];
    auto vfc = src.data[static_cast<size_t>(xdim * yc + xf)];
    auto vcf = src.data[static_cast<size_t>(xdim * yf + xc)];
    auto vcc = src.data[static_cast<size_t>(xdim * yc + xc)];
    return vff * (1.f - xw) * (1.f - yw) + vcf * xw * (1.f - yw) + vfc * (1.f - xw) * yw + vcc * xw * yw;
}

template <typename T>
void PhantomMovie<T>::displace(float &x, float &y, size_t n) const
{
    auto x_shift = skipShift ? 0 : shiftX(req_size.n() - n - 1); // 'reverse' the order (see doc)
    auto y_shift = skipShift ? 0 : shiftY(req_size.n() - n - 1); // 'reverse' the order (see doc)
    if (skipBarrel)
    {
        x += x_shift;
        y += y_shift;
    }
    else
    {
        auto x_center = static_cast<float>(req_size.x()) / 2.f;
        auto y_center = static_cast<float>(req_size.y()) / 2.f;
        auto k1 = k1_start + static_cast<float>(n) * (k1_end - k1_start) / (static_cast<float>(req_size.n()) - 1);
        auto k2 = k2_start + static_cast<float>(n) * (k2_end - k2_start) / (static_cast<float>(req_size.n()) - 1);
        auto y_norm = (y - y_center + (shiftAfterBarrel ? 0 : y_shift)) / y_center;
        auto x_norm = (x - x_center + (shiftAfterBarrel ? 0 : x_shift)) / x_center;
        auto r_out = sqrt(x_norm * x_norm + y_norm * y_norm);
        auto r_out_2 = r_out * r_out;
        auto r_out_4 = r_out_2 * r_out_2;
        auto scale = (1 + k1 * r_out_2 + k2 * r_out_4);
        x = (x_norm * scale * x_center) + x_center + (shiftAfterBarrel ? x_shift : 0);
        y = (y_norm * scale * y_center) + y_center + (shiftAfterBarrel ? y_shift : 0);
    }
}

template <typename T>
void PhantomMovie<T>::addGrid(MultidimArray<T> &frame)
{
    std::cout << "Generating grid" << std::endl;
    // add rows
    for (auto y = ystep - (thickness / 2); y < frame.ydim - (thickness / 2) + 1; y += ystep)
    {
        for (auto t = 0; t < thickness; ++t)
        {
            size_t y_offset = (y + t) * frame.xdim;
            for (size_t x = 0; x < frame.xdim; ++x)
            {
                size_t index = y_offset + x;
                frame.data[index] += signal_val;
            }
        }
    }
    // add columns
    for (auto x = xstep; x < frame.xdim- (thickness / 2) + 1; x += xstep)
    {
        for (int t = 0; t < thickness; ++t)
        {
            size_t x_offset = (x + t);
            for (size_t y = 0; y < frame.ydim; ++y)
            {
                size_t index = x_offset + y * frame.xdim;
                frame.data[index] += signal_val;
            }
        }
    }
}

template <typename T>
MultidimArray<T> PhantomMovie<T>::findWorkSize()
{
    auto x_max_shift = 0.f;
    auto y_max_shift = 0.f;
    const auto x = static_cast<float>(req_size.x());
    const auto y = static_cast<float>(req_size.y());
    for (size_t n = 0; n < req_size.n(); ++n)
    {
        auto x_0 = 0.f;
        auto y_0 = 0.f;
        auto x_n = x - 1.f;
        auto y_n = y - 1.f;
        displace(x_0, y_0, n); // top left corner
        displace(x_n, y_n, n); // bottom right corner
        // barrel deformation moves to center, so we need to read outside of the edge - [0,0] will be negative and [n-1,n-1] will be bigger than that
        auto x_shift = std::max(std::abs(x_0), x_n - x - 1.f);
        auto y_shift = std::max(std::abs(y_0), y_n - y - 1.f);
        x_max_shift = std::max(x_max_shift, x_shift);
        y_max_shift = std::max(y_max_shift, y_shift);
    }
    // new size must incorporate 'gaps' on both sides, min values should be negative, max values positive
    // the shift might not be uniform, so the safer solution is to have the bigger gap on both sides
    auto x_new = req_size.x() + 2 * static_cast<size_t>(std::ceil(x_max_shift));
    auto y_new = req_size.y() + 2 * static_cast<size_t>(std::ceil(y_max_shift));
    printf("Due to displacement, working with frames of size [%lu, %lu]\n", x_new, y_new);
    return MultidimArray<T>(1, 1, static_cast<int>(y_new), static_cast<int>(x_new));
}

template <typename T>
void PhantomMovie<T>::applyLowPass(MultidimArray<T> &frame) const
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
void PhantomMovie<T>::generateIce(MultidimArray<T> &frame) const
{
    std::cout << "Generating ice\n";
    std::mt19937 gen(seed);
    std::normal_distribution<> d(ice_avg, ice_stddev);
    for (size_t i = 0; i < frame.nzyxdim; ++i)
    {
        frame[i] = d(gen);
    }
}

template <typename T>
template<bool SKIP_DOSE>
void PhantomMovie<T>::generateMovie(const MultidimArray<T> &refFrame) const
{
    MultidimArray<T> frame(1, 1, static_cast<int>(req_size.y()), static_cast<int>(req_size.x()));
    fn_out.deleteFile();
    std::mt19937 gen(seed);

    auto genFrame = [&frame, &refFrame, &gen, this](auto n) {
        float x_center = frame.xdim / 2.f;
        float y_center = frame.ydim / 2.f;
        std::cout << "Processing frame " << n << std::endl;
        for (size_t y = 0; y < req_size.y(); ++y) {
            for (size_t x = 0; x < req_size.x(); ++x) {
                auto x_tmp = static_cast<float>(x);
                auto y_tmp = static_cast<float>(y);
                displace(x_tmp, y_tmp, n);
                // move coordinate system to center - [0, 0] will be in the center of the frame
                auto val = bilinearInterpolation(refFrame, x_tmp - x_center, y_tmp - y_center);
                if (!SKIP_DOSE) {
                    auto dist = std::poisson_distribution<int>(val * dose);
                    val = dist(gen);
                }
                frame.data[y * frame.xdim + x] = val;
            }
        }
    };

    for (size_t n = 0; n < req_size.n(); ++n)
    {
        genFrame(n);
        Image<T> tmp(frame);
        tmp.write(fn_out, n + 1, true, WRITE_REPLACE);
    }
}

template <typename T>
void PhantomMovie<T>::run()
{
    auto refFrame = findWorkSize();
    if (!skipIce)
    {
        generateIce(refFrame);
        applyLowPass(refFrame);
        refFrame.rangeAdjust(ice_min, ice_max);
    }
    addGrid(refFrame);
    if (skipDose) {
        generateMovie<true>(refFrame);
    } else {
        generateMovie<false>(refFrame);
    }
}

// template class PhantomMovie<float>; // can't be used because the fourier filter doesn't support it
template class PhantomMovie<double>;
