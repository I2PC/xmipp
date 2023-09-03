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
#include "data/fourier_filter.h"

template <typename T>
auto PhantomMovie<T>::shiftX(size_t t) const
{
    const auto tf = static_cast<float>(t);
    if (dispParams.simple)
        return dispParams.a1 * tf;
    return dispParams.a1 * tf + dispParams.a2 * tf * tf + std::cos(tf / 10.f) / 10.f;
};

template <typename T>
auto PhantomMovie<T>::shiftY(size_t t) const
{
    const auto tf = static_cast<float>(t);
    if (dispParams.simple)
        return dispParams.b1 * tf;
    return dispParams.b1 * tf + dispParams.b2 * tf * tf + (std::sin(tf * tf)) / 5.f;
};

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
    auto x_shift = options.skipShift ? 0 : shiftX(params.req_size.n() - n - 1); // 'reverse' the order (see doc)
    auto y_shift = options.skipShift ? 0 : shiftY(params.req_size.n() - n - 1); // 'reverse' the order (see doc)
    if (options.skipBarrel)
    {
        x += x_shift;
        y += y_shift;
    }
    else
    {
        auto x_center = static_cast<float>(params.req_size.x()) / 2.f;
        auto y_center = static_cast<float>(params.req_size.y()) / 2.f;
        auto k1 = dispParams.k1_start + static_cast<float>(n) * (dispParams.k1_end - dispParams.k1_start) / (static_cast<float>(params.req_size.n()) - 1);
        auto k2 = dispParams.k2_start + static_cast<float>(n) * (dispParams.k2_end - dispParams.k2_start) / (static_cast<float>(params.req_size.n()) - 1);
        auto y_norm = (y - y_center + (options.shiftAfterBarrel ? 0 : y_shift)) / y_center;
        auto x_norm = (x - x_center + (options.shiftAfterBarrel ? 0 : x_shift)) / x_center;
        auto r_out = sqrt(x_norm * x_norm + y_norm * y_norm);
        auto r_out_2 = r_out * r_out;
        auto r_out_4 = r_out_2 * r_out_2;
        auto scale = (1 + k1 * r_out_2 + k2 * r_out_4);
        x = (x_norm * scale * x_center) + x_center + (options.shiftAfterBarrel ? x_shift : 0);
        y = (y_norm * scale * y_center) + y_center + (options.shiftAfterBarrel ? y_shift : 0);
    }
}

template <typename T>
void PhantomMovie<T>::addContent(MultidimArray<T> &frame) const
{
    switch (content.type)
    {
    case PhantomType::grid:
        return addGrid(frame);
    case PhantomType::particleCircle:
        return addCircles(frame);
    case PhantomType::particleCross:
        return addCrosses(frame);
    default:
        throw std::logic_error("Unsupported PhantomType");
    }
}

template <typename T>
void PhantomMovie<T>::addCircles(MultidimArray<T> &frame) const
{
    std::cout << "Generating circles" << std::endl;
    auto drawCircle = [&frame](const auto r, const auto x, const auto y, const int thickness, const auto val)
    {
        const auto xdim = frame.xdim;
        for (int j = 0; j <= r + thickness; ++j)
        {
            for (int i = 0; i <= r + thickness; ++i)
            {
                // test top right quadrant
                int d = sqrt(j * j + i * i);
                if (d >= r - thickness && d <= r + thickness)
                {
                    // modify all 4 quadrants
                    frame.data[(y - j) * xdim - i + x] = val;
                    frame.data[(y - j) * xdim + i + x] = val;
                    frame.data[(y + j) * xdim - i + x] = val;
                    frame.data[(y + j) * xdim + i + x] = val;
                }
            }
        }
    };

    std::mt19937 gen(content.seed);
    std::uniform_int_distribution<size_t> distX(content.maxSize / 2 + content.thickness / 2, frame.xdim - 1 - content.maxSize / 2 - content.thickness / 2);
    std::uniform_int_distribution<size_t> distY(content.maxSize / 2 + content.thickness / 2, frame.ydim - 1 - content.maxSize / 2 - content.thickness / 2);
    std::uniform_int_distribution<> size(content.minSize, content.maxSize);
    for (auto i = 0; i < content.count; ++i) {
        auto r = size(gen) / 2;
        auto x = distX(gen);
        auto y = distY(gen);
        drawCircle(r, x, y, content.thickness, content.signal_val);
    }
}

template <typename T>
void PhantomMovie<T>::addCrosses(MultidimArray<T> &frame) const
{
    std::cout << "Generating crosses" << std::endl;
    auto drawCross = [&frame](const auto s, const auto x, const auto y, const auto val)
    {
        const auto xdim = frame.xdim;
        for (int d = 0; d < s; ++d) 
        {
            frame.data[(y - d) * xdim - d + x] = val;
            frame.data[(y - d) * xdim + d + x] = val;
            frame.data[(y + d) * xdim - d + x] = val;
            frame.data[(y + d) * xdim + d + x] = val;
        }
    };

    std::mt19937 gen(content.seed);
    std::uniform_int_distribution<size_t> distX(content.maxSize / 2 + content.thickness / 2, frame.xdim - 1 - content.maxSize / 2 - content.thickness / 2);
    std::uniform_int_distribution<size_t> distY(content.maxSize / 2 + content.thickness / 2, frame.ydim - 1 - content.maxSize / 2 - content.thickness / 2);
    std::uniform_int_distribution<> size(content.minSize, content.maxSize);
    const auto xdim = frame.xdim;
    const int thickness = content.thickness;
    for (auto i = 0; i < content.count; ++i) {
        auto s = size(gen) / 2;
        auto x = distX(gen);
        auto y = distY(gen);
        // draw cross
        for (int t = 0; t < thickness / 2; ++t)
        {
            // move the center to change the thickness
            drawCross(s, x, y - t, content.signal_val);
            drawCross(s, x - t, y, content.signal_val);
            drawCross(s, x + t, y, content.signal_val);
            drawCross(s, x, y + t, content.signal_val);
        }
    }
}

template <typename T>
void PhantomMovie<T>::addGrid(MultidimArray<T> &frame) const
{
    std::cout << "Generating grid" << std::endl;
    const auto xdim = frame.xdim;
    const auto ydim = frame.ydim;
    // add rows
    for (auto y = content.ystep - (content.thickness / 2); y < ydim - (content.thickness / 2) + 1; y += content.ystep)
    {
        for (auto t = 0; t < content.thickness; ++t)
        {
            size_t y_offset = (y + t) * xdim;
            for (size_t x = 0; x < xdim; ++x)
            {
                size_t index = y_offset + x;
                frame.data[index] += content.signal_val;
            }
        }
    }
    // add columns
    for (auto x = content.xstep; x < xdim - (content.thickness / 2) + 1; x += content.xstep)
    {
        for (int t = 0; t < content.thickness; ++t)
        {
            size_t x_offset = (x + t);
            for (size_t y = 0; y < ydim; ++y)
            {
                size_t index = x_offset + y * xdim;
                frame.data[index] += content.signal_val;
            }
        }
    }
}

template <typename T>
MultidimArray<T> PhantomMovie<T>::findWorkSize() const
{
    auto x_max_shift = 0.f;
    auto y_max_shift = 0.f;
    const auto x = static_cast<float>(params.req_size.x());
    const auto y = static_cast<float>(params.req_size.y());
    for (size_t n = 0; n < params.req_size.n(); ++n)
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
    // + 10 is in case we made a rounding mistake on multiple places :/
    auto x_new = params.req_size.x() + 2 * static_cast<size_t>(std::ceil(x_max_shift)) + 10;
    auto y_new = params.req_size.y() + 2 * static_cast<size_t>(std::ceil(y_max_shift)) + 10;
    std::cout << "Due to displacement, working with frames of size [" << x_new << ", " << y_new << "]\n";
    return MultidimArray<T>(1, 1, static_cast<int>(y_new), static_cast<int>(x_new));
}

template <typename T>
void PhantomMovie<T>::applyLowPass(MultidimArray<T> &frame) const
{
    std::cout << "Applying low-pass filter\n";
    auto filter = FourierFilter();
    filter.w1 = ice.low_w1;
    filter.raised_w = ice.low_raised_w;
    filter.FilterBand = LOWPASS;
    filter.FilterShape = RAISED_COSINE;
    filter.apply(frame);
}

template <typename T>
void PhantomMovie<T>::generateIce(MultidimArray<T> &frame) const
{
    std::cout << "Generating ice\n";
    std::mt19937 gen(ice.seed);
    std::normal_distribution<> d(ice.avg, ice.stddev);
    const auto nzyxdim = frame.nzyxdim;
    for (size_t i = 0; i < nzyxdim; ++i)
    {
        frame[i] = d(gen);
    }
}

template <typename T>
template <bool SKIP_DOSE>
void PhantomMovie<T>::generateMovie(const MultidimArray<T> &refFrame) const
{
    MultidimArray<T> frame(1, 1, static_cast<int>(params.req_size.y()), static_cast<int>(params.req_size.x()));
    params.fn_out.deleteFile();
    std::mt19937 gen(content.seed);

    auto genFrame = [&frame, &refFrame, &gen, this](auto n)
    {
        float x_center = static_cast<float>(frame.xdim) / 2.f;
        float y_center = static_cast<float>(frame.ydim) / 2.f;
        std::cout << "Processing frame " << n << std::endl;
        for (size_t y = 0; y < params.req_size.y(); ++y) {
            for (size_t x = 0; x < params.req_size.x(); ++x) {
                auto x_tmp = static_cast<float>(x);
                auto y_tmp = static_cast<float>(y);
                displace(x_tmp, y_tmp, n);
                // move coordinate system to center - [0, 0] will be in the center of the frame
                auto val = bilinearInterpolation(refFrame, x_tmp - x_center, y_tmp - y_center);
                if (!SKIP_DOSE) {
                    auto dist = std::poisson_distribution<int>(val * content.dose);
                    val = dist(gen);
                }
                frame.data[y * frame.xdim + x] = val;
            }
        }
    };

    for (size_t n = 0; n < params.req_size.n(); ++n)
    {
        genFrame(n);
        Image<T> tmp(frame);
        tmp.write(params.fn_out, n + 1, true, WRITE_REPLACE);
    }
}

template <typename T>
void PhantomMovie<T>::run() const
{
    auto refFrame = findWorkSize();
    if (!options.skipIce)
    {
        generateIce(refFrame);
        applyLowPass(refFrame);
        refFrame.rangeAdjust(ice.min, ice.max);
    }
    addContent(refFrame);
    if (options.skipDose)
    {
        generateMovie<true>(refFrame);
    }
    else
    {
        generateMovie<false>(refFrame);
    }
    if (!params.fn_gain.empty())
    {
        Image<T> gain(static_cast<int>(params.req_size.x()), static_cast<int>(params.req_size.y()));
        gain().initConstant(1);
        gain.write(params.fn_gain);
    }
    if (!params.fn_dark.empty())
    {
        Image<T> dark(static_cast<int>(params.req_size.x()), static_cast<int>(params.req_size.y()));
        dark().initConstant(0);
        dark.write(params.fn_dark);
    }
}

// template class PhantomMovie<float>; // can't be used because the fourier filter doesn't support it
template class PhantomMovie<double>;
