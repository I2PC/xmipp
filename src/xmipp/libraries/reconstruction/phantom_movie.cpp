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

template<typename T>
void PhantomMovie<T>::defineParams()
{
	addParamsLine(size_param + " <x=4096> <y=4096> <n=40>                :"
			" Movie size");
	addParamsLine(step_param + " <x=50> <y=50>                           :"
			" Distance between the lines/rows of the grid (before the transform is applied)");
	addParamsLine("[--thickness <t=5>]                                   :"
			" Thickness of the grid lines" );
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

	addUsageLine("Create phantom movie with grid, using shift and barrel / pincushion transform.");
	addUsageLine("Bear in mind that the following function of the shift is applied in 'backward'"
			" fashion,");
	addUsageLine(" as it's original form produces biggest shift towards the end"
			" as opposed to real movies (which has biggest shift in first frames).");
	addUsageLine("x(t) = a1*t + a2*t*t + cos(t)/10");
	addUsageLine("y(t) = b1*t + b2*t*t + sin(t*t)/5");
	addUsageLine("The barrel/pincushion transform params are linearly interpolated between first and last frame.");
	addUsageLine("For normalized coordinates ([-1..1]) its distance is given by:");
	addUsageLine("r_out = r_in(1 + k1*(r_in)^2 + k2*(r_in)^4" );

	addExampleLine("xmipp_phantom_movie -size 4096 4096 60 -step 50 50 --skipBarrel -o phantom_movie.stk");
}

template<typename T>
void PhantomMovie<T>::readParams()
{
	const char* size_param_ch = size_param.c_str();
	xdim = getIntParam(size_param_ch, 0);
	ydim = getIntParam(size_param_ch, 1);
	ndim = getIntParam(size_param_ch, 2);

	const char* step_param_ch = step_param.c_str();
	xstep = getIntParam(step_param_ch, 0);
	ystep = getIntParam(step_param_ch, 1);

	const char* shift_param_ch = shift_param.c_str();
	a1 = getDoubleParam(shift_param_ch, 0);
	a2 = getDoubleParam(shift_param_ch, 1);
	b1 = getDoubleParam(shift_param_ch, 2);
	b2 = getDoubleParam(shift_param_ch, 3);

	const char* barrel_param_ch = barrel_param.c_str();
	k1_start = getDoubleParam(barrel_param_ch, 0);
	k1_end = getDoubleParam(barrel_param_ch, 1);
	k2_start = getDoubleParam(barrel_param_ch, 2);
	k2_end = getDoubleParam(barrel_param_ch, 3);

	skipBarrel = checkParam("--skipBarrel");
	skipShift = checkParam("--skipShift");
	shiftAfterBarrel = checkParam("--shiftAfterBarrel");

	thickness = getIntParam("--thickness");
	fn_out = getParam("-o");
}

template<typename T>
T PhantomMovie<T>::getValue(Image<T>& src, T x, T y)
{
	if (inRange(x, y)) {
		size_t index = (size_t)y * xdim + (size_t)x;
		return src.data[index];
	}
	return (T)0;
}

template<typename T>
T PhantomMovie<T>::bilinearInterpolation(Image<T>& src, T x, T y)
{
	T xf = std::floor(x);
	T xc = std::ceil(x);
	T yf = std::floor(y);
	T yc = std::ceil(y);
	T xw = x - xf;
	T yw = y - yf;
	T vff = getValue(src, xf, yf);
	T vfc = getValue(src, xf, yc);
	T vcf = getValue(src, xc, yf);
	T vcc = getValue(src, xc, yc);
	return vff * ((T)1 - xw) * ((T)1 - yw)
			+ vcf * xw * ((T)1 - yw)
			+ vfc * ((T)1 - xw) * yw
			+ vcc * xw * yw;
}

template<typename T>
void PhantomMovie<T>::addShiftBarrelDeformation(Image<T> &movie)
{
	// Temporal frame with original data
	Image<T> tmp = Image<T>(xdim, ydim);
	T x_center = xdim / (T)2;
	T y_center = ydim / (T)2;
	for (size_t n = 0; n < ndim; ++n) {
		size_t framePixels = xdim * ydim;
		size_t frameBytes = framePixels * sizeof(T);
		memcpy(tmp.data.data, &(movie.data[n * framePixels]), frameBytes);
		T k1 = k1_start + n * (k1_end - k1_start) / (ndim-1);
		T k2 = k2_start + n * (k2_end - k2_start) / (ndim-1);
		std::cout << "k1 = " << k1 << " k2 = " << k2 << std::endl;
		T x_shift = skipShift ? 0 : shiftX(ndim - n - 1); // 'reverse' the order (see doc)
		T y_shift = skipShift ? 0 : shiftY(ndim - n - 1); // 'reverse' the order (see doc)
		if (!skipShift) std::cout << "shiftX = " << x_shift << " shiftY = " << y_shift << std::endl;
		for (size_t y = 0; y < ydim; ++y) {
			T y_norm = ((T)y - y_center + (shiftAfterBarrel ? 0 : y_shift)) / y_center;
			for (size_t x = 0; x < xdim; ++x) {
				T x_norm = ((T)x - x_center + (shiftAfterBarrel ? 0 : x_shift)) / x_center;
				T r_out = sqrt(x_norm*x_norm + y_norm*y_norm);
				T r_out_2 = r_out * r_out;
				T r_out_4 = r_out_2 * r_out_2;
				T scale = (1 + k1 * r_out_2 + k2 * r_out_4);
				T x_new = (x_norm * scale * x_center) + x_center
						+ (shiftAfterBarrel ? x_shift : 0);
				T y_new = (y_norm * scale * y_center) + y_center
						+ (shiftAfterBarrel ? y_shift : 0);
				size_t index = n * ydim * xdim + y * xdim + x;
				movie.data[index] = bilinearInterpolation(tmp, x_new, y_new);
			}
		}
	}
}

template<typename T>
void PhantomMovie<T>::addShift(Image<T> &movie)
{
	// Temporal frame with original data
	Image<T> tmp = Image<T>(xdim, ydim);
	for (size_t n = 0; n < ndim; ++n) {
		std::cout << "Applying shift " << n << std::endl;
		size_t framePixels = xdim * ydim;
		size_t frameBytes = framePixels * sizeof(T);
		memcpy(tmp.data.data, &(movie.data[n * framePixels]), frameBytes);
		for (size_t y = 0; y < ydim; ++y) {
			for (size_t x = 0; x < xdim; ++x) {
				T x_new = x + shiftX(ndim - n - 1); // 'reverse' the order (see doc)
				T y_new = y + shiftY(ndim - n - 1); // 'reverse' the order (see doc)
				size_t index = n * ydim * xdim + y * xdim + x;
				movie.data[index] = bilinearInterpolation(tmp, x_new, y_new);
			}
		}
	}
}

template<typename T>
void PhantomMovie<T>::generateGrid(Image<T> &movie)
{
	for (size_t n = 0; n < ndim; ++n) {
		std::cout << "Generating grid " << n << std::endl;
		size_t n_offset = n * xdim * ydim;
		// add rows
		for (size_t y = ystep; y < ydim; y+=ystep) {
			for (int t = -(thickness / 2); t < std::ceil(thickness/2.0); ++t) {
				if ((y >= thickness/2) && (y + t < ydim)) {
					size_t y_offset = (y + t) * xdim;
					for (size_t x = 0; x < xdim; ++x) {
						size_t index = n_offset + y_offset + x;
						movie.data[index] = 1;
					}
				}
			}
		}
		// add columns
		for (size_t x = xstep; x < xdim; x+=xstep) {
			for (int t = -(thickness / 2); t < std::ceil(thickness/2.0); ++t) {
				if ((x >= thickness/2) && (x + t < xdim)) {
					size_t x_offset = (x + t);
					for (size_t y = 0; y < ydim; ++y) {
						size_t index = n_offset + x_offset + y * xdim;
						movie.data[index] = 1;
					}
				}
			}
		}
	}
}

template<typename T>
void PhantomMovie<T>::run()
{
	std::cout << xdim  << " " << ydim << " " << ndim << " | "
			<< xstep << " " << ystep << " | "
			<< thickness << " " << fn_out << std::endl;
	Image<T> movie(xdim, ydim, 1, ndim);
	generateGrid(movie);
	if (!skipShift && skipBarrel) {
		addShift(movie);
	}
	if (!skipBarrel) {
		addShiftBarrelDeformation(movie);
	}
	movie.write(fn_out);
}

template class PhantomMovie<float>;
