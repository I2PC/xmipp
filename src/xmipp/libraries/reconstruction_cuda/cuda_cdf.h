/***************************************************************************
 *
 * Authors:    Martin Horacek (horacek1martin@gmail.com)
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
#ifndef CUDA_CDF
#define CUDA_CDF

namespace Gpu {

/** Gpu version of Cumulative density function.
 * This function computes a table with the cumulative density function*/
template< typename T >
struct CDF {
	static constexpr size_t type_size = sizeof(T);

	T* d_V;
	T* d_x;
	T* d_probXLessThanx;

	size_t volume_size;
	T probStep;
	T multConst;
	T Nsteps;


	CDF(size_t volume_size, T multConst = 1.0, T probStep = 0.005);
	~CDF();

	void calculateCDF(const T*  d_filtered1, const T* d_filtered2);
	void calculateCDF(const T* d_S);

		// Functions must be public because they use device lambda
	void _calculateDifference(const T* __restrict__ d_filtered1, const T* __restrict__ d_filtered2);
	void _calculateSquare(const T* __restrict__ d_S);
	void _updateProbabilities();

private:
	void sort();

};

} // namespace Gpu

#endif // CUDA_CDF