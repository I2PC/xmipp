/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
 *              Oier Lauzirika  (olauzirika@cnb.csic.es)
 *
 * Spanish Research Council for Biotechnology, Madrid, Spain
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

#include "masking_base.h"
#include <sys/stat.h>
#include <core/metadata_extension.h>
#include "core/transformations.h"
#include <random>
#include <limits>
#include <type_traits>


template<typename T>
void spherical3DMask(MultidimArray<T> &vol, MultidimArray<T> &mask, int softRange)
{
	Image<double> maskImg;

	int boxsize = XSIZE(vol)*0.5;
	auto halfboxsize = boxsize/2;
	auto halfboxsize2 = halfboxsize*halfboxsize;

	auto smoothLim = halfboxsize + softRange;


	mask.initZeros(vol);

	long n=0;
	for (int k=0; k<boxsize; k++)
	{
		int k2 = (k-halfboxsize);
		k2 = k2*k2;
		for (int i=0; i<boxsize; i++)
		{
			int i2 = i-halfboxsize;
			int i2k2 = i2*i2 +k2 ;
			for (int j=0; j<boxsize; j++)
			{
				int j2 = (j- halfboxsize);
				auto radius = sqrt(i2k2 + j2*j2);
				if (radius<halfboxsize)
				{
					DIRECT_MULTIDIMELEM(mask, n) = 1.0;
				}
				else
				{
					if (radius<smoothLim)
					{
						DIRECT_MULTIDIMELEM(mask, n) = 1 + cos( (smoothLim - radius)*PI / softRange) ;
					}
				}
				n++;
			}
		}
	}
}

template<typename T>
void spherical3DMaskIdx(MultidimArray<T> &vol, std::vector<size_t> &maskIdx)
{
	int boxsize = XSIZE(vol)*0.5;
	auto halfboxsize = boxsize/2;
	
	long n=0;
	for (int k=0; k<boxsize; k++)
	{
		int k2 = (k-halfboxsize);
		k2 = k2*k2;
		for (int i=0; i<boxsize; i++)
		{
			int i2 = i-halfboxsize;
			int i2k2 = i2*i2 +k2 ;
			for (int j=0; j<boxsize; j++)
			{
				int j2 = (j- halfboxsize);
				if (sqrt(i2k2 + j2*j2)>halfboxsize)
				{
					maskIdx.push_back(n);
				}
				n++;
			}
		}
	}
}

template<typename T>
void normalizeSubtomo(MultidimArray<T> &subtomo, std::vector<size_t> &maskIdx)
{
	MultidimArray<double> maskNormalize;

	T sumVal = 0;
	T sumVal2 = 0;

	auto counter = maskIdx.size();
	for (size_t i=0; i<maskIdx.size(); i++)
	{
			auto val = DIRECT_MULTIDIM_ELEM(subtomo, maskIdx[i]);
			sumVal += val;
			sumVal2 += val*val;

	}

	T mean;
	T sigma2;
	mean = sumVal/counter;
	sigma2 = sqrt(sumVal2/counter - mean*mean);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(subtomo)
	{
		DIRECT_MULTIDIM_ELEM(subtomo, n) -= mean;
		DIRECT_MULTIDIM_ELEM(subtomo, n) /= sigma2;
	}
}

