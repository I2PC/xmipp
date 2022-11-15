/***************************************************************************
 *
 * Authors:    Carlos Oscar             coss@cnb.csic.es
 * 			   David Herreros Calero    dherreros@cnb.csic.es
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

#include <fstream>
#include <iterator>
#include <numeric>
#include <algorithm>
#include "data/cpu.h"
#include "compare_views.h"
#include "data/projection.h"
#include "data/fourier_projection.h"

// Params definition =======================================================
// -i ---> V2 (paper) / -r --> V1 (paper)
void ProgCompareViews::defineParams() {
	addUsageLine("Compute the deformation that properly fits two volumes using spherical harmonics");
	addParamsLine("   -v1 <volume>                        : First volume to compare");
	addParamsLine("   -v2 <volume>                        : Second volume to compare");
	addParamsLine("  [-o <image=\"\">]                    : Output correlation image");
	addParamsLine("  [--degstep <d=5.0>]                  : Degrees step size for rot and tilt angles");
	addParamsLine("  [--thr <N=-1>]                       : Maximal number of the processing CPU threads");
	addExampleLine("xmipp_compare_views -v1 vol1.vol -v2 vol2.vol -o corr_img.xmp");
}

// Read arguments ==========================================================
void ProgCompareViews::readParams() {
    std::string aux;
	fnVol1 = getParam("-v1");
	fnVol2 = getParam("-v2");
	fnImgOut = getParam("-o");
	degstep = getDoubleParam("--degstep");

	if (fnImgOut=="")
		fnImgOut="Rot_tilt_corr_map.xmp";

	V1.read(fnVol1);
	V1().setXmippOrigin();
	V2.read(fnVol2);
	V2().setXmippOrigin();

	// Update degstep to have evenly spaced points in the interval [0,180]
	degstep = 360. / ROUND(360./degstep);
	
	int size_rot = 360./degstep;
	int size_tlt = 180./degstep;
	tilt_v.resize(size_tlt + 1);
	rot_v.resize(size_rot + 1);
	CorrImg().initZeros(size_rot + 1, size_tlt + 1);
	CorrImg().setXmippOrigin();

	std::generate(tilt_v.begin(), tilt_v.end(), [&, n = -degstep] () mutable { return n+=degstep; });
    std::generate(rot_v.begin(), rot_v.end(), [&, n = -degstep] () mutable { return n+=degstep; });

    int threads = getIntParam("--thr");
    if (0 >= threads) {
        threads = CPU::findCores();
    }
    m_threadPool.resize(threads);
}

// Show ====================================================================
void ProgCompareViews::show() {
	if (verbose==0)
		return;
	std::cout
	        << "First volume:         " << fnVol1         << std::endl
			<< "Second volume:        " << fnVol2         << std::endl
			<< "Output image:         " << fnImgOut       << std::endl
			<< "Degree step:          " << degstep        << std::endl
	;

}

void ProgCompareViews::computeCorrImage(int i)
{
    Projection P1, P2;
	auto &mCorrImg = CorrImg();
    auto &mV1 = V1();
    auto &mV2 = V2();
    auto &mP1 = P1();
    auto &mP2 = P2();
    int size_x = XSIZE(mV1);
    int size_y = YSIZE(mV1);
	auto rot = rot_v[i];
	for (int j = 0; j < tilt_v.size(); j++)
	{
		projectVolume(mV1, P1, size_y, size_x, rot, tilt_v[j], 0.);
		projectVolume(mV2, P2, size_y, size_x, rot, tilt_v[j], 0.);
		DIRECT_A2D_ELEM(mCorrImg, i, j) = correlationIndex(mP1, mP2);
	}
}

// Run =====================================================================
void ProgCompareViews::run() {
	auto futures = std::vector<std::future<void>>();
	futures.reserve(V1().zdim);

	auto routine = [this](int thrId, int i) {
        computeCorrImage(i);
    };

    for (int i=0; i<rot_v.size(); i++)
    {
        futures.emplace_back(m_threadPool.push(routine, i));
    }

	for (auto &f : futures) {
        f.get();
    }

    CorrImg.write(fnImgOut);
}
