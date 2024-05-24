/***************************************************************************
 *
 * Authors:    Jose Luis Vilas 					  jlvilas@cnb.csic.es
 * 			   Oier Lauzirika Zarrabeitia         oierlauzi@bizkaia.eu
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

#ifndef __TOMO_TWOFOLD_ALIGN
#define __TOMO_TWOFOLD_ALIGN

#include <core/metadata_vec.h>
#include <core/xmipp_program.h>
#include <core/xmipp_filename.h>
#include <data/fourier_projection.h>
#include <data/sampling.h>

#include <vector>

class ProgTomoVolumeAlignTwofold : public XmippProgram
{

private:
    /** Filenames */
    FileName fnInMetadata;
    /** Filenames */
    FileName fnOutMetadata;
    /** Angular sampling rate in degrees*/
    double angularSamplingRate;
    /** Max tilt angle in degrees*/
    double maxTiltAngle;
    /** Maximum frequency for comparisons */
    double maxFrequency;
    /** Padding factor */
    double padding;
    int interp;
    /** Input metadata*/
    MetaDataVec inputVolumesMd;
    /** Output metadata */
    MetaDataVec alignmentMd;
    /** Input volumes */
    std::vector<Image<double>> inputVolumes;
    /** Projectors for volumes */
    std::vector<FourierProjector> projectors;
    /** Central slice */
    std::vector<MultidimArray<double>> centralProjections;
    /** Sampling on sphere */
    Sampling sphereSampling;

private:
    // --------------------------- INFO functions ----------------------------
    void readParams() override;
    void defineParams() override;


    // --------------------------- HEAD functions ----------------------------
    double twofoldAlign(std::size_t i, std::size_t j, double &rot, double &tilt, double &psi);
    static double computeSquareDistance(const MultidimArray<double> &x, const MultidimArray<double> &y);

    // --------------------------- I/O functions -----------------------------
    void defineSampling();
    void readVolumes();
    
    // --------------------------- MAIN --------------------------------------
    void run() override;

};

#endif // __TOMO_TWOFOLD_ALIGN
