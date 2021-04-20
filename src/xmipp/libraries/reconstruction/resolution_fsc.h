/***************************************************************************
 *
 * Authors:    Jose Luis Vilas (jlvilas@cnb.csic.es)
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

#ifndef _PROG_RES_FSC
#define _PROG_RES_FSC

#include <core/xmipp_program.h>
#include <core/xmipp_fftw.h>
#include <core/metadata_extension.h>

class ProgResolutionFsc : public XmippProgram
{
public:

    FileName    fn_ref, fn_root,fn_img, fn_out;
    double       sam;
    double       max_sam, min_sam;
    bool        do_dpr, do_set_of_images, do_o, do_rfactor;

    FileName    fn_sel;
    bool        apply_geo;

    void defineParams();
    void readParams();

    void writeFiles(const FileName &fnRoot,
                        const MultidimArray<double> &freq,
                        const MultidimArray<double> &frc,
                        const MultidimArray<double> &frc_noise,
                        const MultidimArray<double> &dpr,
                        const MultidimArray<double> &error_l2,
    					double max_sam, bool do_dpr, double rFactor);

    bool process_img();

    bool process_sel();

    void run();


};

#endif
