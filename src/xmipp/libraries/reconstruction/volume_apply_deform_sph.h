/***************************************************************************
 *
 * Authors:    David Herreros Calero             dherreros@cnb.csic.es
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

#ifndef _PROG_VOL_APPLY_SPH_DEFORM
#define _PROG_VOL_APPLY_SPH_DEFORM

#include <core/xmipp_program.h>
#include <core/xmipp_image.h>


class ProgApplyVolDeformSph: public XmippProgram
{
private:
    /** PDB file */
    FileName fn_vol;

    /** Deformation coefficients list */
    FileName fn_sph;

    /** Output fileroot */
    FileName fn_out;

    /** Vector containing the deformation coefficients */
	std::vector<double> clnm;

    /** Vector containing the degrees and Rmax of the basis */
	std::vector<double> basisParams;

    /** Zernike and SPH coefficients vectors */
    Matrix1D<int> vL1;
    Matrix1D<int> vN;
    Matrix1D<int> vL2;
    Matrix1D<int> vM;

public:
    /** Params definitions */
    void defineParams() override;

    /** Read from a command line. */
    void readParams() override;

    /** Show parameters. */
    void show() const override;

    /** Run. */
    void run() override;

    /** Read Nth line of file */
    std::string readNthLine(int N) const;

    /** Convert String to Vector */
    std::vector<double> string2vector(std::string const &s) const;

    /** Fill degree and order vectors */
    void fillVectorTerms();

};
//@}
#endif