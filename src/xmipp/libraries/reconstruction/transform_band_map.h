/***************************************************************************
 *
 * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#ifndef _PROG_TRANSFORM_BAND_MAP
#define _PROG_TRANSFORM_BAND_MAP

#include <core/xmipp_program.h>
#include <core/xmipp_filename.h>
#include <core/multidim_array.h>

#include <vector>

/**@defgroup TransformBandMap
   @ingroup ReconsLibrary */
//@{

class ProgTransformBandMap : public XmippProgram
{
public:
    virtual void readParams() override;
    virtual void defineParams() override;
    virtual void show() const override;
    virtual void run() override;

    FileName        fnOutput;
    size_t          nBands;
    size_t          imageSizeX;
    size_t          imageSizeY;
    size_t          imageSizeZ;
    double          lowResLimit;
    double          highResLimit;

private:
    static std::vector<double> computeArithmeticBandFrecuencies(double lowResLimit,
                                                                double highResLimit,
                                                                size_t nBands );
    static std::vector<double> computeGeometricBandFrecuencies( double lowResLimit,
                                                                double highResLimit,
                                                                size_t nBands );

    static MultidimArray<int> computeBands( const size_t nx, 
                                            const size_t ny, 
                                            const size_t nz, 
                                            const std::vector<double>& frecuencies );

};

#endif