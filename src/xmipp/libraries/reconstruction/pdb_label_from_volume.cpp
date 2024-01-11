/***************************************************************************
 *
 * Authors:
 *
 * Erney Ramirez-Aportela (eramirea@cnb.csic.es)
 * Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#include <fstream>
#include <iomanip>
#include "pdb_label_from_volume.h"
#include "core/xmipp_image.h"
#include "core/metadata_vec.h"
#include "data/pdb.h"

/* Usage ------------------------------------------------------------------- */

void ProgPdbValueToVol::defineParams()
{
    addUsageLine("Put a volume value to PDB file.");
    addExampleLine("   xmipp_pdb_label_from_volume --pdb 1o7d.pdb --vol volume.vol -o pdb_label.pdb --sampling 1.6");

    addParamsLine("  --pdb <pdb_file>                   : File to process");
	addParamsLine("  --vol <vol_file=\"\">              : Input volume");
	addParamsLine("  [--mask <vol_file=\"\">]           : Input mask (to calculate inside mask)");
	addParamsLine("  -o  <file>                         : Modified output PDB");
    addParamsLine("   --sampling <Ts=1>                 : Sampling rate (Angstroms/pixel)");
    addParamsLine("   [--origin <...>]                  : Origin of the volume --origin x y z");
    addParamsLine("   [--radius <radius=0.8>]           : Considered as radius of the atom (Angstroms)");
    addParamsLine("   [--md <output=\"params.xmd\">]    : Save mean and absolute mean output");
}
/* Read parameters --------------------------------------------------------- */
void ProgPdbValueToVol::readParams()
{
    fn_pdb = getParam("--pdb");
	fnVol = getParam("--vol");
	fnMask = getParam("--mask");
	fn_out=getParam("-o");
    fnMD = getParam("--md");
    Ts = getDoubleParam("--sampling");
    radius = getDoubleParam("--radius");
    withMask = checkParam("--mask");
    defOrig = checkParam("--origin");
    getListParam("--origin", origin);

}

/* Show -------------------------------------------------------------------- */
void ProgPdbValueToVol::show()
{
    if (verbose==0)
        return;
    std::cout << "PDB file:           " << fn_pdb           << std::endl
    << "Output:       " << fn_out << std::endl
    << "Sampling rate:        " << Ts          << std::endl
    << "Origin:               ";
    for (size_t i=0; i<origin.size(); ++i)
    	std::cout << origin[i] << " ";
    std::cout << std::endl
    << "Radius:               " << radius      << std::endl;
}

/* Produce Side Info ------------------------------------------------------- */
void ProgPdbValueToVol::produceSideInfo()
{
    Image<double> V, M;
    V.read(fnVol);
    V().setXmippOrigin();
    inputVol = V();
    inputVol.setXmippOrigin();

    if (withMask)
    {
        M.read(fnMask);
        inputMask = M();
        inputMask.setXmippOrigin();
    }

    //Origin of the volume
    if (defOrig)
    {
        STARTINGZ(inputVol) = -textToInteger(origin[2]);
        STARTINGY(inputVol) = -textToInteger(origin[1]);
        STARTINGX(inputVol) = -textToInteger(origin[0]);

        STARTINGZ(inputMask) = -textToInteger(origin[2]);
        STARTINGY(inputMask) = -textToInteger(origin[1]);
        STARTINGX(inputMask) = -textToInteger(origin[0]);
    }

    else
    {
        STARTINGZ(inputVol) = 0;
        STARTINGY(inputVol) = 0;
        STARTINGX(inputVol) = 0;

        STARTINGZ(inputMask) = 0;
        STARTINGY(inputMask) = 0;
        STARTINGX(inputMask) = 0;
    }

}

/* Compute protein geometry ------------------------------------------------ */
void ProgPdbValueToVol::computeProteinGeometry()
{
    PDBRichPhantom pdbIn;
    PDBRichPhantom pdbOut;
    pdbIn.read(fn_pdb);

    MetaDataVec mdmean;
    size_t objId;
    objId = mdmean.addObject();

    double suma=0, sumaP=0;
    int numA=0;
    for (const auto& atomIn : pdbIn.atomList)
    {
        const auto& x = atomIn.x;
        const auto& y = atomIn.y;
        const auto& z = atomIn.z;

        // Correct position
        Matrix1D<double> r(3);
        VECTOR_R3(r, x, y, z);
        r *= 1/Ts;

        // Characterize atom
        double radius2=radius*radius;

        // Find the part of the volume that must be updated
        int k0 = XMIPP_MAX(FLOOR(ZZ(r) - radius), STARTINGZ(inputVol));
        int kF = XMIPP_MIN(CEIL(ZZ(r) + radius), FINISHINGZ(inputVol));
        int i0 = XMIPP_MAX(FLOOR(YY(r) - radius), STARTINGY(inputVol));
        int iF = XMIPP_MIN(CEIL(YY(r) + radius), FINISHINGY(inputVol));
        int j0 = XMIPP_MAX(FLOOR(XX(r) - radius), STARTINGX(inputVol));
        int jF = XMIPP_MIN(CEIL(XX(r) + radius), FINISHINGX(inputVol));

        int ka = XMIPP_MAX(FLOOR(ZZ(r)), STARTINGZ(inputVol));
        int ia = XMIPP_MAX(FLOOR(YY(r)), STARTINGZ(inputVol));
        int ja = XMIPP_MAX(FLOOR(XX(r)), STARTINGZ(inputVol));


        float atomS=0.0;
        float atomP=0.0;
        float value=0.0;
        int cont=0;
        for (int k = k0; k <= kF; k++)
        {
            double zdiff=ZZ(r) - k;
            double zdiff2=zdiff*zdiff;
            for (int i = i0; i <= iF; i++)
            {
                double ydiff=YY(r) - i;
                double zydiff2=zdiff2+ydiff*ydiff;
                for (int j = j0; j <= jF; j++)
                {
                    double xdiff=XX(r) - j;
                    double rdiffModule2=zydiff2+xdiff*xdiff;
                    if (withMask)
                    {
                        if ( (rdiffModule2<radius2 || (k==ka && i==ia && j==ja)) && (inputMask(k, i , j)>0.00001) )
                        {
                            atomS += A3D_ELEM(inputVol,k, i, j);
                            ++cont;

                            //Absolute
                            if (A3D_ELEM(inputVol,k, i, j) < 0)
                            {
                                value = -A3D_ELEM(inputVol,k, i, j);
                                atomP += value;
                            }
                            else
                                atomP += A3D_ELEM(inputVol,k, i, j);

                        }
                    }
                    else
                    {
                        if ( (rdiffModule2<radius2) || (k==ka && i==ia && j==ja))
                        {
                            atomS+=A3D_ELEM(inputVol,k, i, j);
                            ++cont;

                            //Absolute
                            if (A3D_ELEM(inputVol,k, i, j) < 0)
                            {
                                value = -A3D_ELEM(inputVol,k, i, j);
                                atomP += value;
                            }
                            else
                                atomP += A3D_ELEM(inputVol,k, i, j);

                        }
                    }

                }
            }
        }

        if (atomS>=0)
            atomS = atomP;
        else
            atomS = -atomP;

        if (atomS != 0)
            atomS=atomS/cont;
        else
            atomS=0.00;

        if (atomP != 0)
            atomP=atomP/cont;
        else
            atomP=0.00;


        ++numA;
        suma+=atomS;
        sumaP+=atomP;

        auto atomOut = atomIn;
        atomOut.occupancy = atomS;
        pdbOut.addAtom(atomOut);
    }

    double mean = suma/numA;
    double meanA = sumaP/numA;
    std::cout << "mean value: = " << mean << std::endl;
    std::cout << "absolute mean value: = " << meanA << std::endl;

    mdmean.setValue(MDL_VOLUME_SCORE1, mean, objId);
    mdmean.setValue(MDL_VOLUME_SCORE2, meanA, objId);
    mdmean.write(fnMD);

    pdbOut.write(fn_out);
}


/* Run --------------------------------------------------------------------- */
void ProgPdbValueToVol::run()
{
    produceSideInfo();
    show();
    computeProteinGeometry();
}
