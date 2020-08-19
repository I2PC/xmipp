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

#include <iomanip>
#include "pdb_label_from_volume.h"
#include "core/xmipp_image.h"
#include "core/metadata.h"

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
    std::ifstream fh_pdb;
    std::ofstream fh_out(fn_out);
    fh_pdb.open(fn_pdb.c_str());

    MetaData mdmean;
    size_t objId;
    objId = mdmean.addObject();

    if (!fh_pdb)
        REPORT_ERROR(ERR_IO_NOTEXIST, fn_pdb);

    double suma=0, sumaP=0;
    int numA=0;


    while (!fh_pdb.eof())
    {
        // Read an ATOM line
        std::string line;
        getline(fh_pdb, line);
        if (line == "")
        {
            fh_out << line << " \n";
            continue;
        }
        std::string kind = line.substr(0,4);
        if (kind != "ATOM" && kind !="HETA")
        {
            fh_out << line << " \n";
            continue;
        }

        // Extract atom type and position
        // Typical line:
        // ATOM    909  CA  ALA A 161      58.775  31.984 111.803  1.00 34.78
        std::string atom_type = line.substr(13,2);
        double x = textToFloat(line.substr(30,8));
        double y = textToFloat(line.substr(38,8));
        double z = textToFloat(line.substr(46,8));

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
        float atomN=0.0;
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

                            //Positivity
                            if (A3D_ELEM(inputVol,k, i, j) < 0)
                            {
                                value = -A3D_ELEM(inputVol,k, i, j);
                                atomP += atomP;
                            }
                            else
                                atomP += A3D_ELEM(inputVol,k, i, j);

                            //Negativity
                            if (A3D_ELEM(inputVol,k, i, j) > 0)
                            {
                                value = -A3D_ELEM(inputVol,k, i, j);
                                atomN += atomN;
                            }
                            else
                                atomN += A3D_ELEM(inputVol,k, i, j);

                        }
                    }
                    else
                    {
                        if ( (rdiffModule2<radius2) || (k==ka && i==ia && j==ja))
                        {
                            atomS+=A3D_ELEM(inputVol,k, i, j);
                            ++cont;

                            //Positivity
                            if (A3D_ELEM(inputVol,k, i, j) < 0)
                            {
                                value = -A3D_ELEM(inputVol,k, i, j);
                                atomP += value;
                            }
                            else
                                atomP += A3D_ELEM(inputVol,k, i, j);


                            //Negativity
                            if (A3D_ELEM(inputVol,k, i, j) > 0)
                            {
                                value = -A3D_ELEM(inputVol,k, i, j);
                                atomN += atomN;
                            }
                            else
                                atomN += A3D_ELEM(inputVol,k, i, j);

                        }
                    }

                }
            }
        }

        if (atomS>=0)
            atomS = atomP;
        else
            atomS = atomN;

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

        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << atomS;
        std::string s = stream.str();

        if (atomS<0)
            line.replace(55, 5, s, 0, 5);
        else
            line.replace(56, 4, s, 0, 4);
//		    std::cout << line << std::endl;

        fh_out << line << " \n";
    }

    double mean = suma/numA;
    double meanA = sumaP/numA;
    std::cout << "mean value: = " << mean << std::endl;
    std::cout << "absolute mean value: = " << meanA << std::endl;

    mdmean.setValue(MDL_VOLUME_SCORE1, mean, objId);
    mdmean.setValue(MDL_VOLUME_SCORE2, meanA, objId);
    mdmean.write(fnMD);

    // Close file
    fh_pdb.close();
    fh_out.close();

}


/* Run --------------------------------------------------------------------- */
void ProgPdbValueToVol::run()
{
    produceSideInfo();
    show();
    computeProteinGeometry();
}
