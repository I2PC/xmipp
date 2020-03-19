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

#include "pdb_from_volume.h"

#include <core/args.h>

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

/* Usage ------------------------------------------------------------------- */

void ProgPdbValueToVol::defineParams()
{
    addUsageLine("Put a volume value to PDB file.");
    addExampleLine("   xmipp_volume_from_pdb -i 1o7d.pdb --sampling 1.6");
    addExampleLine("   xmipp_transform_filter -i 1o7d.vol -o 1o7dFiltered.vol --fourier low_pass 10 raised_cosine 0.05 --sampling 1.6");

    addParamsLine("  --pdb <pdb_file>                   : File to process");
	addParamsLine("  --vol <vol_file=\"\">              : Input volume");
	addParamsLine("  [--mask <vol_file=\"\">]            : Input mask (to calculate inside mask)");
	addParamsLine("  -o  <file>                         : Modified PDB");
    addParamsLine("   --sampling <Ts=1>                 : Sampling rate (Angstroms/pixel)");
    addParamsLine("   [--origin <...>]                  : Origin of the volume --origin x y z");
    addParamsLine("   [--radius <radius=1>]             : Considered as radius of the atom (Angstroms)");
}
/* Read parameters --------------------------------------------------------- */
void ProgPdbValueToVol::readParams()
{
    fn_pdb = getParam("--pdb");
	fnVol = getParam("--vol");
	fnMask = getParam("--mask");
	fn_out=getParam("-o");
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
	inputVol = V();
//	V().setXmippOrigin();

	if (defOrig)
	{
		STARTINGZ(inputVol) = -textToInteger(origin[2]);
		STARTINGY(inputVol) = -textToInteger(origin[1]);
		STARTINGX(inputVol) = -textToInteger(origin[0]);
	}

	if (withMask)
	{
		M.read(fnMask);
		inputMask = M();
	}

}

/* Compute protein geometry ------------------------------------------------ */
void ProgPdbValueToVol::computeProteinGeometry()
{
    std::ifstream fh_pdb;
    std::ofstream fh_out(fn_out);
    fh_pdb.open(fn_pdb.c_str());
//    fh_out.open(fn_out.c_str());
    if (!fh_pdb)
        REPORT_ERROR(ERR_IO_NOTEXIST, fn_pdb);

    double suma=0;
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
//		std::cout << "x: = " << x << "y: = " << y << "z: = " << z<< std::endl;

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

		std::cout << "k0: = " << k0 << "kF: = " << kF << std::endl;
		std::cout << "i0: = " << i0 << "iF: = " << iF << std::endl;
		std::cout << "j0: = " << j0 << "jF: = " << jF << std::endl;

        // Fill the volume with this atom
        float atomS=0;
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
						if ( (rdiffModule2<radius2 || (k==ka && i==ia && j==ja)) && (inputMask(k, i , j)>0.05) )
						{
							atomS+=A3D_ELEM(inputVol,k, i, j);
							++cont;
						}
                	}
                	else
                	{
						if ( (rdiffModule2<radius2) || (k==ka && i==ia && j==ja))
						{
							std::cout << "k i j  =  " << k << i << j << std::endl;
							atomS+=A3D_ELEM(inputVol,k, i, j);
							++cont;
						}
                	}

                }
            }
        }
		std::cout << "suma: = " << atomS << std::endl;
		std::cout << "conteo: = " << cont << std::endl;
        atomS=atomS/cont;
//        if (atomS>3)
//		   atomS=3.00;
//        if (atomS<-3)
// 		   atomS=-3.00;
//        if (atomS>0 && atomS<0.25)
// 		   atomS=0.00;
//        if (atomS<0 && atomS>-0.25)
// 		   atomS=0.00;
//        std::cout << "antes: = " << atomS << std::endl;
//        normalizo
//        if (atomS>=0)
//        	atomS = (-0.333333*atomS)+1;
//        else
//        	atomS = (-0.333333*atomS)-1;
        std::cout << "despues: = " << atomS << std::endl;


        ++numA;
        suma+=atomS;
//		std::cout << "media: = " << atomS << std::endl;

        std::stringstream ss;
        ss<<atomS;
        std::string str1=ss.str();
//        std::string str1 = std::to_string (atomS);
        if (atomS<0)
            line.replace(55, 5, str1, 0, 5);
        else
        	line.replace(56, 4, str1, 0, 4);
//		std::cout << "valor: = " << str1 << std::endl;
//		std::cout << "linea: = " << line << std::endl;

		fh_out << line << " \n";
    }

    std::cout << "la suma total es: = " << suma/numA << std::endl;

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
