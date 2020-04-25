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

#include "compare_pdb.h"

#include <core/args.h>

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

/* Usage ------------------------------------------------------------------- */

void ProgComparePdb::defineParams()
{
    addUsageLine("Compare two PDBs.");

    addParamsLine("  --pdb1 <pdb_file>                   : File to process");
	addParamsLine("  --pdb2 <pdb_file=\"\">              : Input volume");
	addParamsLine("  -o  <file>                         : Modified PDB");
}
/* Read parameters --------------------------------------------------------- */
void ProgComparePdb::readParams()
{
    fn_pdb1 = getParam("--pdb1");
    fn_pdb2 = getParam("--pdb2");
	fn_out=getParam("-o");

}

/* Compute protein geometry ------------------------------------------------ */
void ProgComparePdb::computeProteinGeometry()
{
	float value1, value2;
	int ntotal=0, npartial=0;

    std::ifstream fh_pdb1;
    std::ifstream fh_pdb2;
    std::ofstream fh_out(fn_out);
    fh_pdb1.open(fn_pdb1.c_str());
    fh_pdb2.open(fn_pdb2.c_str());
    if (!fh_pdb1)
        REPORT_ERROR(ERR_IO_NOTEXIST, fn_pdb1);

    while (!fh_pdb1.eof())
    {
        // Read an ATOM line
        std::string line1;
        std::string line2;
        getline(fh_pdb1, line1);
        getline(fh_pdb2, line2);
        if (line1 == "")
        {
            continue;
        }
        std::string kind = line1.substr(0,4);
        if (kind != "ATOM" && kind !="HETA")
        {
            continue;
        }
    	++ntotal;

        value1 = strtof(line1.substr(55,5).c_str(),0);
        value2 = strtof(line2.substr(55,5).c_str(),0);


        if ( (value1<0.4 && value1>=0) && (value2<0.25 && value2>=0) )
        {
    		fh_out << line1 << " \n";
        	++npartial;
        }

//        else  ( (value1>1.0) && (value2>0.5) )
//        {
//    		fh_out << line1 << " \n";
//        	++npartial;
//        }

//        else if ( (value1<-1.0) && (value2>0.3) )
//        {
//    		fh_out << line1 << " \n";
//        	++npartial;
//        }
//
//        else if ( (value1>0.4) && (value2<-0.2) )
//        {
//    		fh_out << line1 << " \n";
//        	++npartial;
//        }

        else
            continue;

    }



    // Close file
    fh_pdb1.close();
    fh_pdb2.close();
    fh_out.close();
    std::cout << "atom total: = " << ntotal << "atom mal: = " << npartial << std::endl;

}


/* Run --------------------------------------------------------------------- */
void ProgComparePdb::run()
{

    computeProteinGeometry();

}
