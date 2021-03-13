/***************************************************************************
 *
 * Authors:    Vahid Abrishami (vabrishami@cnb.csic.es)
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
#include "core/xmipp_program.h"
#include "core/metadata_vec.h"

class ProgTest: public XmippProgram
{
    FileName fnIn,fnOut;
    bool extracPart;

    void defineParams()
    {
        addUsageLine("This program convert Metadata to XML");
        addParamsLine("== Basic ==");
        addParamsLine(" -i <metadata> : metadata input for testing");
        addParamsLine(" -o <file> : output XML file");
        addParamsLine(" [--extractParticlesMD] : If we want to process a MetaData that comes from ExtractParticles protocol and some of the particles have been rejected");
        addExampleLine("Produce XML file from a metadata for benchmark",false);
        addExampleLine("xmipp_metadata_XML -i DefaultFamily_extract_list.xmd  -o XMLFile.xml");

    }
    void readParams()
    {
        fnIn = getParam("-i");
        fnOut = getParam("-o");
        extracPart = checkParam("--extractParticlesMD");

    }

    void run()
    {

        MetaDataVec MD, sortedMD;
        std::ofstream fhOut;
        StringVector blockList;
        fhOut.open(fnOut.c_str());
        fhOut<<"<particlepicking>"<<std::endl;
        int x,y;
        FileName micName,sTemp;

        if (!extracPart)
        {
            getBlocksInMetaDataFile(fnIn,blockList);
            for (size_t i=0; i<blockList.size(); i++)
            {
                MD.read(blockList[i]+"@"+fnIn);
                sTemp=blockList[i];
                micName=sTemp.removeUntilPrefix("_");
                fhOut<<"<micrograph id=\""<<micName<<"\">"<<std::endl;
                for (size_t objId : MD.ids())
                {
                    MD.getValue(MDL_XCOOR, x, objId);
                    MD.getValue(MDL_YCOOR, y, objId);
                    fhOut<<"<coordinate x=\""<<x<<"\" y=\""<<y<<"\"/>"<<std::endl;
                }
                fhOut<<"</micrograph>"<<std::endl;
            }
        }
        else
        {
        	FileName name,newName,nodirName;
            MD.read(fnIn);
            MD.removeDisabled();

            std::cout << fnIn << std::endl;
            sortedMD.sort(MD,MDL_MICROGRAPH);

            sortedMD.getValue(MDL_MICROGRAPH, name, MD.firstRowId());
            nodirName=name.removeDirectories();
            nodirName=nodirName.removeAllExtensions();
            fhOut<<"<micrograph id=\""<<nodirName<<"\">"<<std::endl;

            sortedMD.getValue(MDL_XCOOR, x, sortedMD.firstRowId());
            sortedMD.getValue(MDL_YCOOR, y, sortedMD.firstRowId());
            fhOut<<"<coordinate x=\""<<x<<"\" y=\""<<y<<"\"/>"<<std::endl;

            for (auto idIt = sortedMD.ids().begin(); idIt != sortedMD.ids().end(); ++idIt)
            {
            	if (*idIt == MD.firstRowId())
                    ++idIt;

                size_t objId = *idIt;

            	sortedMD.getValue(MDL_MICROGRAPH, newName, objId);

                if (name == newName)
                {
                	sortedMD.getValue(MDL_XCOOR, x, objId);
                	sortedMD.getValue(MDL_YCOOR, y, objId);
                    fhOut<<"<coordinate x=\""<<x<<"\" y=\""<<y<<"\"/>"<<std::endl;
                }
                else
                {
                    fhOut<<"</micrograph>"<<std::endl;
                    name = newName;
                    nodirName=name.removeDirectories();
                    nodirName=nodirName.removeAllExtensions();
                    fhOut<<"<micrograph id=\""<<nodirName<<"\">"<<std::endl;
                    sortedMD.getValue(MDL_XCOOR, x, objId);
                    sortedMD.getValue(MDL_YCOOR, y, objId);
                    fhOut<<"<coordinate x=\""<<x<<"\" y=\""<<y<<"\"/>"<<std::endl;
                }
            }

        }
        fhOut<<"</micrograph>"<<std::endl;
        fhOut<<"</particlepicking>"<<std::endl;
        fhOut.close();
    }
};
