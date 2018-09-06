#!/usr/bin/env python2
"""/***************************************************************************
 *
 * Authors:     Carlos Oscar Sorzano
 *
 * CSIC
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
"""

import os
import xmipp_base

class ScriptPDBSelect(xmipp_base.XmippScript):
    def __init__(self):
        xmipp_base.XmippScript.__init__(self)
        
    def defineParams(self):
        self.addUsageLine('Select/Exclude alpha helices or beta sheets from a PDB. '
                          'It is assumed that the secondary structure information is written '
                          'in the header')
        ## params
        self.addParamsLine(' -i <pdb>          : PDB file to process')
        self.addParamsLine(' -o <pdb>          : Output PDB')
        self.addParamsLine('[--keep_alpha]     : Keep alpha helices')
        self.addParamsLine('[--keep_beta]      : Keep beta helices')
        self.addParamsLine('[--exclude_alpha]  : Exclude alpha helices')
        self.addParamsLine('[--exclude_beta]   : Exclude beta helices')
        ## examples
        self.addExampleLine('   xmipp_pdb_select -i myfile.pdb -o onlyalpha.pdb --keep_alpha')
            

    def readPDB(self,fnIn):
        self.allAtoms = []
        self.sse = []
        with open(fnIn) as f:
            lines = f.readlines()
        for line in lines:
            # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/framepdbintro.html
            try:
                if line.startswith('ATOM '):
                    self.allAtoms.append(line)
                elif line.startswith('HELIX '):
                    self.sse.append(('Helix',line[19],int(line[21:25]),int(line[33:37])))
                elif line.startswith('SHEET '):
                    self.sse.append(('Sheet',line[21],int(line[22:26]),int(line[33:37])))
            except:
                pass

    def sseForResidue(self,atomLine):
        try:
            residueNumber = int(atomLine[22:26])
            chain = atomLine[21]
            for (ssType, ssChain, ss0, ssF) in self.sse:
                if chain == ssChain and residueNumber>=ss0 and residueNumber<=ssF:
                    return ssType
            return "None"
        except:
            pass

    def selectAtoms(self):
        self.selectedAtoms = []
        for atomLine in self.allAtoms:
            ssType = self.sseForResidue(atomLine)
            ok = False
            if ssType == "None":
                if self.exclude_alpha or self.exclude_beta:
                    ok=True
            elif ssType == "Helix":
                if self.keep_alpha:
                    ok=True
            elif ssType == "Sheet":
                if self.keep_beta:
                    ok=True
            if ok:
                self.selectedAtoms.append(atomLine)

    def run(self):
        fnIn = self.getParam('-i')
        fnOut = self.getParam('-o')
        self.keep_alpha = self.checkParam('--keep_alpha')
        self.keep_beta = self.checkParam('--keep_beta')
        self.exclude_alpha = self.checkParam('--exclude_alpha')
        self.exclude_beta = self.checkParam('--exclude_beta')
        self.readPDB(fnIn)
        self.selectAtoms()
        with open(fnOut,"w") as f:
            for line in self.selectedAtoms:
                f.write("%s"%line)

if __name__ == '__main__':
    ScriptPDBSelect().tryRun()
