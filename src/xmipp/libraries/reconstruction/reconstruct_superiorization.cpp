/***************************************************************************
 *
 * Authors:     Edgar Garduno Angeles (edgargar@ieee.org)
 *
 * Department of Computer Science, Institute for Applied Mathematics
 * and Systems Research (IIMAS), National Autonomous University of
 * Mexico (UNAM)
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

#include "reconstruct_superiorization.h"

void ProgReconsSuper::defineParams()
{
   addUsageLine("Reconstruction of tomography tilt series using superiorization");
   addParamsLine("  -i <tiltseries>    : Metadata with the set of images in the tilt series, and their tilt angles");
   addParamsLine("  --zsize <z=-1>     : Z size of the reconstructed volume. If -1, then a cubic volume is assumed");
}

void ProgReconsSuper::readParams()
{
   fnTiltSeries = getParam("-i");
   Zsize = getIntParam("--zsize");
}

void ProgReconsSuper::show()
{
   if(verbose > 0){
      std::cout << "Input tilt series: " << fnTiltSeries << std::endl;
      std::cout << "Zsize: " << Zsize << std::endl;
   }
}

//#define DEBUG
void ProgReconsSuper::run()
{
   show();
   MetaData mdTS;
   FileName fnImg;
   mdTS.read(fnTiltSeries);

   // Load the tilt series in memory
   Image<double> I, V;
   MultidimArray<double> TS;
   int z=0;
   FOR_ALL_OBJECTS_IN_METADATA(mdTS)
   {
       mdTS.getValue(MDL_IMAGE,fnImg,__iter.objId);
       I.read(fnImg);
       if (ZSIZE(TS)==0)
       {
    	   TS.initZeros(mdTS.size(),YSIZE(I()),XSIZE(I()));
    	   if (Zsize<0)
    		   Zsize=XSIZE(I());
    	   V().initZeros(Zsize,YSIZE(I()),XSIZE(I()));
       }
       memcpy(&A3D_ELEM(TS,z++,0,0),&I(0,0),sizeof(double)*MULTIDIM_SIZE(I()));
   }
   std::vector<double> tiltAngles;
   mdTS.getColumnValues(MDL_ANGLE_TILT,tiltAngles);
   //for (int k=0; k<ZSIZE(TS); k++)
//	   std::cout << tiltAngles[k] << std::endl;

   // Iterate over Y 2D subproblems
   I().resizeNoCopy(ZSIZE(TS),XSIZE(TS));
   for (int i=0; i<YSIZE(TS); i++)
   {
	   // Extract the 2D subproblem
	   for (int k=0; k<ZSIZE(TS); k++)
		   memcpy(&I(k,0),&A3D_ELEM(TS,k,i,0),sizeof(double)*XSIZE(TS));
#ifdef DEBUG
	   I.write("PPP2D.xmp");
	   I().printStats();
	   std::cout << "Press any key" << std::endl;
	   char c; std::cin >> c;
#endif
   }
}
#undef DEBUG
