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
#include <core/alglib/ap.h>

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Local Methods ********************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Method to print the input format for running the software and adding required arguments.
**
*/
void ProgReconsSuper::defineParams()
{
 addUsageLine("Reconstruction of tomography tilt series using superiorization");
 addParamsLine("  -i <tiltseries>    : Metadata with the set of images in the tilt series, and their tilt angles");
 addParamsLine("  -o <output>        : Filename for the resulting reconstruction.");
 addParamsLine("  --zsize <z=-1>     : Z size of the reconstructed volume. If -1, then a cubic volume is assumed");
 addParamsLine("  -a <float>         : a variable used to compute the magnitude of the perturbation (beta = b * a^l).");
 addParamsLine("  -b <float>         : b variable used to compute the magnitude of the perturbation (beta = b * a^l).");
 addParamsLine("  -N <N = 0>         : Maximum number of internal iterations for truly superiorization. By default, there is not superiorization (N = 0)");
 addParamsLine("  --atl <atl = ATL0> : Defines the way the counter l is updated. By default, l follows the original superiorization algorithm (atl = ATL0).");
}

/**
**
** Method to read the arguments from the command line
**
*/
void ProgReconsSuper::readParams()
{
 int m_l;

 fnTiltSeries = getParam("-i");
 fnOut        = getParam("-o");
 Zsize        = getIntParam("--zsize");
 l_method     = getParam("--atl");
 a            = getDoubleParam("-a");
 b            = getDoubleParam("-b");
 N            = getIntParam("-N");
}

/**
**
** Method to inform the user how the software will run
**
*/
void ProgReconsSuper::show()
{
 if(verbose > 0){
    std::cout << "Superiorized "<< "NA" << " with the following arguments:" << std::endl;
    std::cout << "Input tilt series: " << fnTiltSeries << std::endl;
    std::cout << "Output reconstruction: " << fnOut << std::endl;
    std::cout << "Zsize: " << Zsize << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "atl: " << l_method << std::endl;
    std::cout << "phi: " << "NA" << std::endl;
    std::cout << "Pr: " << "NA" << std::endl;
   }
}

/**
**
** Method to initialize variables and status before running but after reading from the command line.
**
*/
ProgReconsSuper::ProgReconsSuper()
{
 mode_l = lmode::ATL0;
 a = 0.0;
 b = 0.0;
 N = 0;
 Zsize = 0;
 phi_method = "";
 nav_method = "";
 l_method = "";
 pr_method = "";
}

/**
**
** Method to initialize variables and status before executing algorithm/method but after reading arguments from the command line.
**
*/
void ProgReconsSuper::produceSideInfo()
{
 if(l_method=="ATL0")
    mode_l = lmode::ATL0;
 else{
    if(l_method=="ATL1")
       mode_l = lmode::ATL1;
    else{
       if(l_method=="ATL2")
          mode_l = lmode::ATL2;
       else{
         std::cout << l_method <<" is not a valid option, ATL0 will be used instead." << std::endl;
        }
      }
   }

 if(N < 0){
    std::cerr << "ERROR: N has to be positive" << std::endl;
    usage("-N", true);
   }
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Main Method *********************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Method to superiorize an iterative reconstruction algorithm
**
*/
//#define DEBUG
void ProgReconsSuper::run()
{
 //
 // Processing the input arguments
 //
 produceSideInfo();

 //
 // Showing the arguments' values used by the program
 show();

 //
 // Reading the Input projections through the Metadata file
 //
 MetaData mdTS;
 FileName fnImg;
 mdTS.read(fnTiltSeries);

 /*
  *
  * Processing the variables according to the input arguments
  *
  */
 //
 // Load the tilt series in memory
 //
 Image<double> I, V;
 MultidimArray<double> TS,x,v,z;
 std::vector<double> tiltAngles;

 int k=0;
 FOR_ALL_OBJECTS_IN_METADATA(mdTS){
     mdTS.getValue(MDL_IMAGE,fnImg,__iter.objId);
     I.read(fnImg);
     if(ZSIZE(TS)==0){
  	    TS.initZeros(mdTS.size(),YSIZE(I()),XSIZE(I()));
  	    if(Zsize<0)
  		   Zsize=XSIZE(I());
  	    V().initZeros(Zsize,YSIZE(I()),XSIZE(I()));
       }
     memcpy(&A3D_ELEM(TS,k++,0,0),&I(0,0),sizeof(double)*MULTIDIM_SIZE(I()));
    }
 //
 // Loading tilting anlges (orientations)
 //
 mdTS.getColumnValues(MDL_ANGLE_TILT,tiltAngles);
 //for (int k=0; k<ZSIZE(TS); k++)
//	   std::cout << tiltAngles[k] << std::endl;

/*
 *
 * Superiorization Section
 *
 */
 //
 // Further Initializing variables
 //
int n = 0;
int l = 0;
bool loop;
double beta;
 //
 // Starting the actual superiorization process
 //
 while(true){
	 // Handling the counter l
	 switch(mode_l){
	     case lmode::ATL1:
	    	 l = k;
	    	 break;
	     case lmode::ATL2:
	    	 l = (int)round(alglib::randomreal()*( l- k )) + k;
	    	 break;
	    }
     // Preparing the variables for the inner loop
	 n = 0;
	 while(n < N){
//         v = phi.nav(x); // Method to obtain the nonascending vector for phi at x^k
		 loop = true;
         while(loop){
             beta = b * pow(a,l);
             l += 1;
//             z = x + beta*v;
             // z in Delta and phi(z)<=phi(x^k)
/*
             if(phi(z) <= phi(x)){
            	n += 1;
            	x = z;
            	loop = false;
               }
*/
            }
        }
//     recon = B;
     k += 1;
#ifdef DEBUG
	 std::cout << "k: " << k <<std::endl;
#endif
    }
}
#undef DEBUG
