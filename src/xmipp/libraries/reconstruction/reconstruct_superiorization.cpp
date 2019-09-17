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
/****************** Definition of Local Methods *******************************/
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
 addParamsLine("  -i <tiltseries>      : Metadata with the set of images in the tilt series, and their tilt angles");
 addParamsLine("  -o <output>          : Filename for the resulting reconstruction.");
 addParamsLine("  -e <epsilon>         : Tolerance value for the reconstruction.");
 addParamsLine("  --zsize <z=-1>       : Z size of the reconstructed volume. If -1, then a cubic volume is assumed");
 addParamsLine("  -a <float>           : variable used to compute the magnitude of the perturbation (beta = b * a^l).");
 addParamsLine("  -b <float>           : variable used to compute the magnitude of the perturbation (beta = b * a^l).");
 addParamsLine("  -N <N = 0>           : Maximum number of internal iterations for truly superiorization. By default, there is not superiorization (N = 0)");
 addParamsLine("  [--rec <rec = ART>]  : Defines the reconstruction algorithm (rec = ART).");
 addParamsLine("  [--atl <atl = ATL0>] : Defines the way the counter l is updated. By default, l follows the original superiorization algorithm (atl = ATL0).");
 addParamsLine("  [--phi <phi = TV>]   : Defines the second criterion. By default, phi is Total Variation (phi = TV).");
 addParamsLine("  [--Pr <prox = L2SQ>] : Defines the proximity function for the solution. By default, the proximity function is the squared Euclidean norm.");
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
 epsilon      = getDoubleParam("-e");
 Zsize        = getIntParam("--zsize");
 a            = getDoubleParam("-a");
 b            = getDoubleParam("-b");
 N            = getIntParam("-N");
 rec_method   = getParam("--rec");
 l_method     = getParam("--atl");
 phi_method   = getParam("--phi");
 pr_method    = getParam("--Pr");
}

/**
**
** Method to inform the user how the software will run
**
*/
void ProgReconsSuper::show()
{
 if(verbose > 0){
    std::cout << "########################################################" << std::endl;
    std::cout << "Superiorized "<< "NA" << " with the following arguments:" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Input tilt series: " << fnTiltSeries << std::endl;
    std::cout << "Output reconstruction: " << fnOut << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "Zsize: " << Zsize << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "ATL: ";
    switch(mode_l){
        case lmode::ATL0:std::cout << "ATL0" << std::endl;
             break;
        case lmode::ATL1:std::cout << "ATL1" << std::endl;
             break;
        case lmode::ATL2:std::cout << "ATL2" << std::endl;
             break;
       }
    std::cout << "phi: " << "NA" << std::endl;
    std::cout << "Pr: " << "NA" << std::endl;
    std::cout << "########################################################" << std::endl;
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
 epsilon = 0.0;
 N = 0;
 Zsize = 0;

 rec_method = "ART";
 phi_method = "ITV";
 l_method = "ATL0";
 pr_method = "L2SQ";

 Pr.set(pr_method);
 phi.set(phi_method);
 //B.set(rec_method);
}

/**
**
** Method to initialize variables and status before executing algorithm/method but after reading arguments from the command line.
**
*/
void ProgReconsSuper::produceSideInfo()
{
 //
 // Checking for the existence of input file
 //
 std::ifstream ifile(fnTiltSeries.c_str());
 if((bool)ifile == 0){
    std::cerr << "ERROR: the input file does not exist." << std::endl;
    usage("-i", true);
   }

 //
 // Checking the value for epsilon
 //
 if(epsilon < 0.0){
	 std::cerr << "ERROR: epsilon has to be positive." << std::endl;
	 usage("-e",true);
   }

 //
 // Checking the value for a
 //
 if(a <= 0.0 || a >= 1.0){
	 std::cerr << "ERROR: a has to meet 0.0 < a < 1.0." << std::endl;
	 usage("-a",true);
   }

 //
 // Checking the value set for N
 //
 if(N < 0){
    std::cerr << "ERROR: N has to be positive" << std::endl;
    usage("-N", true);
   }

 //
 // Setting the method to update l
 //
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

 //
 // Checking the value for reconstruction
 //
 if(B.valid(rec_method) == false){
    std::cerr << "ERROR: invalid method for reconstruction" << std::endl;
    usage("--rec", true);
   }

 //
 // Checking the value for second criterion (phi)
 //
 if(phi.valid(phi_method) == false){
    std::cerr << "ERROR: invalid second criterion for optimization" << std::endl;
    usage("--phi", true);
   }

 //
 // Checking the value for the proximity function
 //
 if(Pr.valid(pr_method) == false){
    std::cerr << "ERROR: invalid Proximity function." << std::endl;
    usage("--Pr", true);
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
 // Load the tilt series into memory
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
 //x().initZeros(Zsize,YSIZE(I()),XSIZE(I()));
 //v().initZeros(Zsize,YSIZE(I()),XSIZE(I()));
 //z().initZeros(Zsize,YSIZE(I()),XSIZE(I()));
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
         phi.nav(x,v); // Method to obtain the nonascending vector for phi at x^k
		 loop = true;
         while(loop){
             beta = b * pow(a,l);
             l += 1;
             z = x + beta*v;
             // z in Delta and phi(z)<=phi(x^k)
             if(phi(z) <= phi(x)){
            	n += 1;
            	x = z;
            	loop = false;
               }
            }
        }
//     B(x,TS,tiltAngles);
     k += 1;
     if(Pr(x) < epsilon){
        break;
       }
#ifdef DEBUG
	 std::cout << "k: " << k <<std::endl;
#endif
    }
}
#undef DEBUG
