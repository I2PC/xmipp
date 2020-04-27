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

#include "superiorization_reconstruct_base.h"
#include "superiorization_proximity_types.h"

#include <core/alglib/ap.h>

#define OUTPUT_PRINTING
#define PRINT_N_LOOP
#define PRINT_BETA
//#define DEBUG
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
 addParamsLine("  -i <tiltseries>         : Metadata with the set of images in the tilt series, and their tilt angles");
 addParamsLine("  -o <output>             : Filename for the resulting reconstruction.");
 addParamsLine("  -a <float>              : variable used to compute the magnitude of the perturbation (beta = b * a^l).");
 addParamsLine("  -N <N = 0>              : Maximum number of internal iterations for truly superiorization. By default, there is not superiorization (N = 0)");
 addParamsLine("  [-b <b = 1.0>]            : variable used to compute the magnitude of the perturbation (beta = b * a^l).");
 addParamsLine("  [--zsize <z=-1>]        : Z size of the reconstructed volume. If -1, then a cubic volume is assumed");
 addParamsLine("  [-e <epsilon = -1.0>]   : Tolerance value for the reconstruction.");
 addParamsLine("  [--NI <iter = -1>]      : Defines that a counter would be used instead of proximity function. A positive integer value is mandatory. By default, its value is -1, indicating a proximity function would be used.");
 addParamsLine("  [--rec <rec = ART>]     : Defines the reconstruction algorithm (rec = ART).");
 addParamsLine("  [--lart <lart = 1.0>]   : Defines the lambda for ART (lambda = 1.0).");
 addParamsLine("  [--lsart <lsart = 1.0>] : Defines the lambda for SART (lambda = 1.0).");
 addParamsLine("  [--atl <atl = ATL0>]    : Defines the way the counter l is updated. By default, l follows the original superiorization algorithm (atl = ATL0).");
 addParamsLine("  [--phi <phi = ITV>]     : Defines the second criterion. By default, phi is isometric Total Variation (phi = TV).");
 addParamsLine("  [--rP <rP = 3>]         : Defines the size/radius for kernel P.");
 addParamsLine("  [--sP <sP = 1.0>]       : Defines the sigma for kernel P.");
 addParamsLine("  [--kP <kP = 1.0>]       : Defines the weight kappa for kernel P.");
 addParamsLine("  [--rM <rM = 3>]         : Defines the size/radius for kernel M.");
 addParamsLine("  [--sM <sM = 1.0>]       : Defines the sigma for kernel M.");
 addParamsLine("  [--kM <kM = 1.0>]       : Defines the weight kappa for kernel M.");
 addParamsLine("  [--Pr <prox = L2SQ>]    : Defines the proximity function for the solution. By default, the proximity function is the L2-squared norm.");
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
 a            = getDoubleParam("-a");
 b            = getDoubleParam("-b");
 N            = getIntParam("-N");
 
 Zsize        = getIntParam("--zsize");
 epsilon      = getDoubleParam("-e");
 iter_cnt     = getIntParam("--NI");
 rec_method   = getParam("--rec");
 lart         = getDoubleParam("--lart");
 lsart        = getDoubleParam("--lsart");
 l_method     = getParam("--atl");
 phi_method   = getParam("--phi");
 rP           = getIntParam("--rP");
 sP           = getDoubleParam("--sP");
 kP           = getIntParam("--kP");
 rM           = getIntParam("--rM");
 sM           = getDoubleParam("--sM");
 kM           = getIntParam("--kM");
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
    std::cout << "Superiorized ";
    switch(B.getType()){
        case reconType::ART: std::cout << "ART";break;
        default: "NA";
       }
    std::cout << " with the following arguments:" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Input tilt series: " << fnTiltSeries << std::endl;
    std::cout << "Output reconstruction: " << fnOut << std::endl;
    if(epsilon<0.0)
       std::cout << "# of iterations: " << iter_cnt << std::endl;
    else
       std::cout << "Epsilon: " << epsilon << std::endl;
    if(Zsize == -1)
       std::cout << "The depth size (Z size) would determined when reading the input projection file" << std::endl;
    else
       std::cout << "The depth size (Z size): " << Zsize << std::endl;
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
    
    std::cout << "phi: " << phi.getName() << std::endl;
    if(phi.getShortName() == "ATV"){
       std::cout<<"Radius kernel P: " << rP << std::endl;
       std::cout<<"std. dev. kernel P: " << sP << std::endl;
       std::cout<<"Weight (kappa)  kernel P: " << kP << std::endl;
       std::cout<<"Radius kernel M: " << rM << std::endl;
       std::cout<<"std. dev. kernel M: " << sM << std::endl;
       std::cout<<"Weight (kappa) kernel M: " << kM << std::endl;
      }
    
    if(iter_cnt == -1){
       std::cout << "Pr: ";
       switch(B.getPrType()){
           case proximityType::L2SQ:
                std::cout<< "L2-squared" << std::endl;
                break;
           default:
                std::cout<< "NONE selected" << std::endl;
                break;
          }
      }
    std::cout << "########################################################" << std::endl;
   }
}

/**
**
** Default constructor.
** Method to initialize variables and status before running but after reading from the command line.
**
*/
ProgReconsSuper::ProgReconsSuper()
{
 mode_l   = lmode::ATL0;
 a        =  0.0;
 b        =  0.0;
 epsilon  = -1.0;
 N        =  0;
 Zsize    = -1;
 lart     =  1.0;
 lsart    =  1.0;
 rP       =  3;
 sP       =  1.0;
 kP       =  1.0;
 rM       =  3;
 sM       =  1.0;
 kM       =  1.0;
 iter_cnt = -1;
 
 l_method = "ATL0";
 
 rec_method = "ART";
 B.set(rec_method);
 B.setParam(lart);
 
 phi_method = "ITV";
 phi.set(phi_method);
 
 pr_method = "L2SQ";
 B.setPr(pr_method);
}

/**
**
** The Destructor, ha ha ha.
**
*/
ProgReconsSuper::~ProgReconsSuper()
{
 /*
 Pr.set(pr_method);
 phi.set(phi_method);
 B.set(rec_method);
 */
}

/**
**
** Method to initialize variables and status before executing algorithm/method but after reading arguments from the command line.
**
*/
void ProgReconsSuper::checkArgsInfo()
{
 //
 // Checking for the existence of input file
 //
 std::ifstream ifile(fnTiltSeries.c_str());
 if((bool)ifile == 0){
    std::cerr << "\e[1m\e[31m" << "ERROR: the input file does not exist." << "\e[m\e[0m" << std::endl;
    usage("-i", true);
   }

 //
 // Checking the value for Zsize
 //
 if(Zsize == -1)
    std::cerr << "\e[1m\e[35m" << "WARNING: zsize was not set. The default value would be equal to the horizontal dimension (X size) of the projection data." << "\e[m\e[0m" << std::endl;

 //
 // Checking the value for epsilon
 //
 if(epsilon < 0.0 && iter_cnt<0){
    std::cerr << "\e[1m\e[36m" << "ERROR: it is necessary to select a stop criterion: by proximity function or number of iterations." << "\e[m\e[0m" << std::endl;
    std::cerr << "\e[1m\e[36m" << "       The values of epsilon and the number of iterations are negative." << "\e[m\e[0m" << std::endl;
    usage("-e",true);
   }

 //
 // Checking the value for a
 //
 if(a <= 0.0 || a >= 1.0){
    std::cerr << "\e[1m\e[36m" << "ERROR: a has to meet 0.0 < a < 1.0." << "\e[m\e[0m" << std::endl;
    usage("-a",true);
   }

 //
 // Checking the value set for N
 //
 if(N < 0){
    std::cerr << "\e[1m\e[36m" << "ERROR: N has to be positive" << "\e[m\e[0m" << std::endl;
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
         std::cout << l_method << "\e[1m\e[31m" <<" ERROR, is not a valid option, ATL0 will be used instead." << "\e[m\e[0m" << std::endl;
        }
      }
   }

 //
 // Checking the value for reconstruction
 //
 if((rec_method == std::string("ART")) && (B.getType()!=reconType::ART)){
    B.set("ART");
    B.setParam(lart);
   }
 
 if((rec_method == std::string("SART")) && (B.getType()!=reconType::SART)){
    B.set("SART");
    B.setParam(lart);
   }
 /*
 if(B.valid(rec_method) == false){
    std::cerr << "\e[1m\e[36m" << "ERROR: invalid method for reconstruction" << "\e[m\e[0m" << std::endl;
    usage("--rec", true);
   }
 */
 //
 // Checking the value for second criterion (phi)
 //
 if(phi.valid(phi_method) == false){
    std::cerr << "\e[1m\e[36m" << "ERROR: invalid second criterion for optimization" << "\e[m\e[0m" << std::endl;
    usage("--phi", true);
   }
 if(phi_method == std::string("ATV") && (phi.getShortName()!=String("ATV"))){
    phi.set("ATV");
    
    if(rP<3){
       std::cerr << "\e[1m\e[36m" << "ERROR: the size/radius for kernel P smaller than 3" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
    if(rM<3){
       std::cerr << "\e[1m\e[36m" << "ERROR: the size/radius for kernel M smaller than 3" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
    if(sP <= 0.0){
       std::cerr << "\e[1m\e[36m" << "ERROR: the std. dev. for kernel P <= 0" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
    if(sM <= 0.0){
       std::cerr << "\e[1m\e[36m" << "ERROR: the std. dev. for kernel M <= 0" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
    if(kP <= 0.0){
       std::cerr << "\e[1m\e[36m" << "ERROR: the weight for kernel P <= 0" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
    if(kM <= 0.0){
       std::cerr << "\e[1m\e[36m" << "ERROR: the weight for kernel M <= 0" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
   }
 if(phi_method == std::string("ITV") && (phi.getShortName()!=String("ITV")))
    phi.set("ITV");
 if(phi_method == std::string("WTV") && (phi.getShortName()!=String("WTV")))
    phi.set("WTV");
 
 //
 // Checking whether to use counter or proximity function
 //
 if(iter_cnt == 0){
    std::cerr << "\e[1m\e[36m" << "ERROR: invalid value for the number of iterations for reconstruction" << "\e[m\e[0m" << std::endl;
    usage("--NI", true);
   }
 //
 // Checking the value for the proximity function
 //
 /*
 if(B.validPr(pr_method) == false){
    std::cerr << "\e[1m\e[36m" << "ERROR: invalid Proximity function." << "\e[m\e[0m" << std::endl;
    usage("--Pr", true);
   }
*/
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
void ProgReconsSuper::run()
{
 //
 // Processing the input arguments
 //
 checkArgsInfo();

 //
 // Showing the arguments' values used by the program
 show();

 //
 // Reading the Input projections through the Metadata file
 //
 MetaData mdTS;
 FileName fnImg;
 mdTS.read(fnTiltSeries);

 //
 // Initialize data, variables,
 // reconstruction algorithm, second criterion
 // and proximity function
 /*
  *
  * Load the tilt series into memory
  *
  */
 Image<double> I, V;
 MultidimArray<double> TS;
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
// for(int k=0; k<ZSIZE(TS); k++)
//     std::cout << "Angle[ "<< k <<" ]: "<< tiltAngles[k] << std::endl;
// 
// exit(0);
 double maxAngle = std::numeric_limits<double>::min();
 double minAngle = std::numeric_limits<double>::max();
 for(ushort i = 0; i < tiltAngles.size(); i ++){
     if(tiltAngles[i] > maxAngle)
        maxAngle = tiltAngles[i];
     if(tiltAngles[i] < minAngle)
        minAngle = tiltAngles[i];
    }

 /*
  *
  * Reconstruction Algorithm
  *
  */
 switch(B.getType()){
     case reconType::ART:
          std::cout<<"Initializing ART"<<std::endl;
          B.setParam(lart);
          B.init(TS.xdim,Zsize,tiltAngles);
          break;
     case reconType::SART:
          std::cout<<"Initializing SART"<<std::endl;
          B.setParam(lsart);
          B.init(TS.xdim,Zsize,tiltAngles);
          break;
     default:
          std::cout << "\e[1m\e[31m" << "Unknown Reconstruction Algorithm" << "\e[m\e[0m" << std::endl;
          std::cout << "\e[1m\e[31m" << "Aborting the execution !!!" << "\e[m\e[0m" << std::endl;
          exit(0);
    }
 
 /*
  *
  * Allocating memory and resizing the final reconstruction and
  * the necessary vectors for superiorization (non-ascending and z)
  *
  */
 MultidimArray<double> x,v,z;
 //std::cout<<"INPUT SIZE: "<<TS.xdim<<" , "<<TS.ydim<<" , "<<TS.zdim<<std::endl;
 x.resize(Zsize,TS.ydim,TS.xdim); // resize --> Ndim, Zdim, Ydim, Xdim (Zdim, Ydim, Xdim)
 v.resize(Zsize,TS.ydim,TS.xdim);
 z.resize(Zsize,TS.ydim,TS.xdim);

 /*
  *
  * Superiorization Section
  *
  */
 //
 // Initializing Reconstruction
 //
 memset(x.data,0,x.xdim*x.ydim*x.zdim*sizeof(double)); //x.initZeros(Zsize,TS.ydim,TS.xdim);
 if(phi.getShortName() == "ATV"){
    phi.init(x,sP,(unsigned short)rP,kP,
               sM,(unsigned short)rM,kM,minAngle,maxAngle);
    if( ((rP%2) != (x.xdim%2)) || ((rP%2) != (x.ydim%2)) || ((rP%2) != (x.zdim%2)) ){
       std::cout << "\e[1m\e[31m" << "ERROR: Mismatch parity between kernel P and reconstruction" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
    if( ((rM%2) != (x.xdim%2)) || ((rM%2) != (x.ydim%2)) || ((rM%2) != (x.zdim%2)) ){
       std::cout << "\e[1m\e[31m" << "ERROR: Mismatch parity between kernel M and reconstruction" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
   }
 else
    phi.init(x);

k=0;
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
#ifdef OUTPUT_PRINTING
     if(N>0)
        std::cout << "Perturbing solution x" << std::endl;
#endif
     n = 0;
     double phixk = phi(x);
     double phiz;
     
     phi.preupdate(x);
     
     while(n < N){
         phi.nav(x,v); // Method to obtain the nonascending vector for phi at x^k
         loop = true;
         while(loop){
             beta = b * pow(a,l);
             l += 1;
             z = x + beta*v;
             // z in Delta and phi(z)<=phi(x^k)
             bool inDelta = true;
/*
             //
             // Section to force positivity
             //
             if(SuperPositivity == true)
                for(int i_z=0;inDelta==true && i_z<x.zdim;i_z++)
                    for(int i_y=0;inDelta==true && i_y<x.ydim;i_y++)
                        for(int i_x=0;inDelta==true && i_x<x.xdim;i_x++)
                            if(x.data[i_z*x.dim*y.dim + i_y*x.xdim + i_x] < 0.0){
                               inDelta == false;
                               break;
                              }
*/
             phiz = phi(z);
             if(inDelta && (phiz <= phixk)){
                n += 1;
                x = z;
                loop = false;
               }
#ifdef PRINT_BETA
             std::cout << "\e[1m\e[31m" << "phi(x)" << "\e[m\e[0m" << ": " << std::setprecision(12) << phixk;
             std::cout << "\e[1m\e[31m" << ", phi(z)" << "\e[m\e[0m" << ": " << std::setprecision(12) << phiz;
             std::cout << "\e[1m\e[31m" << ", l" << "\e[m\e[0m" << ": " << l << "\e[1m\e[31m" << ", beta" << "\e[m\e[0m" << ": " << std::setprecision(12) << beta << std::endl;
#endif
            }
#ifdef PRINT_N_LOOP
         std::cout << "\e[1m\e[36m" << "Internal Iteration" << "\e[m\e[0m" << ": " << n << std::endl;
#endif
        }
     std::cout << "\e[1m\e[33m" << "Executing Operator B" << "\e[m\e[0m" << std::endl;
     B(x,TS,tiltAngles,k);
     phi.postupdate(x);
     k += 1;
     std::cout << "\e[1m\e[32m" << "Iteration" << "\e[m\e[0m" << ": " << k << "\e[1m\e[32m" << " Finished" << "\e[m\e[0m" << std::endl;
     if(iter_cnt > 0){
        if(k >= iter_cnt)
           break;
       }
     else{
        double lsqv = B.Pr(x,TS,tiltAngles);
        fprintf(stdout,"Pr: %20.16f\n",lsqv);
        if(lsqv < epsilon)
           break;
       }
#ifdef DEBUG
     std::cout << "k: " << k <<std::endl;
#endif
    }
 fprintf(stdout,"Pr: %20.16f\n",B.Pr(x,TS,tiltAngles));
 //
 // Finishing the reconstruction
 //
 v.clear();
 z.clear();
 std::cout << "\e[1m\e[35m" << "Saving the Result in " << fnOut << "\e[m\e[0m" << std::endl;
 Image<double> reconstructedvolume;
 reconstructedvolume() = x;
 x.clear();
 reconstructedvolume.write(fnOut);
 std::cout << "Procedure finished. Exiting the Program" << std::endl;
}
#undef DEBUG
