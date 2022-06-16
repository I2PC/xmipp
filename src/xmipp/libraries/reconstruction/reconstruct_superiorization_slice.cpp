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

#include "reconstruct_superiorization_slice.h"

#include "superiorization_reconstruct_base.h"
#include "superiorization_proximity_types.h"

#include <fstream>
#include <iostream>
#include <istream>
#include <chrono>
#include <random>
#include <core/alglib/ap.h>

#define COLOR_OUTPUT
#define FILE_OUTPUT
//#define OUTPUT_NAV
//#define OUTPUT_Z
#define OUTPUT_PRINTING
#define OUTPUT_PROGRESS
//#define PRINT_N_LOOP
//#define PRINT_BETA
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
** Method to carry out the reconstruction of the map using a slice-by-slice scheme
**
*/
void ProgReconsSuperSlice::Reconstruct2D(MultidimArray<double> &X, MultidimArray<double> PD, std::vector<double> Angles)
{
 MultidimArray<double> x,v,z,P;
 int k;
 int n;
 int l;
 bool loop;
 bool flag_first = false;
 double beta;
#ifdef FILE_OUTPUT
 Image<double> outf;
#endif
 
 x.resize(X.ydim,X.xdim);
 v.resize(X.ydim,X.xdim);
 z.resize(X.ydim,X.xdim);

 if(phi.getShortName() == "ATV"){
    double maxAngle = std::numeric_limits<double>::min();
    double minAngle = std::numeric_limits<double>::max();
    
    for(ushort i = 0; i < Angles.size(); i ++){
        if(Angles[i] > maxAngle)
           maxAngle = Angles[i];
        if(Angles[i] < minAngle)
           minAngle = Angles[i];
       }
    phi.init(x,sigma,Ku,Me,delta_angle,minAngle,maxAngle);
   }
 else
    phi.init(x);
 
 flag_first = false;
#ifdef OUTPUT_PROGRESS
 fprintf(stdout,"Reconstructing Plane: %d (%6.2f %%)\n",slice,(100.0*slice)/PD.ydim);
 std::cout << std::flush;
#endif
 //
 // Initializing superiorization process
 //
 k = 0;
 n = 0;
 l = 0;

 std::ofstream outfPr;
 std::string oname;
 oname = "proximity.txt";
 outfPr.open(oname, std::ios::out);
 
 memset(x.data,0,x.xdim*x.ydim*sizeof(double));
 memset(v.data,0,x.xdim*x.ydim*sizeof(double));
 memset(z.data,0,x.xdim*x.ydim*sizeof(double));
 
 LoadSlice(X, slice, x);
 
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
     if(N>0){
     	  std::cout << "\e[1m\e[0;33m" << "Perturbing solution x" << "\e[m\e[0m";
     	  std::cout << std::endl;
     	 }
#endif
     n = 0;
     double phixk = phi(x);
     double phiz;
     phi.preupdate(x);
     while(n < N){
         v = phi.nav(x); // Method to obtain the nonascending vector for phi at x^k
         loop = true;
         while(loop){
             beta = b * pow(a,l);
             l += 1;
             z = x + beta*v;
             // z in Delta and phi(z) <= phi(x^k)
             bool inDelta = true;
             phiz = phi(z);
             if(inDelta && (phiz <= phixk)){
                n += 1;
                x = z;
                loop = false;
                if(flag_first){
                   flag_first = true;
                   std::cout << "\e[1m\e[34m" << "phi(x)" << "\e[m\e[0m";
                   std::cout <<  ": " << std::setprecision(12) << phixk;
                   std::cout << "\e[1m\e[34m" << ", phi(z)" << "\e[m\e[0m";
                   std::cout << ": " << std::setprecision(12) << phiz;
                   std::cout << "\e[1m\e[34m" << ", l" << "\e[m\e[0m";
                   std::cout << ": " << l;
                   std::cout << "\e[1m\e[34m" << ", beta" << "\e[m\e[0m";
                   std::cout << ": " << std::setprecision(12) << beta << std::endl;
                   std::cout << "k: "<< k << std::endl;
                  }
               }
#ifdef PRINT_BETA
     		    std::cout << "\e[1m\e[34m" << "phi(x)" << "\e[m\e[0m";
     		    std::cout <<  ": " << std::setprecision(12) << phixk;
     		    std::cout << "\e[1m\e[34m" << ", phi(z)" << "\e[m\e[0m";
     		    std::cout << ": " << std::setprecision(12) << phiz;
     		    std::cout << "\e[1m\e[34m" << ", l" << "\e[m\e[0m";
     		    std::cout << ": " << l;
     		    std::cout << "\e[1m\e[34m" << ", beta" << "\e[m\e[0m";
     		    std::cout << ": " << std::setprecision(12) << beta << std::endl;
#endif
            }
#ifdef PRINT_N_LOOP
     	   std::cout << "\e[1m\e[36m" << "Internal Iteration" << "\e[m\e[0m";
     	   std::cout << ": " << n << std::endl;
#endif
        }
     std::cout << "\e[1m\e[33m" << "Executing Operator B: " << slice << "\e[m\e[0m";
     std::cout << std::endl;
     B(x,PD,Angles,slice,k);
     phi.postupdate(x);
     k += 1;
     
     double lsqv = B.Pr(x,PD,Angles,slice);
     double tphi = phi(x);
     
     std::cout << "\e[1m\e[32m" << "Iteration" << "\e[m\e[0m";
     std::cout << ": " << k << " ";
     std::cout << "\e[1m\e[32m" << "Finished, Pr " << "\e[m\e[0m";
     std::cout << k << ": ";
     std::cout << std::setprecision(12) << lsqv;
     std::cout << "\e[1m\e[32m" << ", phi " << "\e[m\e[0m";
     std::cout << k << ": ";
     std::cout << std::setprecision(12) << tphi << std::endl;

     outfPr<<lsqv<<"\t";
     outfPr<<tphi<<"\n";
     
     if(iter_cnt > 0){
        if(k >= iter_cnt)
           break;
       }
     else{
        if(lsqv < epsilon)
           break;
       }
#ifdef DEBUG
     std::cout << "k: " << k <<std::endl;
#endif
    }
 
 outfPr<<k;
 outfPr.close();
 
 ReturnSlice(x, slice, X);
}

/*
**
** Method to read the Tilt Series information from a file and into a data structure
**
*/
void ProgReconsSuperSlice::ReadTiltSeries(std::vector<double>& Angles, MultidimArray<double>& TS)
{
 #ifdef XMIPP_21
 MetaDataVec mdTS;  // After 2021
#else
MetaData mdTS; // Before 2021
#endif
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
 Image<double> I;
 
 int k=0;
#ifdef XMIPP_21
 for(size_t objId : mdTS.ids()){
     mdTS.getValue(MDL_IMAGE, fnImg, objId);
     // xmipp_bundle/src/xmipp/applications/tests/function_tests/test_metadata_vec_main.cpp
#else
 FOR_ALL_OBJECTS_IN_METADATA(mdTS){
     mdTS.getValue(MDL_IMAGE,fnImg,__iter.objId);
#endif
     I.read(fnImg);
     if(ZSIZE(TS) == 0)
        TS.initZeros(mdTS.size(),YSIZE(I()),XSIZE(I()));
     memcpy(&A3D_ELEM(TS,k++,0,0),&I(0,0),sizeof(double)*MULTIDIM_SIZE(I()));
    }
 //
 // Loading tilt anlges (orientations)
 //
 mdTS.getColumnValues(MDL_ANGLE_TILT,Angles);
}

/*
**
** Method to create the 3D dataset from the information in the Tilt Series and the Zsize value
**
*/
void ProgReconsSuperSlice::Create3DImage(const MultidimArray<double>& TS, MultidimArray<double>& x) 
{
 x.resize(Zsize,TS.xdim); // resize --> Ndim, Zdim, Ydim, Xdim (Zdim, Ydim, Xdim)
 memset(x.data,0,x.xdim*x.ydim*x.zdim*sizeof(double));
 
 if(X0_rand == true){
    Pmin = std::numeric_limits<double>::max();
    Pmax = std::numeric_limits<double>::lowest();
    
    for(long j=0; j<(TS.xdim*TS.ydim*TS.zdim); j++){
        if(TS.data[j] > Pmax)
           Pmax = TS.data[j];
        else
           Pmin = TS.data[j];
       }
   }
 
 // construct a trivial random generator engine from a time-based seed:
 unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
 std::default_random_engine generator(seed);
#ifdef NORMAL_DIST
 std::normal_distribution<double> distribution(0.0,1.0);  // <--- Normal distribution with 0 mean and std. dev. = 1.0
#else
 std::uniform_real_distribution<double> distribution(Pmin,Pmax);
#endif 
 for(long j=0; j<(x.xdim*x.ydim*x.zdim); j++){
#ifdef NORMAL_DIST
     x.data[j] = Pmin + 0.5*std::fabs(Pmax-Pmin) + 0.25*std::fabs(Pmax-Pmin)*distribution(generator);
#else
     x.data[j] = distribution(generator);
#endif
    }
}

/*
**
** Method to create the 3D dataset from the information in the Tilt Series and the Zsize value
**
*/
void ProgReconsSuperSlice::Save3DImage(MultidimArray<double>& TS, MultidimArray<double>& x) 
{
 Image<double> reconstructedvolume;
 reconstructedvolume() = x;
 reconstructedvolume.write(fnOut);
}

/*
**
** Method to carry out the reconstruction of the map using a volume scheme
**
*/
void ProgReconsSuperSlice::LoadSlice(const MultidimArray<double>& x, const uint plane, MultidimArray<double>& S)
{
 memcpy(&S.data[0], &x.data[0], x.xdim*x.ydim*sizeof(double));
}

/*
**
** Method to carry out the reconstruction of the map using a volume scheme
**
*/
void ProgReconsSuperSlice::ReturnSlice(const MultidimArray<double>& S, const uint plane, MultidimArray<double>& x)
{
 memcpy(&x.data[0], &S.data[0], S.xdim*S.ydim*sizeof(double));
}

/*
**
** Method to compute the final Proximity function.
**
*/
double ProgReconsSuperSlice::Pr(const MultidimArray<double>& X,const MultidimArray<double>& P, const std::vector<double>& Angles)
{
 double lsqv = 0.0;
 
 //lsqv = B.Pr(x,P,Angles,plane);
 lsqv = B.Pr(X,P,Angles);
 
 return lsqv;
}

/*
**
** Method to compute the final phi function.
**
*/
double ProgReconsSuperSlice::Phi(const MultidimArray<double>& X, const MultidimArray<double>& P, const std::string type)
{
 double phiv = 0.0;
 
 //phiv = phi(x);
 SuperRegular<double> TV(type);
 phiv = TV(X);
 
 return phiv;
}

/*
**
** Method to print the input format for running the software and adding required arguments.
**
*/
void ProgReconsSuperSlice::defineParams()
{
 addUsageLine("Reconstruction slice by slice of tomography tilt series using superiorization 3D");
 addParamsLine("  -i <tiltseries>         : Metadata with the set of images in the tilt series, and their tilt angles");
 addParamsLine("  -o <output>             : Filename for the resulting reconstruction.");
 addParamsLine("  -a <float>              : variable used to compute the magnitude of the perturbation (beta = b * a^l).");
 addParamsLine("  -N <N = 0>              : Maximum number of internal iterations for truly superiorization. By default, there is not superiorization (N = 0)");
 addParamsLine("  -s <int>                : Slice to reconstruct.");
 addParamsLine("  [-b <b = 1.0>]          : variable used to compute the magnitude of the perturbation (beta = b * a^l).");
 addParamsLine("  [--zsize <z = -1>]      : Z size of the reconstructed volume. If -1, then a cubic volume is assumed");
 addParamsLine("  [-e <epsilon = -1.0>]   : Tolerance value for the reconstruction.");
 addParamsLine("  [--NI <iter = -1>]      : Defines a counter used instead of a proximity function. A positive integer value is mandatory. By default, its value is -1, indicating a proximity function would be used.");
 addParamsLine("  [--rec <rec = ART3>]    : Defines the reconstruction algorithm (ART2, ART3, SART2, or ART3), by default rec=ART.");
 addParamsLine("  [--lart <lart = 1.0>]   : Defines the lambda for ART (lambda = 1.0).");
 addParamsLine("  [--lsart <lsart = 1.0>] : Defines the lambda for SART (lambda = 1.0).");
 addParamsLine("  [--atl <atl = ATL0>]    : Defines the way the counter l is updated. By default, l follows the original superiorization algorithm (atl = ATL0).");
 addParamsLine("  [--phi <phi = ITV>]     : Defines the second criterion. By default, phi is isometric Total Variation (phi = TV).");
 addParamsLine("  [--sigma <sigma = 1>]   : Defines the sigma for Gaussian kernel.");
 addParamsLine("  [--Ku <Ku = 1.0>]       : Defines the weight Ku for kernel.");
 addParamsLine("  [--Me <Me = 1.0>]       : Defines the weight Me for kernel.");
 addParamsLine("  [--delta <delta = 1.0>] : Defines the angle sampling for kernel.");
 addParamsLine("  [--Pr <prox = L2SQ>]    : Defines the proximity function for the solution. By default, the proximity function is the L2-squared norm.");
 addParamsLine("  [--X0 <X0 = empty>]     : Defines an input file for the initial solution. This flag takes precedence over XR.");
 addParamsLine("  [--XR <XR = 0>]         : Defines a random initial solution with normal distribution with cero mean and std. dev. equal to one.");
}

/**
**
** Method to read the arguments from the command line
**
*/
void ProgReconsSuperSlice::readParams()
{
 fnTiltSeries = getParam("-i");
 fnOut        = getParam("-o");
 a            = getDoubleParam("-a");
 b            = getDoubleParam("-b");
 N            = getIntParam("-N");
 slice        = getIntParam("-s");
 
 Zsize        = getIntParam("--zsize");
 epsilon      = getDoubleParam("-e");
 iter_cnt     = getIntParam("--NI");
 rec_method   = getParam("--rec");
 lart         = getDoubleParam("--lart");
 lsart        = getDoubleParam("--lsart");
 l_method     = getParam("--atl");
 phi_method   = getParam("--phi");
 sigma        = getDoubleParam("--sigma");
 Ku           = getDoubleParam("--Ku");
 Me           = getDoubleParam("--Me");
 delta_angle  = getIntParam("--delta");
 pr_method    = getParam("--Pr");
 X0_name      = getParam("--X0");
 int XR       = getIntParam("--XR");
 if(std::abs(XR) != 0)
    X0_rand = true;
}

/**
**
** Method to inform the user how the software will run
**
*/
void ProgReconsSuperSlice::show()
{
 if(verbose > 0){
    std::cout << "\e[1m\e[1;33m" << "########################################################" << "\e[m\e[0m" << std::endl;
    std::cout << "Superiorized ";
    switch(B.getType()){
        case reconType::ART: std::cout << "ART";break;
        case reconType::SART: std::cout << "SART";break;
        default: "NA";
       }
    std::cout << " in 2D" << std::endl;
    std::cout << " with the following arguments:" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Input tilt series: " << fnTiltSeries << std::endl;
    std::cout << "Slice: " << slice << std::endl;
    if(X0_file == true)
       std::cout << "Reading Initial Solution from: " << X0_name << std::endl;
    if(X0_rand == true)
       std::cout << "Randomly initializing first solution. " << std::endl;
    std::cout << "Output reconstruction: " << fnOut << std::endl;
    if(epsilon<0.0)
       std::cout << "# of iterations: " << iter_cnt << std::endl;
    else
       std::cout << "Epsilon: " << epsilon << std::endl;
    if(Zsize == -1)
       std::cout << "The depth size (Z size) would be determined when reading the input projection file" << std::endl;
    else
       std::cout << "The depth size (Z size): " << Zsize << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "N: " << N << std::endl;
    
    std::cout << "ATL: ";
    switch(mode_l){
        case lmode::ATL0: std::cout << "ATL0" << std::endl;
             break;
        case lmode::ATL1: std::cout << "ATL1" << std::endl;
             break;
        case lmode::ATL2: std::cout << "ATL2" << std::endl;
             break;
       }
    
    std::cout << "phi: " << phi.getName() << std::endl;
    if(phi.getShortName() == "ATV"){
       std::cout<<"std. dev. kernel: " << sigma << std::endl;
       std::cout<<"Weight (Ku)  kernel: " << Ku << std::endl;
       std::cout<<"Weight (Me) kernel: " << Me << std::endl;
       std::cout<<"Angle increment kernel: " << delta_angle << std::endl;
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
    std::cout << "\e[1m\e[1;33m" << "########################################################" << "\e[m\e[0m" << std::endl;
   }
}

/**
**
** Default constructor.
** Method to initialize variables and status before running but after reading from the command line.
**
*/
ProgReconsSuperSlice::ProgReconsSuperSlice()
{
 mode_l      = lmode::ATL0;
 a           =  0.0;
 b           =  0.0;
 epsilon     = -1.0;
 N           =  0;
 slice       = -1;
 Zsize       = -1;
 lart        =  1.0;
 lsart       =  1.0;
 sigma       =  1.0;
 Ku          =  1.0;
 Me          =  1.0;
 delta_angle =  1;
 iter_cnt    = -1;

 Pmin        = 0.0;
 Pmax        = 0.0;
 hdrOff      = 0;

 X0_file = false;
 X0_rand = false;
 X0_name = "";

 l_method = "ATL0";
 
 rec_method = "ART";
 //B.set(rec_method);
 //B.setParam(lart);
 
 phi_method = "ITV";
 phi.set(phi_method);
 
 pr_method = "L2SQ";
 //B.setPr(pr_method);
 I_dims.x_dim = 0;
 I_dims.y_dim = 0;
 I_dims.z_dim = 0;
 
 B_dims.x_dim = 0;
 B_dims.y_dim = 0;
 B_dims.z_dim = 0;
}

/**
**
** The Destructor, ha ha ha.
**
*/
ProgReconsSuperSlice::~ProgReconsSuperSlice()
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
void ProgReconsSuperSlice::checkArgsInfo()
{
 //
 // Checking for the existence of input file
 //
 std::ifstream ifile(fnTiltSeries.c_str());
 if((bool)ifile == 0){
    std::cout << "\e[1m\e[0;31m" << "ERROR: the input file does not exist." << "\e[m\e[0m";
    std::cout << std::endl;
    usage("-i", true);
   }

 //
 // Checking the value for Zsize
 //
 if(Zsize == -1){
    std::cerr << "\e[1m\e[1;31m" << "WARNING: " << "\e[m\e[0m";
    std::cerr << "zsize was not set. The default value would be equal to the horizontal dimension (X size) of the projection data." << std::endl;
   }

 //
 // Checking the value for epsilon
 //
 if(epsilon < 0.0 && iter_cnt<0){
    std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
    std::cerr << "it is necessary to select a stop criterion: by proximity function or number of iterations." << std::endl;
    std::cerr << "       The values of epsilon and the number of iterations are negative." << std::endl;
    usage("-e",true);
   }

 //
 // Checking the value for a
 //
 if(a <= 0.0 || a >= 1.0){
    std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
    std::cerr << "a has to meet 0.0 < a < 1.0." << std::endl;
    usage("-a",true);
   }

 //
 // Checking the value set for N
 //
 if(N < 0){
    std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
    std::cerr << "N has to be positive" << std::endl;
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
          std::cerr << l_method << " ";
          std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
          std::cerr << "it is not a valid option, ATL0 will be used instead." << std::endl;
         }
      }
   }

 //
 // Checking the value for reconstruction
 //
 if(rec_method == std::string("ART")){
    rec_method = "ART";
   }
 
 if(rec_method == std::string("SART")){
    rec_method = "SART";
   }
 
 //
 // Checking the value for second criterion (phi)
 //
 if(phi.valid(phi_method) == false){
    std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
    std::cerr << "invalid second criterion for optimization" << std::endl;
    usage("--phi", true);
   }
 if(phi_method == std::string("ATV") && (phi.getShortName() != String("ATV"))){
    phi.set("ATV");
    
    if(sigma <= 0.0){
       std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
       std::cerr << "the std. dev. for kernel <= 0" << "\e[m\e[0m" << std::endl;
       exit(EXIT_SUCCESS);
      }
    if(Ku <= 0.0){
       std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
       std::cerr << "the weight Ku for kernel <= 0" << std::endl;
       exit(EXIT_SUCCESS);
      }
    if(Me <= 0.0){
       std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
       std::cerr << "the weight Me for kernel <= 0" << std::endl;
       exit(EXIT_SUCCESS);
      }
   }
 if(phi_method == std::string("ITV") && (phi.getShortName()!= String("ITV"))){
    phi.set("ITV");
   }
 if(phi_method == std::string("WTV") && (phi.getShortName()!= String("WTV"))){
    phi.set("WTV");
   }
 
 //
 // Checking whether to use counter or proximity function
 //
 if(iter_cnt == 0){
    std::cerr << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
    std::cerr << "invalid value for the number of iterations for reconstruction" << std::endl;
    usage("--NI", true);
   }
 
 //
 // Checking for the initial solution
 //
 if(X0_name.empty() == false && X0_name != std::string("empty")){
    X0_file = true;
   }
 
 //
 // Checking that the initial solution is random XOR from file
 //
 if(X0_rand && X0_file)
    X0_rand = false;
 
 //
 // Checking the slice selected for reconstruction
 //
 if(slice <= 0){
    std::cout << "\e[1m\e[1;31m" << "ERROR: " << "\e[m\e[0m";
    std::cerr << "the selected slice to reconstruct needs to be greater than zero" << std::endl;
    usage("-s", true);
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
void ProgReconsSuperSlice::run()
{
 //
 // Processing the input arguments
 //
 checkArgsInfo();
 B.set(rec_method);
 B.setParam(lart);
 B.setPr(pr_method);
 
 //
 // Showing the arguments' values used by the program
 show();
 
 //
 // Reading the Input projections through the Metadata file
 //
 MultidimArray<double> TS;
 std::vector<double> tiltAngles;

 ReadTiltSeries(tiltAngles, TS);
 if(Zsize == -1)
    Zsize = TS.xdim;
 
 std::cout<<"Dimensions of the Tilt Series: " << TS.xdim << "," << TS.ydim << "," << TS.zdim << std::endl;
 if(slice > TS.ydim){
    std::cout<<"WARNING: the requested slice (" << slice <<") is beyond the dimension where there is projection data." << std::endl;
    std::cout<<"         Thus, the slice will be set to the size of that dimension." << std::endl;
    slice = TS.ydim;
   }
 slice -= 1;
 /*
  *
  * Setting up the Reconstruction Algorithm
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
          std::cerr << "\e[1m\e[1;31m" << "Unknown Reconstruction Algorithm." << std::endl;
          std::cerr << "Aborting the execution !!!" << "\e[m\e[0m" << std::endl;
          exit(0);
    }
 
 /*
  *
  * Allocating memory and resizing the final reconstruction and
  * the necessary vectors for superiorization (non-ascending and z)
  *
  */
 MultidimArray<double> x;
 Create3DImage(TS,x);
 //std::ofstream outfPr("X0R.bin", std::ios::binary | std::ios::out);
 //outfPr.write(reinterpret_cast<char *>(x.data), x.xdim*x.ydim*x.zdim*sizeof(double));
 //outfPr.close();
 
 /*
  *
  * Superiorization Section
  *
  */
 Reconstruct2D(x,TS,tiltAngles);
 
 // Informing the user about the values for Pr and phi
 std::cout << "\e[1m\e[34m" << "Pr" << "\e[m\e[0m";
 std::cout << ": " << std::setprecision(12);
 std::cout << B.Pr(x,TS,tiltAngles,slice) << std::endl;
 
 std::cout << "\e[1m\e[34m" << "TV" << "\e[m\e[0m";
 std::cout << ": " << std::setprecision(12);
 std::cout << phi(x) << std::endl;
 
 // Saving the Result
 std::cout << "\e[1m\e[35m" << "Saving the Result in " << "\e[m\e[0m";
 std::cout << fnOut << std::endl;
 tiltAngles.clear();
 Save3DImage(TS,x);
 x.clear();
 std::cout << "Procedure finished. Leaving the Program" << std::endl;
}

#undef COLOR_OUTPUT
#undef OUTPUT_NAV
#undef OUTPUT_Z
#undef OUTPUT_PRINTING
#undef PRINT_N_LOOP
#undef PRINT_BETA
#undef DEBUG

