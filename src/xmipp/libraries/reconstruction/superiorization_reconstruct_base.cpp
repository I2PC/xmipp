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

#include "superiorization_reconstruct_base.h"

#include <core/alglib/ap.h>

#include <functional>
#include <cmath>

#include "superiorization_reconstruct_types.h"
#include "superiorization_proximity_types.h"
#include "superiorization_reconstruct_art.h"
#include "superiorization_reconstruct_sart.h"

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Public Methods ******************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/**
**
** Default Constructor
**
*/
ReconBase::ReconBase():R(0)
{
 //R = new RecART;
 //R->setPr(proximityType::L2SQ);
}

/**
**
** Constructor with ID for the desired reconstruction algorithm.
**
*/
ReconBase::ReconBase(const String &StrType):R(0)
{
 if(StrType == std::string("ART")){
    R = new RecART;
    R->setPr(proximityType::L2SQ);
   }
 if(StrType == std::string("SART")){
    R = new RecSART;
    R->setPr(proximityType::L2SQ);
   }
}

/**
**
** Constructor with ID for the desired reconstruction algorithm
** and Proximity function.
**
*/
ReconBase::ReconBase(const String &recType,const String &prType):R(0)
{
 if(recType == std::string("ART"))
    R = new RecART;
 
 if(recType == std::string("SART"))
    R = new RecSART;
 
 if(R != 0){
    if(prType == std::string("L2SQ")){
       R->setPr(proximityType::L2SQ);
      }
   }
}

/**
**
** Method to select the desired reconstruction algorithm
**
*/
void ReconBase::init(const uint xdim, const uint ydim,const std::vector<double>& A)
{
 R->init(xdim,ydim,A);
}

/**
**
** Method to select the desired reconstruction algorithm
**
*/
void ReconBase::set(std::string StrType)
{
 if(R != 0){
    std::cout<<"WARNING: Changing the Reconstruction method (for superiorization)."<<std::endl;
    delete R;
   }
 
 if(StrType == std::string("ART")){
    R = new RecART;
    R->setPr(proximityType::L2SQ);
   }
 
 if(StrType == std::string("SART")){
    R = new RecSART;
    R->setPr(proximityType::L2SQ);
   }
}

/**
**
** Method to select the desired Proximity function
**
*/
void ReconBase::setPr(std::string StrType)
{
 if(R != 0)
    R->setPr(StrType);
 else
    std::cout<<"ERROR: Reconstruction method not defined just yet"<<std::endl;
}

/**
**
** Method to get the name of the reconstruction algorithm
**
*/
std::string ReconBase::getName()
{
 if(R != 0){
    switch(R->getType()){
        case reconType::ART: return std::string("ART");
        case reconType::SART:return std::string("SART");
        default:return std::string("Not Set");
       }
   }

 return std::string("Not Set");
}

/**
**
** Method to get the name of the proximity function
**
*/
std::string ReconBase::getPrName()
{
 if(R != 0){
    switch(R->getPrType()){
        case proximityType::L2SQ: return std::string("Least Squares (L2 Squared)");
        default:return std::string("Not Set");
       }
   }
 
 return std::string("Not Set");
}

/**
**
** Method to obtain the type of selected reconstruction algorithm
**
*/
reconType ReconBase::getType(void)
{
 if(R != 0)
    return R->getType();

 return reconType::none;
}

/**
**
** Method to obtain the selected proximity function
**
*/
proximityType ReconBase::getPrType(void)
{
 if(R != 0)
    return R->getPrType();

 return proximityType::none;
}

/**
**
** Method to select the desired reconstruction algorithm
**
*/
void ReconBase::setParam(const double v)
{
 switch(R->getType()){
     case reconType::ART:
        dynamic_cast<RecART*>(R)->setParam(v);
        break;
     case reconType::SART:
        dynamic_cast<RecSART*>(R)->setParam(v);
        break;
     default:
        break;
    }
}

/**
**
** Calls the selected reconstruction algorithm
**
*/
void ReconBase::operator()(MultidimArray<double>& v,
                            const MultidimArray<double>& P,
			    const std::vector<double>& A,const int k)
{
 R->B(v,P,A,k);
}

/**
**
** Calls the selected reconstruction algorithm
**
*/
void ReconBase::operator()(MultidimArray<double>& v,
                            const MultidimArray<double>& P,
			    const std::vector<double>& A,const uint slice,const int k)
{
 R->B(v,P,A,slice,k);
}

/**
**
** Calls the selected proximity function
**
*/
double ReconBase::Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA)
{
 double val = R->Pr(v,P,LA);
 return val;
}

/**
**
** Calls the selected proximity function
**
*/
double ReconBase::Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA,const uint slice)
{
 double val = R->Pr(v,P,LA,slice);
 return val;
}

