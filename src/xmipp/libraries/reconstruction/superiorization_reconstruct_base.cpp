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
#include "superiorization_reconstruct_art.h"

#include <core/alglib/ap.h>

#include <functional>
#include <cmath>

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
ReconBase::ReconBase()
{
 B = 0;
 RecType = ART;
 B = new RecART;
}

/**
**
** Constructor with ID for the desired reconstruction algorithm
**
*/
ReconBase::ReconBase(const String &StrType)
{
 if(StrType == std::string("ART")){
    RecType = ART;
    B = new RecART;
   }
}

/**
**
** Method to select the desired reconstruction algorithm
**
*/
void ReconBase::set(std::string StrType)
{
 if(B != 0)
    delete B;
 
 if(StrType == std::string("ART")){
    RecType = ART;
    B = new RecART;
   }
}

/**
**
** Method to select the desired reconstruction algorithm
**
*/
std::string ReconBase::getName()
{
 if(RecType == ART){
    return std::string("ART");
   }

 return std::string("Not Set");
}

/**
**
** Method to select the desired reconstruction algorithm
**
*/
ReconBase::classType ReconBase::getType(void)
{
 return this->RecType;
}

/**
**
** Method to select the desired reconstruction algorithm
**
*/
void ReconBase::setParam(const double v)
{
 switch(RecType){
     case ART:
        dynamic_cast<RecART*>(B)->setParam(v);
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
 B->B(v,P,A,k);
}

#undef DEBUG
