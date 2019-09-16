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

#include "superiorization_proximity.h"
#include <core/alglib/ap.h>

#include <functional>

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Local Methods ********************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Constructor from a name (string).
**
*/
double SuperProx::l2SQ(const MultidimArray<double>& x)
{
 return 0.0;
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Public Method *********************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/**
**
** Simple constructor.
**
*/
SuperProx::SuperProx()
{
 //std::function<int(const char*)> f = std::atoi;
 prox = std::bind(&SuperProx::l2SQ,this,std::placeholders::_1);
 PrType = SuperProx::L2SQ;
}

/**
**
** Constructor from a name (string).
**
*/
SuperProx::SuperProx(String &StrType)
{

}

/**
**
** Method to set the desired function to be used as a Proximity function.
**
*/
void SuperProx::set(std::string StrType)
{
 if(StrType == std::string("L2SQ")){
    prox = std::bind(&SuperProx::l2SQ,this,std::placeholders::_1);
    PrType = SuperProx::L2SQ;
   }
}

/**
**
** Method to call the selected method to measure the proximity of the partial result/reconstruction.
**
*/
double SuperProx::Pr(const MultidimArray<double>& x)
{

}

/**
**
** Method to call the selected method to measure the proximity of the partial result/reconstruction.
**
*/
double SuperProx::operator ()(const MultidimArray<double>& x)
{
 return prox(x);
}
#undef DEBUG
