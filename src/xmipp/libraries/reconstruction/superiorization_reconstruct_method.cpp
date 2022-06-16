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

#include "superiorization_reconstruct_method.h"

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
** Executes the Reconstruction Algorithm
**
*/
void RecMeth::init(const uint xdim, const uint ydim,const std::vector<double>& A)
{
}

/**
**
** Executes the Reconstruction Algorithm
**
*/
void RecMeth::B(MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& A,const int k)
{
 memset(v.data,0,v.xdim*v.ydim*v.zdim*sizeof(double));
}

/**
**
** Executes the Reconstruction Algorithm
**
*/
void RecMeth::B(MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& A,const uint slice,const int k)
{
 memset(v.data,0,v.xdim*v.ydim*v.zdim*sizeof(double));
}

/**
**
** Executes the Proximity function
**
*/
double RecMeth::Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& A)
{
 return 0.0;
}

/**
**
** Executes the Proximity function
**
*/
double RecMeth::Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& A, const uint plane)
{
 return 0.0;
}

/**
**
** Returns the type of reconstruction algorithm being used
**
*/
reconType RecMeth::getType(void)
{
 return reconType::none;
}

/**
**
** Returns the type of proximity function being used
**
*/
proximityType RecMeth::getPrType(void)
{
 return proximityType::none;
}

/**
**
** Sets the type of proximity function to be used
**
*/
void RecMeth::setPr(const proximityType type)
{
}

/**
**
** Sets the type of proximity function to be used
**
*/
void RecMeth::setPr(const std::string strType)
{
}
