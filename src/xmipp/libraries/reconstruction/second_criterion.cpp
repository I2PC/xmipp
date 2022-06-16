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

#include "second_criterion.h"
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
** Computes the Second Criterion
**
*/
double secc::phi(const MultidimArray<double>& v)
{
 return 0.0;
}

/**
**
** Computes the normalized non-ascending vector
**
*/
MultidimArray<double> secc::nav(const MultidimArray<double>& v)
{
 MultidimArray<double> w(v);
 memset(w.data,0,v.xdim*v.ydim*v.zdim*sizeof(double));
}

/**
**
** Initializes the parameters for the second criterion
**
*/
void secc::init(MultidimArray<double>& v)
{
 
}

/**
**
** Initializes the parameters for the second criterion
**
*/
void secc::init(MultidimArray<double>& u, const double sigma, const double ku, const double me, const int delta, const double Amin, const double Amax)
{
 
}

/**
**
** Updates, if necessary, the parameters for the second criterion
**
*/
void secc::preupdate(MultidimArray<double>& v)
{
 
}

/**
**
** Updates, if necessary, the parameters for the second criterion
**
*/
void secc::postupdate(MultidimArray<double>& v)
{
 
}
