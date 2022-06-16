/***************************************************************************
 *
 * Authors:     Edgar Garduno Angeles (edgargar@ieee.org)
 * Created on: Sep 6, 2019
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
#ifndef REC_METHOD_ART_HH
#define REC_METHOD_ART_HH

#include "superiorization_reconstruct_method.h"

#include <core/xmipp_program.h>

#include <assert.h>

#include "superiorization_proximity_types.h"
#include "recons_util.h"

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/********************** Declaration of ART Class ******************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
class RecART:public RecMeth
{
 private:
    const double eps_zero = 1e-10;
    double lambda;
    proximityType PrType;
    
    recu::pixRay PR;
    
 private:
    double L2SQ_2(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA, const uint slice);
    double L2SQ_3(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA);
 public:
    RecART();
    void init(const uint xdim, const uint ydim,const std::vector<double>& A);
    void setParam(const double l);
    void B(MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& A,const uint slice,const int k);
    double Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA);
    double Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA, const uint slice);
    void setPr(const proximityType type);
    void setPr(const std::string strType);
    reconType getType(void);
    proximityType getPrType(void);
}; /* virtual class for ART method */

#endif /* REC_METHOD_ART_HH */
