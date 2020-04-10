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
#ifndef REC_METHOD_SART_HH
#define REC_METHOD_SART_HH

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
class RecSART:public RecMeth
{
 private:
    MultidimArray<double> RW,  // Ray Weight by Pixel
                          RWS, // Ray Sum Weight by Pixel
                          Q;

    const double eps_zero = 1e-10;
    double lambda;
    proximityType PrType;
    
    uint NP,NR;
    int M,N;
    recu::pixRay PR;
    
 private:
    double L2SQ(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA);
 public:
    RecSART();
    ~RecSART();
    void init(const uint xdim, const uint ydim,const std::vector<double>& A);
    void setParam(const double l);
    void B(MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& A,const int k);
    double Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA);
    void setPr(const proximityType type);
    void setPr(const std::string strType);
    reconType getType(void);
    proximityType getPrType(void);
}; /* virtual class for SART method */

#endif /* REC_METHOD_SART_HH */
