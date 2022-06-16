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
#ifndef REC_BASE_HH
#define REC_BASE_HH

#include <core/xmipp_program.h>

#include "superiorization_reconstruct_method.h"
#include "superiorization_reconstruct_types.h"
#include "superiorization_proximity_types.h"

class ReconBase
{
 private:
    RecMeth* R;
 
 public:
    ReconBase();
    ReconBase(const String &StrType);
    ReconBase(const String &recType,const String &prType);
    void init(const uint xdim, const uint ydim,const std::vector<double>& A);
    void set(std::string StrType);
    void setPr(std::string StrType);
    std::string getName();
    reconType getType(void);
    std::string getPrName();
    proximityType getPrType(void);
    void setParam(const double v);
    void operator()(MultidimArray<double>& v,const MultidimArray<double>& P,
                    const std::vector<double>& A,const int k);
    void operator()(MultidimArray<double>& v,const MultidimArray<double>& P,
                    const std::vector<double>& A,const uint slice, const int k);
    double Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA);
    double Pr(const MultidimArray<double>& v,const MultidimArray<double>& P, const std::vector<double>& LA, const uint slice);
    
}; /* virtual class for reconstruction method */

#endif /* REC_BASE_HH */
