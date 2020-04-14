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
#ifndef ATV_HH
#define ATV_HH

#include <core/xmipp_program.h>

#include "second_criterion.h"
#include "recons_util.h"

class atv:public secc
{
private:
    MultidimArray<double> w,G,H;
    double eps;
    double minA, maxA;
    MultidimArray<int> MPregionsMask;
private:
    std::vector<recu::reg_R> pixray(const int np,const int nr,const std::vector<double>& LA);
    void GaussKernel(MultidimArray<double>& K, const double sigma, const unsigned short size);
public:
    atv();
    ~atv();
    double phi(const MultidimArray<double>& u);
    void nav(const MultidimArray<double>& u, MultidimArray<double>& v);
    void init(MultidimArray<double>& v,double sigmaG,unsigned short sizeG,double sigmaH,unsigned short sizeH,double Amin,double Amax);
    void update(MultidimArray<double>& u);
    void createMask(const size_t xdim, const size_t ydim, const size_t zdim);
}; /* class for Weighted Total Variation functions */

#endif /* ATV_HH */

