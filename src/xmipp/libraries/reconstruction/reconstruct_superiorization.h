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
#ifndef _PROG_SUPER_HH
#define _PROG_SUPER_HH

#include <core/xmipp_program.h>
#include "superiorization_regularizer.h"
#include "superiorization_proximity.h"

/* The user interface program should make a call to the run routine.
  */
class ProgReconsSuper: public XmippProgram
{
private:
	enum class lmode {ATL0,ATL1,ATL2};
	lmode mode_l;
	double a,b,epsilon;
	int N;
	SuperProx Pr;
	SuperRegular<double> phi;
public:
    FileName fnTiltSeries, fnOut;
    int Zsize;
	String phi_method, l_method, pr_method;
public:
    ///Functions of common reconstruction interface
	ProgReconsSuper();
    void defineParams();
    void readParams();
    void produceSideInfo();
    void show();
    void run();
};

#endif
