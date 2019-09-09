/***************************************************************************
 *
 * superiorization_regularizer.cpp
 *
 * Created on: Sep 6, 2019
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


#include "superiorization_regularizer.h"
/*
 // Iterate over Y 2D subproblems
 I().resizeNoCopy(ZSIZE(TS),XSIZE(TS));
 for(int i=0; i<YSIZE(TS); i++){
     // Extract the 2D subproblem
     for(int k=0; k<ZSIZE(TS); k++)
         memcpy(&I(k,0),&A3D_ELEM(TS,k,i,0),sizeof(double)*XSIZE(TS));
#ifdef DEBUG
     I.write("PPP2D.xmp");
     I().printStats();
     std::cout << "Press any key" << std::endl;
     char c; std::cin >> c;
#endif
    }
*/

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Local Methods ********************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Method to initialize variables and status before running
**
*/
SuperRegular::SuperRegular()
{

}
