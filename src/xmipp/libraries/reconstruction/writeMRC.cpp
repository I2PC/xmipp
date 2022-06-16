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

#include "writeMRC.h"

#include <fstream>
#include <iostream>
#include <istream>
#include <chrono>
#include <random>
#include <core/alglib/ap.h>

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************* Definition of Local Variables ****************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/**
 ** Definition of the  MRC Header
 */
struct MRChead
{             // file header for MRC data
    int nx;              //  0   0       image size
    int ny;              //  1   4
    int nz;              //  2   8
    int mode;            //  3           0=char,1=short,2=float,6=uint16
    int nxStart;         //  4           unit cell offset
    int nyStart;         //  5
    int nzStart;         //  6
    int mx;              //  7           unit cell size in voxels
    int my;              //  8
    int mz;              //  9    1=Image or images stack, if volume mz=nz.
                         //              If ispg=401 then nz=number of volumes in stack * volume z dimension
                         //              mz=volume zdim
    float a;             // 10   40      cell dimensions in A
    float b;             // 11
    float c;             // 12
    float alpha;         // 13           cell angles in degrees
    float beta;          // 14
    float gamma;         // 15
    int mapc;            // 16           column axis
    int mapr;            // 17           row axis
    int maps;            // 18           section axis
    float amin;          // 19           minimum density value
    float amax;          // 20   80      maximum density value
    float amean;         // 21           average density value
    int ispg;            // 22           space group number   0=Image/stack,1=Volume,401=volumes stack
    int nsymbt;          // 23           bytes used for sym. ops. table
    float extra[25];     // 24           user-defined info
    float xOrigin;       // 49           phase origin in pixels FIXME: is in pixels or [L] units?
    float yOrigin;       // 50
    float zOrigin;       // 51
    char map[4];         // 52       identifier for map file ("MAP ")
    char machst[4];      // 53           machine stamp
    float arms;          // 54       RMS deviation
    int nlabl;           // 55           number of labels used
    char labels[800];    // 56-255       10 80-character labels
} ;

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Local Methods *******************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Main Methods ********************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Method to superiorize an iterative reconstruction algorithm
**
*/
int getMRCHdrSize(void)
{
 return sizeof(MRChead);
}

/**
**
** Method to superiorize an iterative reconstruction algorithm
**
*/
int writeMRCHdr(FILE *file, const int Xdim, const int Ydim, const int Zdim, const String &bitDepth)
{
 MRChead hdr;

 std::memset(&hdr,0,sizeof(MRChead));
 hdr.mode = 2;
 hdr.map[0] = 'M';
 hdr.map[1] = 'A';
 hdr.map[2] = 'P';
 hdr.map[3] = 0;

 short int word = 0x0001;
 char *byte = (char *) &word;
 if(byte[0]){ //LITTLE_ENDIAN
    hdr.machst[0] = 68;
    hdr.machst[1] = 65;
   }
 else{ // BIG_ENDIAN
    hdr.machst[0] = 17;
    hdr.machst[1] = 17;
   }
 
 hdr.a = hdr.mx = hdr.nx = Xdim;
 hdr.b = hdr.my = hdr.ny = Ydim;
 hdr.c = hdr.mz = hdr.nz = Zdim;

 hdr.alpha = 90.0;
 hdr.beta  = 90.0;
 hdr.gamma = 90.0;
 
 hdr.mapc = 1;
 hdr.mapr = 2;
 hdr.maps = 3;
 
 hdr.nsymbt = 0;
 hdr.nlabl  = 10;

 hdr.ispg = 1;

 std:fseek(file,0,SEEK_SET);
 std::fwrite(&hdr,sizeof(MRChead),1,file);
}

