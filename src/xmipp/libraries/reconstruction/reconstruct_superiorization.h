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
#ifndef PROG_SUPER_HH
#define PROG_SUPER_HH

#define XMIPP_21

#include <limits>
#include <iomanip>
#include <fstream>

#include <core/xmipp_program.h>
#include <core/xmipp_hdf5.h>
#ifdef XMIPP_21
#include <core/metadata_base.h> // After 2021
#include <core/metadata_vec.h>     // After 2021
#else
#include <core/metadata.h> // Before 2021
#endif

#include <core/xmipp_image.h>
#include <core/xmipp_program.h>
#include <core/xmipp_hdf5.h>
#include <core/xmipp_threads.h>

#include "superiorization_regularizer.h"
#include "superiorization_reconstruct_base.h"

/* The user interface program should make a call to the run routine.
  */
class ProgReconsSuper: public XmippProgram
{
private:
    enum class lmode {ATL0,ATL1,ATL2};
    struct ImageDims
    {
     int x_dim, y_dim, z_dim;
    };
    
    ImageDims     I_dims, B_dims;
    //std::fstream  fX;
    FILE         *fX;
    int           Zsize;
    lmode         mode_l;
    double        a,b,epsilon,lart,lsart, Pmin, Pmax;
    double        sigma, Ku, Me;
    int           delta_angle;
    int           N, iter_cnt,hdrOff;
    ReconBase     B;
    SuperRegular<double> phi;
    bool          X0_file, X0_rand, Y_file;
    std::string   X0_name;
public:
    FileName      fnTiltSeries, fnOut;
    String        phi_method, l_method, pr_method, rec_method;
private:
    void ReadTiltSeries(std::vector<double>& Angles, MultidimArray<double>& TS);
    void Create3DImage(const MultidimArray<double>& TS, MultidimArray<double>& x);
    void Save3DImage(MultidimArray<double>& TS, MultidimArray<double>& x);
    void LoadSlice(const MultidimArray<double>& x, const uint plane, MultidimArray<double>& S);
    void ReturnSlice(const MultidimArray<double>& S, const uint plane, MultidimArray<double>& x);
    void Reconstruct2D(MultidimArray<double> &x,MultidimArray<double> PD, std::vector<double> Angles);
    double Pr(const MultidimArray<double>& x,const MultidimArray<double>& P, const std::vector<double>& Angles);
    double Phi(const MultidimArray<double>& X, const MultidimArray<double>& P, const std::string type);
public:
    ///Functions of common reconstruction interface
	 ProgReconsSuper();
	~ProgReconsSuper();
    void defineParams();
    void readParams();
    void checkArgsInfo();
    void show();
    void run();
};

#endif /* PROG_SUPER_HH */

