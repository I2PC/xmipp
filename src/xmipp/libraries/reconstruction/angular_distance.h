/***************************************************************************
 *
 * Authors:    Carlos Oscar            coss@cnb.csic.es (2002)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
#ifndef _PROG_ANGULAR_DISTANCE
#define _PROG_ANGULAR_DISTANCE

#include <core/xmipp_funcs.h>
#include <core/metadata_vec.h>
#include <core/xmipp_program.h>
#include <core/symmetries.h>

/**@defgroup AngularDistance angular_distance (Distance between two angular assignments)
   @ingroup ReconsLibrary */
//@{
/** Angular Distance parameters. */
class ProgAngularDistance: public XmippProgram
{
public:
    /** Filename angle doc 1 */
    FileName fn_ang1;
    /** Filename angle doc 2 */
    FileName fn_ang2;
    /** Filename symmetry file */
    FileName fn_sym;
    /** Filename of output file with merging */
    FileName fn_out;
    /** Check mirrors for Spider APMQ */
    bool check_mirrors;
    /** Use object rotations */
    bool object_rotation;
    /** Compute weights */
    bool compute_weights;
    /** Minimum angular sigma */
    double minSigma;
    /** Minimum displacement sigma */
    double minSigmaD;
    // Identification label
    String idLabel;
    /// Set of angular difference
    int set;
    /// The set of angles to be used for output
    int ang;
    /// Compute angle mean
    bool compute_average_angle;
    /// Compute shift mean
    bool compute_average_shift;
public:
    // DocFile 1
    MetaDataVec DF1;
    // DocFile 2
    MetaDataVec DF2;
    // Symmetry List
    SymList SL;
public:
    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /** Produce side info.
        Read all document files and symmetry list if any.
        An exception is thrown if both files are not of the same length. */
    void produce_side_info();

    /** Run */
    void run();

    /** computeWeights */
    void computeWeights();

    static void euler2quat( double rot, double tilt, double psi,
                            double q[4] );
    static void quat2Euler( const double q[4],
                            double& rot, double& tilt, double& psi );

    static void computeAverageAngles(double rot1, double tilt1, double psi1,
                                     double rot2, double tilt2, double psi2,
                                     double& rot, double& tilt, double& psi );

    static void computeAverageShifts(double shiftX1, double shiftY1,
                                     double shiftX2, double shiftY2,
                                     double& shiftX, double& shiftY );

};
//@}
#endif
