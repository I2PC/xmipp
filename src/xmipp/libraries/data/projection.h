/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#ifndef CORE_PROJECTION_H
#define CORE_PROJECTION_H

#include "core/xmipp_filename.h"
#include "core/xmipp_threads.h"
#include "core/matrix1d.h"
#include "grids.h"

class Projection;
template<typename T>
class Image;
template<typename T>
class Matrix2D;
template<typename T>
class MultidimArray;
class SimpleGrid;
class Basis;
class XmippProgram;

// These two structures are needed when projecting and backprojecting using
// threads. They make mutual exclusion and synchronization possible.
extern barrier_t project_barrier;
extern pthread_mutex_t project_mutex;

/* Projection parameters for tomography --------------------------- */
/** Projecting parameters. This class reads a set of projection parameters
 *  in a file (see xmipp_project_tomography for more information about the
 *  file structure) and extract the useful information from it.*/
class ParametersProjectionTomography
{
public:
    /** Phantom filename.
        It must be a Xmipp volume. */
    FileName fnPhantom;
    /// Root filename (used for a stack)
    FileName  fnRoot;
    /// Output filename (used for a singleProjection)
    FileName  fnOut;
    /// Only project a single image
    bool singleProjection;
    /// First projection number. By default, 1.
    int      starting;
    /// Extension for projection filenames. This is optional
    std::string   fn_projection_extension;

    /// Projection Xdim
    int      proj_Xdim;
    /// Projection Ydim
    int      proj_Ydim;

    /// Only create angles, do not project
    bool only_create_angles;
    // Show angles calculation in std::out
    bool show_angles;

    /// Rotational angle of the tilt axis
    double axisRot;
    /// Tilt angle of the tilt axis
    double axisTilt;
    /// Offset of the tilt axis
    Matrix1D<double> raxis;
    /// Minimum tilt around this axis
    double tilt0;
    /// Maximum tilt around this axis
    double tiltF;
    /// Step in tilt
    double tiltStep;

    /// Bias to be applied to each pixel grey value */
    double    Npixel_avg;
    /// Standard deviation of the noise to be added to each pixel grey value
    double    Npixel_dev;

    /// Bias to apply to the image center
    double    Ncenter_avg;
    /// Standard deviation of the image center
    double    Ncenter_dev;

    /// Bias to apply to the angles
    double    Nangle_avg;
    /// Standard deviation of the angles
    double    Nangle_dev;

public:

    ParametersProjectionTomography();

    /** Read projection parameters from a file.
    */
    void read(const FileName &fn_proj_param);
    void defineParams(XmippProgram * program);
    void readParams(XmippProgram * program);

    /**
     * Calculate the Euler angles and X-Y shifts from the tilt axis direction and tilt angle.
     */
    void calculateProjectionAngles(Projection &P, double angle, double inplaneRot,
                                   const Matrix1D<double> &sinplane);
};

/** Structure for threaded projections.
   This structure contains all the information needed by a thread
   working on the projecting/backprojecting of a projection. This is
   structure is needed to pass parameters from the master thread to the
   working threads as they run as a function which does not accept passed
   parameters other than a void * structure.
   */
typedef struct
{
    int thread_id;
    int threads_count;
    Image<double> * vol;
    const SimpleGrid * grid;
    const Basis * basis;
    Projection * global_proj;
    Projection * global_norm_proj;
    int FORW;
    int eq_mode;
    const Image<int> *VNeq;
    Matrix2D<double> *M;
    const MultidimArray<int> *mask;
    double ray_length;
    double rot,tilt,psi;
    bool destroy;
}
project_thread_params;

extern project_thread_params * project_threads;

template <class T>
void project_SimpleGrid(Image<T> *vol, const SimpleGrid *grid,
                        const Basis *basis,
                        Projection *proj, Projection *norm_proj, int FORW, int eq_mode,
                        const Image<int> *VNeq, Matrix2D<double> *M,
                        const MultidimArray<int> *mask=NULL,
                        double ray_length = -1.0,
                        int thread_id = -1, int num_threads = 1);

/*---------------------------------------------------------------------------*/
/* PROJECTION GENERATION                                                     */
/*---------------------------------------------------------------------------*/
// Projecting functions ====================================================
#define FORWARD  1
#define BACKWARD 0

#define ARTK     1
#define CAVK     2
#define COUNT_EQ 3
#define CAV      4
#define CAVARTK  5

/** From voxel volumes.
    The voxel volume is projected onto a projection plane defined by
    (rot, tilt, psi) (1st, 2nd and 3rd Euler angles) . The projection
    is previously is resized to Ydim x Xdim and initialized to 0.
    The projection itself, from now on, will keep the Euler angles.

    The offset is a 3D vector specifying the offset that must be applied
    when going from the projection space to the universal space

    rproj=E*r+roffset => r=E^t (rproj-roffset)

    Set it to NULL if you don't want to use it
 */
void projectVolume(MultidimArray<double> &V, Projection &P, int Ydim, int Xdim,
                   double rot, double tilt, double psi,
                   const Matrix1D<double> *roffset=NULL);

/** From voxel volumes, off-centered tilt axis.
    This routine projects a volume that is rotating (angle) degrees
    around the axis defined by the two angles (axisRot,axisTilt) and
    that passes through the point raxis. The projection can be further
    inplane rotated and shifted through the parameters
    (inplaneRot) and (rinplane).

    All vectors involved must be 3D.

    The projection model is rproj=H Rinplane Raxis r +
                                  Rinplane (I-Raxis) raxis + rinplane

    Where Raxis is the 3D rotation matrix given by the axis and
    the angle.
*/
void projectVolumeOffCentered(MultidimArray<double> &V, Projection &P,
                              int Ydim, int Xdim);

/** Single Weighted Back Projection.
   Projects a single particle into a voxels volume by updating its components this way:
    Voxel(i,j,k) = Voxel(i,j,k) + Pixel( x,y) * Distance.

 Where:

 - Voxel(i,j,k) is the voxel the ray is crossing.
 - Pixel( y,z ) is the value of the pixel where the ray departs from.
 - Distance is the distance the ray goes through the Voxel.
*/
void singleWBP(MultidimArray<double> &V, Projection &P);

/** Count equations in volume.
   For Component AVeraing (CAV), the number of equations in which
   each basis is involved is needed. */
void count_eqs_in_projection(GridVolumeT<int> &GVNeq,
                             const Basis &basis, Projection &read_proj);

/** Project a crystal basis volume.
    This function projects a crystal deformed basis volume, ie, in the
    documentation volume g. However the angles given must be those for
    volume f, the undeformed one. You must supply the deformed lattice
    vectors, and the matrix to pass from the deformed to the undeformed
    vectors (D and Dinv). a=D*ai;

    Valid eq_modes are ARTK, CAVARTK and CAV.
*/
void project_Crystal_Volume(GridVolume &vol, const Basis &basis,
                            Projection &proj, Projection &norm_proj,
                            int Ydim, int Xdim,
                            double rot, double tilt, double psi, const Matrix1D<double> &shift,
                            const Matrix1D<double> &aint, const Matrix1D<double> &bint,
                            const Matrix2D<double> &D, const Matrix2D<double> &Dinv,
                            const MultidimArray<int> &mask, int FORW, int eq_mode = ARTK);

// Implementations =========================================================
// Some aliases
#define x0   STARTINGX(IMGMATRIX(*proj))
#define xF   FINISHINGX(IMGMATRIX(*proj))
#define y0   STARTINGY(IMGMATRIX(*proj))
#define yF   FINISHINGY(IMGMATRIX(*proj))
#define xDim XSIZE(IMGMATRIX(*proj))
#define yDim YSIZE(IMGMATRIX(*proj))

// Projections from single particles #######################################
// Projections from basis volumes ==========================================

/* Projection of a simple volume ------------------------------------------- */
// The projection is not precleaned (set to 0) before projecting and its
// angles are supposed to be already written (and all Euler matrices
// precalculated
// The projection plane is supposed to pass through the Universal coordinate
// origin

/* Time measures ...........................................................
   This function has been optimized in time, several approaches have been
   tried and here are the improvements in time
   Case: Base (unit time measure)
         The points within the footprint are calculated inside the
         inner loop instead of computing the foot coordinate for the
         first corner and knowing the sampling rate in the oversampled
         image go on the next points. The image access was done directly
         (physical access).

   Notation simplification:
   Case: VOLVOXEL(V,k,i,j) ----> V(k,i,j):   + 38% (Inacceptable)
   Case: DIRECT_IMGPIXEL   ----> IMGPIXEL:   +  5% (Acceptable)

   Algorithmic changes:
   Case: project each basis position ---->   +325% (Inacceptable)
         get rid of all beginZ, beginY, prjX, ... and project each
         basis position onto the projection plane
   Case: footprint coordinates outside ----> - 33% (Great!!)
         the footprint coordinates are computed outside the inner
         loop, but you have to use the sampling rate instead to
         move over the oversampled image.
*/

const int ART_PIXEL_SUBSAMPLING = 2;

/** Threaded projection for simple grids
*/
template <class T>
void *project_SimpleGridThread( void * params );

/** Projection of a Simple Grid.
    Valid eq_modes are ARTK, CAVARTK and CAV.
*/
template <class T>
void project_SimpleGrid(Image<T> *vol, const SimpleGrid *grid,
                        const Basis *basis,
                        Projection *proj, Projection *norm_proj, int FORW, int eq_mode,
                        const Image<int> *VNeq, Matrix2D<double> *M,
                        const MultidimArray<int> *mask,
                        double ray_length,
                        int thread_id, int numthreads);

/* Project a Grid Volume --------------------------------------------------- */
/** Projection of a Grid Volume.

    Project a grid volume with a basis.
    The Grid volume is projected onto a projection plane defined by
    (rot, tilt, psi) (1st, 2nd and 3rd Euler angles). The projection
    is previously is resized to Ydim x Xdim and initialized to 0.
    The projection itself, from now on, will keep the Euler angles.

    FORWARD process:
       Each volume of the grid is projected on to the projection plane.
       The output is the projection itself and a normalising image, the
       normalising image is the projection of the same grid supposing
       that all basis are of value 1. This normalising image is used by
       the ART process

    BACKWARD process:
       During the backward process the normalising projection contains
       the correction image to apply to the volume (in the ART sense).
       The output is the volume itself, the projection image is useless
       in this case, and the normalising projection is not modified at
       all.

    As for the mode, valid modes are ARTK, CAVK, COUNT_EQ, CAVARTK.

    M is the matrix corresponding to the projection process.
*/
template <class T>
void project_GridVolume(
    GridVolumeT<T> &vol,                  // Volume
    const Basis &basis,                   // Basis
    Projection       &proj,               // Projection
    Projection       &norm_proj,          // Projection of a unitary volume
    int              Ydim,                // Dimensions of the projection
    int              Xdim,
    double rot, double tilt, double psi,  // Euler angles
    int              FORW,                // 1 if we are projecting a volume
    //   norm_proj is calculated
    // 0 if we are backprojecting
    //   norm_proj must be valid
    int              eq_mode,             // ARTK, CAVARTK, CAVK or CAV
    GridVolumeT<int> *GVNeq,              // Number of equations per blob
    Matrix2D<double> *M,                  // System matrix
    const MultidimArray<int> *mask,            // mask(i,j)=0 => do not update this pixel
    double            ray_length,   // Ray length of the projection
    int               threads);

#undef x0
#undef xF
#undef y0
#undef yF
#undef xDim
#undef yDim
//@}
#endif
