/***************************************************************************
 *
 * Authors:    Carlos Oscar             coss@cnb.csic.es
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

#ifndef _PROG_NMA_ALIGNMENT
#define _PROG_NMA_ALIGNMENT

#include "condor/ObjectiveFunction.h"
#include "core/matrix1d.h"
#include "core/xmipp_image.h"
#include "core/xmipp_metadata_program.h"
#include "core/rerunable_program.h"

class ProgPdbConverter;

/**@defgroup NMAAlignmentVol Alignment of volumes with Normal modes
   @ingroup ReconsLibrary */
//@{
/** NMA Alignment Parameters. */
class ProgNmaAlignmentVol: public XmippMetadataProgram, public Rerunable
{
public:
    /** MPI version */
    bool MPIversion;
    
    /** Resume computations */
    bool resume;

    /// Reference atomic or pseudo-atomic structure in PDB format
    FileName fnPDB;

    /// Output PDB
    FileName fnOutPDB;

    /// Output directory
    FileName fnOutDir;

    /// File with a list of mode filenames
    FileName fnModeList;

    /// Pixel size in Angstroms
    double sampling_rate;
    
    /// Mask file for 2D masking of the projections of the deformed volume
    FileName fnmask;

    /// Center the PDB structure
    bool do_centerPDB;

    /// Low-pass filter the volume from PDB 
    bool do_FilterPDBVol;

    /// Low-pass cut-off frequency
    double cutoff_LPfilter;

    /// Use pseudo-atoms instead of atoms
    bool useFixedGaussian;

    /// Gaussian standard deviation for pseudo-atoms
    double sigmaGaussian;
 
    /// Align volumes
    bool alignVolumes;

    /// Parameters required from the CONDOR optimization
    double trustradius_scale;
    double rhoStartBase;
    double rhoEndBase;
    int niter;

    // starting and ending tilt angles for compensating for a single tilt wedge mask for tomography data
    int tilt0, tiltF;

    // maximum search frequency and shift while rigid body alignment
    double frm_freq;
    int frm_shift;


public:

    // Random generator seed
    int rangen;
    
    // All estimated parameters (with the cost)
    Matrix1D<double> parameters;

    // Trial parameters
    Matrix1D<double> trial;

    // Best trial parameters
    Matrix1D<double> trial_best;

    // Best fitness
    double fitness_min;
    
    // Number of modes
    int numberOfModes;

    // Size of the volumes in the selfile
    int imgSize;
    
    // Current volume being considered
    FileName currentVolName;
    
    // Template for temporal filename generation
    char nameTemplate[256];

    // Volume from PDB
    ProgPdbConverter* progVolumeFromPDB;

    // Volume that is being fitted
    Image<double> V, Vdeformed;

    // Mask
    MultidimArray<int> mask;

    // for fetching the rigid-body alignment parameters for each volume
    FILE *AnglesShiftsAndScore;
    float Best_Angles_Shifts[6];
    float fit_value;

    // flag indicates if there is a compensation for the missing wedge (volumes are rotated by 90 degrees about y axis for this purpose)
    bool flip = false;

public:
    /// Empty constructor
    ProgNmaAlignmentVol();

    /// Destructor
    ~ProgNmaAlignmentVol();

    /// Define params
    void defineParams();

    /// Read arguments from command line
    void readParams();

    /// Show
    void show();

   /** Create deformed PDB */
    FileName createDeformedPDB() const;

    /** Computes the fitness of a set of trial parameters */
    double computeFitness(Matrix1D<double> &trial) const;

    /** Update the best fitness and the corresponding best trial*/
    bool updateBestFit(double fitness);

    /** Produce side info.
        An exception is thrown if any of the files is not found*/
    virtual void preProcess();
    /** Assign NMA and Alignment parameters to a volume */
    virtual void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

    /** Write the final parameters. */
    virtual void finishProcessing();

    /** Write the parameters found for one image */
    virtual void writeVolumeParameters(const FileName &fnImg);


  protected:
    virtual void createWorkFiles() {
      return Rerunable::createWorkFiles(resume, getInputMd());
    }

  private:
    using Rerunable::createWorkFiles;
    
    std::vector<MDLabel> getLabelsForEmpty() override {
      return std::vector<MDLabel>{MDL_IMAGE,     MDL_ENABLED,    MDL_IMAGE,
                                  MDL_ANGLE_ROT, MDL_ANGLE_TILT, MDL_ANGLE_PSI,
                                  MDL_SHIFT_X,   MDL_SHIFT_Y,    MDL_SHIFT_Z,
                                  MDL_NMA,       MDL_NMA_ENERGY, MDL_MAXCC,
                                  MDL_ANGLE_Y};
    }
};

class ObjFunc_nma_alignment_vol: public UnconstrainedObjectiveFunction
{
  public:
    ObjFunc_nma_alignment_vol(int _t, int _n=0);
    ~ObjFunc_nma_alignment_vol(){};
    double eval(Vector v, int *nerror=nullptr);
};

#endif
