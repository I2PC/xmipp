/***************************************************************************
 *
 * Authors:  Mohamad Harastani mohamad.harastani@upmc.fr
 *	         Slavica Jonic slavica.jonic@upmc.fr
 *           Carlos Oscar Sanchez Sorzano coss.eps@ceu.es
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
#ifndef _PROG_VOLUMESET_ALIGN
#define _PROG_VOLUMESET_ALIGN

#include <core/xmipp_program.h>
#include <core/metadata.h>
#include <core/xmipp_image.h>

class ProgVolumeSetAlign: public XmippMetadataProgram
{
public:

    /** Resume computations */
    bool resume = false;

    /// Reference volume structure
    FileName fnREF;

    /// Output directory
    FileName fnOutDir;
 
    /// Align volumes
    bool alignVolumes;

    // starting and ending tilt angles for compensating for a single tilt wedge mask for tomography data
    int tilt0;
    int tiltF;

    // maximum search frequency and shift while rigid body alignment
    double frm_freq;
    int frm_shift;
    
    // for fetching the rigid-body alignment parameters for each volume
    FILE *fnAnglesAndShifts;
    float Matrix_Angles_Shifts[6];
    float fitness;
    
    // flag indicates if there is a compensation for the missing wedge (volumes are rotated by 90 degrees about y axis for this purpose)
    bool flipped = false;
    
    // Random generator seed
    int rangen = 0;
    
    // All estimated parameters (with the cost)
    Matrix1D<double> parameters;

    // Current volume being considered
    FileName currentVolName = "";
    
    // Template for temporal filename generation
    char nameTemplate[256];

    /// Empty constructor
    ProgVolumeSetAlign();

    /// Define params
    void defineParams();

    /// Read arguments from command line
    void readParams();
    
    /** Create the processing working files.
     * The working files are:
     * To_Do.xmd for images to process (nmaTodo = mdIn - nmaDone)
     * Done.xmd image already processed (could exists from a previous run)
     */
    virtual void createWorkFiles();

    /** Produce side info.
        An exception is thrown if any of the files is not found*/
    virtual void preProcess();
    
    /** Assign NMA and Alignment parameters to a volume */
    void processImage(const FileName &fnImg, const FileName &, const MDRow &, MDRow &);

    /** Write the final parameters. */
    virtual void finishProcessing();

    /** Write the parameters found for one image */
    virtual void writeVolumeParameters(const FileName &fnImg);

private:
    void computeFitness();
   
};

#endif

