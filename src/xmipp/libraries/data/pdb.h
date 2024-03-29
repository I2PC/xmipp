/***************************************************************************
 *
 * Authors:     Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es)
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
/*****************************************************************************/
/* INTERACTION WITH PDBs                                                     */
/*****************************************************************************/

#ifndef _XMIPP_PDB_HH
#define _XMIPP_PDB_HH

#include <vector>
#include "cif++.hpp"
#include "core/xmipp_error.h"

template<typename T>
class Matrix1D;
template<typename T>
class Matrix2D;
template<typename T>
class MultidimArray;
class FileName;
class Projection;
class Histogram1D;

/**@defgroup PDBinterface PDB
   @ingroup InterfaceLibrary */
//@{
/** Returns the charge of an atom.
    Returns 0 if the atom is not within the short list (H, C, N, O, S, P, Fe)
    of valid atoms. */
int atomCharge(const std::string &atom);

/** Returns the radius of an atom.
    Returns 0 if the atom is not within the short list (H, C, N, O, S, P, Fe)
    of valid atoms.
    
    The radius data is taken from http://www.webelements.com as the empirical
    radius. */
double atomRadius(const std::string &atom);

/** Returns the covalent radius of an atom.
    Returns 0 if the atom is not within the short list (H, C, N, O, S, P, Fe)
    of valid atoms.
    The radius data is taken from http://www.webelements.com as the empirical
    radius. */
double atomCovalentRadius(const std::string &atom);

/** Compute the center of mass and limits of a PDB file.
    The intensity column is used only for the pseudoatoms. It specifies
    from which column we should read the intensity. Valid columns are
    Bfactor or occupancy.
*/
void computePDBgeometry(const std::string &fnPDB,
                        Matrix1D<double> &centerOfMass,
                        Matrix1D<double> &limit0, Matrix1D<double> &limitF,
                        const std::string &intensityColumn);

/** Apply geometry transformation to an input PDB.
    The result is written in the output PDB. Set centerPDB if you
    want to compute the center of mass first and apply the transformation
    after centering the PDB. */

void applyGeometryToPDBFile(const std::string &fn_in, const std::string &fn_out,
                   const Matrix2D<double> &A, bool centerPDB=true,
                   const std::string &intensityColumn="occupancy");

/** pdbdata is an struct that contains the coordiantes of the atom positions defined
as x, y, z, the b factor, b, the residue of each atom, and the covalent radiues. These
variables are defined as vectors*/
struct pdbInfo
{
	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> z;
	std::vector<double> b;
	std::vector<std::string> chain;
	std::vector<int> residue;
	std::vector<double> atomCovRad;
};

/** ANALYZEPDBDATA takes as input a filename of a pdb file (atomic model) and selects only
the typeOfAtom, (for instance the C-alpha atoms) storing the atom positions, b- factor,
the residue of each atom, and the covalent radiues in a struct vector at_pos. Also the number
of atoms is kept.*/
void analyzePDBAtoms(const FileName &fn_pdb, const std::string &typeOfAtom, int &numberOfAtoms, pdbInfo &at_pos);


/** Atom class. */
class Atom
{
public:
    /// Type
    char atomType;

    /// Position X
    double x;

    /// Position Y
    double y;

    /// Position Z
    double z;
};

/** Phantom description using atoms. */
class PDBPhantom
{
public:
    /// List of atoms
    std::vector<Atom> atomList;

    // Whole data block
    cif::datablock dataBlock;

    /// Add Atom
    void addAtom(const Atom &atom)
    {
        atomList.push_back(atom);
    }

    /// Get Atom at position i
    const Atom& getAtom(int i) const
    {
        return atomList[i];
    }

    /// Get number of atoms
    size_t getNumberOfAtoms() const
    {
        return atomList.size();
    }

    /**
     * @brief Read phantom from either a PDB of CIF file.
     * 
     * This function reads the given PDB or CIF file and inserts the found atoms inside in class's atom list.
     * 
     * @param fnPDB PDB/CIF file.
    */
    void read(const FileName &fnPDB);

    /// Apply a shift to all atoms
    void shift(double x, double y, double z);

    /** Produce side info.
        The side info produces the radial profiles of each atom
    and its projections.
    */
    void produceSideInfo();
};

/** Atom class. */
class RichAtom
{
public:
    /// Record Type ("ATOM  " or "HETATM")
    std::string record;

    /// atom serial number
    int serial;

    /// Position X
    double x;

    /// Position Y
    double y;

    /// Position Z
    double z;

    /// Name
    std::string name;

    /// Alternate location
    std::string altloc;

    /// Residue name
    std::string resname;

    /// ChainId
    char chainid;

    /// Residue sequence
    int resseq;

    /// Icode
    std::string icode;

    /// Occupancy
    double occupancy;

    /// Bfactor
    double bfactor;

    /// atom element type
    std::string atomType;

    /// 2-char charge with sign 2nd (e.g. 1- or 2+)
    std::string charge;

    /* PDB Specific values */
    /// segment name
    std::string segment;

    /* CIF Specific values */
    // Alternative id
    std::string altId;

    // Sequence id
    int seqId;

    // Author sequence id
    int authSeqId;

    // Author chain name
    std::string authCompId;

    // Author chain location
    std::string authAsymId;

    // Author atom name
    std::string authAtomId;

    // PDB model number
    int pdbNum;
};

/** Phantom description using atoms. */
class PDBRichPhantom
{
public:
	/// List of remarks
	std::vector<std::string> remarks;

    /// List of atoms
    std::vector<RichAtom> atomList;
    std::vector<double> intensities;

    // Whole data block
    cif::datablock dataBlock;

    /// Add Atom
    void addAtom(const RichAtom &atom)
    {
        atomList.push_back(atom);
    }

    /// Get number of atoms
    size_t getNumberOfAtoms() const
    {
        return atomList.size();
    }

    /**
     * @brief Read rich phantom from either a PDB of CIF file.
     * 
     * This function reads the given PDB or CIF file and stores the found atoms, remarks, and intensities.
     * 
     * @param fnPDB PDB/CIF file.
     * @param pseudoatoms Flag for returning intensities (stored in B-factors) instead of atoms.
     *  **false** (default) is used when there are no pseudoatoms or when using a threshold.
     * @param threshold B factor threshold for filtering out for pdb_reduce_pseudoatoms.
    */
    void read(const FileName &fnPDB, const bool pseudoatoms = false, const double threshold = 0.0);

    /**
     * @brief Write rich phantom to PDB or CIF file.
     * 
     * This function stores all the data of the rich phantom into a PDB or CIF file.
     * Note: Conversion is not enabled yet, so if a file read from a PDB is written into a CIF file,
     * results might not be great. Atoms should be properly translated, but remarks and intensities probably not.
     * 
     * @param fnPDB PDB/CIF file to write to.
     * @param renumber Flag for determining if atom's serial numbers must be renumbered or not.
    */
    void write(const FileName &fnPDB, const bool renumber = false);

};

/** Description of the electron scattering factors.
    The returned descriptor is descriptor(0)=Z (number of electrons of the
    atom), descriptor(1-5)=a1-5, descriptor(6-10)=b1-5.
    The electron scattering factor at a frequency f (Angstroms^-1)
    is computed as f_el(f)=sum_i(ai exp(-bi*x^2)). Use the function
    electronFormFactorFourier or 
    
    See Peng, Ren, Dudarev, Whelan. Robust parameterization of elastic and
    absorptive electron atomic scattering factors. Acta Cryst. A52: 257-276
    (1996). Table 3 and equation 3.*/
void atomDescriptors(const std::string &atom, Matrix1D<double> &descriptors);

/** Compute the electron Form Factor in Fourier space.
    The electron scattering factor at a frequency f (Angstroms^-1)
    is computed as f_el(f)=sum_i(ai exp(-bi*x^2)). */
double electronFormFactorFourier(double f,
                                 const Matrix1D<double> &descriptors);

/** Compute the electron Form Factor in Real space.
    Konwing the electron form factor in Fourier space is easy to make an
    inverse Fourier transform and express it in real space. r is the
    distance to the center of the atom in Angstroms. */
double electronFormFactorRealSpace(double r,
                                   const Matrix1D<double> &descriptors);

/** Atom radial profile.
    Returns the radial profile of a given atom, i.e., the electron scattering
    factor convolved with a suitable low pass filter for sampling the volume
    at a sampling rate M*T. The radial profile is sampled at T Angstroms/pixel.
*/
void atomRadialProfile(int M, double T, const std::string &atom,
                       Matrix1D<double> &profile);

/** Atom projection radial profile.
    Returns the radial profile of the atom described by its profileCoefficients
    (Bspline coefficients). */
void atomProjectionRadialProfile(int M,
                                 const Matrix1D<double> &profileCoefficients,
                                 Matrix1D<double> &projectionProfile);

/** Class for Atom interpolations. */
class AtomInterpolator
{
public:
    // Vector of radial volume profiles
    std::vector< MultidimArray<double> > volumeProfileCoefficients;
    // Vector of radial projection profiles
    std::vector< MultidimArray<double> > projectionProfileCoefficients;
    // Vector of atom radii
    std::vector<double> radii;
    // Downsampling factor
    int M;
    // Fine sampling rate
    double highTs;

    /** Setup.
        HighTs is a fine sampling rate, M is an integer number so that
    the final sampling rate is Ts=M*highTs; */
    void setup(int m, double hights, bool computeProjection=false);

    /// Add atom
    void addAtom(const std::string &atomType, bool computeProjection=false);

    /// Get atom index
    int getAtomIndex(char atom) const
    {
        int idx=-1;
        switch (atom)
        {
        case 'H':
            idx=0;
            break;
        case 'C':
            idx=1;
            break;
        case 'N':
            idx=2;
            break;
        case 'O':
            idx=3;
            break;
        case 'P':
            idx=4;
            break;
        case 'S':
            idx=5;
            break;
        case 'E': // Iron Fe
            idx=6;
            break;
        case 'K':
            idx=7;
            break;
        case 'F':
             idx=8;
             break;
        case 'G': // Magnesium Mg
             idx=9;
             break;
        case 'L': // Chlorine Cl
             idx=10;
             break;
        case 'A': // Calcium Ca
             idx=11;
             break;
        default:
            REPORT_ERROR(ERR_VALUE_INCORRECT,(std::string)
                         "AtomInterpolator::getAtomIndex: Atom "+atom+" unknown");
        }
        return idx;
    }

    /** Radius of an atom in the final sampling rate M*highTs. */
    double atomRadius(char atom) const
    {
        return radii[getAtomIndex(atom)];
    }

    /** Volume value at a distance r of the atom whose first letter
        is the one provided as atom. */
    double volumeAtDistance(char atom, double r) const;

    /** Projection value at a distance r of the atom whose first letter
        is the one provided as atom. */
    double projectionAtDistance(char atom, double r) const;
};

/** Project PDB.
    Project the PDB following a certain projection direction. */
void projectPDB(const PDBPhantom &phantomPDB,
                const AtomInterpolator &interpolator, Projection &proj,
                int Ydim, int Xdim, double rot, double tilt, double psi);

/** Compute distance histogram of a PDB phantom.
 * Consider the distance between each atom and its N nearest neighbours. Then, compute the histogram of these distances
 * with Nbin samples.
 */
void distanceHistogramPDB(const PDBPhantom &phantomPDB, size_t Nnearest, double maxDistance, int Nbins, Histogram1D &hist);
//@}
#endif

const char*
hy36encode(unsigned width, int value, char* result);

const char*
hy36decode(unsigned width, const char* s, unsigned s_size, int* result);

void
hy36encodeSafe(unsigned width, int value, char* result);

void
hy36decodeSafe(unsigned width, const char* s, unsigned s_size, int* result);