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

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "cif++.hpp"
#include "pdb.h"
#include "core/matrix2d.h"
#include "core/multidim_array.h"
#include "core/transformations.h"
#include "core/xmipp_fftw.h"
#include "core/xmipp_strings.h"
#include "data/fourier_projection.h"
#include "data/integration.h"
#include "data/mask.h"
#include "data/numerical_tools.h"

/* If you change the include guards, please be sure to also rename the
   functions below. Otherwise your project will clash with the original
   iotbx declarations and definitions.
 */
#ifndef IOTBX_PDB_HYBRID_36_C_H
#define IOTBX_PDB_HYBRID_36_C_H

#ifdef __cplusplus
extern "C" {
#endif

#define HY36_WIDTH_4_MIN -999
#define HY36_WIDTH_4_MAX 2436111 /* 10000 + 2*26*36*36*36 - 1 */
#define HY36_WIDTH_5_MIN -9999
#define HY36_WIDTH_5_MAX 87440031 /* 100000 + 2*26*36*36*36*36 - 1 */

#ifdef __cplusplus
}
#endif
#endif /* IOTBX_PDB_HYBRID_36_C_H */

void analyzePDBAtoms(const FileName &fn_pdb, const std::string &typeOfAtom, int &numberOfAtoms, pdbInfo &at_pos)
{
	//Open the pdb file
	std::ifstream f2parse;
	f2parse.open(fn_pdb.c_str());

	numberOfAtoms = 0;

    // Initializing and reading pdb file
    PDBRichPhantom pdbFile;

    // Read centered pdb
    pdbFile.read(fn_pdb.c_str());

    // For each atom, store necessary info if type matches
    for (auto& atom : pdbFile.atomList) {
        if (atom.name == typeOfAtom) {
            numberOfAtoms++;

            // Storing coordinates
            at_pos.x.push_back(atom.x);
            at_pos.y.push_back(atom.y);
            at_pos.z.push_back(atom.z);
            at_pos.chain.push_back(std::string(1, atom.chainid));

            // Residue Number
            at_pos.residue.push_back(atom.resseq);

            // Getting the bfactor = 8pi^2*u
            at_pos.b.push_back(atom.bfactor); //sqrt(atom.bfactor/(8*PI*PI));

            // Covalent radius of the atom
            at_pos.atomCovRad.push_back(atomCovalentRadius(atom.name));
        }
    }
}

double AtomInterpolator::volumeAtDistance(char atom, double r) const
{
    int idx=getAtomIndex(atom);
    if (r>radii[idx])
        return 0;
    else
        return volumeProfileCoefficients[idx].
               interpolatedElementBSpline1D(r*M,3);
}

double AtomInterpolator::projectionAtDistance(char atom, double r) const
{
    int idx=getAtomIndex(atom);
    if (r>radii[idx])
        return 0;
    else
        return projectionProfileCoefficients[idx].
               interpolatedElementBSpline1D(r*M,3);
}

/* Atom charge ------------------------------------------------------------- */
int atomCharge(const std::string &atom)
{
    switch (atom[0])
    {
    case 'H':
        return 1;
        break;
    case 'C':
        return 6;
        break;
    case 'N':
        return 7;
        break;
    case 'O':
        return 8;
        break;
    case 'P':
        return 15;
        break;
    case 'S':
        return 16;
        break;
    case 'E': // Iron Fe
        return 26;
        break;
    case 'K':
        return 19;
        break;
    case 'F':
        return 9;
        break;
    case 'G': // Magnesium Mg
        return 12;
        break;
    case 'L': // Chlorine Cl
        return 17;
        break;
    case 'A': // Calcium Ca
        return 20;
        break;
    default:
        return 0;
    }
}

/* Atom radius ------------------------------------------------------------- */
double atomRadius(const std::string &atom)
{
    switch (atom[0])
    {
    case 'H':
        return 0.25;
        break;
    case 'C':
        return 0.70;
        break;
    case 'N':
        return 0.65;
        break;
    case 'O':
        return 0.60;
        break;
    case 'P':
        return 1.00;
        break;
    case 'S':
        return 1.00;
        break;
    case 'E': // Iron Fe
        return 1.40;
        break;
    case 'K':
        return 2.20;
        break;
    case 'F':
        return 0.50;
        break;
    case 'G': // Magnesium Mg
        return 1.50;
        break;
    case 'L': // Chlorine Cl
        return 1.00;
        break;
    case 'A': // Calcium Ca
        return 1.80;
        break;
    default:
        return 0;
    }
}

/* Atom Covalent radius ------------------------------------------------------------- */
double atomCovalentRadius(const std::string &atom)
{
    switch (atom[0])
    {
    case 'H':
        return 0.38;
    case 'C':
        return 0.77;
    case 'N':
        return 0.75;
    case 'O':
        return 0.73;
    case 'P':
        return 1.06;
    case 'S':
        return 1.02;
    case 'F': // Iron
        return 1.25;
    default:
        return 0;
    }
}

/* Compute geometry -------------------------------------------------------- */
void computePDBgeometry(const std::string &fnPDB,
                        Matrix1D<double> &centerOfMass,
                        Matrix1D<double> &limit0, Matrix1D<double> &limitF,
                        const std::string &intensityColumn)
{
    // Initialization
    centerOfMass.initZeros(3);
    limit0.resizeNoCopy(3);
    limitF.resizeNoCopy(3);
    limit0.initConstant(1e30);
    limitF.initConstant(-1e30);
    double total_mass = 0;

    // Initialize PDBRichPhantom and read atom struct file
    PDBRichPhantom pdbFile;

    // Read centered pdb
    pdbFile.read(fnPDB);

    // For each atom, correct necessary info
    bool useBFactor = intensityColumn=="Bfactor";
    for (auto& atom : pdbFile.atomList) {
        // Update center of mass and limits
        XX(limit0) = std::min(XX(limit0), atom.x);
        YY(limit0) = std::min(YY(limit0), atom.y);
        ZZ(limit0) = std::min(ZZ(limit0), atom.z);

        XX(limitF) = std::max(XX(limitF), atom.x);
        YY(limitF) = std::max(YY(limitF), atom.y);
        ZZ(limitF) = std::max(ZZ(limitF), atom.z);

        double weight;
        if (atom.name == "EN")
        {
            if (useBFactor)
                weight = atom.bfactor;
            else
                weight = atom.occupancy;
        }
        else
        {
            if (atom.record == "HETATM")
                continue;
            weight = (double) atomCharge(atom.name);
        }
        total_mass += weight;
        XX(centerOfMass) += weight * atom.x;
        YY(centerOfMass) += weight * atom.y;
        ZZ(centerOfMass) += weight * atom.z;
    }

    // Finish calculations
    centerOfMass /= total_mass;
}

/* Apply geometry ---------------------------------------------------------- */
void applyGeometryToPDBFile(const std::string &fn_in, const std::string &fn_out,
                   const Matrix2D<double> &A, bool centerPDB,
                   const std::string &intensityColumn)
{
    Matrix1D<double> centerOfMass;
    Matrix1D<double> limit0;
    Matrix1D<double> limitF;
    if (centerPDB)
    {
        computePDBgeometry(fn_in, centerOfMass,limit0, limitF,
                           intensityColumn);
        limit0 -= centerOfMass;
        limitF -= centerOfMass;
    }

    // Open files
    std::ifstream fh_in;
    fh_in.open(fn_in.c_str());
    if (!fh_in)
        REPORT_ERROR(ERR_IO_NOTEXIST, fn_in);
    std::ofstream fh_out;
    fh_out.open(fn_out.c_str());
    if (!fh_out)
        REPORT_ERROR(ERR_IO_NOWRITE, fn_out);

    // Process all lines of the file
    while (!fh_in.eof())
    {
        // Read an ATOM line
        std::string line;
        getline(fh_in, line);
        if (line == "")
        {
            fh_out << "\n";
            continue;
        }
        std::string kind = line.substr(0,4);
        if (kind != "ATOM" && kind != "HETA")
        {
            fh_out << line << std::endl;
            continue;
        }

        // Extract atom type and position
        // Typical line:
        // ATOM    909  CA  ALA A 161      58.775  31.984 111.803  1.00 34.78
        double x = textToFloat(line.substr(30,8));
        double y = textToFloat(line.substr(38,8));
        double z = textToFloat(line.substr(46,8));

        Matrix1D<double> v(4);
        if (centerPDB)
        {
            VECTOR_R3(v,x-XX(centerOfMass),
                      y-YY(centerOfMass),z-ZZ(centerOfMass));
        }
        else
        {
            VECTOR_R3(v,x,y,z);
        }
        v(3)=1;
        v=A*v;

        char aux[15];
        sprintf(aux,"%8.3f",XX(v));
        line.replace(30,8,aux);
        sprintf(aux,"%8.3f",YY(v));
        line.replace(38,8,aux);
        sprintf(aux,"%8.3f",ZZ(v));
        line.replace(46,8,aux);

        fh_out << line << std::endl;
    }

    // Close files
    fh_in.close();
    fh_out.close();
}

/**
 * @brief Checks if the file uses a supported extension type.
 * 
 * This function checks if the given file path has one of the given supported extensions, with or without compression
 * in any of the accepted compressions.
 * 
 * @param filePath File including path.
 * @param acceptedExtensions List of accepted extensions.
 * @param acceptedCompressions List of accepted compressions.
 * @return true if the extension is valid, false otherwise.
*/
bool checkExtension(const std::filesystem::path &filePath, const std::list<std::string> &acceptedExtensions, const std::list<std::string> &acceptedCompressions) {
    // File extension is invalid by default 
    bool validExtension = false;

    // Checking if file extension is in accepted extensions with or without an accepted compression
    if (find(acceptedExtensions.begin(), acceptedExtensions.end(), filePath.extension()) != acceptedExtensions.end()) {
        // Accepted extension without compression
        validExtension = true;
    } else {
        if (find(acceptedCompressions.begin(), acceptedCompressions.end(), filePath.extension()) != acceptedCompressions.end()) {
            // Accepted compression detected
            // Checking if next extension is valid
            const std::filesystem::path shortedPath = filePath.parent_path().u8string() + "/" + filePath.stem().u8string();
            if (find(acceptedExtensions.begin(), acceptedExtensions.end(), shortedPath.extension()) != acceptedExtensions.end()) {
                // Accepted extension with compression
                validExtension = true;
            }
        }
    }

    // Returning calculated validity
    return validExtension;
}

template<typename callable>
/**
 * @brief Read phantom from PDB.
 * 
 * This function reads the given PDB file and inserts the found atoms inside in the class's atom list.
 * 
 * @param fnPDB PDB file.
 * @param addAtom Function to add atoms to class's atom list.
*/
void readPDB(const FileName &fnPDB, const callable &addAtom)
{
    // Open file
    std::ifstream fh_in;
    fh_in.open(fnPDB.c_str());
    if (!fh_in)
        REPORT_ERROR(ERR_IO_NOTEXIST, fnPDB);

    // Process all lines of the file
    std::string line;
    std::string kind;
    Atom atom;
    while (!fh_in.eof())
    {
        // Read an ATOM line
        getline(fh_in, line);
        if (line == "")
            continue;
        kind = line.substr(0, 4);
        if (kind != "ATOM" && kind != "HETA")
            continue;

        // Extract atom type and position
        // Typical line:
        // ATOM    909  CA  ALA A 161      58.775  31.984 111.803  1.00 34.78
        atom.atomType = line[13];
        atom.x = textToFloat(line.substr(30, 8));
        atom.y = textToFloat(line.substr(38, 8));
        atom.z = textToFloat(line.substr(46, 8));
        addAtom(atom);
    }

    // Close files
    fh_in.close();
}

template<typename callable>
/**
 * @brief Read phantom from CIF.
 * 
 * This function reads the given CIF file and inserts the found atoms inside in the class's atom list.
 * 
 * @param fnPDB CIF file path.
 * @param addAtom Function to add atoms to class's atom list.
 * @param dataBlock Data block used to store all of CIF file's fields.
*/
void readCIF(const std::string &fnCIF, const callable &addAtom, cif::datablock &dataBlock)
{
    // Parsing mmCIF file
    cif::file cifFile;
    cifFile.load(fnCIF);

    // Extrayendo datos del archivo en un DataBlock
    cif::datablock& db = cifFile.front();

    // Reading Atom section
    // Typical line:
    // ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
    cif::category& atom_site = db["atom_site"];

    // Iterating through atoms and heteroatoms getting atom id and x,y,z positions
    Atom atom;
	for (const auto& [atom_id, x_pos, y_pos, z_pos]: atom_site.find
        <std::string,float,float,float>
        (
            cif::key("group_PDB") == "ATOM" || cif::key("group_PDB") == "HETATM",
            "label_atom_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z"
        ))
	{
        // Obtaining:
        // ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
        //               *                        ******  ******* ******
        atom.atomType = atom_id[0];
        atom.x = x_pos;
        atom.y = y_pos;
        atom.z = z_pos;
        addAtom(atom);
	}

    // Storing whole datablock
    dataBlock = db;
}

void PDBPhantom::read(const FileName &fnPDB)
{
    // Checking if extension is .cif or .pdb
    if (checkExtension(fnPDB.getString(), {".cif"}, {".gz"})) {
        readCIF(fnPDB.getString(), bind(&PDBPhantom::addAtom, this, std::placeholders::_1), dataBlock);
    } else {
        readPDB(fnPDB, bind(&PDBPhantom::addAtom, this, std::placeholders::_1));
    }
}

/* Shift ------------------------------------------------------------------- */
void PDBPhantom::shift(double x, double y, double z)
{
    int imax=atomList.size();
    for (int i=0; i<imax; i++)
    {
        atomList[i].x+=x;
        atomList[i].y+=y;
        atomList[i].z+=z;
    }
}

template<typename callable>
/**
 * @brief Read rich phantom from either a PDB of CIF file.
 * 
 * This function reads the given PDB or CIF file and stores the found atoms, remarks, and intensities.
 * 
 * @param fnPDB PDB/CIF file.
 * @param addAtom Function to add atoms to class's atom list.
 * @param intensities List of atom intensities.
 * @param remarks List of file remarks.
 * @param pseudoatoms Flag for returning intensities (stored in B-factors) instead of atoms. false (default) is used when there are no pseudoatoms or when using a threshold.
 * @param threshold B factor threshold for filtering out for pdb_reduce_pseudoatoms.
*/
void readRichPDB(const FileName &fnPDB, const callable &addAtom, std::vector<double> &intensities,
    std::vector<std::string> &remarks, const bool pseudoatoms, const double threshold)
{
    // Open file
    std::ifstream fh_in;
    fh_in.open(fnPDB.c_str());
    if (!fh_in)
        REPORT_ERROR(ERR_IO_NOTEXIST, fnPDB);

    // Process all lines of the file
    auto line = std::string(80, ' ');
    std::string kind;

    RichAtom atom;
    while (!fh_in.eof())
    {
        // Read an ATOM line
        getline(fh_in, line);
        if (line == "")
        {
            continue;
        }

        // Reading and storing type of atom
        kind = simplify(line.substr(0, 6)); // Removing extra spaces if there are any

        if (kind == "ATOM" || kind == "HETATM")
        {
			line.resize (80,' ');

			// Extract atom type and position
			// Typical line:
			// ATOM    909  CA  ALA A 161      58.775  31.984 111.803  1.00 34.78
			// ATOM      2  CA  ALA A   1      73.796  56.531  56.644  0.50 84.78           C
			atom.record = kind;
			hy36decodeSafe(5, line.substr(6, 5).c_str(), 5, &atom.serial);
			atom.name = simplify(line.substr(12, 4)); // Removing extra spaces if there are any
			atom.altloc = line[16];
			atom.resname = line.substr(17, 3);
			atom.chainid = line[21];
			hy36decodeSafe(4, line.substr(22, 4).c_str(), 4, &atom.resseq);
			atom.icode = line[26];
			atom.x = textToFloat(line.substr(30, 8));
			atom.y = textToFloat(line.substr(38, 8));
			atom.z = textToFloat(line.substr(46, 8));
			atom.occupancy = textToFloat(line.substr(54, 6));
			atom.bfactor = textToFloat(line.substr(60, 6));
            if (line.length() >= 76 && simplify(line.substr(72, 4)) != "")
			    atom.segment = line.substr(72, 4);
            if (line.length() >= 78 && simplify(line.substr(77, 1)) != "")
			    atom.atomType = line.substr(77, 1);
            else
                atom.atomType = atom.name[0];
            if (line.length() >= 80 && simplify(line.substr(79, 1)) != "")
			    atom.charge = simplify(line.substr(79, 1)); // Converting into empty string if it is a space

			if(pseudoatoms)
				intensities.push_back(atom.bfactor);
            
			if(!pseudoatoms && atom.bfactor >= threshold)
				addAtom(atom);

		} else if (kind == "REMARK")
			remarks.push_back(line);
    }

    // Close files
    fh_in.close();
}

template<typename callable>
/**
 * @brief Read rich phantom from CIF.
 * 
 * This function reads the given CIF file and stores the found atoms, remarks, and intensities.
 * Note: CIF files do not contain segment name, so that data won't be read.
 * 
 * @param fnPDB CIF file path.
 * @param addAtom Function to add atoms to class's atom list.
 * @param intensities List of atom intensities.
 * @param pseudoatoms Flag for returning intensities (stored in B-factors) instead of atoms. false (default) is used when there are no pseudoatoms or when using a threshold.
 * @param threshold B factor threshold for filtering out for pdb_reduce_pseudoatoms.
 * @param dataBlock Data block used to store all of CIF file's fields.
*/
void readRichCIF(const std::string &fnCIF, const callable &addAtom, std::vector<double> &intensities,
    const bool pseudoatoms, const double threshold, cif::datablock &dataBlock)
{
    // Parsing mmCIF file
    cif::file cifFile;
    cifFile.load(fnCIF);

    // Extrayendo datos del archivo en un DataBlock
    cif::datablock& db = cifFile.front();

    // Reading Atom section
    cif::category& atom_site = db["atom_site"];

    // Iterating through atoms and heteroatoms getting atom id and x,y,z positions
    RichAtom atom;
	for (const auto& [record, serialNumber, atomId, altId, resName, chain, resSeq, seqId, iCode, xPos, yPos, zPos,
            occupancy, bFactor, charge, authSeqId, authCompId, authAsymId, authAtomId, pdbNum]:
        atom_site.find<std::string,int,std::string,std::string,std::string,std::string,int,int,std::string,float,
            float,float,float,float,std::string,int,std::string,std::string,std::string,int>
        (
            // Note: search by key is needed to iterate list of atoms. Workaround: use every possible record type for the search
            cif::key("group_PDB") == "ATOM" || cif::key("group_PDB") == "HETATM",
            "group_PDB",            // Record:          -->ATOM<--   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "id",                   // Serial number:   ATOM   -->8<--      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "label_atom_id",        // Id:              ATOM   8      C  -->CD1<-- . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "label_alt_id",         // Alt id:          ATOM   8      C  CD1 -->.<-- ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "label_comp_id",        // Chain name:      ATOM   8      C  CD1 . -->ILE<-- A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "label_asym_id",        // Chain location:  ATOM   8      C  CD1 . ILE -->A<--  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "label_entity_id",      // Residue sequence:ATOM   8      C  CD1 . ILE A  -->1<-- 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "label_seq_id",         // Sequence id:     ATOM   8      C  CD1 . ILE A  1 -->3<--    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "pdbx_PDB_ins_code",    // Ins code:        ATOM   8      C  CD1 . ILE A  1 3    -->?<-- 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "Cartn_x",              // X position:      ATOM   8      C  CD1 . ILE A  1 3    ? -->48.271<--  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "Cartn_y",              // Y position:      ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  -->183.605<-- 19.253  1.00 35.73  ? 3    ILE A CD1 1
            "Cartn_z",              // Z position:      ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 -->19.253<--  1.00 35.73  ? 3    ILE A CD1 1
            "occupancy",            // Occupancy:       ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  -->1.00<-- 35.73  ? 3    ILE A CD1 1
            "B_iso_or_equiv",       // B factor:        ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 -->35.73<--  ? 3    ILE A CD1 1
            "pdbx_formal_charge",   // Author charge:   ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  -->?<-- 3    ILE A CD1 1
            "auth_seq_id",          // Author seq id:   ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? -->3<--    ILE A CD1 1
            "auth_comp_id",         // Author chainname:ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    -->ILE<-- A CD1 1
            "auth_asym_id",         // Author chain loc:ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE -->A<-- CD1 1
            "auth_atom_id",         // Author id:       ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A -->CD1<-- 1
            "pdbx_PDB_model_num"    // PDB model number:ATOM   8      C  CD1 . ILE A  1 3    ? 48.271  183.605 19.253  1.00 35.73  ? 3    ILE A CD1 -->1<--
        ))
	{
        // Storing values in atom list
        atom.record = record;
        atom.serial = serialNumber;
        atom.name = atomId;
        atom.atomType = atomId;
        atom.altId = altId;
        atom.resname = resName;
        atom.altloc = chain;
        atom.chainid = chain[0];
        atom.resseq = resSeq;
        atom.seqId = seqId;
        atom.icode = iCode;
        atom.x = xPos;
        atom.y = yPos;
        atom.z = zPos;
        atom.occupancy = occupancy;
        atom.bfactor = bFactor;
        atom.charge = charge;
        atom.authSeqId = authSeqId;
        atom.authCompId = authCompId;
        atom.authAsymId = authAsymId;
        atom.authAtomId = authAtomId;
        atom.pdbNum = pdbNum;

        // If it is a pseudoatom, insert B factor into intensities
        if(pseudoatoms) {
            intensities.push_back(bFactor);
        } else {
            // Adding atom if is not pseudoatom and B factor is not greater than set threshold
            if (bFactor >= threshold)
                addAtom(atom);
        }
	}

    // Storing whole datablock
    dataBlock = db;
}

void PDBRichPhantom::read(const FileName &fnPDB, const bool pseudoatoms, const double threshold)
{
    // Checking if extension is .cif or .pdb
    if (checkExtension(fnPDB.getString(), {".cif"}, {".gz"})) {
        readRichCIF(fnPDB.getString(), bind(&PDBRichPhantom::addAtom, this, std::placeholders::_1), intensities, pseudoatoms, threshold, dataBlock);
    } else {
        readRichPDB(fnPDB, bind(&PDBRichPhantom::addAtom, this, std::placeholders::_1), intensities, remarks, pseudoatoms, threshold);
    }
}

template<typename callable>
/**
 * @brief Write rich phantom to PDB file.
 * 
 * This function stores all the data of the rich phantom into a PDB file.
 * 
 * @param fnPDB PDB file to write to.
 * @param renumber Flag for determining if atom's serial numbers must be renumbered or not.
 * @param remarks List of remarks.
 * @param atomList List of atoms to be stored.
*/
void writePDB(const FileName &fnPDB, bool renumber, const std::vector<std::string> &remarks, const callable &atomList)
{
    FILE* fh_out=fopen(fnPDB.c_str(),"w");
    if (!fh_out)
        REPORT_ERROR(ERR_IO_NOWRITE, fnPDB);
    size_t imax=remarks.size();
    for (size_t i=0; i<imax; ++i)
        fprintf(fh_out,"%s\n",remarks[i].c_str());

    imax=atomList.size();
    for (size_t i=0; i<imax; ++i)
    {
    	const RichAtom &atom=atomList[i];
        char serial[5+1];
        if (!renumber) {
            auto* errmsg3 = hy36encode(5, atom.serial, serial);
            if (errmsg3) {
                reportWarning("Failed to use atom.serial. Using i+1 instead.");
                renumber=true;
                hy36encodeSafe(5, (int)i + 1, serial);
            }
        }
        else {
            // use i+1 instead
            hy36encodeSafe(5, (int)i + 1, serial);
        }
        char resseq[4+1];
        hy36encodeSafe(4, atom.resseq, resseq);
        fprintf (fh_out,"%-6s%5s %-4s%s%-4s%c%4s%s   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%-2s\n",
				atom.record.c_str(),serial,atom.name.c_str(),
				atom.altloc.c_str(),atom.resname.c_str(),atom.chainid,
				resseq,atom.icode.c_str(),
				atom.x,atom.y,atom.z,atom.occupancy,atom.bfactor,
				atom.segment.c_str(),atom.atomType.c_str(),atom.charge.c_str());
    }
    fclose(fh_out);
}

template<typename callable>
/**
 * @brief Write rich phantom to CIF file.
 * 
 * This function stores all the data of the rich phantom into a CIF file.
 * 
 * @param fnPDB PDB file path to write to.
 * @param atomList List of atoms to be stored.
 * @param dataBlock Data block containing the full CIF file.
*/
void writeCIF(const std::string &fnCIF, const callable &atomList, cif::datablock &dataBlock)
{
    // Opening CIF file
    std::ofstream cifFile(fnCIF);

    // Creating atom_site category
    cif::category atomSite("atom_site");
    cif::row_initializer atomSiteInserter;

    // Declaring temporary variables for occupancy, coords, and Bfactor
    std::stringstream tempStream;
    std::string occupancy;
    std::string xPos;
    std::string yPos;
    std::string zPos;
    std::string bFactor;

    // Inserting data from atom list
    for (RichAtom atom : atomList) {
        // Converting occupancy to string with 2 fixed decimals
        tempStream << std::fixed << std::setprecision(2) << atom.occupancy;
        occupancy = tempStream.str();
        tempStream.clear();
        tempStream.str("");

        // Converting xPos to string with 3 fixed decimals
        tempStream << std::fixed << std::setprecision(3) << atom.x;
        xPos = tempStream.str();
        tempStream.clear();
        tempStream.str("");

        // Converting yPos to string with 3 fixed decimals
        tempStream << std::fixed << std::setprecision(3) << atom.y;
        yPos = tempStream.str();
        tempStream.clear();
        tempStream.str("");

        // Converting zPos to string with 3 fixed decimals
        tempStream << std::fixed << std::setprecision(3) << atom.z;
        zPos = tempStream.str();
        tempStream.clear();
        tempStream.str("");

        // Converting bFactor to string with 2 fixed decimals
        tempStream << std::fixed << std::setprecision(2) << atom.bfactor;
        bFactor = tempStream.str();
        tempStream.clear();
        tempStream.str("");

        // Defining row
        // Empty string or char values are substitued with "." or '.' (no value)
        // No "?" are inserted since it is not allowed by spec
        atomSiteInserter = {
            {"group_PDB", atom.record.empty() ? "." : atom.record},
            {"id", atom.serial},
            {"type_symbol", atom.name.empty() ? '.' : atom.name[0]},
            {"label_atom_id", atom.name.empty() ? "." : atom.name},
            {"label_alt_id", atom.altId.empty() ? "." : atom.altId},
            {"label_comp_id", atom.resname.empty() ? "." : atom.resname},
            {"label_asym_id", atom.altloc.empty() ? "." : atom.altloc},
            {"label_entity_id", atom.resseq},
            {"label_seq_id", atom.seqId},
            {"pdbx_PDB_ins_code", atom.icode.empty() ? "." : atom.icode},
            {"Cartn_x", xPos},
            {"Cartn_y", yPos},
            {"Cartn_z", zPos},
            {"occupancy", occupancy},
            {"B_iso_or_equiv", bFactor},
            {"pdbx_formal_charge", atom.charge.empty() ? "." : atom.charge},
            {"auth_seq_id", atom.authSeqId},
            {"auth_comp_id", atom.authCompId.empty() ? "." : atom.authCompId},
            {"auth_asym_id", atom.authAsymId.empty() ? "." : atom.authAsymId},
            {"auth_atom_id", atom.authAtomId.empty() ? "." : atom.authAtomId},
            {"pdbx_PDB_model_num", atom.pdbNum}
        };

        // Inserting row
        atomSite.emplace(std::move(atomSiteInserter));
    }

    // Updating atom list in stored data block
    auto categoryInsertPosition = std::find_if(
        dataBlock.cbegin(), dataBlock.cend(), 
        [](const cif::category& cat) { 
            return cat.name() == "atom_site"; 
        }
    ); 
    if (categoryInsertPosition != dataBlock.cend()) {
        categoryInsertPosition = dataBlock.erase(categoryInsertPosition);
    }
    dataBlock.insert(categoryInsertPosition, atomSite);

    // Writing datablock to file
    dataBlock.write(cifFile);

    // Closing file
    cifFile.close();
}

void PDBRichPhantom::write(const FileName &fnPDB, const bool renumber)
{
    // Checking if extension is .cif or .pdb
    if (checkExtension(fnPDB.getString(), {".cif"}, {".gz"})) {
        writeCIF(fnPDB.getString(), atomList, dataBlock);
    } else {
        writePDB(fnPDB, renumber, remarks, atomList);
    }
}

/* Atom descriptors -------------------------------------------------------- */
void atomDescriptors(const std::string &atom, Matrix1D<double> &descriptors)
{
    descriptors.initZeros(11);
    if (atom=="H")
    {
        descriptors( 0)= 1;     // Z
        descriptors( 1)= 0.0088; // a1
        descriptors( 2)= 0.0449; // a2
        descriptors( 3)= 0.1481; // a3
        descriptors( 4)= 0.2356; // a4
        descriptors( 5)= 0.0914; // a5
        descriptors( 6)= 0.1152; // b1
        descriptors( 7)= 1.0867; // b2
        descriptors( 8)= 4.9755; // b3
        descriptors( 9)=16.5591; // b4
        descriptors(10)=43.2743; // b5
    }
    else if (atom=="C")
    {
        descriptors( 0)= 6;     // Z
        descriptors( 1)= 0.0489; // a1
        descriptors( 2)= 0.2091; // a2
        descriptors( 3)= 0.7537; // a3
        descriptors( 4)= 1.1420; // a4
        descriptors( 5)= 0.3555; // a5
        descriptors( 6)= 0.1140; // b1
        descriptors( 7)= 1.0825; // b2
        descriptors( 8)= 5.4281; // b3
        descriptors( 9)=17.8811; // b4
        descriptors(10)=51.1341; // b5
    }
    else if (atom=="N")
    {
        descriptors( 0)= 7;     // Z
        descriptors( 1)= 0.0267; // a1
        descriptors( 2)= 0.1328; // a2
        descriptors( 3)= 0.5301; // a3
        descriptors( 4)= 1.1020; // a4
        descriptors( 5)= 0.4215; // a5
        descriptors( 6)= 0.0541; // b1
        descriptors( 7)= 0.5165; // b2
        descriptors( 8)= 2.8207; // b3
        descriptors( 9)=10.6297; // b4
        descriptors(10)=34.3764; // b5
    }
    else if (atom=="O")
    {
        descriptors( 0)= 8;     // Z
        descriptors( 1)= 0.0365; // a1
        descriptors( 2)= 0.1729; // a2
        descriptors( 3)= 0.5805; // a3
        descriptors( 4)= 0.8814; // a4
        descriptors( 5)= 0.3121; // a5
        descriptors( 6)= 0.0652; // b1
        descriptors( 7)= 0.6184; // b2
        descriptors( 8)= 2.9449; // b3
        descriptors( 9)= 9.6298; // b4
        descriptors(10)=28.2194; // b5
    }
    else if (atom=="P")
    {
        descriptors( 0)=15;     // Z
        descriptors( 1)= 0.1005; // a1
        descriptors( 2)= 0.4615; // a2
        descriptors( 3)= 1.0663; // a3
        descriptors( 4)= 2.5854; // a4
        descriptors( 5)= 1.2725; // a5
        descriptors( 6)= 0.0977; // b1
        descriptors( 7)= 0.9084; // b2
        descriptors( 8)= 4.9654; // b3
        descriptors( 9)=18.5471; // b4
        descriptors(10)=54.3648; // b5
    }
    else if (atom=="S")
    {
        descriptors( 0)=16;     // Z
        descriptors( 1)= 0.0915; // a1
        descriptors( 2)= 0.4312; // a2
        descriptors( 3)= 1.0847; // a3
        descriptors( 4)= 2.4671; // a4
        descriptors( 5)= 1.0852; // a5
        descriptors( 6)= 0.0838; // b1
        descriptors( 7)= 0.7788; // b2
        descriptors( 8)= 4.3462; // b3
        descriptors( 9)=15.5846; // b4
        descriptors(10)=44.6365; // b5
    }
    else if (atom=="Fe")
    {
        descriptors( 0)=26;     // Z
        descriptors( 1)= 0.1929; // a1
        descriptors( 2)= 0.8239; // a2
        descriptors( 3)= 1.8689; // a3
        descriptors( 4)= 2.3694; // a4
        descriptors( 5)= 1.9060; // a5
        descriptors( 6)= 0.1087; // b1
        descriptors( 7)= 1.0806; // b2
        descriptors( 8)= 4.7637; // b3
        descriptors( 9)=22.8500; // b4
        descriptors(10)=76.7309; // b5
    }
    else if (atom=="K")
    {
        descriptors( 0)=19;     // Z
        descriptors( 1)= 0.2149; // a1
        descriptors( 2)= 0.8703; // a2
        descriptors( 3)= 2.4999; // a3
        descriptors( 4)= 2.3591; // a4
        descriptors( 5)= 3.0318; // a5
        descriptors( 6)= 0.1660; // b1
        descriptors( 7)= 1.6906; // b2
        descriptors( 8)= 8.7447; // b3
        descriptors( 9)=46.7825; // b4
        descriptors(10)=165.6923; // b5
    }
    else if (atom=="F")
    {
        descriptors( 0)=9;     // Z
        descriptors( 1)= 0.0382; // a1
        descriptors( 2)= 0.1822; // a2
        descriptors( 3)= 0.5972; // a3
        descriptors( 4)= 0.7707; // a4
        descriptors( 5)= 0.2130; // a5
        descriptors( 6)= 0.0613; // b1
        descriptors( 7)= 0.5753; // b2
        descriptors( 8)= 2.6858; // b3
        descriptors( 9)= 8.8214; // b4
        descriptors(10)=25.6668; // b5
    }
    else if (atom=="Mg")
    {
        descriptors( 0)=12;     // Z
        descriptors( 1)= 0.1130; // a1
        descriptors( 2)= 0.5575; // a2
        descriptors( 3)= 0.9046; // a3
        descriptors( 4)= 2.1580; // a4
        descriptors( 5)= 1.4735; // a5
        descriptors( 6)= 0.1356; // b1
        descriptors( 7)= 1.3579; // b2
        descriptors( 8)= 6.9255; // b3
        descriptors( 9)=32.3165; // b4
        descriptors(10)=92.1138; // b5
    }
    else if (atom=="Cl")
    {
        descriptors( 0)=17;     // Z
        descriptors( 1)= 0.0799; // a1
        descriptors( 2)= 0.3891; // a2
        descriptors( 3)= 1.0037; // a3
        descriptors( 4)= 2.3332; // a4
        descriptors( 5)= 1.0507; // a5
        descriptors( 6)= 0.0694; // b1
        descriptors( 7)= 0.6443; // b2
        descriptors( 8)= 3.5351; // b3
        descriptors( 9)=12.5058; // b4
        descriptors(10)=35.8633; // b5
    }
    else if (atom=="Ca")
    {
        descriptors( 0)=20;     // Z
        descriptors( 1)= 0.2355; // a1
        descriptors( 2)= 0.9916; // a2
        descriptors( 3)= 2.3959; // a3
        descriptors( 4)= 3.7252; // a4
        descriptors( 5)= 2.5647; // a5
        descriptors( 6)= 0.1742; // b1
        descriptors( 7)= 1.8329; // b2
        descriptors( 8)= 8.8407; // b3
        descriptors( 9)=47.4583; // b4
        descriptors(10)=134.9613; // b5
    }
    else
        REPORT_ERROR(ERR_VALUE_INCORRECT,(std::string)"atomDescriptors: Unknown atom "+atom);
}

/* Electron form factor in Fourier ----------------------------------------- */
double electronFormFactorFourier(double f, const Matrix1D<double> &descriptors)
{
    double retval=0;
    for (int i=1; i<=5; i++)
    {
        double ai=VEC_ELEM(descriptors,i);
        double bi=VEC_ELEM(descriptors,i+5);
        retval+=ai*exp(-bi*f*f);
    }
    return retval;
}

/* Electron form factor in real space -------------------------------------- */
/* We know the transform pair
   sqrt(pi/b)*exp(-x^2/(4*b)) <----> exp(-b*W^2)
   
   We also know that 
   
   X(f)=sum_i ai exp(-bi*f^2)
   
   Therefore, using W=2*pi*f
   
   X(W)=sum_i ai exp(-bi/(2*pi)^2*W^2)
   
   Thus, the actual b for the inverse Fourier transform is bi/(2*pi)^2
   And we have to divide by 2*pi to account for the Jacobian of the
   transformation.
*/
double electronFormFactorRealSpace(double r,
                                   const Matrix1D<double> &descriptors)
{
    double retval=0;
    for (int i=1; i<=5; i++)
    {
        double ai=descriptors(i);
        double bi=descriptors(i+5);
        double b=bi/(4*PI*PI);
        retval+=ai*sqrt(PI/b)*exp(-r*r/(4*b));
    }
    retval/=2*PI;
    return retval;
}

/* Computation of the low pass filter -------------------------------------- */
// Returns the impulse response of the lowpass filter
void hlpf(MultidimArray<double> &f, int M,  const std::string &filterType,
          MultidimArray<double> &filter, double reductionFactor=0.8,
          double ripple=0.01, double deltaw=1.0/8.0)
{
    filter.initZeros(XSIZE(f));
    filter.setXmippOrigin();

    auto Nmax=(int)CEIL(M/2.0);
    if (filterType=="SimpleAveraging")
    {
        FOR_ALL_ELEMENTS_IN_ARRAY1D(filter)
        if (ABS(i)<=Nmax)
            filter(i)=1.0/(2*Nmax+1);
    }
    else if (filterType=="SincKaiser")
    {
    	int lastIdx=FINISHINGX(filter);
    	SincKaiserMask(filter,reductionFactor*PI/M,ripple,deltaw);
    	int newLastIdx=FINISHINGX(filter);
    	if (newLastIdx>3*lastIdx)
    		filter.selfWindow(-lastIdx,lastIdx);
        filter/=filter.sum();
        if (FINISHINGX(f)>FINISHINGX(filter))
            filter.selfWindow(STARTINGX(f),FINISHINGX(f));
        else
            f.selfWindow(STARTINGX(filter),FINISHINGX(filter));
    }
}

/* Convolution between f and the hlpf -------------------------------------- */
void fhlpf(const MultidimArray<double> &f, const MultidimArray<double> &filter,
           int M, MultidimArray<double> &convolution)
{
	// Expand the two input signals
    int Nmax=FINISHINGX(filter);
    MultidimArray<double> auxF;
    MultidimArray<double> auxFilter;
    auxF=f;
    auxFilter=filter;
    auxF.selfWindow(STARTINGX(f)-Nmax,FINISHINGX(f)+Nmax);
    auxFilter.selfWindow(STARTINGX(filter)-Nmax,FINISHINGX(filter)+Nmax);

    // Convolve in Fourier
    MultidimArray< std::complex<double> > F;
    MultidimArray< std::complex<double> > Filter;
    FourierTransform(auxF,F);
    FourierTransform(auxFilter,Filter);
    F*=Filter;

    // Correction for the double phase factor and the double
    // amplitude factor
    const double K1=2*PI*(STARTINGX(auxFilter)-1);
    const double K2=XSIZE(auxFilter);
    std::complex<double> aux;
    auto * ptrAux=(double*)&aux;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(F)
    {
        double w;
        FFT_IDX2DIGFREQ(i,XSIZE(F),w);
        double arg=w*K1;
        sincos(arg,ptrAux+1,ptrAux);
        *ptrAux*=K2;
        *(ptrAux+1)*=K2;
        A1D_ELEM(F,i)*=aux;
    }
    InverseFourierTransform(F,convolution);
    convolution.setXmippOrigin();
}

/* Optimization of the low pass filter to fit a given atom ----------------- */
Matrix1D<double> globalHlpfPrm(3);
MultidimArray<double> globalf;
int globalM;
double globalT;
std::string globalAtom;

//#define DEBUG
double Hlpf_fitness(double *p, void *prm)
{
    double reductionFactor=p[1];
    double ripple=p[2];
    double deltaw=p[3];

    if (reductionFactor<0.7 || reductionFactor>1.3)
        return 1e38;
    if (ripple<0 || ripple>0.2)
        return 1e38;
    if (deltaw<1e-3 || deltaw>0.2)
        return 1e38;

    // Construct the filter with the current parameters
    MultidimArray<double> filter;
    MultidimArray<double> auxf;
    auxf=globalf;
    hlpf(auxf, globalM, "SincKaiser", filter, reductionFactor,
         ripple, deltaw);

    // Convolve the filter with the atomic profile
    MultidimArray<double> fhlpfFinelySampled;
    fhlpf(auxf, filter, globalM, fhlpfFinelySampled);

    // Coarsely sample
    double Rmax=FINISHINGX(fhlpfFinelySampled)*globalT;
    int imax=CEIL(Rmax/(globalM*globalT));
    MultidimArray<double> fhlpfCoarselySampled(2*imax+1);
    MultidimArray<double> splineCoeffsfhlpfFinelySampled;
    produceSplineCoefficients(xmipp_transformation::BSPLINE3,splineCoeffsfhlpfFinelySampled,fhlpfFinelySampled);
    fhlpfCoarselySampled.setXmippOrigin();
    FOR_ALL_ELEMENTS_IN_ARRAY1D(fhlpfCoarselySampled)
    {
        double r=i*(globalM*globalT)/globalT;
        fhlpfCoarselySampled(i)=
            splineCoeffsfhlpfFinelySampled.interpolatedElementBSpline1D(r,3);
    }

    // Build the frequency response of the convolved and coarsely sampled
    // atom
    MultidimArray<double> aux;
    MultidimArray<double> FfilterMag;
    MultidimArray<double> freq;
    MultidimArray< std::complex<double> > Ffilter;
    aux=fhlpfCoarselySampled;
    aux.selfWindow(-10*FINISHINGX(aux),10*FINISHINGX(aux));
    FourierTransform(aux,Ffilter);
    FFT_magnitude(Ffilter,FfilterMag);
    freq.initZeros(XSIZE(Ffilter));

    FOR_ALL_ELEMENTS_IN_ARRAY1D(FfilterMag)
    FFT_IDX2DIGFREQ(i,XSIZE(FfilterMag),freq(i));
    freq/=globalM*globalT;
    double amplitudeFactor=fhlpfFinelySampled.sum()/
                           fhlpfCoarselySampled.sum();

    // Compute the error in representation
    double error=0;
    Matrix1D<double> descriptors;
    atomDescriptors(globalAtom, descriptors);
    double iglobalT=1.0/globalT;
#ifdef DEBUG
    MultidimArray<double> f1array, f2array;
    f1array.initZeros(FfilterMag);
    f2array.initZeros(FfilterMag);
#endif
    double Npoints=0;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(FfilterMag)
    if (A1D_ELEM(freq,i)>=0)
    {
        double f1=log10(A1D_ELEM(FfilterMag,i)*XSIZE(FfilterMag)*amplitudeFactor);
        double f2=log10(iglobalT*
                           electronFormFactorFourier(A1D_ELEM(freq,i),descriptors));
        Npoints++;
        double diff=(f1-f2);
        error+=diff*diff;
#ifdef DEBUG
        f1array(i)=f1;
        f2array(i)=f2;
        std::cout << "i=" << i << " wi=" << A1D_ELEM(freq,i) << " f1=" << f1 << " f2=" << f2 << " diff=" << diff << " err2=" << diff*diff << std::endl;
#endif
    }
#ifdef DEBUG
        f1array.write("PPPf1.txt");
        f2array.write("PPPf2.txt");
        std::cout << "Error " << error/Npoints << " " << Npoints << " M=" << globalM << " T=" << globalT << std::endl;
#endif

    return error/Npoints;
}

/** Optimize the low pass filter.
    The optimization is so that the Fourier response of the coarsely
    downsampled and convolved atom profile resembles as much as possible
    the ideal atomic response up to the maximum frequency provided by the
    Nyquist frequency associated to M*T.
    
    f is the electron scattering factor in real space sampled at a sampling
    rate T. M is the downsampling factor. atom is the name of the atom
    being optimized. filter is an output parameter with the optimal impulse
    response sampled at a sampling rate T. bestPrm(0)=reduction function
    of the cutoff frequency, bestPrm(1)=ripple of the Kaiser selfWindow,
    bestPrm(2)=deltaw of the Kaiser selfWindow.
*/
void optimizeHlpf(MultidimArray<double> &f, int M, double T, const std::string &atom,
		MultidimArray<double> &filter, Matrix1D<double> &bestPrm)
{
    globalHlpfPrm(0)=1.0;     // reduction factor
    globalHlpfPrm(1)=0.01;    // ripple
    globalHlpfPrm(2)=1.0/8.0; // deltaw
    globalf=f;
    globalM=M;
    globalT=T;
    globalAtom=atom;
    double fitness;
    int iter;
    Matrix1D<double> steps(3);
    steps.initConstant(1);
    powellOptimizer(globalHlpfPrm, 1, 3,
                    &Hlpf_fitness, nullptr, 0.05, fitness, iter, steps, false);
    bestPrm=globalHlpfPrm;
    hlpf(f, M, "SincKaiser", filter, bestPrm(0), bestPrm(1), bestPrm(2));
}

/* Atom radial profile ----------------------------------------------------- */
void atomRadialProfile(int M, double T, const std::string &atom,
		MultidimArray<double> &profile)
{
    // Compute the electron form factor in real space
    double largestb1=76.7309/(4*PI*PI);
    double Rmax=4*sqrt(2*largestb1);
    auto imax=(int)CEIL(Rmax/T);
    Matrix1D<double> descriptors;
    atomDescriptors(atom, descriptors);
    MultidimArray<double> f(2*imax+1);
    f.setXmippOrigin();
    for (int i=-imax; i<=imax; i++)
    {
        double r=i*T;
        f(i)=electronFormFactorRealSpace(r,descriptors);
    }

    // Compute the optimal filter
    MultidimArray<double> filter;
    Matrix1D<double> bestPrm;
    optimizeHlpf(f, M, T, atom, filter, bestPrm);

    // Perform the convolution
    fhlpf(f, filter, M, profile);

    // Remove zero values
    int ileft=STARTINGX(profile);
    if (fabs(profile(ileft))<1e-3)
        for (ileft=STARTINGX(profile)+1; ileft<=0; ileft++)
            if (fabs(profile(ileft))>1e-3)
                break;
    int iright=FINISHINGX(profile);
    if (fabs(profile(iright))<1e-3)
        for (iright=FINISHINGX(profile)-1; iright>=0; iright--)
            if (fabs(profile(iright))>1e-3)
                break;
    profile.selfWindow(ileft,iright);
}

/** Atom projection profile ------------------------------------------------ */
class AtomValueFunc: public doubleFunction
{
public:
    int M;
    double r0_2;
    double z;
    const MultidimArray<double> *profileCoefficients;
    virtual double operator()()
    {
        double r=M*sqrt(r0_2+z*z);
        if (ABS(r)>FINISHINGX(*profileCoefficients))
            return 0;
        return profileCoefficients->interpolatedElementBSpline1D(r,3);
    }
};

#define INTEGRATION 2
void atomProjectionRadialProfile(int M,
                                 const MultidimArray<double> &profileCoefficients,
                                 MultidimArray<double> &projectionProfile)
{
    AtomValueFunc atomValue;
    atomValue.profileCoefficients=&profileCoefficients;
    atomValue.M=M;
    double radius=(double)FINISHINGX(profileCoefficients)/M;
    double r2=radius*radius;

    projectionProfile.initZeros(profileCoefficients);
    FOR_ALL_ELEMENTS_IN_ARRAY1D(projectionProfile)
    {
        double r0=(double)i/M;
        atomValue.r0_2=r0*r0;
        if (atomValue.r0_2>r2)
            continue; // Because of numerical instabilities

        double maxZ=sqrt(r2-atomValue.r0_2);

#if INTEGRATION==1

        double dz=1/24.0;
        double integral=0;
        for (atomValue.z=-maxZ; atomValue.z<=maxZ; atomValue.z+=dz)
        {
            projectionProfile(i)+=atomValue();
        }
        projectionProfile(i)*=dz;
#else

        Romberg Rom(atomValue, atomValue.z,-maxZ,maxZ);
        projectionProfile(i) = Rom();
#endif

    }
}

/** Atom interpolations ---------------------------------------------------- */
void AtomInterpolator::setup(int m, double hights, bool computeProjection)
{
    M=m;
    highTs=hights;
    if (volumeProfileCoefficients.size()==12)
    	return;
    addAtom("H",computeProjection);
    addAtom("C",computeProjection);
    addAtom("N",computeProjection);
    addAtom("O",computeProjection);
    addAtom("P",computeProjection);
    addAtom("S",computeProjection);
    addAtom("Fe",computeProjection);
    addAtom("K",computeProjection);
    addAtom("F",computeProjection);
    addAtom("Mg",computeProjection);
    addAtom("Cl",computeProjection);
    addAtom("Ca",computeProjection);
}

void AtomInterpolator::addAtom(const std::string &atom, bool computeProjection)
{
    MultidimArray<double> profile;
    MultidimArray<double> splineCoeffs;

    // Atomic profile
    atomRadialProfile(M, highTs, atom, profile);
    produceSplineCoefficients(xmipp_transformation::BSPLINE3,splineCoeffs,profile);
    volumeProfileCoefficients.push_back(splineCoeffs);

    // Radius
    radii.push_back((double)FINISHINGX(profile)/M);

    // Projection profile
    if (computeProjection)
    {
        atomProjectionRadialProfile(M, splineCoeffs, profile);
        produceSplineCoefficients(xmipp_transformation::BSPLINE3,splineCoeffs,profile);
        projectionProfileCoefficients.push_back(splineCoeffs);
    }
}

/** PDB projection --------------------------------------------------------- */
// Taken from phantom.cpp, Feature::project_to
void projectAtom(const Atom &atom, Projection &P,
                 const Matrix2D<double> &VP, const Matrix2D<double> &PV,
                 const AtomInterpolator &interpolator)
{
constexpr float SUBSAMPLING = 2;                  // for every measure 2x2 line
    // integrals will be taken to
    // avoid numerical errors
constexpr float SUBSTEP = 1/(SUBSAMPLING*2.0);

    Matrix1D<double> origin(3);
    Matrix1D<double> direction;
    VP.getRow(2, direction);
    direction.selfTranspose();
    Matrix1D<double> corner1(3);
    Matrix1D<double> corner2(3);
    Matrix1D<double> act(3);
    SPEED_UP_temps012;

    // Find center of the feature in the projection plane ...................
    // Step 1). Project the center to the plane, the result is in the
    //          universal coord system
    Matrix1D<double> Center(3);
    VECTOR_R3(Center, atom.x, atom.y, atom.z);
    double max_distance=interpolator.atomRadius(atom.atomType);
    M3x3_BY_V3x1(origin, VP, Center);

    //#define DEBUG_LITTLE
#ifdef DEBUG_LITTLE

    std::cout << "Actual atom\n"        << atom.atomType << " ("
    << atom.x << "," << atom.y << "," << atom.z << ")\n";
    std::cout << "Center              " << Center.transpose() << std::endl;
    std::cout << "VP matrix\n"          << VP << std::endl;
    std::cout << "P.direction         " << P.direction.transpose() << std::endl;
    std::cout << "direction           " << direction.transpose() << std::endl;
    std::cout << "P.euler matrix      " << P.euler << std::endl;
    std::cout << "max_distance        " << max_distance << std::endl;
    std::cout << "origin              " << origin.transpose() << std::endl;
#endif

    // Find limits for projection ...........................................
    // Choose corners for the projection of this feature. It is supposed
    // to have at the worst case a projection of size max_distance
    VECTOR_R3(corner1, max_distance, max_distance, max_distance);
    VECTOR_R3(corner2, -max_distance, -max_distance, -max_distance);
#ifdef DEBUG_LITTLE

    std::cout << "Corner1 : " << corner1.transpose() << std::endl
    << "Corner2 : " << corner2.transpose() << std::endl;
#endif

    box_enclosing(corner1, corner2, VP, corner1, corner2);
#ifdef DEBUG_LITTLE

    std::cout << "Corner1 moves to : " << corner1.transpose() << std::endl
    << "Corner2 moves to : " << corner2.transpose() << std::endl;
#endif

    V3_PLUS_V3(corner1, origin, corner1);
    V3_PLUS_V3(corner2, origin, corner2);
#ifdef DEBUG_LITTLE

    std::cout << "Corner1 finally is : " << corner1.transpose() << std::endl
    << "Corner2 finally is : " << corner2.transpose() << std::endl;
#endif
    // Discard not necessary components
    corner1.resize(2);
    corner2.resize(2);

    // Clip to image size
    sortTwoVectors(corner1, corner2);
    XX(corner1) = CLIP(ROUND(XX(corner1)), STARTINGX(P()), FINISHINGX(P()));
    YY(corner1) = CLIP(ROUND(YY(corner1)), STARTINGY(P()), FINISHINGY(P()));
    XX(corner2) = CLIP(ROUND(XX(corner2)), STARTINGX(P()), FINISHINGX(P()));
    YY(corner2) = CLIP(ROUND(YY(corner2)), STARTINGY(P()), FINISHINGY(P()));

#ifdef DEBUG_LITTLE

    std::cout << "corner1      " << corner1.transpose() << std::endl;
    std::cout << "corner2      " << corner2.transpose() << std::endl;
    std::cout.flush();
#endif

    // Check if there is something to project
    if (XX(corner1) == XX(corner2))
        return;
    if (YY(corner1) == YY(corner2))
        return;

    // Study the projection for each point in the projection plane ..........
    // (u,v) are in the deformed projection plane (if any deformation)
    for (auto v = (int)YY(corner1); v <= (int)YY(corner2); v++)
        for (auto u = (int)XX(corner1); u <= (int)XX(corner2); u++)
        {
            double length = 0;
            //#define DEBUG_EVEN_MORE
#ifdef DEBUG_EVEN_MORE

            std::cout << "Studying point (" << u << "," << v << ")\n";
            std::cout.flush();
#endif

            // Perform subsampling ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            double u0 = u - (int)(SUBSAMPLING / 2.0) * SUBSTEP;
            double v0 = v - (int)(SUBSAMPLING / 2.0) * SUBSTEP;
            double actv = v0;
            for (int subv = 0; subv < SUBSAMPLING; subv++)
            {
                double actu = u0;
                for (int subu = 0; subu < SUBSAMPLING; subu++)
                {
                    // Compute the coordinates of point (subu,subv) which is
                    // within the plane in the universal coordinate system
                    XX(act) = actu;
                    YY(act) = actv;
                    ZZ(act) = 0;
                    M3x3_BY_V3x1(act, PV, act);

                    // Compute the intersection of a ray which passes through
                    // this point and its direction is perpendicular to the
                    // projection plane
                    double r=point_line_distance_3D(Center, act, direction);
                    double possible_length=interpolator.
                                           projectionAtDistance(atom.atomType,r);
                    if (possible_length > 0)
                        length += possible_length;

#ifdef DEBUG_EVEN_MORE

                    std::cout << "Averaging at (" << actu << "," << actv << ")\n";
                    std::cout << "   which in univ. coords is " << act.transpose() << std::endl;
                    std::cout << "   r=" << r << std::endl;
                    std::cout << "   intersection there " << possible_length << std::endl;
                    std::cout.flush();
#endif
                    // Prepare for next iteration
                    actu += SUBSTEP * 2.0;
                }
                actv += SUBSTEP * 2.0;
            }
            length /= (SUBSAMPLING * SUBSAMPLING);
            //#define DEBUG
#ifdef DEBUG

            std::cout << "Final value added at position (" << u << "," << v << ")="
            << length << std::endl;
            std::cout.flush();
#endif

            // Add at the correspondent pixel the found intersection ,,,,,,,,,,
            IMGPIXEL(P, v, u) += length;
        }
}
#undef DEBUG
#undef DEBUG_LITTLE
#undef DEBUG_EVEN_MORE

void projectPDB(const PDBPhantom &phantomPDB,
                const AtomInterpolator &interpolator, Projection &proj,
                int Ydim, int Xdim, double rot, double tilt, double psi)
{
    // Initialise projection
    proj().initZeros(Ydim, Xdim);
    proj().setXmippOrigin();
    proj.setAngles(rot, tilt, psi);

    // Compute volume to Projection matrix
    Matrix2D<double> VP = proj.euler;
    Matrix2D<double> PV = VP.inv();

    // Project all elements
    for (size_t i = 0; i < phantomPDB.getNumberOfAtoms(); i++)
    {
        try
        {
            projectAtom(phantomPDB.getAtom(i), proj, VP, PV, interpolator);
        }
        catch (XmippError &XE) {}
    }
}

void distanceHistogramPDB(const PDBPhantom &phantomPDB, size_t Nnearest, double maxDistance, int Nbins, Histogram1D &hist)
{
    // Compute the histogram of distances
	const std::vector<Atom> &atoms=phantomPDB.atomList;
    int Natoms=atoms.size();
    MultidimArray<double> NnearestDistances;
    NnearestDistances.resize((Natoms-1)*Nnearest);
    for (int i=0; i<Natoms; i++)
    {
        std::vector<double> NnearestToThisAtom;
        const Atom& atom_i=atoms[i];
        for (int j=i+1; j<Natoms; j++)
        {
            const Atom& atom_j=atoms[j];
            double diffx=atom_i.x-atom_j.x;
            double diffy=atom_i.y-atom_j.y;
            double diffz=atom_i.z-atom_j.z;
            double dist=sqrt(diffx*diffx+diffy*diffy+diffz*diffz);
            if (maxDistance>0 && dist>maxDistance)
            	continue;
        	//std::cout << "Analyzing " << i << " and " << j << " -> d=" << dist << std::endl;
            size_t nearestSoFar=NnearestToThisAtom.size();
            if (nearestSoFar==0)
            {
                NnearestToThisAtom.push_back(dist);
            	//std::cout << "Pushing d" << std::endl;
            }
            else
            {
                size_t idx=0;
                while (idx<nearestSoFar && NnearestToThisAtom[idx]<dist)
                    idx++;
                if (idx<nearestSoFar)
                {
                    NnearestToThisAtom.insert(NnearestToThisAtom.begin()+idx,1,dist);
                    if (NnearestToThisAtom.size()>Nnearest)
                        NnearestToThisAtom.erase(NnearestToThisAtom.begin()+Nnearest);
                }
                if (idx==nearestSoFar && nearestSoFar<Nnearest)
                {
                    NnearestToThisAtom.push_back(dist);
                	//std::cout << "Pushing d" << std::endl;
                }
            }
        }
		if (i<Natoms-1)
			for (size_t k=0; k<Nnearest; k++)
				NnearestDistances(i*Nnearest+k)=NnearestToThisAtom[k];
    }
    compute_hist(NnearestDistances, hist, 0, NnearestDistances.computeMax(), Nbins);
}


/*! C port of the hy36encode() and hy36decode() functions in the
    hybrid_36.py Python prototype/reference implementation.
    See the Python script for more information.
    This file has no external dependencies, NOT even standard C headers.
    Optionally, use hybrid_36_c.h, or simply copy the declarations
    into your code.
    This file is unrestricted Open Source (cctbx.sf.net).
    Please send corrections and enhancements to cctbx@cci.lbl.gov .
    See also: http://cci.lbl.gov/hybrid_36/
    Ralf W. Grosse-Kunstleve, Feb 2007.
 */

/* The following #include may be commented out.
   It is here only to enforce consistency of the declarations
   and the definitions.
 */
// #include <iotbx/pdb/hybrid_36_c.h>

/* All static functions below are implementation details
   (and not accessible from other translation units).
 */

static
const char*
digits_upper() { return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; }

static
const char*
digits_lower() { return "0123456789abcdefghijklmnopqrstuvwxyz"; }

static
const char*
value_out_of_range() { return "value out of range."; }

static
const char* invalid_number_literal() { return "invalid number literal."; }

static
const char* unsupported_width() { return "unsupported width."; }

static
void
fill_with_stars(unsigned width, char* result)
{
  while (width) {
    *result++ = '*';
    width--;
  }
  *result = '\0';
}

static
void
encode_pure(
  const char* digits,
  unsigned digits_size,
  unsigned width,
  int value,
  char* result)
{
  char buf[16];
  int rest;
  unsigned i, j;
  i = 0;
  j = 0;
  if (value < 0) {
    j = 1;
    value = -value;
  }
  while (1) {
    rest = value / digits_size;
    buf[i++] = digits[value - rest * digits_size];
    if (rest == 0) break;
    value = rest;
  }
  if (j) buf[i++] = '-';
  for(j=i;j<width;j++) *result++ = ' ';
  while (i != 0) *result++ = buf[--i];
  *result = '\0';
}

static
const char*
decode_pure(
  const int* digits_values,
  unsigned digits_size,
  const char* s,
  unsigned s_size,
  int* result)
{
  int si, dv;
  int have_minus = 0;
  int have_non_blank = 0;
  int value = 0;
  unsigned i = 0;
  for(;i<s_size;i++) {
    si = s[i];
    if (si < 0 || si > 127) {
      *result = 0;
      return invalid_number_literal();
    }
    if (si == ' ') {
      if (!have_non_blank) continue;
      value *= digits_size;
    }
    else if (si == '-') {
      if (have_non_blank) {
        *result = 0;
        return invalid_number_literal();
      }
      have_non_blank = 1;
      have_minus = 1;
      continue;
    }
    else {
      have_non_blank = 1;
      dv = digits_values[si];
      if (dv < 0 || dv >= digits_size) {
        *result = 0;
        return invalid_number_literal();
      }
      value *= digits_size;
      value += dv;
    }
  }
  if (have_minus) value = -value;
  *result = value;
  return 0;
}

/*! hybrid-36 encoder: converts integer value to string result
      width: must be 4 (e.g. for residue sequence numbers)
                  or 5 (e.g. for atom serial numbers)
      value: integer value to be converted
      result: pointer to char array of size width+1 or greater
              on return result is null-terminated
      return value: pointer to error message, if any,
                    or 0 on success
    Example usage (from C++):
      char result[4+1];
      const char* errmsg = hy36encode(4, 12345, result);
      if (errmsg) throw std::runtime_error(errmsg);
 */
const char*
hy36encode(unsigned width, int value, char* result)
{
  int i = value;
  if (width == 4U) {
    if (i >= -999) {
      if (i < 10000) {
        encode_pure(digits_upper(), 10U, 4U, i, result);
        return 0;
      }
      i -= 10000;
      if (i < 1213056 /* 26*36**3 */) {
        i += 466560 /* 10*36**3 */;
        encode_pure(digits_upper(), 36U, 0U, i, result);
        return 0;
      }
      i -= 1213056;
      if (i < 1213056) {
        i += 466560;
        encode_pure(digits_lower(), 36U, 0U, i, result);
        return 0;
      }
    }
  }
  else if (width == 5U) {
    if (i >= -9999) {
      if (i < 100000) {
        encode_pure(digits_upper(), 10U, 5U, i, result);
        return 0;
      }
      i -= 100000;
      if (i < 43670016 /* 26*36**4 */) {
        i += 16796160 /* 10*36**4 */;
        encode_pure(digits_upper(), 36U, 0U, i, result);
        return 0;
      }
      i -= 43670016;
      if (i < 43670016) {
        i += 16796160;
        encode_pure(digits_lower(), 36U, 0U, i, result);
        return 0;
      }
    }
  }
  else {
    fill_with_stars(width, result);
    return unsupported_width();
  }
  fill_with_stars(width, result);
  return value_out_of_range();
}

/*! hybrid-36 decoder: converts string s to integer result
      width: must be 4 (e.g. for residue sequence numbers)
                  or 5 (e.g. for atom serial numbers)
      s: string to be converted
         does not have to be null-terminated
      s_size: size of s
              must be equal to width, or an error message is
              returned otherwise
      result: integer holding the conversion result
      return value: pointer to error message, if any,
                    or 0 on success
    Example usage (from C++):
      int result;
      const char* errmsg = hy36decode(width, "A1T5", 4, &result);
      if (errmsg) throw std::runtime_error(errmsg);
 */
const char*
hy36decode(unsigned width, const char* s, unsigned s_size, int* result)
{
  static int first_call = 1;
  static int digits_values_upper[128U];
  static int digits_values_lower[128U];
  static const char*
    ie_range = "internal error hy36decode: integer value out of range.";
  unsigned i;
  int di;
  const char* errmsg;
  if (first_call) {
    first_call = 0;
    for(i=0;i<128U;i++) digits_values_upper[i] = -1;
    for(i=0;i<128U;i++) digits_values_lower[i] = -1;
    for(i=0;i<36U;i++) {
      di = digits_upper()[i];
      if (di < 0 || di > 127) {
        *result = 0;
        return ie_range;
      }
      digits_values_upper[di] = i;
    }
    for(i=0;i<36U;i++) {
      di = digits_lower()[i];
      if (di < 0 || di > 127) {
        *result = 0;
        return ie_range;
      }
      digits_values_lower[di] = i;
    }
  }
  if (s_size == width) {
    di = s[0];
    if (di >= 0 && di <= 127) {
      if (digits_values_upper[di] >= 10) {
        errmsg = decode_pure(digits_values_upper, 36U, s, s_size, result);
        if (errmsg == 0) {
          /* result - 10*36**(width-1) + 10**width */
          if      (width == 4U) (*result) -= 456560;
          else if (width == 5U) (*result) -= 16696160;
          else {
            *result = 0;
            return unsupported_width();
          }
          return 0;
        }
      }
      else if (digits_values_lower[di] >= 10) {
        errmsg = decode_pure(digits_values_lower, 36U, s, s_size, result);
        if (errmsg == 0) {
          /* result + 16*36**(width-1) + 10**width */
          if      (width == 4U) (*result) += 756496;
          else if (width == 5U) (*result) += 26973856;
          else {
            *result = 0;
            return unsupported_width();
          }
          return 0;
        }
      }
      else {
        errmsg = decode_pure(digits_values_upper, 10U, s, s_size, result);
        if (errmsg) return errmsg;
        if (!(width == 4U || width == 5U)) {
          *result = 0;
          return unsupported_width();
        }
        return 0;
      }
    }
  }
  *result = 0;
  return invalid_number_literal();
}

// safe function for hy36decode
void hy36decodeSafe(unsigned width, const char* s, unsigned s_size, int* result)
{
    auto* errmsg = hy36decode(width, s, s_size, result); 
    if (errmsg) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, errmsg);
    }
}

// safe function for hy36encode
void hy36encodeSafe(unsigned width, int value, char* result)
{
    const char* errmsg = hy36encode(width, value, result); 
    if (errmsg) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, errmsg);
    }
}