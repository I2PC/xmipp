/***************************************************************************
 *
 * Authors:
 *
 * Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#include "volume_from_pdb.h"
#include "core/transformations.h"
#include <core/args.h>
#include "data/pdb.h"

#include <fstream>

/* Empty constructor ------------------------------------------------------- */
ProgPdbConverter::ProgPdbConverter()
{
    blob.radius = 2;   // Blob radius in voxels
    blob.order  = 2;   // Order of the Bessel function
    blob.alpha  = 3.6; // Smoothness parameter
    output_dim_x = -1;
    output_dim_y = -1;
    output_dim_z = -1;

    fn_pdb = "";
    Ts = 1;
    highTs = 1.0/12.0;
    useBlobs=false;
    usePoorGaussian=false;
    useFixedGaussian=false;
    doCenter=false;
    noHet=false;

    // Periodic table for the blobs
    periodicTable.resize(12, 2);
    periodicTable(0, 0) = atomRadius("H");
    periodicTable(0, 1) = atomCharge("H");
    periodicTable(1, 0) = atomRadius("C");
    periodicTable(1, 1) = atomCharge("C");
    periodicTable(2, 0) = atomRadius("N");
    periodicTable(2, 1) = atomCharge("N");
    periodicTable(3, 0) = atomRadius("O");
    periodicTable(3, 1) = atomCharge("O");
    periodicTable(4, 0) = atomRadius("P");
    periodicTable(4, 1) = atomCharge("P");
    periodicTable(5, 0) = atomRadius("S");
    periodicTable(5, 1) = atomCharge("S");
    periodicTable(6, 0) = atomRadius("Fe");
    periodicTable(6, 1) = atomCharge("Fe");
    periodicTable(7, 0) = atomRadius("K");
    periodicTable(7, 1) = atomCharge("K");
    periodicTable(8, 0) = atomRadius("F");
    periodicTable(8, 1) = atomCharge("F");
    periodicTable(9, 0) = atomRadius("Mg");
    periodicTable(9, 1) = atomCharge("Mg");
    periodicTable(10, 0) = atomRadius("Cl");
    periodicTable(10, 1) = atomCharge("Cl");
    periodicTable(11, 0) = atomRadius("Ca");
    periodicTable(11, 1) = atomCharge("Ca");

    // Correct the atom weights by the blob weight
    for (size_t i = 0; i < MAT_YSIZE(periodicTable); i++)
    {
        periodicTable(i, 1) /=
            basvolume(periodicTable(i, 0) / highTs, blob.alpha, blob.order, 3);
    }
}

/* Produce Side Info ------------------------------------------------------- */
void ProgPdbConverter::produceSideInfo()
{
    if (useFixedGaussian && sigmaGaussian<0)
    {
        // Check if it is a pseudodensity volume
        std::ifstream fh_pdb;
        fh_pdb.open(fn_pdb.c_str());
        if (!fh_pdb)
            REPORT_ERROR(ERR_IO_NOTEXIST, fn_pdb);
        while (!fh_pdb.eof())
        {
            // Read an ATOM line
            std::string line;
            getline(fh_pdb, line);
            if (line == "")
                continue;
            std::string kind = line.substr(0,6);
            if (kind!="REMARK")
                continue;
            std::vector< std::string > results;
            splitString(line," ",results);
            if (useFixedGaussian && results[1]=="fixedGaussian")
                sigmaGaussian=textToFloat(results[2]);
            if (useFixedGaussian && results[1]=="intensityColumn")
                intensityColumn=results[2];
        }
        fh_pdb.close();
    }

    if (!useBlobs && !usePoorGaussian && !useFixedGaussian)
    {
        // Compute the downsampling factor
        M=(int)ROUND(Ts/highTs);

        // Atom profiles for the electron scattering method
        atomProfiles.setup(M,highTs,false);
    }
}

/* Atom description ------------------------------------------------------- */
void ProgPdbConverter::atomBlobDescription(
    const std::string &_element, double &weight, double &radius) const
{
    int idx = -1;
    weight = radius = 0;
    switch (_element[0])
    {
    case 'H':
        idx = 0;
        break;
    case 'C':
        idx = 1;
        break;
    case 'N':
        idx = 2;
        break;
    case 'O':
        idx = 3;
        break;
    case 'P':
        idx = 4;
        break;
    case 'S':
        idx = 5;
        break;
    case 'E': //iron Fe
        idx = 6;
        break;
    case 'K':
        idx = 7;
        break;
    case 'F':
        idx = 8;
        break;
    case 'G': // Magnesium Mg
        idx = 9;
        break;
    case 'L': // Chlorine Cl
        idx = 10;
        break;
    case 'A': // Calcium Ca
        idx = 11;
        break;
    default:
    	if (verbose>0)
    		std::cout << "Unknown :" << _element << std::endl;
        return;
    }
    radius = periodicTable(idx, 0);
    weight = periodicTable(idx, 1);
}
/* Usage ------------------------------------------------------------------- */
void ProgPdbConverter::defineParams()
{
    addUsageLine("Covert a PDB file to a volume.");
    addExampleLine("Sample at 1.6A and limit the frequency to 10A",false);
    addExampleLine("   xmipp_volume_from_pdb -i 1o7d.pdb --sampling 1.6");
    addExampleLine("   xmipp_transform_filter -i 1o7d.vol -o 1o7dFiltered.vol --fourier low_pass 10 raised_cosine 0.05 --sampling 1.6");

    addParamsLine("   -i <pdb_file>                     : File to process");
    addParamsLine("  [-o <fn_root>]                     : Root name for output");
    addParamsLine("  [--sampling <Ts=1>]                : Sampling rate (Angstroms/pixel)");
    addParamsLine("  [--high_sampling_rate <highTs=0.08333333>]: Sampling rate before downsampling");
    addParamsLine("  [--size <output_dim_x=-1> <output_dim_y=-1> <output_dim_z=-1>]: Final size in pixels (must be a power of 2, if blobs are used)");
    addParamsLine("  [--orig <orig_x=0> <orig_y=0> <orig_z=0>]: Define origin of the output volume");
    addParamsLine("  				                     : If just one dimension is introduced dim_x = dim_y = dim_z");
    addParamsLine("  [--centerPDB]                       : Center PDB with the center of mass");
    addParamsLine("  [--oPDB]                            : Save centered PDB");
    addParamsLine("  [--noHet]                           : Heteroatoms are not converted");
    addParamsLine("  [--blobs]                           : Use blobs instead of scattering factors");
    addParamsLine("  [--poor_Gaussian]                   : Use a simple Gaussian adapted to each atom");
    addParamsLine("  [--fixed_Gaussian <std=-1>]         : Use a fixed Gausian for each atom with");
    addParamsLine("                                     :  this standard deviation");
    addParamsLine("                                     :  If not given, the standard deviation is taken from the PDB file");
    addParamsLine("  [--intensityColumn <intensity_type=occupancy>]   : Where to write the intensity in the PDB file");
    addParamsLine("     where <intensity_type> occupancy Bfactor     : Valid values: occupancy, Bfactor");
}
/* Read parameters --------------------------------------------------------- */
void ProgPdbConverter::readParams()
{
    fn_pdb = getParam("-i");
    fn_out = checkParam("-o") ? getParam("-o") : fn_pdb.withoutExtension();
    Ts = getDoubleParam("--sampling");
    highTs = getDoubleParam("--high_sampling_rate");
    output_dim_x = getIntParam("--size", 0);
    output_dim_y = getIntParam("--size", 1);
    output_dim_z = getIntParam("--size", 2);
    if (checkParam("--orig"))
    {
    	orig_x = getIntParam("--orig", 0);
        orig_y = getIntParam("--orig", 1);
        orig_z = getIntParam("--orig", 2);
        origGiven=true;
    }
    else
    {
    	origGiven=false;
    	orig_x=orig_y=orig_z=0;
    }
    useBlobs = checkParam("--blobs");
    usePoorGaussian = checkParam("--poor_Gaussian");
    useFixedGaussian = checkParam("--fixed_Gaussian");
    if (useFixedGaussian)
        sigmaGaussian = getDoubleParam("--fixed_Gaussian");
    doCenter = checkParam("--centerPDB");
    fn_outPDB = checkParam("--oPDB") ? (fn_out + "_centered.pdb") : FileName();
    noHet = checkParam("--noHet");
    intensityColumn = getParam("--intensityColumn");
}

/* Show -------------------------------------------------------------------- */
void ProgPdbConverter::show()
{
    if (verbose==0)
        return;
    std::cout << "PDB file:           " << fn_pdb           << std::endl
    << "Sampling rate:      " << Ts               << std::endl
    << "High sampling rate: " << highTs           << std::endl
    << "Size:               " << output_dim_x << " " << output_dim_y << " " << output_dim_z << std::endl
    << "Origin:             " << orig_x << " " << orig_y << " " << orig_z << std::endl
    << "Center PDB:         " << doCenter         << std::endl
    << "Do not Hetatm:      " << noHet         << std::endl
    << "Use blobs:          " << useBlobs         << std::endl
    << "Use poor Gaussian:  " << usePoorGaussian  << std::endl
    << "Use fixed Gaussian: " << useFixedGaussian << std::endl
    ;
    if (useFixedGaussian)
        std::cout << "Intensity Col:      " << intensityColumn  << std::endl
        << "Sigma:              " << sigmaGaussian  << std::endl;
}

/* Compute protein geometry ------------------------------------------------ */
void ProgPdbConverter::computeProteinGeometry()
{
    Matrix1D<double> limit0(3), limitF(3);
    computePDBgeometry(fn_pdb, centerOfMass, limit0, limitF, intensityColumn);
    if (doCenter)
    {
        limit0-=centerOfMass;
        limitF-=centerOfMass;
    }
    limit.resize(3);
    XX(limit) = XMIPP_MAX(ABS(XX(limit0)), ABS(XX(limitF)));
    YY(limit) = XMIPP_MAX(ABS(YY(limit0)), ABS(YY(limitF)));
    ZZ(limit) = XMIPP_MAX(ABS(ZZ(limit0)), ABS(ZZ(limitF)));

    // Update output size if necessary
    if (output_dim_x == -1)
    {
        int max_dim = XMIPP_MAX(CEIL(ZZ(limit) * 2 / Ts) + 5, CEIL(YY(limit) * 2 / Ts) + 5);
        max_dim = XMIPP_MAX(max_dim, CEIL(XX(limit) * 2 / Ts) + 5);
        if (useBlobs)
        {
            output_dim_x = (int)NEXT_POWER_OF_2(max_dim);
        	output_dim_y = output_dim_x;
			output_dim_z = output_dim_x;
        }
        else
        {
            output_dim_x = max_dim+10;
            output_dim_y = output_dim_x;
			output_dim_z = output_dim_x;
        }
    }
    else
    {
    	if (output_dim_y == -1)
    	{
    		output_dim_y = output_dim_x;
    		output_dim_z = output_dim_x;
    	}
    }
}

/* Create protein at a high sampling rate ---------------------------------- */
void ProgPdbConverter::createProteinAtHighSamplingRate()
{
    // Create an empty volume to hold the protein
    int finalDim_x, finalDim_y, finalDim_z;
    if (highTs!=Ts)
    {
        finalDim_x=(int)NEXT_POWER_OF_2(output_dim_x / (highTs/Ts));
        finalDim_y=(int)NEXT_POWER_OF_2(output_dim_y / (highTs/Ts));
        finalDim_z=(int)NEXT_POWER_OF_2(output_dim_z / (highTs/Ts));
    }
    else
    {
        finalDim_x=output_dim_x;
        finalDim_y=output_dim_y;
        finalDim_z=output_dim_z;
    }
    Vhigh().initZeros(finalDim_x,finalDim_y,finalDim_z);
    if (!origGiven)
	    Vhigh().setXmippOrigin();
    else
    {
		STARTINGX(Vhigh()) = orig_x;
		STARTINGY(Vhigh()) = orig_y;
		STARTINGZ(Vhigh()) = orig_z;
    }
    if (verbose)
    	std::cout << "The highly sampled volume is of size " << XSIZE(Vhigh())
    	<< std::endl;
    std::cout << "Size: "; Vhigh().printShape(); std::cout << std::endl;

    // Declare centered PDB
    PDBRichPhantom centered_pdb;

    // Read centered pdb
    centered_pdb.read(fn_pdb.c_str(), true);

    Matrix1D<double> r(3);
    bool useBFactor = intensityColumn=="Bfactor";
    // Iterate the list of atoms modifying data if needed
    for (auto& atom : centered_pdb.atomList) {
        if (doCenter) {
            atom.x -= XX(centerOfMass);
            atom.y -= YY(centerOfMass);
            atom.z -= ZZ(centerOfMass);
        }
        VECTOR_R3(r, atom.x, atom.y, atom.z);
        r /= highTs;

        // Characterize atom
        double weight, radius;
        if (!useFixedGaussian)
        {
            if (noHet && atom.record == "HETATM")
                continue;
            atomBlobDescription(atom.atomType, weight, radius);
        }
        else
        {
            radius=4.5*sigmaGaussian;
            if (useBFactor)
                weight=atom.bfactor;
            else
                weight=atom.occupancy;
        }
        blob.radius = radius;
        if (usePoorGaussian)
            radius=XMIPP_MAX(radius/Ts,4.5);
        double GaussianSigma2=(radius/(3*sqrt(2.0)));
        if (useFixedGaussian)
            GaussianSigma2=sigmaGaussian;
        GaussianSigma2*=GaussianSigma2;
        double GaussianNormalization = 1.0/pow(2*PI*GaussianSigma2,1.5);

        // Find the part of the volume that must be updated
        int k0 = XMIPP_MAX(FLOOR(ZZ(r) - radius), STARTINGZ(Vhigh()));
        int kF = XMIPP_MIN(CEIL(ZZ(r) + radius), FINISHINGZ(Vhigh()));
        int i0 = XMIPP_MAX(FLOOR(YY(r) - radius), STARTINGY(Vhigh()));
        int iF = XMIPP_MIN(CEIL(YY(r) + radius), FINISHINGY(Vhigh()));
        int j0 = XMIPP_MAX(FLOOR(XX(r) - radius), STARTINGX(Vhigh()));
        int jF = XMIPP_MIN(CEIL(XX(r) + radius), FINISHINGX(Vhigh()));

        // Fill the volume with this atom
        Matrix1D<double> rdiff(3);
        for (int k = k0; k <= kF; k++)
            for (int i = i0; i <= iF; i++)
                for (int j = j0; j <= jF; j++)
                {
                    VECTOR_R3(rdiff, XX(r) - j, YY(r) - i, ZZ(r) - k);
                    rdiff*=highTs;
                    if (useBlobs)
                        Vhigh(k, i, j) += weight * blob_val(rdiff.module(), blob);
                    else if (usePoorGaussian || useFixedGaussian)
                        Vhigh(k, i, j) += weight *
                                          exp(-rdiff.module()*rdiff.module()/(2*GaussianSigma2))*
                                          GaussianNormalization;
                }
    }
    
    // Save centered PDB
    if (doCenter && !fn_outPDB.empty())
    {
        centered_pdb.write(fn_outPDB);
    }
}

/* Create protein at a low sampling rate ----------------------------------- */
void ProgPdbConverter::createProteinAtLowSamplingRate()
{
    // Compute the integer downsapling factor
    int M = FLOOR(Ts / highTs);
    double current_Ts = highTs;

    // Use Bsplines pyramid if possible
    int levels = FLOOR(log10((double)M) / log10(2.0) + XMIPP_EQUAL_ACCURACY);
    pyramidReduce(xmipp_transformation::BSPLINE3, Vlow(), Vhigh(), levels);
    current_Ts *= pow(2.0, levels);
    Vhigh.clear();

    // Now scale using Bsplines
    int new_output_dim = CEIL(XSIZE(Vlow()) * current_Ts / Ts);
    scaleToSize(xmipp_transformation::BSPLINE3, Vhigh(), Vlow(),
                new_output_dim, new_output_dim, new_output_dim);
    Vlow() = Vhigh();
    if (!origGiven)
    	Vlow().setXmippOrigin();

    // Return to the desired size
    Vlow().selfWindow(FIRST_XMIPP_INDEX(output_dim_x), FIRST_XMIPP_INDEX(output_dim_y),
                  FIRST_XMIPP_INDEX(output_dim_z), LAST_XMIPP_INDEX(output_dim_x),
                  LAST_XMIPP_INDEX(output_dim_y), LAST_XMIPP_INDEX(output_dim_z));
}

/* Blob properties --------------------------------------------------------- */
void ProgPdbConverter::blobProperties() const
{
    std::ofstream fh_out;
    fh_out.open((fn_out + "_Fourier_profile.txt").c_str());
    if (!fh_out)
        REPORT_ERROR(ERR_IO_NOWRITE, fn_out);
    fh_out << "# Freq(1/A) 10*log10(|Blob(f)|^2) Ts=" << highTs << std::endl;
    

    for(double w=0; w < 1.0 / (2*highTs); w += 1.0 / (highTs * 500))
    {
        double H = kaiser_Fourier_value(w * highTs, periodicTable(0, 0) / highTs,
                                        blob.alpha, blob.order);
        fh_out << w << " " << 10*log10(H*H) << std::endl;
    }
    fh_out.close();
}

/* Create protein using scattering profiles -------------------------------- */
void ProgPdbConverter::createProteinUsingScatteringProfiles()
{
    // Create an empty volume to hold the protein
    Vlow().initZeros(output_dim_x,output_dim_y,output_dim_z);
    if (!origGiven) {
    	Vlow().setXmippOrigin();
    }
    else
    {
		STARTINGX(Vlow()) = orig_x;
		STARTINGY(Vlow()) = orig_y;
		STARTINGZ(Vlow()) = orig_z;
    }

    //Save centered PDB
    PDBRichPhantom centered_pdb;

    // Read centered pdb
    centered_pdb.read(fn_pdb.c_str(), true);

    Matrix1D<double> r(3);
    bool useBFactor = intensityColumn=="Bfactor";
    // Iterate the list of atoms modifying data if needed
    for (auto& atom : centered_pdb.atomList) {
        // Check if heteroatoms are allowed and current atom is one of them
        if (noHet && atom.record == "HETATM")
            continue;

        if (doCenter) {
            atom.x -= XX(centerOfMass);
            atom.y -= YY(centerOfMass);
            atom.z -= ZZ(centerOfMass);
        }
        VECTOR_R3(r, atom.x, atom.y, atom.z);
        r /= Ts;

        // Characterize atom
        try
        {
            double radius=atomProfiles.atomRadius(atom.atomType[0]);
            double radius2=radius*radius;

            // Find the part of the volume that must be updated
            const MultidimArray<double> &mVlow=Vlow();
            int k0 = XMIPP_MAX(FLOOR(ZZ(r) - radius), STARTINGZ(mVlow));
            int kF = XMIPP_MIN(CEIL(ZZ(r) + radius), FINISHINGZ(mVlow));
            int i0 = XMIPP_MAX(FLOOR(YY(r) - radius), STARTINGY(mVlow));
            int iF = XMIPP_MIN(CEIL(YY(r) + radius), FINISHINGY(mVlow));
            int j0 = XMIPP_MAX(FLOOR(XX(r) - radius), STARTINGX(mVlow));
            int jF = XMIPP_MIN(CEIL(XX(r) + radius), FINISHINGX(mVlow));

            // Fill the volume with this atom
            for (int k = k0; k <= kF; k++)
            {
                double zdiff=ZZ(r) - k;
                double zdiff2=zdiff*zdiff;
                for (int i = i0; i <= iF; i++)
                {
                    double ydiff=YY(r) - i;
                    double zydiff2=zdiff2+ydiff*ydiff;
                    for (int j = j0; j <= jF; j++)
                    {
                        double xdiff=XX(r) - j;
                        double rdiffModule2=zydiff2+xdiff*xdiff;
                        if (rdiffModule2<radius2)
                        {
                            double rdiffModule=sqrt(rdiffModule2);
                            A3D_ELEM(mVlow,k, i, j) += atomProfiles.volumeAtDistance(
                                                 atom.atomType[0],rdiffModule);
                        }
                    }
                }
            }
        }
        catch (XmippError XE)
        {
        	if (verbose)
        		std::cerr << "Ignoring atom of type *" << atom.atomType << "*" << std::endl;
        }
    }

    // Save centered PDB
    if (doCenter && !fn_outPDB.empty())
    {
        centered_pdb.write(fn_outPDB);
    }
}

/* Run --------------------------------------------------------------------- */
void ProgPdbConverter::run()
{
    produceSideInfo();
    show();
    computeProteinGeometry();
    if (useBlobs)
    {
        createProteinAtHighSamplingRate();
        createProteinAtLowSamplingRate();
        blobProperties();
    }
    else if (usePoorGaussian || useFixedGaussian)
    {
        highTs=Ts;
        createProteinAtHighSamplingRate();
        Vlow=Vhigh;
        Vhigh.clear();
    }
    else
    {
        createProteinUsingScatteringProfiles();
    }
    if (fn_out!="")
        Vlow.write(fn_out + ".vol");
}
