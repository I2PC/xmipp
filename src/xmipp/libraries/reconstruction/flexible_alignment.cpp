/***************************************************************************
 *
 * Authors:    Slavica Jonic                slavica.jonic@impmc.jussieu.fr
 *             Carlos Oscar Sanchez Sorzano coss.eps@ceu.es
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

#include <algorithm>
#include <fstream>

#include "core/bilib/messagedisplay.h"
#include "core/bilib/error.h"
#include "core/bilib/configs.h"
#include "core/bilib/linearalgebra.h"
#include "core/multidim_array.h"
#include "core/transformations.h"
#include "core/xmipp_image.h"
#include "flexible_alignment.h"
#include "data/pdb.h"
#include "program_extension.h"

// Empty constructor =======================================================
ProgFlexibleAlignment::ProgFlexibleAlignment() : Rerunable("")
{
	rangen = 0;
	resume = false;
	currentImgName = "";
	each_image_produces_an_output = false;
	produces_an_output = true;
}

// Params definition ============================================================
void ProgFlexibleAlignment::defineParams() {
	addUsageLine(
			"Compute deformation parameters according to a set of NMA modes");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment(
			"Metadata with output alignment and deformations");
	XmippMetadataProgram::defineParams();
	addParamsLine(
			"   --pdb <PDB_filename>                : PDB Model to compute NMA");
	addParamsLine("  [--odir <outputDir=\".\">]           : Output directory");
	addParamsLine("  [--resume]                           : Resume processing");
	addParamsLine("==Generation of the deformed volumes==");
	addParamsLine(
			"   --modes <filename>                  : File with a list of mode filenames");
	addParamsLine(
			"  [--defampsampling <s=200>]           : Deformation sampling");
	addParamsLine(
			"  [--maxdefamp <s=2000>]               : Maximum deformation amplitude");
	addParamsLine(
			"  [--translsampling <s=2>]             : Translational sampling");
	addParamsLine(
			"  [--maxtransl <max=10>]               : Maximum translation");
	addParamsLine(
			"  [--sampling_rate <Ts=1>]             : in Angstroms/pixel");
	addParamsLine(
			"  [--filterVol <cutoff=15.>]           : Filter the volume after deforming. Default cut-off is 15 A.");
	addParamsLine(
			"  [--centerPDB]                        : Center the PDB structure");
	addParamsLine(
			"  [--fixed_Gaussian <std=-1>]          : For pseudo atoms fixed_Gaussian must be used.");
	addParamsLine(
			"                                       : Default standard deviation <std> is read from PDB file.");
	addParamsLine("==Angular assignment and mode detection==");
	addParamsLine(
			"  [--mask <m=\"\">]                    : 2D Mask applied to the reference images of the deformed volume");
	addParamsLine(
			"                                       :+Note that wavelet assignment needs the input images to be of a size power of 2");
	addParamsLine(
			"  [--minAngularSampling <ang=3>]       : Minimum angular sampling rate");
	addParamsLine(
			"  [--gaussian_Real    <s=0.5>]         : Weighting sigma in Real space");
	addParamsLine(
			"  [--zerofreq_weight  <s=0.>]          : Zero-frequency weight");
	addParamsLine("  [--sigma    <s=10>]                  : Sigma");
	addParamsLine(
			"  [--max_iter  <N=60>]                 : Maximum number of iterations");
	addExampleLine(
			"xmipp_nma_alignment -i images.sel --pdb 2tbv.pdb --modes modelist.xmd --sampling_rate 6.4 -o output.xmd --resume");
	Rerunable::setFileName(fnOutDir+"/nmaDone.xmd");
}

// Read arguments ==========================================================
void ProgFlexibleAlignment::readParams() {
	XmippMetadataProgram::readParams();
	fnPDB = getParam("--pdb");
	fnOutDir = getParam("--odir");
	fnModeList = getParam("--modes");
	resume = checkParam("--resume");
	maxdefamp = getDoubleParam("--maxdefamp");
	defampsampling = getDoubleParam("--defampsampling");
	translsampling = getDoubleParam("--translsampling");
	maxtransl = getDoubleParam("--maxtransl");
	sampling_rate = getDoubleParam("--sampling_rate");
	fnmask = getParam("--mask");
	gaussian_Real_sigma = getDoubleParam("--gaussian_Real");
	weight_zero_freq = getDoubleParam("--zerofreq_weight");
	do_centerPDB = checkParam("--centerPDB");
	do_FilterPDBVol = checkParam("--filterVol");
	if (do_FilterPDBVol)
		cutoff_LPfilter = getDoubleParam("--filterVol");
	useFixedGaussian = checkParam("--fixed_Gaussian");
	//if (useFixedGaussian)
	sigmaGaussian = getDoubleParam("--fixed_Gaussian");
	minAngularSampling = getDoubleParam("--minAngularSampling");
	sigma = getDoubleParam("--sigma");
	max_no_iter = getIntParam("--max_iter");
}

// Show ====================================================================
void ProgFlexibleAlignment::show() {
	XmippMetadataProgram::show();
	std::cout << "Output directory:     " << fnOutDir << std::endl
			<< "PDB:                  " << fnPDB << std::endl
			<< "Resume:               " << resume << std::endl
			<< "Mode list:            " << fnModeList << std::endl
			<< "Deformation sampling: " << defampsampling << std::endl
			<< "Maximum amplitude:    " << maxdefamp << std::endl
			<< "Transl. sampling:     " << translsampling << std::endl
			<< "Max. Translation:     " << maxtransl << std::endl
			<< "Sampling rate:        " << sampling_rate << std::endl
			<< "Mask:                 " << fnmask << std::endl
			<< "Center PDB:           " << do_centerPDB << std::endl
			<< "Filter PDB volume     " << do_FilterPDBVol << std::endl
			<< "Use fixed Gaussian:   " << useFixedGaussian << std::endl
			<< "Sigma of Gaussian:    " << sigmaGaussian << std::endl
			<< "minAngularSampling:   " << minAngularSampling << std::endl
			<< "Gaussian Real:        " << gaussian_Real_sigma << std::endl
			<< "Zero-frequency weight:" << weight_zero_freq << std::endl
			<< "Sigma:                " << sigma << std::endl
			<< "Max. Iter:            " << max_no_iter << std::endl;
}

// Produce side information ================================================
ProgFlexibleAlignment *global_flexible_prog;

void ProgFlexibleAlignment::preProcess() {
	MetaDataVec SF(fnModeList);
	numberOfModes = SF.size();
	SF.getColumnValues(MDL_NMA_MODEFILE, modeList);

	// Get the size of the images in the selfile
	imgSize = xdimOut;
	// Set the pointer of the program to this object
	global_flexible_prog = this;
	//create some neededs files
	createWorkFiles();
}

void ProgFlexibleAlignment::finishProcessing() {
	XmippMetadataProgram::finishProcessing();
	rename(Rerunable::getFileName().c_str(), fn_out.c_str());
}

// Create deformed PDB =====================================================
FileName ProgFlexibleAlignment::createDeformedPDB() const {
	String program;
	String arguments;
	FileName fnRandom;
	fnRandom.initUniqueName(nameTemplate, fnOutDir);
	const char *randStr = fnRandom.c_str();
	program = "xmipp_pdb_nma_deform";
	arguments = formatString(
			"--pdb %s -o %s_deformedPDB.pdb --nma %s --deformations ",
			fnPDB.c_str(), randStr, fnModeList.c_str());
	for (size_t i = 5; i < VEC_XSIZE(trial); i++)
		arguments += floatToString(trial(i)) + " ";
	runSystem(program, arguments, false);

	return fnRandom;
}

void ProjectionRefencePoint(Matrix1D<double> &Parameters, int dim, double *R,
		double *Tr, MultidimArray<double> &proj_help_test,
		MultidimArray<double> &P_esp_image,
		//double   *S_mu,
		int Xwidth, int Ywidth, double sigma)
{
//    double *coord_gaussian;
//    double *ksi_v,*coord_img;
//    double *ro_ksi_v, *ro_coord_img, *ro_coord_gaussian;
    //double S_mu;
    auto    psi_max = (int) (sqrt(3) * Ywidth / 2);
//    int    kx,ky;
    double    a0,a1,a2;
    double centre_Xwidth, centre_Ywidth;
    double sum2,hlp;

	proj_help_test.initZeros(Xwidth, Ywidth);
	std::vector<double> ksi_v(4,0);
	for (int i = 0; i < dim + 5; i++) {
		global_flexible_prog->trial(i) = Parameters(i);
	}
	std::string command;
	String program;
	String arguments;
	FileName fnRandom;
	fnRandom.initUniqueName(global_flexible_prog->nameTemplate,
			global_flexible_prog->fnOutDir);
	const char *randStr = fnRandom.c_str();

	program = "xmipp_pdb_nma_deform";
	arguments = formatString(
			"--pdb %s -o %s_deformedPDB.pdb --nma %s --deformations ",
			global_flexible_prog->fnPDB.c_str(), randStr,
			global_flexible_prog->fnModeList.c_str());
	for (size_t i = 5; i < VEC_XSIZE(global_flexible_prog->trial); i++) {
		float aaa = global_flexible_prog->scdefamp * Parameters(i);
		arguments += floatToString(aaa) + " ";
	}
	runSystem(program, arguments, false);

	String deformed_pdb = formatString("%s_deformedPDB.pdb", randStr);
	centre_Xwidth = double(Xwidth - 1) / 2.0;
	centre_Ywidth = double(Ywidth - 1) / 2.0;
	Matrix1D<double> limit0(3), limitF(3), centerOfMass(3);
	const char *intensityColumn = " ";
	computePDBgeometry(global_flexible_prog->fnPDB, centerOfMass, limit0,
			limitF, intensityColumn);
	centerOfMass = (limit0 + limitF) / 2;
	std::ifstream fh_deformedPDB;
	fh_deformedPDB.open(deformed_pdb.c_str());

	if (!fh_deformedPDB)
		REPORT_ERROR(ERR_UNCLASSIFIED,
				(std::string )"Prog_PDBPhantom_Parameters::protein_geometry:"
						"Cannot open " + deformed_pdb + " for reading");

	// Process all lines of the filem+".pdb").c_str());
	if (!fh_deformedPDB)
		REPORT_ERROR(ERR_UNCLASSIFIED,
				(std::string )"Prog_PDBPhantom_Parameters::protein_geometry:"
						"Cannot open " + deformed_pdb + " for reading");

	std::vector<double> ro_ksi_v(4,0);
	ksi_v[0] = 0.0;
	ksi_v[1] = 0.0;
	ksi_v[3] = 1.0;

	std::vector<double> coord_gaussian(4,0);
	std::vector<double> ro_coord_gaussian(4,0);

	coord_gaussian[3] = 1.0;

	std::vector<double> coord_img(4,0);
	std::vector<double> ro_coord_img(4,0);

	coord_img[2] = 0.0;
	coord_img[3] = 1.0;
	std::string kind;
	std::string line;
	std::string atom_type;
	int ttt = 0;
	int kx, ky;
	// Reading PDB/CIF file
	PDBPhantom pdb;
	FileName fileNameDeformedPdb(deformed_pdb.c_str());
	pdb.read(fileNameDeformedPdb);

	int nAtom = 0;
	while (!fh_deformedPDB.eof()) {
		ttt++;
		// Read an ATOM line
		getline(fh_deformedPDB, line);
		if (line == "")
			continue;
		kind = line.substr(0, 4);
		if (kind != "ATOM" && kind != "HETA")
			continue;

		// Extract atom type and position
		// Typical line:
		// ATOM    909  CA  ALA A 161      58.775  31.984 111.803  1.00 34.78
		const auto& atom = pdb.atomList[nAtom];
		atom_type = atom.atomType;
		double x = atom.x;
		double y = atom.y;
		double z = atom.z;
		coord_gaussian[0] = (x - centerOfMass(0))
				/ (global_flexible_prog->sampling_rate);
		coord_gaussian[1] = (y - centerOfMass(1))
				/ (global_flexible_prog->sampling_rate);
		coord_gaussian[2] = (z - centerOfMass(2))
				/ (global_flexible_prog->sampling_rate);

		for (int ksi = -psi_max; ksi <= psi_max; ksi++) {
			ksi_v[2] = (double) ksi;
			MatrixMultiply(R, ksi_v.data(), ro_ksi_v.data(), 4L, 4L, 1L);

			for (ky = 0; ky < Ywidth; ky++) {
				coord_img[1] = (double) ky - centre_Ywidth;
				for (kx = 0; kx < Xwidth; kx++) {
					coord_img[0] = (double) kx - centre_Xwidth;
					MatrixMultiply(Tr, coord_img.data(), ro_coord_img.data(), 4L, 4L, 1L);

					a0 = ro_coord_img[0] + ro_ksi_v[0] + coord_gaussian[0];
					a1 = ro_coord_img[1] + ro_ksi_v[1] + coord_gaussian[1];
					a2 = ro_coord_img[2] + ro_ksi_v[2] + coord_gaussian[2];
					proj_help_test(kx - Xwidth / 2, ky - Ywidth / 2) +=
							(double) exp(
									-(a0 * a0 + a1 * a1 + a2 * a2)
											/ (sigma * sigma));
				}
			}
		}
		nAtom++;
	}

	Image<double> test1;
	test1() = proj_help_test;

	// Close file
	fh_deformedPDB.close();
	// To calculate the value of cost function
	for (sum2 = 0.0, kx = 0; kx < Xwidth; kx++) {
		for (ky = 0; ky < Ywidth; ky++) {
			hlp = (double) (proj_help_test(kx - Xwidth / 2, ky - Ywidth / 2)
					- P_esp_image(kx - Xwidth / 2, ky - Ywidth / 2));
			hlp = (double) hlp * hlp;
			sum2 += (double) hlp;
		}
	}
	global_flexible_prog->costfunctionvalue = (double) sum2;
}

/* ------------------------------------------------------------------------- */
/* Calculate the values of partial valur of P_mu_image                       */
/* ------------------------------------------------------------------------- */
int partialpfunction(Matrix1D<double> &Parameters,
		Matrix1D<double> &centerOfMass, double *R, double *Tr, double *DR0,
		double *DR1, double *DR2, MultidimArray<double> &DP_Rz1,
		MultidimArray<double> &DP_Ry, MultidimArray<double> &DP_Rz2,
		MultidimArray<double> &DP_x, MultidimArray<double> &DP_y,
		MultidimArray<double> &DP_q,
		//double            *cost,
//		MultidimArray<double> &P_mu_image, MultidimArray<double> &P_esp_image,
		int Xwidth, int Ywidth) {
	auto psi_max = (int) (sqrt(3) * 128 / (global_flexible_prog->sampling_rate));
	double help, a0, a1, a2;
	help = 0.0;
	int Line_number = 0;
	int kx, ky;
	int dim = global_flexible_prog->numberOfModes;
	double centre_Xwidth, centre_Ywidth;
	centre_Xwidth = (double) (Xwidth - 1) / 2.0;
	centre_Ywidth = (double) (Ywidth - 1) / 2.0;

	DP_Rz1.initZeros(Xwidth, Ywidth);
	DP_Ry.initZeros(Xwidth, Ywidth);
	DP_Rz2.initZeros(Xwidth, Ywidth);
	DP_x.initZeros(Xwidth, Ywidth);
	DP_y.initZeros(Xwidth, Ywidth);
	DP_q.initZeros(dim, Xwidth, Ywidth);

	std::ifstream ModeFile;
	//std::string modefilename = modeList[0];
	std::string modefilename = global_flexible_prog->modeList[0];
	ModeFile.open(modefilename.c_str());
	if (ModeFile.fail())
		REPORT_ERROR(ERR_UNCLASSIFIED,
				(std::string ) modefilename + " for reading");
	std::string line;
	while (getline(ModeFile, line)) {
		Line_number++;
	}
	ModeFile.close();

	std::vector<double> ModeValues(3 * Line_number * dim);
	std::string x, y, z;

	for (int i = 0; i < dim; i++) {
		modefilename = global_flexible_prog->modeList[i];
		ModeFile.open(modefilename.c_str());
		int n = 0;
		while (getline(ModeFile, line)) {
			x = line.substr(3, 10);
			y = line.substr(14, 10);
			z = line.substr(27, 10);
			ModeValues[i * 3 * Line_number + n * 3 + 0] = atof(x.c_str());
			ModeValues[i * 3 * Line_number + n * 3 + 1] = atof(y.c_str());
			ModeValues[i * 3 * Line_number + n * 3 + 2] = atof(z.c_str());
			n++;
		}
		ModeFile.close();
	}

	for (int i = 0; i < dim + 5; i++) {
		global_flexible_prog->trial(i) = Parameters(i);
	}

	FileName fnRandom = global_flexible_prog->createDeformedPDB();
	std::ifstream fh_deformedPDB;
	fh_deformedPDB.open((fnRandom + "_deformedPDB.pdb").c_str());
	if (!fh_deformedPDB)
		REPORT_ERROR(ERR_UNCLASSIFIED,
				(std::string )"Prog_PDBPhantom_Parameters::protein_geometry:" "Cannot open "
						+ fnRandom + "_deformedPDB.pdb" + " for reading");

	// Process all lines of the file
	std::vector<double> help_v(4,0);
	help_v[0] = 0.0;
	help_v[1] = 0.0;
	help_v[3] = 1.0;

	std::vector<double> coord_gaussian(4,0);
	coord_gaussian[3] = 1.0;

	std::vector<double> coord_img(4,0);
	coord_img[2] = 0.0;
	coord_img[3] = 1.0;
	int k = 0;

	// Reading PDB/CIF file
	PDBPhantom pdb;
	FileName fileNameDeformedPdb((fnRandom + "_deformedPDB.pdb").c_str());
	pdb.read(fileNameDeformedPdb);

	std::string kind;
	int nAtom = 0;
	while (!fh_deformedPDB.eof()) {
		// Read an ATOM line
		getline(fh_deformedPDB, line);
		if (line == "")
			continue;
		kind = line.substr(0, 4);
		if (kind != "ATOM")
			continue;

		// Extract atom type and position
		// Typical line:
		// ATOM    909  CA  ALA A 161      58.775  31.984 111.803  1.00 34.78
		const auto& atom = pdb.atomList[nAtom];
		char atom_type = atom.atomType;
		double xPos = atom.x;
		double yPos = atom.y;
		double zPos = atom.z;

		// Correct position
		coord_gaussian[0] = (xPos - centerOfMass(0))
				/ global_flexible_prog->sampling_rate;
		coord_gaussian[1] = (yPos - centerOfMass(1))
				/ global_flexible_prog->sampling_rate;
		coord_gaussian[2] = (zPos - centerOfMass(2))
				/ global_flexible_prog->sampling_rate;

		//MatrixMultiply( Tr, coord_gaussian, coord_gaussian,4L, 4L, 1L);
		if (MatrixMultiply(Tr, coord_gaussian.data(), coord_gaussian.data(), 4L, 4L, 1L) == ERROR) {
			WRITE_ERROR(partialpfunction, "Error returned by MatrixMultiply");
			return (ERROR);
		}

		for (int ksi = -psi_max; ksi <= psi_max; ksi++) {
			coord_img[3] = (double) ksi;
			for (kx = 0; kx < Xwidth; kx++) {
				for (ky = 0; ky < Ywidth; ky++) {

					coord_img[0] = (double) kx - centre_Xwidth;
					coord_img[1] = (double) ky - centre_Ywidth;

					//MatrixTimesVector( Tr, coord_img, help_v,4L, 4L);
					if (MatrixTimesVector(Tr, coord_img.data(), help_v.data(), 4L, 4L) == ERROR) {
						WRITE_ERROR(partialpfunction, "Error returned by MatrixMultiply");
						return (ERROR);
					}
					a0 = help_v[0] + coord_gaussian[0];
					a1 = help_v[1] + coord_gaussian[1];
					a2 = help_v[2] + coord_gaussian[2];

					help = (double) exp(
							-(a0 * a0 + a1 * a1 + a2 * a2)
									/ (global_flexible_prog->sigma
											* global_flexible_prog->sigma));
					if (MatrixTimesVector(DR0, coord_img.data(), help_v.data(), 4L, 4L) == ERROR) {
						WRITE_ERROR(partialpfunction, "Error returned by MatrixMultiply");
						return (ERROR);
					}

					DP_Rz1(kx, ky) += help
									* (a0 * help_v[0] + a1 * help_v[1]
											+ a2 * help_v[2]);
					if (MatrixTimesVector(DR1, coord_img.data(), help_v.data(), 4L, 4L) == ERROR) {
						WRITE_ERROR(partialpfunction, "Error returned by MatrixMultiply");
						return (ERROR);
					}
					DP_Ry(kx, ky) += help
									* (a0 * help_v[0] + a1 * help_v[1]
											+ a2 * help_v[2]);
					if (MatrixTimesVector(DR2, coord_img.data(), help_v.data(), 4L, 4L) == ERROR) {
						WRITE_ERROR(partialpfunction, "Error returned by MatrixMultiply");
						return (ERROR);
					}
					DP_Rz2(kx, ky) += help
									* (a0 * help_v[0] + a1 * help_v[1]
											+ a2 * help_v[2]);
					DP_x(kx, ky) += help * (a0 * R[0] + a1 * R[1] + a2 * R[2]); //global_flexible_prog->sctrans * DP_x(kx,ky)
					DP_y(kx, ky) += help * (a0 * R[4] + a1 * R[5] + a2 * R[6]); //global_flexible_prog->sctrans * DP_y(kx,ky)
					for (int i = 0; i < dim; i++) {
						DP_q(i, kx, ky) += global_flexible_prog->scdefamp * help
								* (a0 * ModeValues[i * 3 * Line_number + k * 3]
										+ a1 * ModeValues[i * 3 * Line_number + k * 3 + 1]
										+ a2 * ModeValues[i * 3 * Line_number + k * 3 + 2]);
					}
				}
			}
		}
		k++;
		nAtom++;
	}
	fh_deformedPDB.close();
	return (!ERROR);
}/*end of partialpfunction*/

/* ------------------------------------------------------------------------- */
/* Gradient and Hessian at pixel                                             */
/* ------------------------------------------------------------------------- */
void gradhesscost_atpixel(double *Gradient, double *Hessian, double *helpgr,
		double difference) {
	int trialSize = VEC_XSIZE(global_flexible_prog->trial);

	for (int i = 0; i < trialSize; i++) {
		Gradient[i] += difference * helpgr[i];
		for (int j = 0; j <= i; j++) {
			Hessian[i * trialSize + j] += helpgr[j] * helpgr[i];
		}
	}
	//return(!ERROR);
}/* End of gradhesscost_atpixel */

/* ------------------------------------------------------------------------- */
/* Calculate the values of Gradient and Hessian                              */
/* ------------------------------------------------------------------------- */
// int Prog_flexali_gauss_prm::return_gradhesscost(
int return_gradhesscost(Matrix1D<double> &centerOfMass, double *Gradient,
		double *Hessian, Matrix1D<double> &Parameters, int dim,
		MultidimArray<double> &rg_projimage, MultidimArray<double> &P_esp_image,
		int Xwidth, int Ywidth) {
	double phi, theta, psi, x0, y0;
	int i, j;
	double difference;
	double SinPhi, CosPhi, SinPsi, CosPsi, SinTheta, CosTheta;
	int trialSize = VEC_XSIZE(global_flexible_prog->trial);
	double lambda = 1000.0;
	double sigmalocal = global_flexible_prog->sigma;
	int half_Xwidth = Xwidth / 2;
	int half_Ywidth = Ywidth / 2;

	//global_flexible_prog->costfunctionvalue = 0.0;

	for (int i = 0; i < dim + 5; i++) {
		global_flexible_prog->trial(i) = Parameters(i);
	}
	phi = Parameters(0);
	theta = Parameters(1);
	psi = Parameters(2);
	x0 = Parameters(3);
	y0 = Parameters(4);
	//defamp = (double *)malloc((size_t) dim * sizeof(double));
	//if (defamp == (double *)NULL)
	//{
	//    WRITE_ERROR(return_gradhesscost, "ERROR - Not enough memory for defamp");
	//    return(ERROR);
	//}
	SinPhi = sin(phi);
	CosPhi = cos(phi);
	SinPsi = sin(psi);
	CosPsi = cos(psi);
	SinTheta = sin(theta);
	CosTheta = cos(theta);
	std::vector<double> Rz1(16,0);
	std::vector<double> Ry(16,0);
	std::vector<double> Rz2(16,0);
	if (GetIdentitySquareMatrix(Rz2.data(), 4L) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by GetIdentitySquareMatrix");
		return (ERROR);
	}
	double *hlp = Rz2.data();
	*hlp++ = CosPsi;
	*hlp = SinPsi;
	hlp += (std::ptrdiff_t) 3L;
	*hlp++ = -SinPsi;
	*hlp = CosPsi;

	if (GetIdentitySquareMatrix(Rz1.data(), 4L) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by GetIdentitySquareMatrix");
		return (ERROR);
	}

	hlp = Rz1.data();
	*hlp++ = CosPhi;
	*hlp = SinPhi;
	hlp += (std::ptrdiff_t) 3L;
	*hlp++ = -SinPhi;
	*hlp = CosPhi;
	if (GetIdentitySquareMatrix(Ry.data(), 4L) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by GetIdentitySquareMatrix");
		return (ERROR);
	}

	hlp = Ry.data();
	*hlp = CosTheta;
	hlp += (std::ptrdiff_t) 2L;
	*hlp = -SinTheta;
	hlp += (std::ptrdiff_t) 6L;
	*hlp = SinTheta;
	hlp += (std::ptrdiff_t) 2L;
	*hlp = CosTheta;

	std::vector<double> R(16,0);
	if (multiply_3Matrices(Rz2.data(), Ry.data(), Rz1.data(), R.data(), 4L, 4L, 4L, 4L) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by multiply_3Matrices");
		return (ERROR);
	}

	std::vector<double> DRz1(16,0);
	std::vector<double> DRy(16,0);
	std::vector<double> DRz2(16,0);

	hlp = DRz2.data();
	*hlp++ = -SinPsi;
	*hlp = CosPsi;
	hlp += (std::ptrdiff_t) 3L;
	*hlp++ = -CosPsi;
	*hlp = -SinPsi;

	hlp = DRz1.data();
	*hlp++ = -SinPhi;
	*hlp = CosPhi;
	hlp += (std::ptrdiff_t) 3L;
	*hlp++ = -CosPhi;
	*hlp = -SinPhi;

	hlp = DRy.data();
	*hlp = -SinTheta;
	hlp += (std::ptrdiff_t) 2L;
	*hlp = -CosTheta;
	hlp += (std::ptrdiff_t) 6L;
	*hlp = CosTheta;
	hlp += (std::ptrdiff_t) 2L;
	*hlp = -SinTheta;

	std::vector<double> DR0(16,0);
	std::vector<double> DR1(16,0);
	std::vector<double> DR2(16,0);

	if (multiply_3Matrices(Rz2.data(), Ry.data(), DRz1.data(), DR0.data(), 4L, 4L, 4L, 4L) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by multiply_3Matrices");
		return (ERROR);
	}

	if (multiply_3Matrices(Rz2.data(), DRy.data(), Rz1.data(), DR1.data(), 4L, 4L, 4L, 4L) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by multiply_3Matrices");
		return (ERROR);
	}

	if (multiply_3Matrices(DRz2.data(), Ry.data(), Rz1.data(), DR2.data(), 4L, 4L, 4L, 4L) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by multiply_3Matrices");
		return (ERROR);
	}

	std::vector<double> Tr(16,0);
	if (GetIdentitySquareMatrix(Tr.data(), 4L) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by GetIdentitySquareMatrix");
		return (ERROR);
	}

	hlp = Tr.data();
	hlp += (std::ptrdiff_t) 3L;
	*hlp = -x0; //global_flexible_prog->sctrans * x0
	hlp += (std::ptrdiff_t) 5L;
	*hlp = -y0; //global_flexible_prog->sctrans * y0
	ProjectionRefencePoint(Parameters, dim, R.data(), Tr.data(), rg_projimage, P_esp_image,
			Xwidth, Ywidth, sigmalocal);

	MultidimArray<double> DP_Rx(Xwidth, Ywidth), DP_Ry(Xwidth, Ywidth), DP_Rz2(
			Xwidth, Ywidth);
	MultidimArray<double> DP_x(Xwidth, Ywidth), DP_y(Xwidth, Ywidth);
	MultidimArray<double> DP_q(dim, Xwidth, Ywidth);
	if (partialpfunction(Parameters, centerOfMass, R.data(), Tr.data(), DR0.data(), DR1.data(), DR2.data(), DP_Rx,
			DP_Ry, DP_Rz2, DP_x, DP_y, P_esp_image, Xwidth,
			Ywidth) == ERROR) {
		WRITE_ERROR(return_gradhesscost, "Error returned by partialpfunction");
		return (ERROR);
	}

	std::vector<double> helpgr(trialSize,0);

	for (i = 0; i < trialSize; i++) {
		Gradient[i] = 0.0;
		for (j = 0; j < trialSize; j++) {
			Hessian[i * trialSize + j] = 0.0;
		}
	}

	for (int kx = 0; kx < Xwidth; kx++)
		for (int ky = 0; ky < Ywidth; ky++) {
			difference = rg_projimage(kx - half_Xwidth, ky - half_Ywidth)
					- P_esp_image(kx - half_Xwidth, ky - half_Ywidth);
			helpgr[0] = DP_Rx(kx, ky);
			helpgr[1] = DP_Ry(kx, ky);
			helpgr[2] = DP_Rz2(kx, ky);
			helpgr[3] = DP_x(kx, ky);
			helpgr[4] = DP_y(kx, ky);
			for (j = 0; j < dim; j++) {
				helpgr[5 + j] = DP_q(j, kx, ky);
			}
			gradhesscost_atpixel(Gradient, Hessian, helpgr.data(), difference);
		}

	for (int i = 0; i < trialSize; i++) {
		Hessian[i * trialSize + i] += lambda * Hessian[i * trialSize + i];
		if (i + 1 < trialSize)
			for (int j = i + 1; j < trialSize; j++) {
				Hessian[i * trialSize + j] = Hessian[j * trialSize + i];
			}
	}
	return (!ERROR);
}/* End of return_gradhesscost */

/* ------------------------------------------------------------------------- */
/* Optimizer                                                                 */
/* ------------------------------------------------------------------------- */
int levenberg_cst2(MultidimArray<double> &lc_P_mu_image,
		MultidimArray<double> &P_esp_image, Matrix1D<double> &centerOfMass,
		double *beta, double *alpha,
		//double *cost,
		Matrix1D<double> &Parameters, double OldCost, double *lambda,
		double LambdaScale, long *iter, double tol_angle, double tol_shift,
		double tol_defamp, int *IteratingStop, size_t Xwidth, size_t Ywidth) {

	auto ma = (long) VEC_XSIZE(global_flexible_prog->trial);
	if (ma <= 0) {
		WRITE_ERROR(levenberg_cst2, "ERROR - degenerated vector");
		return (ERROR);
	}
	double max_defamp;
	int dim = global_flexible_prog->numberOfModes;
	int Status = !ERROR;

	std::vector<double> a(ma,0);
	for (size_t i = 0; i < ma; i++)
		a[i] = Parameters(i);

	std::vector<double> da(ma,0);
	std::vector<double> u(ma*ma,0);
	std::vector<double> v(ma*ma,0);
	std::vector<double> w(ma,0);

	double *t = u.data();
	double *uptr = u.data();
	for (size_t i = 0L; (i < ma); t += (std::ptrdiff_t) (ma + 1L), i++) {
		for (size_t j = 0L; (j < ma); alpha++, j++) {
			*uptr++ = -*alpha;
		}

		*t *= 1.0 + *lambda;
	}
	alpha -= (std::ptrdiff_t) (ma * ma);

	SingularValueDecomposition(u.data(), ma, ma, w.data(), v.data(), SVDMAXITER, &Status);
	double wmax = *std::max_element(w.begin(), w.end());
	double thresh = DBL_EPSILON * wmax;
	size_t j = ma;
	while (--j >= 0L) {
		if (w[j] < thresh) {
			w[j] = 0.0;
			for (size_t i = 0; (i < ma); i++) {
				u[i * ma + j] = 0.0;
				v[i * ma + j] = 0.0;
			}
		}
	}

	SingularValueBackSubstitution(u.data(), w.data(), v.data(), ma, ma, beta, da.data(), &Status);
	double *vptr = (double*) memcpy(v.data(), a.data(), (size_t) ma * sizeof(double));
	t = vptr + (std::ptrdiff_t) ma;
	double *aptr = a.data()+(std::ptrdiff_t) ma;
	double *daptr = da.data()+(std::ptrdiff_t) ma;
	while (--t >= vptr) {
		daptr--;
		*t = *--aptr;
		*t += *daptr;
	}
	for (size_t i = 0; i < ma; i++)
		Parameters(i) = v[i];

	return_gradhesscost(centerOfMass, w.data(), u.data(), Parameters, dim, lc_P_mu_image,
			P_esp_image, Xwidth, Ywidth);
	double costchanged = global_flexible_prog->costfunctionvalue;

	size_t i;
	for (max_defamp = 0.0, i = 0L; i < dim; i++) {
		double hlp = fabs(a[i + 5] - v[i + 5]);
		if (hlp > max_defamp)
			max_defamp = fabs(a[i + 5] - v[i + 5]);
	}

	(*iter)++;
	if (costchanged < OldCost) {
		if ((fabs(a[0] - v[0]) < tol_angle) && (fabs(a[1] - v[1]) < tol_angle)
				&& (fabs(a[2] - v[2]) < tol_angle)) {
			if ((fabs(a[3] - v[3]) < tol_shift)
					&& (fabs(a[4] - v[4]) < tol_shift)) {
				if (max_defamp < tol_defamp) {
					*IteratingStop = 1;
				}
			}
		}
	}

	if (global_flexible_prog->costfunctionvalue_cst == -1.0) {
		if (costchanged < OldCost) {
			for (i = 0L; (i < ma); i++) {
				for (j = 0L; (j < ma); j++)
					alpha[i * ma + j] = u[i * ma + j];
				beta[i] = w[i];
				a[i] = v[i];
			}
			global_flexible_prog->costfunctionvalue_cst = costchanged;
		}

		return (!ERROR);
	}

	for (i = 0L; (i < ma); i++) {
		for (j = 0L; (j < ma); j++)
			alpha[i * ma + j] = u[i * ma + j];
		beta[i] = w[i];
		a[i] = v[i];
	}
	global_flexible_prog->costfunctionvalue_cst = costchanged;

#ifndef DBL_MIN
#define DBL_MIN 1e-26
#endif
#ifndef DBL_MAX
#define DBL_MAX 1e+26
#endif

	if (costchanged < OldCost) {
		if (*lambda > DBL_MIN) {
			*lambda /= LambdaScale;
		} else {
			*IteratingStop = 1;
		}
	} else {
		if (*lambda < DBL_MAX) {
			*lambda *= LambdaScale;
		} else {
			*IteratingStop = 1;
		}
	}
	return (!ERROR);
} /* End of levenberg_cst2 */

/* Registration ------------------------------------------------------------ */
//cstregistrationcontinuous(centerOfMass,cost,Parameters,P_mu_image,P_esp_image,Xwidth,Ywidth);
int cstregistrationcontinuous(Matrix1D<double> &centerOfMass,
		//double            *cost,
		Matrix1D<double> &Parameters, MultidimArray<double> &cst_P_mu_image,
		MultidimArray<double> &P_esp_image, size_t Xwidth, size_t Ywidth) {
	int DoDesProj, IteratingStop, FlagMaxIter;
	long MaxIter, MaxIter1, iter;
	long MaxNumberOfFailures, SatisfNumberOfSuccesses, nSuccess, nFailure;
	double LambdaScale = 2., OldCost, tol_angle, tol_shift, tol_defamp;
	double OneIterInSeconds;
	time_t *tp1 = nullptr;
	time_t *tp2 = nullptr;
	auto dim = (long) global_flexible_prog->numberOfModes;
	long MaxNoIter, MaxNoFailure, SatisfNoSuccess;

	double lambda = 1000.;
	time_t time1 = time(tp1);
	DoDesProj = 0;
	MaxNoIter = global_flexible_prog->max_no_iter;
	MaxNoFailure = (long) (0.3 * MaxNoIter);
	SatisfNoSuccess = (long) (0.7 * MaxNoIter);
	MaxIter = MaxNoIter;
	MaxIter1 = MaxIter - 1L;

	MaxNumberOfFailures = MaxNoFailure;
	SatisfNumberOfSuccesses = SatisfNoSuccess;
	tol_angle = 0.0;
	tol_shift = 0.0;
	tol_defamp = 0.0;

	std::vector<double> Gradient(dim + 5,0);
	std::vector<double> Hessian((dim + 5) * (dim + 5),0);

	if (DoDesProj && (global_flexible_prog->currentStage == 2)) {
		Parameters(0) = 10;
		Parameters(1) = 0;
		Parameters(2) = 0;
		Parameters(3) = 0;
		Parameters(4) = 0;
		Parameters(5) = 0.5;
		//Parameters(6)=0;
	}

	if (return_gradhesscost(centerOfMass, Gradient.data(), Hessian.data(), Parameters, dim,
			cst_P_mu_image, P_esp_image, Xwidth, Ywidth) == ERROR) {
		WRITE_ERROR(cstregistrationcontinuous, "Error returned by return_gradhesscost");
		return (ERROR);
	}
	time_t time2 = time(tp2);
	OneIterInSeconds = difftime(time2, time1);
	if (DoDesProj && (global_flexible_prog->currentStage == 2)) {
		Image<double> Itemp;
		Itemp() = cst_P_mu_image;
		Itemp.write("test_refimage.spi");
		return (!ERROR);
	}
	if ((MaxIter == 0L) && (!DoDesProj))
		return (!ERROR);
	nSuccess = 0L;
	nFailure = 0L;
	OldCost = global_flexible_prog->costfunctionvalue_cst;
	iter = -1L;
	IteratingStop = 0;
	FlagMaxIter = (MaxIter != 1L);
	if (!FlagMaxIter)
		global_flexible_prog->costfunctionvalue_cst = -1.0;

	do {
		if (levenberg_cst2(cst_P_mu_image, P_esp_image, centerOfMass, Gradient.data(),
				Hessian.data(), Parameters, OldCost, &lambda, LambdaScale, &iter,
				tol_angle, tol_shift, tol_defamp, &IteratingStop, Xwidth,
				Ywidth) == ERROR) {
			WRITE_ERROR(cstregistrationcontinuous, "Error returned by levenberg_cst2");
			return (ERROR);
		}

		if (global_flexible_prog->costfunctionvalue_cst < OldCost) {
			OldCost = global_flexible_prog->costfunctionvalue_cst;
			nSuccess++;
			if (nSuccess >= SatisfNumberOfSuccesses) {
				break;
			}
			if (IteratingStop) {
				break;
			}
		} else {
			nFailure++;
		}

	} while ((nFailure <= MaxNumberOfFailures) && (iter < MaxIter1)
			&& FlagMaxIter);
	return (!ERROR);
}

void ProgFlexibleAlignment::performCompleteSearch(int pyramidLevel) {

	int dim = numberOfModes;
	int ModMaxdeDefamp, ModpowDim, help;
	double SinPhi, CosPhi, SinPsi, CosPsi;
	double S_muMin = 1e30;
	//double    *cost;
	costfunctionvalue = 0.0;
	Matrix1D<double> Parameters(dim + 5);
	Matrix1D<double> limit0(3), limitF(3), centerOfMass(3);
	const char *intensityColumn = "Bfactor";
	computePDBgeometry(fnPDB, centerOfMass, limit0, limitF, intensityColumn);
	centerOfMass = (limit0 + limitF) / 2;

	FileName fnRandom;
	fnRandom.initUniqueName(nameTemplate, fnOutDir);
	std::string command;

	// Reduce the image
	fnDown = formatString("%s_downimg.xmp", fnRandom.c_str());
	if (pyramidLevel != 0) {
		Image<double> I;
		I.read(currentImgName);
		selfPyramidReduce(xmipp_transformation::BSPLINE3, I(), pyramidLevel);
		I.write(fnDown);
	}
	Image<double> imgtemp;
	imgtemp.read(fnDown);

	imgtemp().setXmippOrigin();
	MultidimArray<double> P_mu_image;
	P_mu_image.resize(imgtemp());
	int Xwidth = XSIZE(imgtemp());
	int Ywidth = YSIZE(imgtemp());
	double reduce_rate = 1;

	ModMaxdeDefamp = (int) floor(maxdefamp / defampsampling) + 1;
	ModpowDim = (int) pow(ModMaxdeDefamp, dim);

	std::vector<double> Rz1(16,0);
	std::vector<double> Ry(16,0);
	std::vector<double> Rz2(16,0);
	std::vector<double> R(16,0);
	std::vector<double> Tr(16,0);

	int phiSteps=360/minAngularSampling;
	int thetaSteps=180/minAngularSampling;
	int psiSteps=360/minAngularSampling;
	int transSteps=maxtransl/translsampling;
	for (int iphi = 0; iphi <= phiSteps; iphi++) {
		double phi=iphi*minAngularSampling;
		Parameters(0) = phi;
		trial(0) = phi;
		SinPhi = sin(phi / 180 * PI);
		CosPhi = cos(phi / 180 * PI);

		GetIdentitySquareMatrix(Ry.data(), 4L);

		double *hlp = Ry.data();
		*hlp = CosPhi;
		hlp += (std::ptrdiff_t) 2L;
		*hlp = -SinPhi;
		hlp += (std::ptrdiff_t) 6L;
		*hlp = SinPhi;
		hlp += (std::ptrdiff_t) 2L;
		*hlp = CosPhi;

		for (int itheta = 0; itheta <=thetaSteps; itheta++) {
			double theta=itheta*minAngularSampling;
			Parameters(1) = theta;
			trial(1) = theta;

			GetIdentitySquareMatrix(Rz1.data(), 4L);

			hlp = Rz1.data();
			*hlp++ = CosPhi;
			*hlp = SinPhi;
			hlp += (std::ptrdiff_t) 3L;
			*hlp++ = -SinPhi;
			*hlp = CosPhi;

			for (int ipsi = 0; ipsi<=psiSteps; ipsi++) {
				double psi=ipsi*minAngularSampling;
				Parameters(2) = psi;
				trial(2) = psi;
				SinPsi = sin(psi / 180 * PI);
				CosPsi = cos(psi / 180 * PI);

				GetIdentitySquareMatrix(Rz2.data(), 4L);

				hlp = Rz2.data();
				*hlp++ = CosPsi;
				*hlp = SinPsi;
				hlp += (std::ptrdiff_t) 3L;
				*hlp++ = -SinPsi;
				*hlp = CosPsi;

				multiply_3Matrices(Rz2.data(), Ry.data(), Rz1.data(), R.data(), 4L, 4L, 4L, 4L);

				for (int ix0=0; ix0<=transSteps; ix0++) {
					double x0=ix0*translsampling;
					Parameters(3) = x0 / reduce_rate;
					trial(3) = x0;
					for (int iy0=0; iy0<=transSteps; iy0++) {
						double y0=iy0*translsampling;
						Parameters(4) = y0 / reduce_rate;
						trial(4) = y0;

						GetIdentitySquareMatrix(Tr.data(), 4L);
						hlp = Tr.data();
						hlp += (std::ptrdiff_t) 3L;
						*hlp = -x0;
						hlp += (std::ptrdiff_t) 5L;
						*hlp = -y0;

						MatrixMultiply(R.data(), Tr.data(), Tr.data(), 4L, 4L, 4L);

						for (int i = 0; i < ModpowDim; i++) {
							help = i;
							for (int j = dim - 1; j >= 0; j--) {
								Parameters(j + 5) = (double) (help
										% ModMaxdeDefamp) * defampsampling;
								trial(j + 5) = (double) (help % ModMaxdeDefamp)
										* defampsampling;
								help = (int) floor(help / ModMaxdeDefamp);

							}

							ProjectionRefencePoint(Parameters, dim, R.data(), Tr.data(),
									P_mu_image, imgtemp(), Xwidth, Ywidth,
									sigma);

							if (costfunctionvalue < S_muMin) { //2
								S_muMin = costfunctionvalue;
								for (int k = 0; k < dim + 5; k++) {
									trial_best(k) = trial(k);
								}

							}
							std::cout << "trial" << trial << "  cost="
									<< costfunctionvalue << std::endl;
							std::cout << "trial_best" << trial_best
									<< "  cost_best=" << S_muMin << std::endl;
						}
					}
				}
			}
		}

	}

	std::cout << "trial" << trial << costfunctionvalue << std::endl;
	std::cout << "trial_best" << trial_best << S_muMin << std::endl;
	for (int i = 0; i < dim + 5; i++) {
		trial(i) = trial_best(i);
		parameters(i) = trial_best(i);
	}

	MetaDataVec DF_out_discrete;
	size_t id = DF_out_discrete.addObject();
	DF_out_discrete.setValue(MDL_IMAGE, currentImgName, id);
	std::vector<double> aux;
	aux.resize(VEC_XSIZE(parameters));
	FOR_ALL_ELEMENTS_IN_MATRIX1D(parameters)
		aux[i] = parameters(i);
	DF_out_discrete.setValue(MDL_NMA, aux, id);
	DF_out_discrete.write("result_angular_discrete.xmd");
}

// Continuous assignment ===================================================
double ProgFlexibleAlignment::performContinuousAssignment(int pyramidLevel) {
	costfunctionvalue = 0.0;
	costfunctionvalue_cst = 0.0;

	Matrix1D<double> Parameters;

	Parameters = trial_best;

	std::string command;
	if (pyramidLevel == 0) {
		// Make links
		FileName fnRandom;
		fnRandom.initUniqueName(nameTemplate, fnOutDir);
		fnDown = formatString("%s_downimg.xmp", fnRandom.c_str());

		Image<double> I;
		I.read(currentImgName);
		selfPyramidReduce(xmipp_transformation::BSPLINE3, I(), pyramidLevel);
		I.write(fnDown);
	}

	//std::string arguments;
	//arguments = formatString("%s_downimg.xmp",fnRandom.c_str());
	Image<double> imgtemp;
	imgtemp.read(fnDown);

	imgtemp().setXmippOrigin();
	MultidimArray<double> P_mu_image, P_esp_image;
	P_esp_image = imgtemp();
	P_mu_image.resize(imgtemp());
	int Xwidth = XSIZE(imgtemp());
	int Ywidth = YSIZE(imgtemp());
	//double P_mu_image[Xwidth*Xwidth];
	Matrix1D<double> limit0(3), limitF(3), centerOfMass(3);
	const char *intensityColumn = "Bfactor";
	//Matrix1D<double> test=centerOfMass;
	computePDBgeometry(fnPDB, centerOfMass, limit0, limitF, intensityColumn);

	centerOfMass = (limit0 + limitF) / 2;

	cstregistrationcontinuous(centerOfMass, Parameters, P_mu_image, P_esp_image,
			Xwidth, Ywidth);
	/* Insert code for continious alignment */

	trial(3) *= pow(2.0, (double) pyramidLevel);
	trial(4) *= pow(2.0, (double) pyramidLevel);

	trial_best = trial;

	double outcost = costfunctionvalue_cst;
	return outcost;
}

// Compute fitness =========================================================
double ProgFlexibleAlignment::eval() {
	int pyramidLevelDisc = 1;
	int pyramidLevelCont = (currentStage == 1) ? 1 : 0;

	if (currentStage == 1) {
		performCompleteSearch(pyramidLevelDisc);
	} else {
		//link(currentImgName.c_str(), fnDown.c_str());
	}
	double fitness = performContinuousAssignment(pyramidLevelCont);

	std::cout << "Fitness" << std::endl;
	return fitness;
}

void ProgFlexibleAlignment::processImage(const FileName &fnImg,
		const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) {
	static size_t imageCounter = 0;
	++imageCounter;

	int dim = numberOfModes;

	parameters.initZeros(dim + 5);
	currentImgName = fnImg;
	snprintf(nameTemplate, 256, "_node%d_img%ld_XXXXXX", rangen,
			(long int) imageCounter);

	trial.initZeros(dim + 5);
	trial_best.initZeros(dim + 5);

	currentStage = 1;
#ifdef DEBUG

              std::cerr << std::endl << "DEBUG: ===== Node: " << rangen
              <<" processing image " << fnImg <<"(" << objId << ")"
              << " at stage: " << currentStage << std::endl;
#endif

	eval(); // FIXME is this call necessary?
	bestStage1 = trial = parameters = trial_best;

	currentStage = 2;
#ifdef DEBUG

              std::cerr << std::endl << "DEBUG: ===== Node: " << rangen
              <<" processing image " << fnImg <<"(" << objId << ")"
              << " at stage: " << currentStage << std::endl;
#endif

	double fitness = eval();

	trial = trial_best;
	parameters = trial_best;

	parameters.resize(VEC_XSIZE(parameters) + 1);
	parameters(VEC_XSIZE(parameters) - 1) = fitness;

	writeImageParameters(fnImg);
}

void ProgFlexibleAlignment::writeImageParameters(const FileName &fnImg) {

	MetaDataVec md;
	size_t objId = md.addObject();
	md.setValue(MDL_IMAGE, fnImg, objId);
	md.setValue(MDL_ENABLED, 1, objId);
	md.setValue(MDL_ANGLE_ROT, parameters(0), objId);
	md.setValue(MDL_ANGLE_TILT, parameters(1), objId);
	md.setValue(MDL_ANGLE_PSI, parameters(2), objId);
	md.setValue(MDL_SHIFT_X, parameters(3), objId);
	md.setValue(MDL_SHIFT_Y, parameters(4), objId);

	int dim = numberOfModes;
	std::vector<double> vectortemp;

	for (int j = 5; j < 5 + dim; j++)
		vectortemp.push_back(parameters(j));
	md.setValue(MDL_NMA, vectortemp, objId);
	md.setValue(MDL_COST, parameters(5 + dim), objId);
	md.append(Rerunable::getFileName());
}
