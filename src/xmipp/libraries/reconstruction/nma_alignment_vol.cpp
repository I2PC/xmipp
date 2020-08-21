/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
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

#include <limits>
#include "nma_alignment_vol.h"
#include "volume_from_pdb.h"
#include "program_extension.h"
#include "condor/Solver.h"

// Empty constructor =======================================================
ProgNmaAlignmentVol::ProgNmaAlignmentVol() {
	rangen = 0;
	resume = false;
	currentVolName = "";
	each_image_produces_an_output = false;
	produces_an_output = true;
	progVolumeFromPDB = new ProgPdbConverter();
	alignVolumes=false;
}

ProgNmaAlignmentVol::~ProgNmaAlignmentVol() {
	delete progVolumeFromPDB;
}

// Params definition ============================================================
void ProgNmaAlignmentVol::defineParams() {
	addUsageLine("Align volumes with an atomic or pseudo-atomic (from EM volume) structure by computing deformation amplitudes along normal modes");
	addUsageLine("You may also align the two volumes during the process");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with volume filenames");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Metadata with output Euler angles, shifts, and deformation amplitudes");
	XmippMetadataProgram::defineParams();
	addParamsLine("   --pdb <PDB_filename>                : Reference atomic or pseudo-atomic structure in PDB format");
	addParamsLine("  [--odir <outputDir=\".\">]           : Output directory");
	addParamsLine("  [--resume]                           : Resume processing");
	addParamsLine("  [--opdb <PDB_filename=\"\">]         : Output PDB which is the deformed input PDB");
	addParamsLine("==Generation of the deformed reference volumes==");
	addParamsLine("   --modes <filename>                  : File with a list of mode filenames");
	addParamsLine("  [--sampling_rate <Ts=1>]             : Pixel size in Angstroms");
	addParamsLine("  [--filterVol <cutoff=15.>]           : Filter the volume after deforming. If this option is used, the default cut-off is 15 A.");
	addParamsLine("  [--centerPDB]                        : Center the PDB structure");
	addParamsLine("  [--fixed_Gaussian <std=-1>]          : For pseudo-atoms fixed_Gaussian must be used.");
	addParamsLine("                                       : Default standard deviation <std> is read from the PDB file.");
	addParamsLine("  [--trustradius_scale <s=1>]          : Positive scaling factor to scale the initial trust region radius");
	addParamsLine("==Combined elastic and rigid-body alignment==");
	addParamsLine("  [--alignVolumes <frm_freq=0.25> <frm_shift=10>]  : Align the deformed volume to the input volume before comparing, this is using frm method for volume alignment with frm_freq and frm_shift as parameters");
	addParamsLine("                                       : You need to compile Xmipp with SHALIGNMENT support (see install.sh)");
	addParamsLine("  [--mask <m=\"\">]                    : 3D masking  of the projections of the deformed volume");
	addParamsLine("  [--tilt_values <tilt0=-90> <tiltF=90>]  : only use if you are trying to compensate for the missing wedge");
	addParamsLine("  [--condor_params <rhoStartBase=250.> <rhoEndBase=50.> <niter=10000>]  : parameters for the CONDOR optimiser (recommended to keep default)");
	addParamsLine("                                       : rhoStartBase > 0  : the lower the better, yet the slower");
	addParamsLine("                                       : rhoEndBase no specific rule, however it is better to keep it < 1000, if set very high we risk distortions");
	addParamsLine("                                       : niter should be big enough to guarantee that the search converges to the right set of nma deformation amplitudes");
	addExampleLine("xmipp_nma_alignment_vol -i volumes.xmd --pdb 2tbv.pdb --modes modelist.xmd --sampling_rate 3.2 -o output.xmd --resume");
}

// Read arguments ==========================================================
void ProgNmaAlignmentVol::readParams() {
	XmippMetadataProgram::readParams();
	fnPDB = getParam("--pdb");
	fnOutPDB = getParam("--opdb");
	fnOutDir = getParam("--odir");
	fnModeList = getParam("--modes");
	resume = checkParam("--resume");
	sampling_rate = getDoubleParam("--sampling_rate");
	fnmask = getParam("--mask");
	do_centerPDB = checkParam("--centerPDB");
	do_FilterPDBVol = checkParam("--filterVol");
	trustradius_scale = abs(getDoubleParam("--trustradius_scale"));
	if (do_FilterPDBVol)
		cutoff_LPfilter = getDoubleParam("--filterVol");
	useFixedGaussian = checkParam("--fixed_Gaussian");
	if (useFixedGaussian)
		sigmaGaussian = getDoubleParam("--fixed_Gaussian");
	alignVolumes=checkParam("--alignVolumes");
	if (alignVolumes){
		frm_freq = getDoubleParam("--alignVolumes",0);
		frm_shift= getIntParam("--alignVolumes",1);
	}
	tilt0=getIntParam("--tilt_values",0);
	tiltF=getIntParam("--tilt_values",1);

	rhoStartBase = getDoubleParam("--condor_params",0);
	rhoEndBase   = getDoubleParam("--condor_params",1);
	niter        = getIntParam("--condor_params",2);

}
// Show ====================================================================
void ProgNmaAlignmentVol::show() {
	XmippMetadataProgram::show();
	std::cout
            << "Output directory:     " << fnOutDir << std::endl
	        << "PDB:                  " << fnPDB << std::endl
			<< "Resume:               " << resume << std::endl
			<< "Mode list:            " << fnModeList << std::endl
			<< "Pixel size:           " << sampling_rate << std::endl
			<< "Mask:                 " << fnmask << std::endl
			<< "Center PDB:           " << do_centerPDB << std::endl
			<< "Filter PDB volume:    " << do_FilterPDBVol << std::endl
			<< "Use pseudo-atoms:     " << useFixedGaussian << std::endl
			<< "Pseudo-atom sigma:    " << sigmaGaussian << std::endl
			<< "Trust-region scale:   " << trustradius_scale << std::endl;
}

// Produce side information ================================================
ProgNmaAlignmentVol *global_nma_vol_prog;

void ProgNmaAlignmentVol::createWorkFiles() {
	MetaData *pmdIn = getInputMd();
	MetaData mdTodo, mdDone;
	mdTodo = *pmdIn;
	FileName fn(fnOutDir+"/nmaDone.xmd");
	if (fn.exists() && resume) {
		mdDone.read(fn);
		mdTodo.subtraction(mdDone, MDL_IMAGE);
	} else //if not exists create metadata only with headers
	{
		mdDone.addLabel(MDL_IMAGE);
		mdDone.addLabel(MDL_ENABLED);
		mdDone.addLabel(MDL_IMAGE);
		mdDone.addLabel(MDL_ANGLE_ROT);
		mdDone.addLabel(MDL_ANGLE_TILT);
		mdDone.addLabel(MDL_ANGLE_PSI);
		mdDone.addLabel(MDL_SHIFT_X);
		mdDone.addLabel(MDL_SHIFT_Y);
		mdDone.addLabel(MDL_SHIFT_Z);
		mdDone.addLabel(MDL_NMA);
		mdDone.addLabel(MDL_NMA_ENERGY);
		mdDone.addLabel(MDL_MAXCC);
		mdDone.addLabel(MDL_ANGLE_Y);
		mdDone.write(fn);
	}
	*pmdIn = mdTodo;
}

void ProgNmaAlignmentVol::preProcess() {
	MetaData SF(fnModeList);
	SF.removeDisabled();
	numberOfModes = SF.size();
	// Get the size of the images in the selfile
	imgSize = xdimOut;
	// Set the pointer of the program to this object
	global_nma_vol_prog = this;
	//create some neededs files
	createWorkFiles();
	if (fnmask!="")
	{
		Image<double> aux;
		aux.read(fnmask);
		typeCast(aux(),mask);
	}
}

void ProgNmaAlignmentVol::finishProcessing() {
	XmippMetadataProgram::finishProcessing();
	rename((fnOutDir+"/nmaDone.xmd").c_str(), fn_out.c_str());
}

// Create deformed PDB =====================================================
void deformPDB(const FileName &PDBin, const FileName &PDBout, const FileName &fnModeList,
		const Matrix1D<double> &trial)
{
	String program = "xmipp_pdb_nma_deform";
	String arguments = formatString("--pdb %s -o %s --nma %s --deformations ",PDBin.c_str(), PDBout.c_str(), fnModeList.c_str());
	for (size_t i = 0; i < VEC_XSIZE(trial); ++i)
		arguments += floatToString(trial(i)) + " ";
	runSystem(program, arguments, false);
}

FileName ProgNmaAlignmentVol::createDeformedPDB() const {
	String program;
	String arguments;
	FileName fnRandom;
	fnRandom.initUniqueName(nameTemplate,fnOutDir);
	const char * randStr = fnRandom.c_str();

	deformPDB(fnPDB,formatString("%s_deformedPDB.pdb",randStr),fnModeList,trial);
	program = "xmipp_volume_from_pdb";
	arguments = formatString(
			"-i %s_deformedPDB.pdb --size %i --sampling %f -v 0", randStr,
			imgSize, sampling_rate);

	if (do_centerPDB)
		arguments.append(" --centerPDB ");

	if (useFixedGaussian) {
		arguments.append(" --intensityColumn Bfactor --fixed_Gaussian ");
		if (sigmaGaussian >= 0)
			arguments += formatString("%f",sigmaGaussian);
	}
		
	progVolumeFromPDB->read(arguments);
	progVolumeFromPDB->tryRun();

	if (do_FilterPDBVol) {
		program = "xmipp_transform_filter";
		arguments = formatString(
						"-i %s_deformedPDB.vol --sampling %f --fourier low_pass %f  -v 0",
						randStr, sampling_rate, cutoff_LPfilter);
		runSystem(program, arguments, false);
	}

	return fnRandom;
}

bool ProgNmaAlignmentVol::updateBestFit(double fitness, int dim) {
	if (fitness < fitness_min) {
		fitness_min = fitness;
		trial_best = trial;
		return true;
	}
	return false;
}

// Compute fitness =========================================================
double ObjFunc_nma_alignment_vol::eval(Vector X, int *nerror) {
	int dim = global_nma_vol_prog->numberOfModes;

	for (int i = 0; i < dim; i++) {
		global_nma_vol_prog->trial(i) = X[i];
	}

	FileName fnRandom = global_nma_vol_prog->createDeformedPDB();
	const char * randStr = fnRandom.c_str();
	double retval=std::numeric_limits<double>::max();

	FileName fnShiftsAngles = fnRandom + "_angles_shifts.txt";
	const char * shifts_angles = fnShiftsAngles.c_str();

	String fnVolume1 = global_nma_vol_prog->currentVolName;
	String fnVolume2 = fnRandom + "_deformedPDB.vol";

	if (global_nma_vol_prog->tilt0!=-90 || global_nma_vol_prog->tiltF!=90 ){
		global_nma_vol_prog->flip = true;
		runSystem("xmipp_transform_geometry",formatString("-i %s -o %s_currentvolume.vol --rotate_volume euler 0 90 0 -v 0",fnVolume1.c_str(),randStr));
		fnVolume1 = fnRandom+"_currentvolume.vol";
	}

	const char * Volume1 = fnVolume1.c_str();
	const char * Volume2 = fnVolume2.c_str();
	
	int err;

	if (global_nma_vol_prog->alignVolumes){
		runSystem("xmipp_volume_align",formatString("--i1 %s --i2 %s --frm %f %d %d %d --store %s -v 0 ",
				Volume1,Volume2,global_nma_vol_prog->frm_freq, global_nma_vol_prog->frm_shift, global_nma_vol_prog->tilt0, global_nma_vol_prog->tiltF, shifts_angles));
		//first just see what is the score
		global_nma_vol_prog->AnglesShiftsAndScore = fopen(shifts_angles, "r");
		//fit_value is the 7th element in a single line CSV
		for (int i = 0; i < 7; i++){
			err = fscanf(global_nma_vol_prog->AnglesShiftsAndScore, "%f,", &global_nma_vol_prog->fit_value);
			if (1!=err)
				REPORT_ERROR(ERR_IO, "reading the fitness value was not successful");
		}
		fclose(global_nma_vol_prog->AnglesShiftsAndScore);
		retval = 1 + global_nma_vol_prog->fit_value;
	}

	else{
		global_nma_vol_prog->Vdeformed.read(formatString("%s_deformedPDB.vol",fnRandom.c_str()));
		auto mask = (0 == XSIZE(global_nma_vol_prog->mask)) ? nullptr : &global_nma_vol_prog->mask;
		retval = 1 - correlationIndex(global_nma_vol_prog->V(), global_nma_vol_prog->Vdeformed(), mask);
		//global_nma_vol_prog->V().printStats();
		//global_nma_vol_prog->Vdeformed().printStats();
		//std::cout << correlationIndex(global_nma_vol_prog->V(),global_nma_vol_prog->Vdeformed()) << std::endl;
	}

	if(global_nma_vol_prog->updateBestFit(retval, dim) && global_nma_vol_prog->alignVolumes){
		global_nma_vol_prog->AnglesShiftsAndScore = fopen(shifts_angles, "r");
		for (int i = 0; i < 6; i++){
			err = fscanf(global_nma_vol_prog->AnglesShiftsAndScore, "%f,", &global_nma_vol_prog->Best_Angles_Shifts[i]);
			if (1!=err)
				REPORT_ERROR(ERR_IO, "reading the angles and shifts was not successful");
		    }
		fclose(global_nma_vol_prog->AnglesShiftsAndScore);
	}

	runSystem("rm", formatString("-rf %s* &", randStr));
	//std::cout << global_nma_vol_prog->trial << " -> " << retval << std::endl;
	return retval;
}

ObjFunc_nma_alignment_vol::ObjFunc_nma_alignment_vol(int _t, int _n) {
}

void ProgNmaAlignmentVol::processImage(const FileName &fnImg,
		const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) {
	static size_t imageCounter = 0;
	++imageCounter;

	ObjectiveFunction *of;

	int dim = numberOfModes;

	parameters.initZeros(dim);
	V.read(fnImg);
	currentVolName = fnImg;
	sprintf(nameTemplate, "_node%d_img%lu_XXXXXX", rangen, (long unsigned int)imageCounter);

	trial.initZeros(dim);
	trial_best.initZeros(dim);

	fitness_min=1000000.0;

	of = new ObjFunc_nma_alignment_vol(1, dim);

	of->xStart.setSize(dim);
	for (int i = 0; i < dim; i++)
		of->xStart[i] = 0.;

	double rhoStart=trustradius_scale*rhoStartBase;
    double rhoEnd=trustradius_scale*rhoEndBase;

	CONDOR(rhoStart, rhoEnd, niter, of);

	trial = parameters = trial_best;

	parameters.resize(VEC_XSIZE(parameters) + 1);
	parameters(VEC_XSIZE(parameters) - 1) = fitness_min;

	writeVolumeParameters(fnImg);
	if (fnOutPDB!="")
		deformPDB(fnPDB,fnOutPDB,fnModeList,trial_best);
	delete of;
}

void ProgNmaAlignmentVol::writeVolumeParameters(const FileName &fnImg) {
	MetaData md;
	size_t objId = md.addObject();
	md.setValue(MDL_IMAGE, fnImg, objId);
	md.setValue(MDL_ENABLED, 1, objId);

	std::vector<double> vectortemp;
	double energy=0;
	for (int j = 0; j < numberOfModes; j++)
	{
		double lambdaj=parameters(j);
		vectortemp.push_back(lambdaj);
		energy+=lambdaj*lambdaj;
	}


	energy/=numberOfModes;
	md.setValue(MDL_ANGLE_ROT, (double)global_nma_vol_prog->Best_Angles_Shifts[0], objId);
	md.setValue(MDL_ANGLE_TILT, (double)global_nma_vol_prog->Best_Angles_Shifts[1], objId);
	md.setValue(MDL_ANGLE_PSI, (double)global_nma_vol_prog->Best_Angles_Shifts[2], objId);
	md.setValue(MDL_SHIFT_X, (double)global_nma_vol_prog->Best_Angles_Shifts[3], objId);
	md.setValue(MDL_SHIFT_Y, (double)global_nma_vol_prog->Best_Angles_Shifts[4], objId);
	md.setValue(MDL_SHIFT_Z, (double)global_nma_vol_prog->Best_Angles_Shifts[5], objId);
	md.setValue(MDL_NMA, vectortemp, objId);
	md.setValue(MDL_NMA_ENERGY, energy, objId);
	md.setValue(MDL_MAXCC, 1-parameters(numberOfModes), objId);
	if (global_nma_vol_prog->flip){
		md.setValue(MDL_ANGLE_Y, 90.0 , objId);
	}
	else{
		md.setValue(MDL_ANGLE_Y, 0.0 , objId);
	}

	md.append(fnOutDir+"/nmaDone.xmd");
}
