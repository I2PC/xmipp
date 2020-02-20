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

#include <iostream>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>

#include <core/metadata_extension.h>
#include <data/filters.h>
#include "program_extension.h"
#include "volumeset_align.h"

// Empty constructor =======================================================
ProgVolumeSetAlign::ProgVolumeSetAlign() {
	rangen = 0;
	resume = false;
	currentVolName = "";
	each_image_produces_an_output = false;
	produces_an_output = true;
	flipped = false;
}

// Params definition ============================================================
void ProgVolumeSetAlign::defineParams() {
	addUsageLine("Align a set of volumes with a reference volume");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with volume filenames");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Metadata with output Euler angles and shifts");
	XmippMetadataProgram::defineParams();
	addParamsLine("   --ref <VOL_filename>                : Reference volume");
	addParamsLine("  [--odir <outputDir=\".\">]           : Output directory");
	addParamsLine("  [--resume]                           : Resume processing");
	addParamsLine("  [--frm_parameters <frm_freq=0.25> <frm_shift=10>]  : This is using frm method for volume alignment with frm_freq and frm_shift as parameters");
	addParamsLine("  [--tilt_values <tilt0=-90> <tiltF=90>]  : Optional compensation for the missing wedge. Tested extensively with tilt between [-60 60]");
	addExampleLine("xmipp_many_volume_align -i volumes.xmd --ref reference.vol -o output.xmd --resume");
}

// Read arguments ==========================================================
void ProgVolumeSetAlign::readParams() {
	XmippMetadataProgram::readParams();
	fnREF = getParam("--ref");
	fnOutDir = getParam("--odir");
	resume = checkParam("--resume");
	frm_freq = getDoubleParam("--frm_parameters",0);
	frm_shift= getIntParam("--frm_parameters",1);
	tilt0=getIntParam("--tilt_values",0);
	tiltF=getIntParam("--tilt_values",1);
}

// Produce side information ================================================
ProgVolumeSetAlign *global_volumeset_align_prog;

void ProgVolumeSetAlign::createWorkFiles() {
	MetaData *pmdIn = getInputMd();
	// this will serve to resume
	MetaData mdTodo, mdDone; 
	mdTodo = *pmdIn;
	FileName fn(fnOutDir+"/AlignedSoFar.xmd");
	if (fn.exists() && resume) {
		mdDone.read(fn);
		mdTodo.subtraction(mdDone, MDL_IMAGE);
	} 
	else //if not exists create metadata only with headers
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
		mdDone.addLabel(MDL_MAXCC);
		mdDone.addLabel(MDL_ANGLE_Y);
		mdDone.write(fn);
	}
	*pmdIn = mdTodo;
}

void ProgVolumeSetAlign::preProcess() {
	// Set the pointer of the program to this object
	global_volumeset_align_prog = this;
	createWorkFiles();
}

void ProgVolumeSetAlign::finishProcessing() {
	XmippMetadataProgram::finishProcessing();
	rename((fnOutDir+"/AlignedSoFar.xmd").c_str(), fn_out.c_str());
}

// Compute fitness =========================================================
void ProgVolumeSetAlign::computeFitness(){
	FileName fnRandom;
	fnRandom.initUniqueName(nameTemplate,fnOutDir);
	const char * randStr = fnRandom.c_str();
	
	FileName fnShiftsAngles = fnRandom + "_angles_shifts.txt";
	const char * shifts_angles = fnShiftsAngles.c_str();
	
	String fnVolume1 = global_volumeset_align_prog->currentVolName;
	String fnVolume2 = global_volumeset_align_prog->fnREF;

	if (global_volumeset_align_prog->tilt0!=-90 || global_volumeset_align_prog->tiltF!=90 ){
		global_volumeset_align_prog->flipped = true;
		runSystem("xmipp_transform_geometry",formatString("-i %s -o %s_currentvolume.vol --rotate_volume euler 0 90 0 -v 0",fnVolume1.c_str(),randStr));
		fnVolume1 = fnRandom+"_currentvolume.vol";
	}

	const char * Volume1 = fnVolume1.c_str();
	const char * Volume2 = fnVolume2.c_str();

	runSystem("xmipp_volume_align",formatString("--i1 %s --i2 %s --frm %f %d %d %d --store %s -v 0 ",
			Volume1,Volume2,global_volumeset_align_prog->frm_freq, global_volumeset_align_prog->frm_shift, global_volumeset_align_prog->tilt0, global_volumeset_align_prog->tiltF, shifts_angles));
	//The first 6 parameters are angles and shifts, and the 7th is the fitness value
	fnAnglesAndShifts = fopen(shifts_angles, "r");
	for (int i = 0; i < 6; i++){
	       fscanf(fnAnglesAndShifts, "%f,", &global_volumeset_align_prog->Matrix_Angles_Shifts[i]);
	    }
	fscanf(fnAnglesAndShifts, "%f,", &global_volumeset_align_prog->fitness);
	fclose(fnAnglesAndShifts);

	runSystem("rm", formatString("-rf %s* &", randStr));
}


void ProgVolumeSetAlign::processImage(const FileName &fnImg,
		const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) {
	static size_t imageCounter = 0;
	++imageCounter;
	currentVolName = fnImg;
	sprintf(nameTemplate, "_node%d_img%lu_XXXXXX", rangen, (long unsigned int)imageCounter);
	computeFitness();
	writeVolumeParameters(fnImg);
}

void ProgVolumeSetAlign::writeVolumeParameters(const FileName &fnImg) {
	MetaData md;
	size_t objId = md.addObject();
	md.setValue(MDL_IMAGE, fnImg, objId);
	md.setValue(MDL_ENABLED, 1, objId);
	md.setValue(MDL_ANGLE_ROT,  (double)global_volumeset_align_prog->Matrix_Angles_Shifts[0], objId);
	md.setValue(MDL_ANGLE_TILT, (double)global_volumeset_align_prog->Matrix_Angles_Shifts[1], objId);
	md.setValue(MDL_ANGLE_PSI,  (double)global_volumeset_align_prog->Matrix_Angles_Shifts[2], objId);
	md.setValue(MDL_SHIFT_X,    (double)global_volumeset_align_prog->Matrix_Angles_Shifts[3], objId);
	md.setValue(MDL_SHIFT_Y,    (double)global_volumeset_align_prog->Matrix_Angles_Shifts[4], objId);
	md.setValue(MDL_SHIFT_Z,    (double)global_volumeset_align_prog->Matrix_Angles_Shifts[5], objId);
	md.setValue(MDL_MAXCC,    1-(double)global_volumeset_align_prog->fitness, objId);
	if (global_volumeset_align_prog->flipped){
		md.setValue(MDL_ANGLE_Y, 90.0 , objId);
	}
	else{
		md.setValue(MDL_ANGLE_Y, 0.0 , objId);
	}
	md.append(fnOutDir+"/AlignedSoFar.xmd");
}

