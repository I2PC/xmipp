# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307 USA
# *
# * All comments concerning this program package may be sent to the
# * e-mail address 'scipion@cnb.csic.es'
# ***************************************************************************/

import os, sys
from .utils import runJob
from .constants import SCIPION_SOFTWARE_EM, MODELS_URL, DEFAULT_MODELS_DIR
from .logger import blue, red, yellow, logger


def addModels(login: str, modelPath: str, update: bool):
	modelPath = modelPath.rstrip("/")
	modelName = os.path.basename(modelPath)
	modelsDir = os.path.dirname(modelPath)
	update = '--update' if update else ''
	tgzFileModel = f"xmipp_model_{modelName}.tgz"
	localFileModel = os.path.join(modelsDir, tgzFileModel)
	logger(f"Creating the {tgzFileModel} model.",forceConsoleOutput=True)
	runJob(f"tar czf {tgzFileModel} {modelName}", cwd=modelsDir)
	
	logger(yellow("Warning: Uploading, please BE CAREFUL! This can be dangerous."),forceConsoleOutput=True)
	logger(f'You are going to be connected to {login} to write in folder {SCIPION_SOFTWARE_EM}.',forceConsoleOutput=True)
	if input("Continue? YES/no\n") != 'YES':
		sys.exit()
	
	logger(f"Trying to upload the model using {login} as login",forceConsoleOutput=True)
	args = "%s %s %s %s" % (login, os.path.abspath(localFileModel), SCIPION_SOFTWARE_EM, update)
	retCode, log = runJob(f"dist/bin/xmipp_sync_data upload {args}", showCommand=False, showError=True)
	if retCode == 0:
		logger(f"{modelName} model successfully uploaded! Removing the local .tgz")
		runJob("rm %s" % localFileModel)
		
		

def downloadDeepLearningModels(dest:str):
    if not os.path.exists('dist/bin/xmipp_sync_data'):
        logger(red('Xmipp has not been installed. Please, first install it '))
        sys.exit()
    if dest == DEFAULT_MODELS_DIR:
        modelsPath = os.path.join(dest, 'models')
    else:
        modelsPath = dest
    dataSet = "DLmodels"

    # downloading/updating the DLmodels
    if os.path.isdir(modelsPath):
        logger("Updating the Deep Learning models (in backgound)")
        task = "update"
    else:
        logger("Downloading Deep Learning models (in backgound)")
        task = "download"
    global pDLdownload

    retCode, log = runJob(f"dist/bin/xmipp_sync_data {task} {modelsPath} {MODELS_URL} {dataSet}",
           showCommand=True, showError=True, showOutput=True)
    if retCode == 0:
        logger(f"Models successfully {task}")

