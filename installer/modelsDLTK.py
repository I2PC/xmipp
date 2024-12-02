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
from .constants import SCIPION_NOLAN, MODELS_URL
from .logger import blue, red, yellow, logger


def addModels(login: str, modelPath: str, update: bool):
	modelPath = modelPath.rstrip("/")
	modelName = os.path.basename(modelPath)
	modelsDir = os.path.dirname(modelPath)
	if update: update='--update'
	else: update = ''
	tgzFn = "xmipp_model_%s.tgz" % modelName
	localFn = os.path.join(modelsDir, tgzFn)
	
	logger("Creating the '%s' model." % tgzFn)
	runJob("tar czf %s %s" % (tgzFn, modelName), cwd=modelsDir)
	
	logger(yellow("Warning: Uploading, please BE CAREFUL! This can be dangerous."))
	logger('You are going to be connected to "%s" to write in folder '
	      '"%s".' % (login, SCIPION_NOLAN))
	if input("Continue? YES/no\n").lower() == 'no':
		sys.exit()
	
	logger("Trying to upload the model using '%s' as login" % login)
	args = "%s %s %s %s" % (login, os.path.abspath(localFn), SCIPION_NOLAN, update)
	log = ''
	retCode, log = runJob("dist/bin/xmipp_sync_data upload %s" % args, showCommand=True, showError=True)
	if retCode:
		logger("'%s' model successfully uploaded! Removing the local .tgz" % modelName)
		runJob("rm %s" % localFn)
		
		
		