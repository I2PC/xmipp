# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *							Martín Salinas (martin.salinas@cnb.csic.es)
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
import os
from .utils import runJob, yellow
from .constants import LOG_FILE, CONFIG_FILE, VERSIONS_FILE, COMPRESED_FILE
def createTar():
		try:
				os.remove(COMPRESED_FILE)
				retCode, outputStr = runJob('./xmipp version'.format(VERSIONS_FILE))
				with open(VERSIONS_FILE, 'w') as f:
						f.write(outputStr)
				if retCode == 0:
					retCode, outputStr =runJob('tar -czf {} {} {} {} '.format(
						COMPRESED_FILE, CONFIG_FILE,VERSIONS_FILE,LOG_FILE))
					if retCode != 0:
						print(yellow('reportTar file not created'))
				else:
						print(yellow('reportTar file not created'))
		except Exception as e:
				print(yellow('reportTar file not created'))