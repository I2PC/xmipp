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

"""
Module containing all constants needed for the installation of Xmipp.
"""

from .main import *
from .parser import *
from .errors import *
from .versions import *

#URL's repositoreis
cmakeInstallURL = 'https://i2pc.github.io/docs/Installation/InstallationNotes/index.html#cmake'
urlModels = "http://scipion.cnb.csic.es/downloads/scipion/software/em"
remotePath = "scipionfiles/downloads/scipion/software/em"
urlTest = "http://scipion.cnb.csic.es/downloads/scipion/data/tests"
HOST_TEST = 'google.com'
HOST_TEST_2 = 'https://github.com'
warningToHidden = 'using serial compilation of 2 LTRANS jobs'
