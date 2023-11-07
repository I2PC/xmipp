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

# General imports
import subprocess, pkg_resources, sys
from os import environ
from typing import Union


# Installer imports
from .constants import SCONS_MINIMUM

def showError(errorMsg: str, retCode: int=1) -> Union[None, int]:
	""" This function prints an error message and exits with the given return code. """
	# Print the error message in red color
	print(f"\033[91m{errorMsg}\033[0m")
	sys.exit(retCode)

def versionToNumber(strVersion):
    listVersion = strVersion.split('.')
    numberVersion = int(listVersion[0]) * 100 + int(listVersion[1]) * 10
    try:
        numberVersion = numberVersion + int(listVersion[2]) * 1
    except Exception:
        pass
    return numberVersion

def runJob(cmd, cwd='./', showOutput=True, logOut=None, logErr=None, showError=True, showCommand=True):
    p = subprocess.Popen(cmd, cwd=cwd, env=environ, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()

    if showCommand == True:
        print(blue(cmd))

    if showOutput == True:
        print('{}\n'.format(output.decode("utf-8")))
    if logOut != None:
        logOut.append(output.decode("utf-8"))

    if err:
        if showError == True:
            print(red(err.decode("utf-8")))
        if logErr != None:
            logErr.append(err.decode("utf-8"))
        if err.decode("utf-8") != '':
            return False
    else:
        return True


def green(text):
    return "\033[92m "+text+"\033[0m"

def yellow(text):
    return "\033[93m " + text + "\033[0m"

def red(text):
    return "\033[91m "+text+"\033[0m"

def blue(text):
    return "\033[34m "+text+"\033[0m"

def bold(text):
    return "\033[1m "+text+"\033[0m"



def sconsVersion():
    try:
        textVersion = pkg_resources.get_distribution("scons").version
        if versionToNumber(textVersion) >= versionToNumber(SCONS_MINIMUM):
            return True
    except Exception:
        pass

    if isScipionVersion():
        outlog = []
        errlog = []
        if runJob('pip install scons', logOut=outlog, logErr=errlog, showError=False, showCommand=True):
            return True
        else:
            print(red(errlog[0]))
            return 1, False
    else:
        print(blue('Scipion enviroment not found, please install manually scons library'))
        return 2, False


def isScipionVersion():
    condaEnv = []
    if runJob('echo $CONDA_PREFIX', logOut=condaEnv, showError=True):
        if condaEnv[0].find('scipion3') != -1:
            return True
        else:
            return False
    else:
        return False

