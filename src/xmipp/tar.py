#!/usr/bin/env python3
# ***************************************************************************
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# ***************************************************************************/
import subprocess

import sys
import os
import shutil
from os.path import join, isfile, exists


def usage(error=''):
    errorStr = 'error\n' if error else ''
    print("\n"
          "    %s"
          "\n"
          "    Usage: python tar.py <mode> [v=version] [br=branch]\n"
          "\n"
          "             mode: BinXXXX: Just the binaries. (XXXX is a OS label like 'Debian', 'Centos'...) \n"
          "                   Sources: Just the source code.\n"
          "\n"
          "             branch if for the sources (default: master)\n"
          "\n"
          "             version: X.YY.MM  (version, year and month)\n"
          "    " % errorStr)
    sys.exit(1)


def run(label, version, branch):
    MODES = {'Binaries': 'build', 'Sources': 'src'}

    def makeTarget(target, label):
        if exists(target):
            print("'%s' already exists. Removing it..." % target)
            os.system("rm -rf %s" % target)
        print("...preparing the bundle...")
        cwd = os.getcwd()
        os.mkdir(target)
        shutil.copy('xmipp', target)
        os.chdir(target)
        os.environ['CUDA'] = 'True'
        os.system('./xmipp config')  # just to write the config file
        os.system('./xmipp get_dependencies')
        os.system('./xmipp get_devel_sources %s' % branch)
        os.system('rm -rf tmp xmipp xmipp.conf')
        os.chdir(cwd)

    excludeTgz = ''
    if label.startswith('Bin'):
        print("Recompiling to make sure that last version is there...")
        sublabel = label.split('Bin')[1]
        target = 'xmippBin_%s-%s' % (sublabel, version)
        try:
            # doing compilation and install separately to skip overwriting config
            os.system("./xmipp compile 4")
            os.system("./xmipp install %s" % target)
        except:
            raise Exception("  ...some error occurred during the compilation!!!\n")
        checkFile = isfile(join(target, 'bin', 'xmipp_cuda_movie_alignment_correlation'))
        if not checkFile:
            print("\n"
                  "     ERROR: %s not found. \n"
                  "            Xmipp should be compiled using CUDA to make the binaries.tgz."
                  % checkFile)
            sys.exit(1)
        os.system("cp xmipp.conf %s/xmipp.conf" % target)
        os.system("rm %s/v%s" % (target, version))
        os.system("touch %s/v%s_%s" % (target, version, sublabel))
        excludeTgz = "--exclude='*.tgz' --exclude='*.h' --exclude='*.cpp' " \
                     "--exclude='*.java' --exclude='resources/test' " \
                     "--exclude='*xmipp_test*main'"
    elif label == 'Sources':
        target = 'xmippSrc-v'+version
        makeTarget(target, label)
        excludeTgz = " --exclude='models/*' --exclude='src/*/bin/*' --exclude='*.so'"
    else:
        usage("Incorrect <mode>")

    # FIXME: This is breaking the Sources bundle. Please, use a clean dir and skip this
    # excludeTgz += " --exclude='*.o' --exclude='*.os' --exclude='*pyc'"
    # excludeTgz += " --exclude='*.gz' --exclude='*.bashrc' --exclude='*.fish'"
    # excludeTgz += " --exclude=tests/data --exclude='*.scons*' --exclude=.git"
    excludeTgz = ("--exclude=.git --exclude=.idea "
                  "--exclude='xmipp.bashrc' --exclude='xmipp.fish'")

    cmdStr = "tar czf %(target)s.tgz %(excludeTgz)s %(target)s"

    args = {'excludeTgz': excludeTgz, 'target': target}

    cmd = cmdStr % args

    if exists(target+'.tgz'):
        print("%s.tgz already exists. Removing it..." % target)
        os.system("rm -rf %s.tgz" % target)

    print(cmd)
    os.system(cmd)
    os.system("rm -rf %s" % target)


if __name__ == '__main__':

    if not len(sys.argv) > 3:
        usage("Incorrect number of input parameters")

    label = sys.argv[1]
    version = sys.argv[2]
    branch = sys.argv[3] if len(sys.argv)>3 else 'master'

    run(label, version, branch)
