#!/usr/bin/env python

""" FIXME: Only tested for Sources mode. Please, test for the other 2 modes! """

import sys
import os
import shutil
from os.path import dirname, realpath, join

XMIPP_PATH = realpath(dirname(dirname(dirname(realpath(__file__)))))
MODES = {'Binaries': 'build', 'Sources': 'src'}

def usage(error):
    print ("\n"
           "    ERROR: %s\n"
           "\n"
           "    Usage: python tar.py <mode> <version>\n"
           "\n"
           "             mode: Binaries: Just the binaries \n"
           "                   Sources: Just the source code.\n"
           "\n"
           "             version: YY.MM  (year and month)\n"
           "    ") % error
    sys.exit(1)


if len(sys.argv) != 3:
    usage("Incorrect number of input parameters")

label = sys.argv[1]
version = sys.argv[2]


def makeTarget(target, label):
    os.mkdir(target)
    shutil.copytree(MODES[label], join(target, target), symlinks=True)


excludeTgz = ''
tgzPath = "xmipp%s-%s"
if label == 'Binaries':
    target = tgzPath % ('', version)
    if not os.path.isfile(join(XMIPP_PATH, 'build', 'bin', 'xmipp_reconstruct_significant')):
        print("\n"
              "     ERROR: %s not found. \n"
              "            Xmipp needs to be compiled to make the binaries.tgz."
              % target)
        sys.exit(1)
    excludeTgz = "--exclude='*.tgz' --exclude='*.h' --exclude='*.cpp' --exclude='*.java'"
    makeTarget(target, label)
elif label == 'Sources':
    # We use Sources as Default, so no label is added
    target = tgzPath % ('Src', version)
    makeTarget(target, label)
else:
    usage("Incorrect <mode>")


args = {'excludeTgz': excludeTgz,
        'target': target}

cmdStr = "tar czf %(target)s.tgz --exclude=.git --exclude='software/tmp/*' " \
         "--exclude='*.o' --exclude='*.os' --exclude='*pyc' " \
         "--exclude='*.mrc' --exclude='*.stk' --exclude='*.gz' %(excludeTgz)s" \
         "--exclude='*.scons*' --exclude='config/*.conf' %(target)s"

cmd = cmdStr % args

print(cmd)
os.system(cmd)



