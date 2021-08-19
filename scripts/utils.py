#!/usr/bin/env python3
# ***************************************************************************
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              David Maluenda (dmaluenda@cnb.csic.es)
# *              David Strelak (dstrelak@cnb.csic.es)
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
from os import environ


def green(text):
    return "\033[92m "+text+"\033[0m"


def yellow(text):
    return "\033[93m " + text + "\033[0m"


def red(text):
    return "\033[91m "+text+"\033[0m"


def blue(text):
    return "\033[34m "+text+"\033[0m"


def runJob(cmd, cwd='./', show_output=True, log=None, show_command=True,
           inParallel=False):
    if show_command:
        print(green(cmd))
    p = subprocess.Popen(cmd, cwd=cwd, env=environ,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    while not inParallel:
        output = p.stdout.readline().decode("utf-8")
        if output == '' and p.poll() is not None:
            break
        if output:
            l = output.rstrip()
            if show_output:
                print(l)
            if log is not None:
                log.append(l)
    if inParallel:
        return p
    else:
        return 0 == p.poll()
