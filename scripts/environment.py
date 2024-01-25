#!/usr/bin/env python3
# ***************************************************************************
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              David Maluenda (dmaluenda@cnb.csic.es)
# *              David Strelak (dstrelak@cnb.csic.es)
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

from os import path, pathsep
import json


class Environment:
    def __init__(self):
        self.env = {}

    def update(self, pos='begin', realPath=True, **kwargs):
        """ Add/update a variable in self.env dictionary
            pos = {'begin', 'end', 'replace'}
        """
        for key, value in kwargs.items():
            isString = isinstance(value, str)
            if isString and realPath:
                value = path.realpath(value)
            if key in self.env:
                if pos == 'begin' and isString:
                    self.env[key] = value + pathsep + self.env[key]
                elif pos == 'end' and isString:
                    self.env[key] = self.env[key] + pathsep + value
                elif pos == 'replace':
                    self.env[key] = str(value)
            else:
                self.env[key] = str(value)

    def write(self):
        with open('xmippEnv.json', 'w') as f:
            json.dump(self.env, f, indent=4)
