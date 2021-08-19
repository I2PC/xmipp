#!/usr/bin/env python3
# ***************************************************************************
# * Authors:     David Strelak (dstrelak@cnb.csic.es)
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


class Config:
    FILE_NAME = "xmipp.conf"

    def __init__(self):
        self.configDict = {}

    def set(self, d):
        self.configDict = d

    def get(self):
        return self.configDict

    def is_true(self, key):
        return self.configDict and (key in self.configDict) and (self.configDict[key].lower() == 'true')
