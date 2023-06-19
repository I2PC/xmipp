# ***************************************************************************
# *
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
# ****************************************************************************
""" This is an example of a module which can be imported by any Xmipp script.
    It will be installed at build/pylib/xmippPyModules.
"""

print("Inside example_inmodule2.py")

def anyFunction2():
    return "returningFromFunction (II)"

class anyClass2():

    A_CONSTANT = "A class constant. (II)"

    def __init__(self):
        print("Inside the anyClass.__init__() (II)")
        self.inVar = "An object var. (II)"

    @classmethod
    def getFromClassMethod2(cls):
        return("Getting '%s' (II)" % cls.A_CONSTANT)

    def getFromObjectMethod2(self):
        return("Getting '%s' (II)" % self.inVar)
