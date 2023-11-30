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

import numpy as np

class Redundancy():
    def __init__(cls):
        cls.A = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
                          [2, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 2],
                          [1, -1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0], [0, 0, 1, -1, 0, 0],
                          [0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 1, -1], [-1, 0, 0, 0, 0, 1],
                          [2, 0, 0, -1, 0, 0], [0, 2, 0, 0, -1, 0], [0, 0, 2, 0, 0, -1],
                          [-1, 0, 0, 2, 0, 0], [0, -1, 0, 0, 2, 0], [0, 0, -1, 0, 0, 2],
                          [1, 0, 1, 0, -1, 0], [0, 1, 0, 1, 0, -1], [-1, 0, 1, 0, 1, 0],
                          [0, -1, 0, 1, 0, 1], [1, 0, -1, 0, 1, 0], [0, 1, 0, -1, 0, 1],
                          [1, 0, 0, 0, 0, -1], [-1, 1, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0],
                          [0, 0, -1, 1, 0, 0], [0, 0, 0, -1, 1, 0], [0, 0, 0, 0, -1, 1],
                          [1, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, -1], [-1, 0, 1, 0, 0, 0],
                          [0, -1, 0, 1, 0, 0], [0, 0, -1, 0, 1, 0], [0, 0, 0, -1, 0, 1]])

        cls.Apinv = np.linalg.pinv(cls.A)

    def make_redundant(cls, x):
        return np.matmul(cls.A, x)

    def make_nonredundant(cls, x):
        return np.matmul(cls.Apinv, np.reshape(x,(cls.A.shape[0])))