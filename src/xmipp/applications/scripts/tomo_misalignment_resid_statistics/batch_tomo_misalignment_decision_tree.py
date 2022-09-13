"""
/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
"""
 
from re import S
from sklearn import tree


class ScriptTomoDecisionTree():

  def __init__(self):
    self.dtc = tree.DecisionTreeClassifier()

    self.testSplit = 0.2

    self.infoData_train = []
    self.classData_train = []
    self.infoData_test = []
    self.classData_test = []

    self.readInputData()

    self.trainDecisionTree()

    self.testDecisionTree()


  def readInputData(self):
    """
      Read input data
    """

  def trainDecisionTree(self):
    """
      Train decission tree
    """

    self.dtc.fit(self.infoData_train, self.classData_train)

    tree.plot_tree(self.dtc)

  def testDecisionTree(self):
    """
      Test decission tree
    """

    for d in self.infoData_test:
      self.dtc.predict(d)
    

if __name__ == '__main__':
  cdt = ScriptTomoDecisionTree()

