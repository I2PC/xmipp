/***************************************************************************
 *
 * Authors:     Alberto Pascual Montano (pascual@cnb.csic.es)
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

//-----------------------------------------------------------------------------
// GaussianKerDenSOM.hh
// Implements Smoothly Distributed Kernel Probability Density Estimator Self-Organizing Map
// Uses a Gaussian Kernel Function.
//-----------------------------------------------------------------------------

#ifndef XMIPPTGAUSSIANKERDENSOM_H
#define XMIPPTGAUSSIANKERDENSOM_H

#include "kerdensom.h"

/**@defgroup SmoothlyGaussianStudent Smoothly Distributed Gaussian Kernel Probability Density Estimator Self Organizing Map
   @ingroup ClassificationLibrary */
//@{
/**
 * This class trains a Smoothly Distributed Kernel Probability Density Estimator Self Organizing Map
 * using a Gaussian Kernel function
 */
class GaussianKerDenSOM : public KerDenSOM
{
public:

    /**
     * Constructs the algorithm
     * Parameter: _reg0       Initial regularization factor
     * Parameter: _reg1       Final regularization factor
     * Parameter: _annSteps   Number of steps in deterministic annealing
     * Parameter: _epsilon    Stopping criterion
     * Parameter: _nSteps     Number of training steps
     */
    GaussianKerDenSOM(double _reg0, double _reg1, unsigned long _annSteps,
                           double _epsilon, unsigned long _nSteps)
            : KerDenSOM(_reg0, _reg1, _annSteps, _epsilon, _nSteps)
    {};

    /**
     * Virtual destructor
     */
    virtual ~GaussianKerDenSOM()
    {};


    /**
     * Trains the GaussianKerDenSOM
     * Parameter: _som  The KerDenSom to train
     * Parameter: _ts   The training set
     * Parameter: _update True if uses _som as starting point for training.
     * Parameter: _sigma If update = true, uses this sigma for the training.
     */
    virtual void train(FuzzyMap& _som, TS& _examples, FileName& _fn_vectors,
    		           bool _update = false, double _sigma = 0,
    		           bool _saveIntermediate = false);


    /**
     * Determines the functional value.
     * Returns the likelihood and penalty parts of the functional
     */
    virtual double functional(const TS* _examples, const FuzzyMap* _som, double _sigma, double _reg, double& _likelihood, double& _penalty);


protected:

    // Update Us
    virtual double updateU(FuzzyMap* _som, const TS* _examples, const double& _sigma, double& _alpha);

    // Estimate Sigma II
    virtual double updateSigmaII(FuzzyMap* _som, const TS* _examples, const double& _reg, const double& _alpha);

    // Estimate the PD (Method 1: Using the code vectors)
    virtual double codeDens(const FuzzyMap* _som, const FeatureVector* _example, double _sigma) const;
#ifdef UNUSED // detected as unused 29.6.2018
    // Estimate the PD (Method 2: Using the data)
    virtual double dataDens(const TS* _examples, const FeatureVector* _example, double _sigma) const;
#endif
};

//@}
#endif
