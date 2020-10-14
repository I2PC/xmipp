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
// Sammon.h
// Sammon Projection Algorithm
//-----------------------------------------------------------------------------

#ifndef XMIPPSAMMON_H
#define XMIPPSAMMON_H

#include <stdexcept>
#include <vector>
#include <core/xmipp_funcs.h>
#include "training_vector.h"

// definition of max
#ifndef max
#define max(a,b) ((a)>=(b))?(a):(b)
#endif

//-----------------------------------------------------------------------------
// Sammon: Sammon Maps
//-----------------------------------------------------------------------------

/**@defgroup Sammon Sammon Projection
   @ingroup ClassificationLibrary */
//@{
/**
 * This class implements the Sammon non-linear projection algorithm
 */
class Sammon
{
public:
    /// An xmipp training vector is taken as input
    typedef ClassicTrainingVectors In;
    /// An xmipp training vector is given as output
    typedef ClassicTrainingVectors Out;


    /**
     * Constructs the algorithm
     * Parameter: _mapped     the dimension of the output space
     * Parameter: _num_iterations Number of iterations to be used
     * Parameter: _learning_rate "Magic number"
     */
    Sammon(const unsigned _mapped = 2,
                const unsigned _num_iterations = 100,
                const double _learning_rate = 0.2);


    /**
     * () operator that trains the algorithm.
     * Parameter: in Input features vector (in N-dimmensional space)
     * Parameter: out Output  vector (in _mapped-dimmensional space)
     */
    void operator()(const In& in, Out& out);


    /**
     * Gets the sammon stress (mapping error)
     */
    double getStress() const
    {
        return stress;
    };

    /**
     * Sets the listener class
     */
    void setListener(BaseListener* _listener)
    {
        listener = _listener;
    };

private:
    unsigned mapped;           // mapped space dimension
    unsigned num_iterations;   // number of iterations
    double learning_rate;      // learning rate
    double stress;  // Sammon stress (error in distances)
    BaseListener* listener;  // Listener class
    unsigned verbosity;      // defines verbosity level
    // 0: No verbosity
    // 1: Only shows progress bar
    // 2: Only shows stress
    // 3: Shows progress and stress
};

//@}
#endif//XMIPPSAMMON_H
