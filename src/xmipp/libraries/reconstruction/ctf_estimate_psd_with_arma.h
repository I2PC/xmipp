/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *              Javier �ngel Vel�zquez Muriel (javi@cnb.csic.es)
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

#ifndef _CORE_PROG_SPARMA_HH
#define _CORE_PROG_SPARMA_HH
#define AR          0   // To distinguish between an AR parameter or a MA
#define MA        1   // parameter, or sigma in the output ARMAParameters matrix
#define SIGMA       2   // returned by functions.

#include <core/multidim_array.h>
#include <core/xmipp_program.h>

/**@defgroup SpARMA Spectrum modelling by ARMA filters
   @ingroup ReconsLibrary */
//@{
/** Class to perform the ARMA estimate of the PSD from the command line */
class ARMA_parameters
{
public:
    int       N_AR;               // order in the Rows direction of the AR part of the model
    int       M_AR;               // order in the Cols direction of the AR part of the model
    int       N_MA;               // order in the Rows direction of the MA part of the model
    int       M_MA;               // order in the Cols direction of the MA part of the model
public:
    MultidimArray<double> ARParameters; // Output parameters
    MultidimArray<double> MAParameters; // Output parameters
    double dSigma;                      // Noise variance
public:
    /// Read parameters from command line
    void readParams(XmippProgram *program);
    /// Define parameters
    static void defineParams(XmippProgram *program);
};

/** CausalARMA.

   This function determines the coeficients of an 2D - ARMA model

   Img(y,x)=sum( AR(p,q)*Img(y+p,x+q)) + sqrt(sigma) * e(y,x)


   Where:
        (p,q) is a vector belonging to a support region N1
  AR(p,q) is the matrix with the AR part coeficients of the model
  sigma is the variance of the random correlated input e(x,y)
  e(x,y) is random correlated input with zero mean and cross-spectral
  density:

  E[e(x,y)*Img(x+a,y+b)]=
                             sqrt(sigma)            if (p,q)=(0,0)
           sqrt(sigma)*MA(p,q)    if (p,q) belongs to N2
           0                      otherwise



  N1 - The support region for the AR part is the first quadrant region
       defined by N_AR and M_AR
  N2 - The support region the the MA part  is the second quadrant region
       defined by N_MA and M_MA

   This model is determined following the method depicted in:

   R. L. Kashyap, "Characterization and Estimation of Two-Dimensional
   ARMA Models, " IEEE Transactions on Information Theory, Vol. IT-30, No.
   5, pp. 736-745, September 1984.

    PARAMETERS:
    Img - The matrix - Here it's supposed that it comes from an image
    N_AR, M_AR - The order in Rows and Columns directions of the AR part
                 of the model.
    N_AR, M_AR - The order in Rows and Columns directions of the MA part
                 of the model.
    ARParameters - The matrix to return the resulting parameteres
                 for the AR part of the model.
    MAParameters - The matrix to return the resulting parameteres
                 for the MA part of the model.

   OUTPUT:  The function stores the ARMA parameters into ARParameters and
            MAParameters
            Every row of this output matrices has 3 values:
   1nd and 2nd- indicate the support point (p,q) for the coeficient
   3th column - indicates the value of the coeficient

   DATE:        26-3-2001
*/
void CausalARMA(MultidimArray<double> &Img, ARMA_parameters &prm);


/** ARMAFilter.
   This function returns the ARMA Filter associated to an ARMA model.
   This filter should be applied to a random matrix of independent identically
   distributed variables. In Xmipp the program to do that is Fourierfilter

    PARAMETERS:   Img - The matrix - Here it's supposed that it comes from
                        an input image
                  Filter - The matrix that will contain the filter.
    ARParameters - The matrix with the AR model coeficients, as
      is returned by CausalARMA or NonCausalARMA
    MAParameters - The matrix with the MA model coeficients, as
                           is returned by CausalARMA or NonCausalARMA
    dSigma        - The Sigma Coeficient for the ARMA model

   OUTPUT:  The function stores the output in Filter.

   DATE:        26-3-2001
*/
void ARMAFilter(MultidimArray<double> &Img, MultidimArray< double > &Filter,
                ARMA_parameters &prm);
//@}

#endif
