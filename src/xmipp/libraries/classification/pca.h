/***************************************************************************
 *
 * Authors:     Jorge Garcia de la Nava Ruiz (gdl@ac.uma.es)
 *              Carlos Oscar Sanchez Sorzano
 *              Alberto Pascual Montano (pascual@cnb.csic.es)
 *
 * Departamento de Arquitectura de Computadores, Universidad de M�laga
 *
 * Copyright (c) 2001 , CSIC/UMA.
 *
 * Permission is granted to copy and distribute this file, for noncommercial
 * use, provided (a) this copyright notice is preserved, (b) no attempt
 * is made to restrict redistribution of this file, and (c) this file is
 * restricted by a compilation copyright.
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'pascual@cnb.csic.es'
 *
 *****************************************************************************/

#ifndef XMIPPPC_H
#define XMIPPPC_H

#include "base_algorithm.h"
#include "data_types.h"
#include "training_vector.h"

#include <core/xmipp_funcs.h>
#include <core/matrix2d.h>
#include <core/multidim_array.h>

/**@defgroup PCA Principal Component Analysis
   @ingroup ClassificationLibrary */
//@{
/** Basic PCA class */
class PCAAnalyzer
{
public:

    /**
    * Make an empty PCAAnalyzer
    */
    PCAAnalyzer(void)
    {}

    /**
    * Construct a PCAAnalyzer object with eigenvectors & eigenvalues
    * from ts.
    * Parameter: ts The vectors.
    */
    PCAAnalyzer(ClassicTrainingVectors const &ts)
    {
        reset(ts);
    }

    /**
    * Construct a PCAAnalyzer object with eigenvectors & eigenvalues
    * from ts, using only the items given in idx.
    * Parameter: ts The vectors.
    * Parameter: idx The indexes of the vectors to use
    */
    PCAAnalyzer(ClassicTrainingVectors const &ts, std::vector<unsigned> const & idx)
    {
        reset(ts, idx);
    }

    /**
    * Calculate the eigenval/vecs
    * Parameter: ts The vectors.
    */
    void reset(ClassicTrainingVectors const &ts);

    /**
    * Calculate the eigenval/vecs
    * Parameter: ts The vectors.
    * Parameter: idx The indexes of the vectors to use
    */
    void reset(ClassicTrainingVectors const &ts, std::vector<unsigned> const & idx);

    /**
    * The eigenvectors
    */
    std::vector<FeatureVector> eigenvec;

    /**
    * The eigenvalues
    */
    FeatureVector eigenval;

    /**
    * Mean of the input training set
    */
    FeatureVector mean;

    /** Number of relevant eigenvectors */
    int D;

    /** Product <mean,mean> */
    double prod_mean_mean;

    /** Products <ei,mean> */
    std::vector<double> prod_ei_mean;

    /** Products <ei,ei> */
    std::vector<double> prod_ei_ei;

    /** Average of the mean vector */
    double avg_mean;

    /** Average of the ei vectors */
    std::vector<double> avg_ei;
#ifdef UNUSED // detected as unused 29.6.2018
    /**Set identity matrix as eigenvector matrix
     Parameter: n The number of eigenvectors*/
    void setIdentity(int n);
#endif

    /** Clear.
    Clean the eigenvector, eigenvalues and D */
    void clear();

#ifdef UNUSED // detected as unused 29.6.2018
    /** Prepare for correlation.
    This function computes the inner products emong the ei,
    and the mean vector, among the ei, and the average of all
    vectors*/
    void prepare_for_correlation();
#endif

    /** Set the number of relevant eigenvalues. */
    void set_Dimension(int _D)
    {
        D = _D;
    }

    /** Get the number of relevant eigenvalues. */
    int get_Dimension() const
    {
        return D;
    }

    /** Get the dimension of the eigenvectors. */
    int get_eigenDimension() const
    {
        if (eigenvec.size() > 1)
            return eigenvec[0].size();
        else
            return 0;
    }

#ifdef UNUSED // detected as unused 29.6.2018
    /** Number of components for a given accuracy explanation.
    This function returns the number of components to be taken
    if th_var% of the variance should be explained */
    int Dimension_for_variance(double th_var);

    /** Project onto PCA space.
    Given a vector of the same size as the PCA vectors this function
    returns the projection of size D onto the first D eigenvectors.
    D is set via the set_Dimension function

    An exception is thrown if the input vectors are not of the same size
    as the PCA ones.*/
    void Project(FeatureVector &input, FeatureVector &output);
#endif

    /** Defines Listener class
      */
    void setListener(BaseListener* _listener)
    {
        listener = _listener;
    };

    /** Show relevant eigenvectors and eigenvalues */
    friend std::ostream& operator << (std::ostream &out, const PCAAnalyzer &PC);

    /** Read a set of PCA just as shown */
    friend std::istream& operator >> (std::istream &in, PCAAnalyzer &PC);

private:

    BaseListener* listener;   // Listener class


};

/** Set of PCA classes */
class PCA_set
{
public:
    /** Set of PCA analysis. */
    std::vector<PCAAnalyzer *> PCA;

public:
    /** Destructor */
    ~PCA_set();

#ifdef UNUSED // detected as unused 29.6.2018
    /** Create empty PCA.
        Creates space for n new PCA and returns the index of the first one */
    int create_empty_PCA(int n = 1);
#endif

    /** Returns the number of PCA analysis.*/
    int PCANo() const
    {
        return PCA.size();
    }

    /** Returns a pointer to PCA number i*/
    PCAAnalyzer * operator()(int i) const
    {
        return PCA[i];
    }

    /** Show all PCA */
    friend std::ostream& operator << (std::ostream &out, const PCA_set &PS);

    /** Read a set of PCA just as shown */
    friend std::istream& operator >> (std::istream &in, PCA_set &PS);
};

/** Running PCA.
    Running PCA is an algorithm that estimates iteratively the
    principal components of a dataset that is provided to the algorithm
    as vectors are available.

    See J. Weng, Y. Zhang, W.S. Hwang. Candid covariance-free incremental
    principal component analysis. IEEE Trans. On Pattern Analysis and
    Machine Intelligence, 25(8): 1034-1040 (2003).
*/
class Running_PCA
{
public:
    /// Total number of eigenvectors to be computed.
    int J;

    /// Dimension of the sample vectors
    int d;

    /// Current estimate of the population mean
    Matrix1D<double> current_sample_mean;

    /// Current number of samples seen
    long n;

    /** Current estimate of the eigenvectors.
        Each column is an eigenvector. */
    Matrix2D<double> eigenvectors;

    /** Constructor.
        J is the number of eigenvectors to compute. d is the
        dimension of the sample vectors. */
    Running_PCA(int _J, int _d);

#ifdef UNUSED // detected as unused 29.6.2018
    /** Update estimates with a new sample. */
    void new_sample(const Matrix1D<double> &sample);
#endif

    /** Project a sample vector on the PCA space. */
    void project(const Matrix1D<double> &input, Matrix1D<double> &output) const;

    /// Get a certain eigenvector.
    void get_eigenvector(int j, Matrix1D<double> &result) const
    {
        eigenvectors.getCol(j, result);
    }

    /// Get the variance associated to a certain eigenvector.
    double get_eigenvector_variance(int j) const
    {
        if (n <= j)
            return 0.0;
        double mean_proj = sum_proj(j) / n;
        return sum_proj2(j) / n - mean_proj*mean_proj;
    }
public:
    // Sum of all samples so far
    Matrix1D<double> sum_all_samples;

    // Sum of all projections so far
    Matrix1D<double> sum_proj;

    // Sum of all projections squared so far
    Matrix1D<double> sum_proj2;
};

//@}
#endif

