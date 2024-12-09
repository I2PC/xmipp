/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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
#include <vector>
#include "numerical_tools.h"
#include "core/matrix2d.h"
#include "core/numerical_recipes.h"
#include "core/xmipp_funcs.h"
#include "core/xmipp_memory.h"

/* Random permutation ------------------------------------------------------ */
void randomPermutation(int N, MultidimArray<int>& result)
{
    MultidimArray<double> aux;
    aux.resize(N);
    aux.initRandom(0,1);

    aux.indexSort(result);
    result-=1;
}

/* Powell's optimizer ------------------------------------------------------ */
void powellOptimizer(Matrix1D<double> &p, int i0, int n,
                     double(*f)(double *x, void *), void * prm,
                     double ftol, double &fret,
                     int &iter, const Matrix1D<double> &steps, bool show)
{
        // Adapt indexes of p
    double *pptr = p.adaptForNumericalRecipes();
    double *auxpptr = pptr + (i0 - 1);

    // Form direction matrix
    std::vector<double> buffer(n*n);
auto *xi= buffer.data()-1;
    for (int i = 1, ptr = 1; i <= n; i++)
        for (int j = 1; j <= n; j++, ptr++)
            xi[ptr] = (i == j) ? steps(i - 1) : 0;

    // Optimize
    powell(auxpptr, xi -n, n, ftol, iter, fret, f, prm, show); // xi - n because NR works with matrices starting at [1,1]
    }

/* Gaussian interpolator -------------------------------------------------- */
void GaussianInterpolator::initialize(double _xmax, int N, bool normalize)
{
    xmax=_xmax;
    xstep=xmax/N;
    ixstep=1.0/xstep;
    v.initZeros(N);
    double inorm=1.0/sqrt(2*PI);
    FOR_ALL_ELEMENTS_IN_ARRAY1D(v)
    {
        double x=i*xstep;
        v(i)=exp(-x*x/2);
        if (normalize)
            v(i)*=inorm;
    }
}

/* Solve Cx=d, nonnegative x */
double solveNonNegative(const Matrix2D<double> &C, const Matrix1D<double> &d,
                        Matrix1D<double> &result)
{
    if (C.Xdim() == 0)
        REPORT_ERROR(ERR_MATRIX_EMPTY, "Solve_nonneg: Matrix is empty");
    if (C.Ydim() != d.size())
        REPORT_ERROR(ERR_MATRIX_SIZE, "Solve_nonneg: Different sizes of Matrix and Vector");
    if (d.isRow())
        REPORT_ERROR(ERR_MATRIX_DIM, "Solve_nonneg: Not correct vector shape");

    Matrix2D<double> Ct;
    Ct=C.transpose();

    result.initZeros(Ct.Ydim());
    double rnorm;

    // Watch out that matrix Ct is transformed.
    int success = nnls(MATRIX2D_ARRAY(Ct), Ct.Xdim(), Ct.Ydim(),
                       MATRIX1D_ARRAY(d),
                       MATRIX1D_ARRAY(result),
                       &rnorm, nullptr, nullptr, nullptr);
    if (success == 1)
        std::cerr << "Warning, too many iterations in nnls\n";
    else if (success == 2)
        REPORT_ERROR(ERR_MEM_NOTENOUGH, "Solve_nonneg: Not enough memory");
    return rnorm;
}

/* Solve Ax=b, A definite positive and symmetric --------------------------- */
void solveViaCholesky(const Matrix2D<double> &A, const Matrix1D<double> &b,
                      Matrix1D<double> &result)
{
    Matrix2D<double> Ap = A;
    Matrix1D<double> p(A.Xdim());
    result.resize(A.Xdim());
    choldc(Ap.adaptForNumericalRecipes2(), A.Xdim(),
           p.adaptForNumericalRecipes());
    cholsl(Ap.adaptForNumericalRecipes2(), A.Xdim(),
           p.adaptForNumericalRecipes(), b.adaptForNumericalRecipes(),
           result.adaptForNumericalRecipes());
}


/* Quadratic form ---------------------------------------------------------- */
void evaluateQuadratic(const Matrix1D<double> &x, const Matrix1D<double> &c,
                       const Matrix2D<double> &H, double &val, Matrix1D<double> &grad)
{
    if (x.size() != c.size())
        REPORT_ERROR(ERR_MATRIX_SIZE, "Eval_quadratic: Not compatible sizes in x and c");
    if (H.Xdim() != x.size())
        REPORT_ERROR(ERR_MATRIX_SIZE, "Eval_quadratic: Not compatible sizes in x and H");

    // H*x, store in grad
    grad.initZeros(x.size());
    for (size_t i = 0; i < H.Ydim(); i++)
        for (size_t j = 0; j < x.size(); j++)
            grad(i) += H(i, j) * x(j);

    // Now, compute c^t*x+1/2*x^t*H*x
    // Add c to the gradient
    double quad = 0;
    val = 0;
    for (size_t j = 0; j < x.size(); j++)
    {
        quad += grad(j) * grad(j); // quad=x^t*H^t*H*x
        val += c(j) * x(j);  // val=c^t*x

        grad(j) += c(j);     // grad+=c
    }
    val += 0.5 * quad;
}

/* Quadprog and Lsqlin ----------------------------------------------------- */
/* Structure to pass the objective function and constraints to cfsqp*/
typedef struct
{
    Matrix2D<double> C;
    Matrix2D<double> D;
    Matrix2D<double> A;
    Matrix2D<double> B;
}
CDAB;

/*/////////////////////////////////////////////////////////////////////////
      Internal functions used by the quadraticProgramming function
/////////////////////////////////////////////////////////////////////////*/
/* To calculate the value of the objective function */
void quadraticProgramming_obj32(int nparam, int j, double* x, double* fj, void* cd)
{
    auto* in = (CDAB *)cd;
    Matrix2D<double> X(nparam,1);
    for (int i=0; i<nparam; ++i)
        X(i,0)=x[i];
    Matrix2D<double> result;
    result = 0.5 * X.transpose() * in->C * X + in->D.transpose() * X;

    *fj = result(0, 0);
}

/* To calculate the value of the jth constraint */
void quadraticProgramming_cntr32(int nparam, int j, double* x, double* gj, void* cd)
{
    auto* in = (CDAB *)cd;
    *gj = 0;
    for (int k = 0; k < nparam; k++)
        *gj += in->A(j - 1, k) * x[k];
    *gj -= in->B(j - 1, 0);
}

/* To calculate the value of the derivative of objective function */
void quadraticProgramming_grob32(int nparam,  double* x, double* gradfj, void *cd)
{
    auto* in = (CDAB *)cd;
    Matrix2D<double> X(1,nparam);
    for (int i=0; i<nparam; ++i)
        X(0,i)=x[i];

    Matrix2D<double> gradient;
    gradient = in->C * X + in->D;
    for (int k = 0; k < nparam; k++)
        gradfj[k] = gradient(k, 0);
}

/* To calculate the value of the derivative of jth constraint */
void quadraticProgramming_grcn32(int nparam, int j, double *gradgj, void *cd)
{
    auto* in = (CDAB *)cd;
    for (int k = 0; k < nparam; k++)
        gradgj[k] = in->A(j - 1, k);
}

/**************************************************************************

   Solves Quadratic programming subproblem.

  min 0.5*x'Cx + d'x   subject to:  A*x <= b
   x                                Aeq*x=beq
                                bl<=x<=bu

**************************************************************************/
void quadraticProgramming(const Matrix2D<double> &C, const Matrix1D<double> &d,
                          const Matrix2D<double> &A,   const Matrix1D<double> &b,
                          const Matrix2D<double> &Aeq, const Matrix1D<double> &beq,
                          Matrix1D<double> &bl,        Matrix1D<double> &bu,
                          Matrix1D<double> &x)
{
    CDAB prm;
    prm.C = C;
    prm.D.fromVector(d);
    prm.A.initZeros(A.Ydim() + Aeq.Ydim(), A.Xdim());
    prm.B.initZeros(prm.A.Ydim(), 1);


    // Copy Inequalities
    for (size_t i = 0; i < A.Ydim(); i++)
    {
        for (size_t j = 0; j < A.Xdim(); j++)
            prm.A(i, j) = A(i, j);
        prm.B(i, 0) = b(i);
    }

    // Copy Equalities
    for (size_t i = 0; i < Aeq.Ydim(); i++)
    {
        for (size_t j = 0; j < Aeq.Xdim(); j++)
            prm.A(i + A.Ydim(), j) = Aeq(i, j);
        prm.B(i + A.Ydim(), 0) = beq(i);
    }

    double bigbnd = 1e30;
    // Bounds
    if (bl.size() == 0)
    {
        bl.resize(C.Xdim());
        bl.initConstant(-bigbnd);
    }
    if (bu.size() == 0)
    {
        bu.resize(C.Xdim());
        bu.initConstant(bigbnd);
    }

    // Define intermediate variables
    int    mode = 100;  // CFSQP mode
    int    iprint = 0;  // Debugging
    int    miter = 1000;  // Maximum number of iterations
    double eps = 1e-4; // Epsilon
    double epsneq = 1e-4; // Epsilon for equalities
    double udelta = 0.e0; // Finite difference approximation
    // of the gradients. Not used in this function
    int    nparam = C.Xdim(); // Number of variables
    int    nf = 1;          // Number of objective functions
    int    neqn = Aeq.Ydim();        // Number of nonlinear equations
    int    nineqn = A.Ydim();      // Number of nonlinear inequations
    int    nineq = A.Ydim();  // Number of linear inequations
    int    neq = Aeq.Ydim();  // Number of linear equations
    int    inform;
    int    ncsrl = 0;
    int    ncsrn = 0;
    int    nfsr = 0;
    int    mesh_pts[] = {0};

    if (x.size() == 0)
        x.initZeros(nparam);
    Matrix1D<double> f(nf);
    Matrix1D<double> g(nineq + neq);
    Matrix1D<double> lambda(nineq + neq + nf + nparam);

    // Call the minimization routine
    cfsqp(nparam, nf, nfsr, nineqn, nineq, neqn, neq, ncsrl, ncsrn, mesh_pts,
          mode, iprint, miter, &inform, bigbnd, eps, epsneq, udelta,
          MATRIX1D_ARRAY(bl), MATRIX1D_ARRAY(bu),
          MATRIX1D_ARRAY(x),
          MATRIX1D_ARRAY(f), MATRIX1D_ARRAY(g),
          MATRIX1D_ARRAY(lambda),
          //  quadprg_obj32,quadprog_cntr32,quadprog_grob32,quadprog_grcn32,
          quadraticProgramming_obj32, quadraticProgramming_cntr32, grobfd, grcnfd,
          (void*)&prm);

#ifdef DEBUG

    if (inform == 0)
        std::cout << "SUCCESSFUL RETURN. \n";
    if (inform == 1 || inform == 2)
        std::cout << "\nINITIAL GUESS INFEASIBLE.\n";
    if (inform == 3)
        printf("\n MAXIMUM NUMBER OF ITERATIONS REACHED.\n");
    if (inform > 3)
        printf("\ninform=%d\n", inform);
#endif
}

/**************************************************************************

   Solves the least square problem

  min 0.5*(Norm(C*x-d))   subject to:  A*x <= b
   x                                   Aeq*x=beq
                                       bl<=x<=bu
**************************************************************************/
void leastSquare(const Matrix2D<double> &C, const Matrix1D<double> &d,
                 const Matrix2D<double> &A,   const Matrix1D<double> &b,
                 const Matrix2D<double> &Aeq, const Matrix1D<double> &beq,
                 Matrix1D<double> &bl,        Matrix1D<double> &bu,
                 Matrix1D<double> &x)
{
    // Convert d to Matrix2D for multiplication
    Matrix2D<double> P;
    P.fromVector(d);
    P = -2.0 * (P.transpose() * C);
    P = P.transpose();

    //Convert back to vector for passing it to quadraticProgramming
    Matrix1D<double> newd;
    P.toVector(newd);

    quadraticProgramming(C.transpose()*C, newd, A, b, Aeq, beq, bl, bu, x);
}

/* Regularized least squares ----------------------------------------------- */
void regularizedLeastSquare(const Matrix2D< double >& A,
                            const Matrix1D< double >& d, double lambda,
                            const Matrix2D< double >& G, Matrix1D< double >& x)
{
    int Nx=A.Xdim(); // Number of variables

    Matrix2D<double> X(Nx,Nx); // X=(A^t * A +lambda *G^t G)
    // Compute A^t*A
    FOR_ALL_ELEMENTS_IN_MATRIX2D(X)
    // Compute the dot product of the i-th and j-th columns of A
    for (size_t k=0; k<A.Ydim(); k++)
        X(i,j) += A(k,i) * A(k,j);

    // Compute lambda*G^t*G
    if (G.Xdim()==0)
        for (int i=0; i<Nx; i++)
            X(i,i)+=lambda;
    else
        FOR_ALL_ELEMENTS_IN_MATRIX2D(X)
        // Compute the dot product of the i-th and j-th columns of G
        for (size_t k=0; k<G.Ydim(); k++)
            X(i,j) += G(k,i) * G(k,j);

    // Compute A^t*d
    Matrix1D<double> Atd(Nx);
    FOR_ALL_ELEMENTS_IN_MATRIX1D(Atd)
    // Compute the dot product of the i-th column of A and d
    for (size_t k=0; k<A.Ydim(); k++)
        Atd(i) += A(k,i) * d(k);

    // Compute the inverse of X
    Matrix2D<double> Xinv;
    X.inv(Xinv);

    // Now multiply Xinv * A^t * d
    matrixOperation_Ax(Xinv, Atd, x);
}


////////DE_solver

#define Element(a,b,c)  a[b*nDim+c]
#define RowVector(a,b)  (&a[b*nDim])
#define CopyVector(a,b) memcpy((a),(b),nDim*sizeof(double))

DESolver::DESolver(int dim, int popSize) :
nDim(dim), nPop(popSize),
generations(0), strategy(stRand1Exp),
scale(0.7), probability(0.5), bestEnergy(0.0),
trialSolution(0), bestSolution(0),
popEnergy(0), population(0)
{
    trialSolution = new double[nDim];
    bestSolution  = new double[nDim];
    popEnergy   = new double[nPop];
    population   = new double[nPop * nDim];

    randomize_random_generator();
}

DESolver::~DESolver(void)
{
    if (trialSolution)
        delete [] trialSolution;
    if (bestSolution)
        delete [] bestSolution;
    if (popEnergy)
        delete [] popEnergy;
    if (population)
        delete [] population;

    trialSolution = bestSolution = popEnergy = population = nullptr;
    return;
}

void DESolver::Setup(double min[], double max[],
                     int deStrategy, double diffScale, double crossoverProb)
{
    int i;
    int j;

    strategy = deStrategy;
    scale  = diffScale;
    probability     = crossoverProb;
    bestEnergy      = 1.0E20;

    for (i = 0; i < nPop; i++)
    {
        for (j = 0; j < nDim; j++)
        {
            population[i*nDim+j] = rnd_unif(min[j], max[j]);
            //Element(population, i, j) = rnd_unif(min[j], max[j]);
        }

        popEnergy[i] = 1.0E20;
    }

    for (i = 0; i < nDim; i++)
        bestSolution[i] = (min[i] + max[i]) / 2.0;

    switch (strategy)
    {
    case stBest1Exp:
        calcTrialSolution = &DESolver::Best1Exp;
        break;

    case stRand1Exp:
        calcTrialSolution = &DESolver::Rand1Exp;
        break;

    case stRandToBest1Exp:
        calcTrialSolution = &DESolver::RandToBest1Exp;
        break;

    case stBest2Exp:
        calcTrialSolution = &DESolver::Best2Exp;
        break;

    case stRand2Exp:
        calcTrialSolution = &DESolver::Rand2Exp;
        break;

    case stBest1Bin:
        calcTrialSolution = &DESolver::Best1Bin;
        break;

    case stRand1Bin:
        calcTrialSolution = &DESolver::Rand1Bin;
        break;

    case stRandToBest1Bin:
        calcTrialSolution = &DESolver::RandToBest1Bin;
        break;

    case stBest2Bin:
        calcTrialSolution = &DESolver::Best2Bin;
        break;

    case stRand2Bin:
        calcTrialSolution = &DESolver::Rand2Bin;
        break;
    }

    return;
}

bool DESolver::Solve(int maxGenerations)
{
    bool bAtSolution = false;
    int generation;

    for (generation = 0;(generation < maxGenerations) && !bAtSolution;generation++)
        for (int candidate = 0; candidate < nPop; candidate++)
        {
            (this->*calcTrialSolution)(candidate);
            trialEnergy = EnergyFunction(trialSolution, bAtSolution);

            if (trialEnergy < popEnergy[candidate])
            {
                // New low for this candidate
                popEnergy[candidate] = trialEnergy;
                CopyVector(RowVector(population, candidate), trialSolution);

                // Check if all-time low
                if (trialEnergy < bestEnergy)
                {
                    bestEnergy = trialEnergy;
                    CopyVector(bestSolution, trialSolution);
                }
            }
        }

    generations = generation;
    return bAtSolution;
}

void DESolver::Best1Exp(int candidate)
{
    int r1;
    int r2;
    int n;

    SelectSamples(candidate, &r1, &r2);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; (rnd_unif(0.0, 1.0) < probability) && (i < nDim); i++)
    {
        trialSolution[n] = bestSolution[n]
                           + scale * (Element(population, r1, n)
                                      - Element(population, r2, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::Rand1Exp(int candidate)
{
    int r1;
    int r2;
    int r3;
    int n;

    SelectSamples(candidate, &r1, &r2, &r3);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; (rnd_unif(0.0, 1.0) < probability) && (i < nDim); i++)
    {
        trialSolution[n] = Element(population, r1, n)
                           + scale * (Element(population, r2, n)
                                      - Element(population, r3, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::RandToBest1Exp(int candidate)
{
    int r1;
    int r2;
    int n;

    SelectSamples(candidate, &r1, &r2);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; (rnd_unif(0.0, 1.0) < probability) && (i < nDim); i++)
    {
        trialSolution[n] += scale * (bestSolution[n] - trialSolution[n])
                            + scale * (Element(population, r1, n)
                                       - Element(population, r2, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::Best2Exp(int candidate)
{
    int r1;
    int r2;
    int r3;
    int r4;
    int n;

    SelectSamples(candidate, &r1, &r2, &r3, &r4);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; (rnd_unif(0.0, 1.0) < probability) && (i < nDim); i++)
    {
        trialSolution[n] = bestSolution[n] +
                           scale * (Element(population, r1, n)
                                    + Element(population, r2, n)
                                    - Element(population, r3, n)
                                    - Element(population, r4, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::Rand2Exp(int candidate)
{
    int r1;
    int r2;
    int r3;
    int r4;
    int r5;
    int n;

    SelectSamples(candidate, &r1, &r2, &r3, &r4, &r5);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; (rnd_unif(0.0, 1.0) < probability) && (i < nDim); i++)
    {
        trialSolution[n] = Element(population, r1, n)
                           + scale * (Element(population, r2, n)
                                      + Element(population, r3, n)
                                      - Element(population, r4, n)
                                      - Element(population, r5, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::Best1Bin(int candidate)
{
    int r1;
    int r2;
    int n;

    SelectSamples(candidate, &r1, &r2);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; i < nDim; i++)
    {
        if ((rnd_unif(0.0, 1.0) < probability) || (i == (nDim - 1)))
            trialSolution[n] = bestSolution[n]
                               + scale * (Element(population, r1, n)
                                          - Element(population, r2, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::Rand1Bin(int candidate)
{
    int r1;
    int r2;
    int r3;
    int n;

    SelectSamples(candidate, &r1, &r2, &r3);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; i < nDim; i++)
    {
        if ((rnd_unif(0.0, 1.0) < probability) || (i  == (nDim - 1)))
            trialSolution[n] = Element(population, r1, n)
                               + scale * (Element(population, r2, n)
                                          - Element(population, r3, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::RandToBest1Bin(int candidate)
{
    int r1;
    int r2;
    int n;

    SelectSamples(candidate, &r1, &r2);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; i < nDim; i++)
    {
        if ((rnd_unif(0.0, 1.0) < probability) || (i  == (nDim - 1)))
            trialSolution[n] += scale * (bestSolution[n] - trialSolution[n])
                                + scale * (Element(population, r1, n)
                                           - Element(population, r2, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::Best2Bin(int candidate)
{
    int r1;
    int r2;
    int r3;
    int r4;
    int n;

    SelectSamples(candidate, &r1, &r2, &r3, &r4);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; i < nDim; i++)
    {
        if ((rnd_unif(0.0, 1.0) < probability) || (i  == (nDim - 1)))
            trialSolution[n] = bestSolution[n]
                               + scale * (Element(population, r1, n)
                                          + Element(population, r2, n)
                                          - Element(population, r3, n)
                                          - Element(population, r4, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::Rand2Bin(int candidate)
{
    int r1, r2, r3, r4, r5;
    int n;

    SelectSamples(candidate, &r1, &r2, &r3, &r4, &r5);
    n = (int)rnd_unif(0.0, (double)nDim);

    CopyVector(trialSolution, RowVector(population, candidate));
    for (int i = 0; i < nDim; i++)
    {
        if ((rnd_unif(0.0, 1.0) < probability) || (i  == (nDim - 1)))
            trialSolution[n] = Element(population, r1, n)
                               + scale * (Element(population, r2, n)
                                          + Element(population, r3, n)
                                          - Element(population, r4, n)
                                          - Element(population, r5, n));
        n = (n + 1) % nDim;
    }

    return;
}

void DESolver::SelectSamples(int candidate, int *r1, int *r2,
                             int *r3, int *r4, int *r5)
{
    if (r1)
    {
        do
        {
            *r1 = (int)rnd_unif(0.0, (double)nPop);
        }
        while (*r1 == candidate);
    } else
    	return;

    if (r2)
    {
        do
        {
            *r2 = (int)rnd_unif(0.0, (double)nPop);
        }
        while ((*r2 == candidate) || (*r2 == *r1));
    } else
    	return;

    if (r3)
    {
        do
        {
            *r3 = (int)rnd_unif(0.0, (double)nPop);
        }
        while ((*r3 == candidate) || (*r3 == *r2) || (*r3 == *r1));
    } else
    	return;

    if (r4)
    {
        do
        {
            *r4 = (int)rnd_unif(0.0, (double)nPop);
        }
        while ((*r4 == candidate) || (*r4 == *r3) || (*r4 == *r2) || (*r4 == *r1));
    } else
    	return;

    if (r5)
    {
        do
        {
            *r5 = (int)rnd_unif(0.0, (double)nPop);
        }
        while ((*r5 == candidate) || (*r5 == *r4) || (*r5 == *r3)
               || (*r5 == *r2) || (*r5 == *r1));
    }
}

/* Check Randomness ------------------------------------------------------- */
/* See http://home.ubalt.edu/ntsbarsh/Business-stat/opre504.htm for the formulas */
/* This is called runs test */
double checkRandomness(const std::string &sequence)
{
    int imax=sequence.size();
    if (imax<=1)
        return 0;
    double n0=1;
    double n1=0;
    double R=1;
    int current=0;
    for (int i=1; i<imax; ++i)
    {
        if (sequence[i]!=sequence[i-1])
        {
        	current=(current+1)%2;
        	R++;
            if (current==0)
                n0++;
            else
                n1++;
        }
        else
        {
            if (current==0)
                n0++;
            else
                n1++;
        }
    }
    double m=1+2*n0*n1/(n0+n1);
    double s=sqrt(2*n0*n1*(2*n0*n1-n0-n1)/((n0+n1)*(n0+n1)*(n0+n1-1)));
    double z=(R-m)/s;
    return ABS(z);
}

#ifdef NEVERDEFINED
double ZernikeSphericalHarmonics(int l1, int n, int l2, int m, double xr, double yr, double zr, double r)
{
	// General variables
	double r2=r*r,xr2=xr*xr,yr2=yr*yr,zr2=zr*zr;

	//Variables needed for l>=5
	double tht=0.0,phi=0.0,costh=0.0,sinth=0.0,costh2=0.0,sinth2=0.0;
	if (l2>=5)
	{
		tht = atan2(yr,xr);
		phi = atan2(zr,sqrt(xr2 + yr2));
		sinth = sin(phi); costh = cos(tht);
		sinth2 = sinth*sinth; costh2 = costh*costh;
	}

	// Zernike polynomial
	double R=0.0;

	switch (l1)
	{
	case 0:
		R = 1.0;
		break;
	case 1:
		R = r;
		break;
	case 2:
		switch (n)
		{
		case 0:
			R = -1+2*r2;
			break;
		case 2:
			R = r2;
			break;
		} break;
	case 3:
		switch (n)
		{
		case 1:
			R = 3*r2*r-2*r;
			break;
		case 3:
			R = r2*r;
		} break;
	case 4:
		switch (n)
		{
		case 0:
			R = 6*r2*r2-6*r2+1;
			break;
		case 2:
			R = 4*r2*r2-3*r2;
			break;
		case 4:
			R = r2*r2;
			break;
		} break;
	case 5:
		switch (n)
		{
		case 1:
			R = 10.0*r2*r2*r-12.0*r2*r+3.0*r;;
			break;
		case 3:
			R = 5.0*r2*r2*r-4.0*r2*r;
			break;
		case 5:
			R = r2*r2*r;
			break;
		}break;
	}

	// Spherical harmonic
	double Y=0.0;

	switch (l2)
	{
	case 0:
		Y = (1.0/2.0)*sqrt(1.0/PI);
		break;
	case 1:
		switch (m)
		{
		case -1:
			Y = sqrt(3.0/(4.0*PI))*yr;
			break;
		case 0:
			Y = sqrt(3.0/(4.0*PI))*zr;
			break;
		case 1:
			Y = sqrt(3.0/(4.0*PI))*xr;
			break;
		} break;
	case 2:
		switch (m)
		{
		case -2:
			Y = sqrt(15.0/(4.0*PI))*xr*yr;
			break;
		case -1:
			Y = sqrt(15.0/(4.0*PI))*zr*yr;
			break;
		case 0:
			Y = sqrt(5.0/(16.0*PI))*(-xr2-yr2+2.0*zr2);
			break;
		case 1:
			Y = sqrt(15.0/(4.0*PI))*xr*zr;
			break;
		case 2:
			Y = sqrt(15.0/(16.0*PI))*(xr2-yr2);
			break;
		} break;
	case 3:
		switch (m)
		{
		case -3:
			Y = sqrt(35.0/(16.0*2.0*PI))*yr*(3.0*xr2-yr2);
			break;
		case -2:
			Y = sqrt(105.0/(4.0*PI))*zr*yr*xr;
			break;
		case -1:
			Y = sqrt(21.0/(16.0*2.0*PI))*yr*(4.0*zr2-xr2-yr2);
			break;
		case 0:
			Y = sqrt(7.0/(16.0*PI))*zr*(2.0*zr2-3.0*xr2-3.0*yr2);
			break;
		case 1:
			Y = sqrt(21.0/(16.0*2.0*PI))*xr*(4.0*zr2-xr2-yr2);
			break;
		case 2:
			Y = sqrt(105.0/(16.0*PI))*zr*(xr2-yr2);
			break;
		case 3:
			Y = sqrt(35.0/(16.0*2.0*PI))*xr*(xr2-3.0*yr2);
			break;
		} break;
	case 4:
		switch (m)
		{
		case -4:
			Y = sqrt((35.0*9.0)/(16.0*PI))*yr*xr*(xr2-yr2);
			break;
		case -3:
			Y = sqrt((9.0*35.0)/(16.0*2.0*PI))*yr*zr*(3.0*xr2-yr2);
			break;
		case -2:
			Y = sqrt((9.0*5.0)/(16.0*PI))*yr*xr*(7.0*zr2-(xr2+yr2+zr2));
			break;
		case -1:
			Y = sqrt((9.0*5.0)/(16.0*2.0*PI))*yr*zr*(7.0*zr2-3.0*(xr2+yr2+zr2));
			break;
		case 0:
			Y = sqrt(9.0/(16.0*16.0*PI))*(35.0*zr2*zr2-30.0*zr2+3.0);
			break;
		case 1:
			Y = sqrt((9.0*5.0)/(16.0*2.0*PI))*xr*zr*(7.0*zr2-3.0*(xr2+yr2+zr2));
			break;
		case 2:
			Y = sqrt((9.0*5.0)/(8.0*8.0*PI))*(xr2-yr2)*(7.0*zr2-(xr2+yr2+zr2));
			break;
		case 3:
			Y = sqrt((9.0*35.0)/(16.0*2.0*PI))*xr*zr*(xr2-3.0*yr2);
			break;
		case 4:
			Y = sqrt((9.0*35.0)/(16.0*16.0*PI))*(xr2*(xr2-3.0*yr2)-yr2*(3.0*xr2-yr2));
			break;
		} break;
	case 5:
		switch (m)
		{
		case -5:
			Y = (3.0/16.0)*sqrt(77.0/(2.0*PI))*sint2*sint2*sinth*sin(5.0*phi);
			break;
		case -4:
			Y = (3.0/8.0)*sqrt(385.0/(2.0*PI))*sint2*sint2*sin(4.0*phi);
			break;
		case -3:
			Y = (1.0/16.0)*sqrt(385.0/(2.0*PI))*sint2*sinth*(9.0*cost2-1.0)*sin(3.0*phi);
			break;
		case -2:
			Y = (1.0/4.0)*sqrt(1155.0/(4.0*PI))*sint2*(3.0*cost2*costh-costh)*sin(2.0*phi);
			break;
		case -1:
			Y = (1.0/8.0)*sqrt(165.0/(4.0*PI))*sinth*(21.0*cost2*cost2-14.0*cost2+1)*sin(phi);
			break;
		case 0:
			Y = (1.0/16.0)*sqrt(11.0/PI)*(63.0*cost2*cost2*costh-70.0*cost2*costh+15.0*costh);
			break;
		case 1:
			Y = (1.0/8.0)*sqrt(165.0/(4.0*PI))*sinth*(21.0*cost2*cost2-14.0*cost2+1)*cos(phi);
			break;
		case 2:
			Y = (1.0/4.0)*sqrt(1155.0/(4.0*PI))*sint2*(3.0*cost2*costh-costh)*cos(2.0*phi);
			break;
		case 3:
			Y = (1.0/16.0)*sqrt(385.0/(2.0*PI))*sint2*sinth*(9.0*cost2-1.0)*cos(3.0*phi);
			break;
		case 4:
			Y = (3.0/8.0)*sqrt(385.0/(2.0*PI))*sint2*sint2*cos(4.0*phi);
			break;
		case 5:
			Y = (3.0/16.0)*sqrt(77.0/(2.0*PI))*sint2*sint2*sinth*cos(5.0*phi);
			break;
		}break;
	}

	return R*Y;
}
#endif

template<int L1, int L2>
double ZernikeSphericalHarmonics(int n, int m, double xr, double yr, double zr, double r)
{
	// General variables
	double r2=r*r;
    double xr2=xr*xr;
    double yr2=yr*yr;
    double zr2=zr*zr;

	//Variables needed for l>=5
	double tht=0.0;
    double phi=0.0;
    double costh=0.0;
    double sinth=0.0;
    double costh2=0.0;
    double sinth2=0.0;
    double cosph=0.0;
    double cosph2=0.0;
	if (L2>=5)
	{
		tht = atan2(yr,xr);
		phi = atan2(zr,sqrt(xr2 + yr2));
		sinth = sin(abs(m)*phi); costh = cos(tht); cosph = cos(abs(m)*phi);
		sinth2 = sinth*sinth; costh2 = costh*costh; cosph2 = cosph * cosph;
	}

	// Zernike polynomial
	double R=0.0;

	switch (L1)
	{
	case 0:
		R = std::sqrt(3.0);
		break;
	case 1:
		R = std::sqrt(5.0)*r;
		break;
	case 2:
		switch (n)
		{
		case 0:
			R = -0.5*std::sqrt(7.0)*(2.5*(1-2*r2)+0.5);
			break;
		case 2:
			R = std::sqrt(7.0)*r2;
			break;
		} break;
	case 3:
		switch (n)
		{
		case 1:
			R = -1.5*r*(3.5*(1-2*r2)+1.5);
			break;
		case 3:
			R = 3.0*r2*r;
		} break;
	case 4:
		switch (n)
		{
		case 0:
			R = std::sqrt(11.0)*((63.0*r2*r2/8.0)-(35.0*r2/4.0)+(15.0/8.0));
			break;
		case 2:
			R = -0.5*std::sqrt(11.0)*r2*(4.5*(1-2*r2)+2.5);
			break;
		case 4:
			R = std::sqrt(11.0)*r2*r2;
			break;
		} break;
	case 5:
		switch (n)
		{
		case 1:
			R = std::sqrt(13.0)*r*((99.0*r2*r2/8.0)-(63.0*r2/4.0)+(35.0/8.0));
			break;
		case 3:
			R = -0.5*std::sqrt(13.0)*r2*r*(5.5*(1-2*r2)+3.5);
			break;
        case 5:
            R = std::sqrt(13.0)*r2*r2*r;
            break;
		} break;
    case 6:
        switch (n)
        {
        case 0:
            R = 103.8 * r2 * r2 * r2 - 167.7 * r2 * r2 + 76.25 * r2 - 8.472;
            break;
        case 2:
            R = 69.23 * r2 * r2 * r2 - 95.86 * r2 * r2 + 30.5 * r2;
            break;
        case 4:
            R = 25.17 * r2 * r2 * r2 - 21.3 * r2 * r2;
            break;
        case 6:
            R = 3.873 * r2 * r2 * r2;
            break;
        } break;
    case 7:
        switch (n)
        {
        case 1:
            R = 184.3 * r2 * r2 * r2 * r - 331.7 * r2 * r2 * r + 178.6 * r2 * r - 27.06 * r;
            break;
        case 3:
            R = 100.5 * r2 * r2 * r2 * r - 147.4 * r2 * r2 * r + 51.02 * r2 * r;
            break;
        case 5:
            R = 30.92 * r2 * r2 * r2 * r - 26.8 * r2 * r2 * r;
            break;
        case 7:
            R = 4.123 * r2 * r2 * r2 * r;
            break;
        } break;
    case 8:
        switch (n)
        {
        case 0:
            R = 413.9*r2*r2*r2*r2 - 876.5*r2*r2*r2 + 613.6*r2*r2 - 157.3*r2 + 10.73;
            break;
        case 2:
            R = 301.0*r2*r2*r2*r2 - 584.4*r2*r2*r2 + 350.6*r2*r2 - 62.93*r2;
            break;
        case 4:
            R = 138.9*r2*r2*r2*r2 - 212.5*r2*r2*r2 + 77.92*r2*r2;
            break;
        case 6:
            R = 37.05*r2*r2*r2*r2 - 32.69*r2*r2*r2;
            break;
        case 8:
            R = 4.359*r2*r2*r2*r2;
            break;
        } break;
    case 9:
        switch (n)
        {
        case 1:
            R = 751.6*r2*r2*r2*r2*r - 1741.0*r2*r2*r2*r + 1382.0*r2*r2*r - 430.0*r2*r + 41.35*r;
            break;
        case 3:
            R = 462.6*r2*r2*r2*r2*r - 949.5*r2*r2*r2*r + 614.4*r2*r2*r - 122.9*r2*r;
            break;
        case 5:
            R = 185.0*r2*r2*r2*r2*r - 292.1*r2*r2*r2*r + 111.7*r2*r2*r;
            break;
        case 7:
            R = 43.53*r2*r2*r2*r2*r - 38.95*r2*r2*r2*r;
            break;
        case 9:
            R = 4.583*r2*r2*r2*r2*r;
            break;
        } break;
    case 10:
        switch (n)
        {
        case 0:
            R = 1652.0*r2*r2*r2*r2*r2 - 4326.0*r2*r2*r2*r2 + 4099.0*r2*r2*r2 - 1688.0*r2*r2 + 281.3*r2 - 12.98;
            break;
        case 2:
            R = 1271.0*r2*r2*r2*r2*r2 - 3147.0*r2*r2*r2*r2 + 2732.0*r2*r2*r2 - 964.4*r2*r2 + 112.5*r2;
            break;
        case 4:
            R = 677.7*r2*r2*r2*r2*r2 - 1452.0*r2*r2*r2*r2 + 993.6*r2*r2*r2 - 214.3*r2*r2;
            break;
        case 6:
            R = 239.2*r2*r2*r2*r2*r2 - 387.3*r2*r2*r2*r2 + 152.9*r2*r2*r2;
            break;
        case 8:
            R = 50.36*r2*r2*r2*r2*r2 - 45.56*r2*r2*r2*r2;
            break;
        case 10:
            R = 4.796*r2*r2*r2*r2*r2;
            break;
        } break;
    case 11:
        switch (n)
        {
        case 1:
            R = r*-5.865234375E+1+(r*r*r)*8.7978515625E+2-(r*r*r*r*r)*4.2732421875E+3+(r*r*r*r*r*r*r)*9.0212890625E+3-(r*r*r*r*r*r*r*r*r)*8.61123046875E+3+pow(r,1.1E+1)*3.04705078125E+3;
            break;
        case 3:
            R = (r*r*r)*2.513671875E+2-(r*r*r*r*r)*1.89921875E+3+(r*r*r*r*r*r*r)*4.920703125E+3-(r*r*r*r*r*r*r*r*r)*5.29921875E+3+pow(r,1.1E+1)*2.0313671875E+3;
            break;
        case 5:
            R = (r*r*r*r*r)*-3.453125E+2+(r*r*r*r*r*r*r)*1.5140625E+3-(r*r*r*r*r*r*r*r*r)*2.1196875E+3+pow(r,1.1E+1)*9.559375E+2;
            break;
        case 7:
            R = (r*r*r*r*r*r*r)*2.01875E+2-(r*r*r*r*r*r*r*r*r)*4.9875E+2+pow(r,1.1E+1)*3.01875E+2;
            break;
        case 9:
            R = (r*r*r*r*r*r*r*r*r)*-5.25E+1+pow(r,1.1E+1)*5.75E+1;
            break;
        case 11:
            R = pow(r,1.1E+1)*5.0;
            break;
        } break;
    case 12:
        switch (n)
        {
        case 0:
            R = (r*r)*-4.57149777110666E+2+(r*r*r*r)*3.885773105442524E+3-(r*r*r*r*r*r)*1.40627979054153E+4+(r*r*r*r*r*r*r*r)*2.460989633446932E+4-pow(r,1.0E+1)*2.05828223888278E+4+pow(r,1.2E+1)*6.597058457955718E+3+1.523832590368693E+1;
            break;
        case 2:
            R = (r*r)*-1.828599108443595E+2+(r*r*r*r)*2.220441774539649E+3-(r*r*r*r*r*r)*9.375198603600264E+3+(r*r*r*r*r*r*r*r)*1.789810642504692E+4-pow(r,1.0E+1)*1.583294029909372E+4+pow(r,1.2E+1)*5.277646766364574E+3;
            break;
        case 4:
            R = (r*r*r*r)*4.934315054528415E+2-(r*r*r*r*r*r)*3.409163128584623E+3+(r*r*r*r*r*r*r*r)*8.260664503872395E+3-pow(r,1.0E+1)*8.444234826177359E+3+pow(r,1.2E+1)*3.104498097866774E+3;
            break;
        case 6:
            R = (r*r*r*r*r*r)*-5.244866351671517E+2+(r*r*r*r*r*r*r*r)*2.202843867704272E+3-pow(r,1.0E+1)*2.98031817394495E+3+pow(r,1.2E+1)*1.307157093837857E+3;
            break;
        case 8:
            R = (r*r*r*r*r*r*r*r)*2.591581020820886E+2-pow(r,1.0E+1)*6.274354050420225E+2+pow(r,1.2E+1)*3.734734553815797E+2;
            break;
        case 10:
            R = pow(r,1.0E+1)*-5.975575286115054E+1+pow(r,1.2E+1)*6.49519052838441E+1;
            break;
        case 12:
            R = pow(r,1.2E+1)*5.19615242270811;
            break;
        } break;
    case 13:
        switch (n)
        {
        case 1:
            R = r*7.896313435467891E+1-(r*r*r)*1.610847940832376E+3+(r*r*r*r*r)*1.093075388422608E+4-(r*r*r*r*r*r*r)*3.400678986203671E+4+(r*r*r*r*r*r*r*r*r)*5.332882955634594E+4-pow(r,1.1E+1)*4.102217658185959E+4+pow(r,1.3E+1)*1.230665297454596E+4;
            break;
        case 3:
            R = (r*r*r)*-4.602422688100487E+2+(r*r*r*r*r)*4.858112837433815E+3-(r*r*r*r*r*r*r)*1.854915810656548E+4+(r*r*r*r*r*r*r*r*r)*3.281774126553535E+4-pow(r,1.1E+1)*2.734811772125959E+4+pow(r,1.3E+1)*8.687049158513546E+3;
            break;
        case 5:
            R = (r*r*r*r*r)*8.832932431697845E+2-(r*r*r*r*r*r*r)*5.707433263555169E+3+(r*r*r*r*r*r*r*r*r)*1.312709650617838E+4-pow(r,1.1E+1)*1.286970245704055E+4+pow(r,1.3E+1)*4.572131136059761E+3;
            break;
        case 7:
            R = (r*r*r*r*r*r*r)*-7.60991101808846E+2+(r*r*r*r*r*r*r*r*r)*3.088728589691222E+3-pow(r,1.1E+1)*4.064116565383971E+3+pow(r,1.3E+1)*1.741764242306352E+3;
            break;
        case 9:
            R = (r*r*r*r*r*r*r*r*r)*3.251293252306059E+2-pow(r,1.1E+1)*7.741174410264939E+2+pow(r,1.3E+1)*4.54373280601576E+2;
            break;
        case 11:
            R = pow(r,1.1E+1)*-6.731456008902751E+1+pow(r,1.3E+1)*7.269972489634529E+1;
            break;
        case 13:
            R = pow(r,1.3E+1)*5.385164807128604;
            break;
        } break;
    case 14:
        switch (n)
        {
        case 0:
            R = (r*r)*6.939451623205096E+2-(r*r*r*r)*7.910974850460887E+3+(r*r*r*r*r*r)*3.955487425231934E+4-(r*r*r*r*r*r*r*r)*1.010846786448956E+5+pow(r,1.0E+1)*1.378427436065674E+5-pow(r,1.2E+1)*9.542959172773361E+4+pow(r,1.4E+1)*2.63567443819046E+4-1.749441585683962E+1;
            break;
        case 2:
            R = (r*r)*2.775780649287626E+2-(r*r*r*r)*4.520557057410479E+3+(r*r*r*r*r*r)*2.636991616821289E+4-(r*r*r*r*r*r*r*r)*7.351612992358208E+4+pow(r,1.0E+1)*1.060328796973228E+5-pow(r,1.2E+1)*7.634367338204384E+4+pow(r,1.4E+1)*2.170555419689417E+4;
            break;
        case 4:
            R = (r*r*r*r)*-1.004568234980106E+3+(r*r*r*r*r*r)*9.589060424804688E+3-(r*r*r*r*r*r*r*r)*3.393052150321007E+4+pow(r,1.0E+1)*5.655086917197704E+4-pow(r,1.2E+1)*4.490804316592216E+4+pow(r,1.4E+1)*1.370877107170224E+4;
            break;
        case 6:
            R = (r*r*r*r*r*r)*1.475240065354854E+3-(r*r*r*r*r*r*r*r)*9.04813906750083E+3+pow(r,1.0E+1)*1.99591302959919E+4-pow(r,1.2E+1)*1.8908649754107E+4+pow(r,1.4E+1)*6.527986224621534E+3;
            break;
        case 8:
            R = (r*r*r*r*r*r*r*r)*-1.064486949119717E+3+pow(r,1.0E+1)*4.201922167569399E+3-pow(r,1.2E+1)*5.402471358314157E+3+pow(r,1.4E+1)*2.270603904217482E+3;
            break;
        case 10:
            R = pow(r,1.0E+1)*4.001830635787919E+2-pow(r,1.2E+1)*9.395602362267673E+2+pow(r,1.4E+1)*5.44944937011227E+2;
            break;
        case 12:
            R = pow(r,1.2E+1)*-7.516481889830902E+1+pow(r,1.4E+1)*8.073258326109499E+1;
            break;
        case 14:
            R = pow(r,1.4E+1)*5.567764362829621;
            break;
        } break;
    case 15:
        switch (n)
        {
        case 1:
            R = r*-1.022829477079213E+2+(r*r*r)*2.720726409032941E+3-(r*r*r*r*r)*2.448653768128157E+4+(r*r*r*r*r*r*r)*1.042945123462677E+5-(r*r*r*r*r*r*r*r*r)*2.370329826049805E+5+pow(r,1.1E+1)*2.953795629386902E+5-pow(r,1.3E+1)*1.903557183384895E+5+pow(r,1.5E+1)*4.958846444106102E+4;
            break;
        case 3:
            R = (r*r*r)*7.773504025805742E+2-(r*r*r*r*r)*1.088290563613176E+4+(r*r*r*r*r*r*r)*5.688791582524776E+4-(r*r*r*r*r*r*r*r*r)*1.458664508337975E+5+pow(r,1.1E+1)*1.969197086257935E+5-pow(r,1.3E+1)*1.343687423563004E+5+pow(r,1.5E+1)*3.653886853551865E+4;
            break;
        case 5:
            R = (r*r*r*r*r)*-1.978710115659982E+3+(r*r*r*r*r*r*r)*1.750397410005331E+4-(r*r*r*r*r*r*r*r*r)*5.834658033359051E+4+pow(r,1.1E+1)*9.266809817695618E+4-pow(r,1.3E+1)*7.072039071393013E+4+pow(r,1.5E+1)*2.08793534488678E+4;
            break;
        case 7:
            R = (r*r*r*r*r*r*r)*2.333863213345408E+3-(r*r*r*r*r*r*r*r*r)*1.372860713732243E+4+pow(r,1.1E+1)*2.926360995060205E+4-pow(r,1.3E+1)*2.694110122436285E+4+pow(r,1.5E+1)*9.077979760378599E+3;
            break;
        case 9:
            R = (r*r*r*r*r*r*r*r*r)*-1.445116540770978E+3+pow(r,1.1E+1)*5.57402094297111E+3-pow(r,1.3E+1)*7.028113362878561E+3+pow(r,1.5E+1)*2.90495352332294E+3;
            break;
        case 11:
            R = pow(r,1.1E+1)*4.846974733015522E+2-pow(r,1.3E+1)*1.124498138058931E+3+pow(r,1.5E+1)*6.455452274046838E+2;
            break;
        case 13:
            R = pow(r,1.3E+1)*-8.329615837475285E+1+pow(r,1.5E+1)*8.904072102135979E+1;
            break;
        case 15:
            R = pow(r,1.5E+1)*5.744562646534177;
            break;
        } break;
    default: break;
	}

	// Spherical harmonic
	double Y=0.0;

	switch (L2)
	{
	case 0:
		Y = (1.0/2.0)*sqrt(1.0/PI);
		break;
	case 1:
		switch (m)
		{
		case -1:
			Y = sqrt(3.0/(4.0*PI))*yr;
			break;
		case 0:
			Y = sqrt(3.0/(4.0*PI))*zr;
			break;
		case 1:
			Y = sqrt(3.0/(4.0*PI))*xr;
			break;
		} break;
	case 2:
		switch (m)
		{
		case -2:
			Y = sqrt(15.0/(4.0*PI))*xr*yr;
			break;
		case -1:
			Y = sqrt(15.0/(4.0*PI))*zr*yr;
			break;
		case 0:
			Y = sqrt(5.0/(16.0*PI))*(-xr2-yr2+2.0*zr2);
			break;
		case 1:
			Y = sqrt(15.0/(4.0*PI))*xr*zr;
			break;
		case 2:
			Y = sqrt(15.0/(16.0*PI))*(xr2-yr2);
			break;
		} break;
	case 3:
		switch (m)
		{
		case -3:
			Y = sqrt(35.0/(16.0*2.0*PI))*yr*(3.0*xr2-yr2);
			break;
		case -2:
			Y = sqrt(105.0/(4.0*PI))*zr*yr*xr;
			break;
		case -1:
			Y = sqrt(21.0/(16.0*2.0*PI))*yr*(4.0*zr2-xr2-yr2);
			break;
		case 0:
			Y = sqrt(7.0/(16.0*PI))*zr*(2.0*zr2-3.0*xr2-3.0*yr2);
			break;
		case 1:
			Y = sqrt(21.0/(16.0*2.0*PI))*xr*(4.0*zr2-xr2-yr2);
			break;
		case 2:
			Y = sqrt(105.0/(16.0*PI))*zr*(xr2-yr2);
			break;
		case 3:
			Y = sqrt(35.0/(16.0*2.0*PI))*xr*(xr2-3.0*yr2);
			break;
		} break;
	case 4:
		switch (m)
		{
		case -4:
			Y = sqrt((35.0*9.0)/(16.0*PI))*yr*xr*(xr2-yr2);
			break;
		case -3:
			Y = sqrt((9.0*35.0)/(16.0*2.0*PI))*yr*zr*(3.0*xr2-yr2);
			break;
		case -2:
			Y = sqrt((9.0*5.0)/(16.0*PI))*yr*xr*(7.0*zr2-(xr2+yr2+zr2));
			break;
		case -1:
			Y = sqrt((9.0*5.0)/(16.0*2.0*PI))*yr*zr*(7.0*zr2-3.0*(xr2+yr2+zr2));
			break;
		case 0:
			Y = sqrt(9.0/(16.0*16.0*PI))*(35.0*zr2*zr2-30.0*zr2+3.0);
			break;
		case 1:
			Y = sqrt((9.0*5.0)/(16.0*2.0*PI))*xr*zr*(7.0*zr2-3.0*(xr2+yr2+zr2));
			break;
		case 2:
			Y = sqrt((9.0*5.0)/(8.0*8.0*PI))*(xr2-yr2)*(7.0*zr2-(xr2+yr2+zr2));
			break;
		case 3:
			Y = sqrt((9.0*35.0)/(16.0*2.0*PI))*xr*zr*(xr2-3.0*yr2);
			break;
		case 4:
			Y = sqrt((9.0*35.0)/(16.0*16.0*PI))*(xr2*(xr2-3.0*yr2)-yr2*(3.0*xr2-yr2));
			break;
		} break;
	case 5:
		switch (m)
		{
		case -5:
			Y = (3.0/16.0)*sqrt(77.0/(2.0*PI))*sinth2*sinth2*sinth*sin(5.0*phi);
			break;
		case -4:
			Y = (3.0/8.0)*sqrt(385.0/(2.0*PI))*sinth2*sinth2*sin(4.0*phi);
			break;
		case -3:
			Y = (1.0/16.0)*sqrt(385.0/(2.0*PI))*sinth2*sinth*(9.0*costh2-1.0)*sin(3.0*phi);
			break;
		case -2:
			Y = (1.0/4.0)*sqrt(1155.0/(4.0*PI))*sinth2*(3.0*costh2*costh-costh)*sin(2.0*phi);
			break;
		case -1:
			Y = (1.0/8.0)*sqrt(165.0/(4.0*PI))*sinth*(21.0*costh2*costh2-14.0*costh2+1)*sin(phi);
			break;
		case 0:
			Y = (1.0/16.0)*sqrt(11.0/PI)*(63.0*costh2*costh2*costh-70.0*costh2*costh+15.0*costh);
			break;
		case 1:
			Y = (1.0/8.0)*sqrt(165.0/(4.0*PI))*sinth*(21.0*costh2*costh2-14.0*costh2+1)*cos(phi);
			break;
		case 2:
			Y = (1.0/4.0)*sqrt(1155.0/(4.0*PI))*sinth2*(3.0*costh2*costh-costh)*cos(2.0*phi);
			break;
		case 3:
			Y = (1.0/16.0)*sqrt(385.0/(2.0*PI))*sinth2*sinth*(9.0*costh2-1.0)*cos(3.0*phi);
			break;
		case 4:
			Y = (3.0/8.0)*sqrt(385.0/(2.0*PI))*sinth2*sinth2*cos(4.0*phi);
			break;
		case 5:
			Y = (3.0/16.0)*sqrt(77.0/(2.0*PI))*sinth2*sinth2*sinth*cos(5.0*phi);
			break;
		}break;
    case 6:
        switch (m)
		{
		case -6:
			Y = -0.6832*sinth*pow(costh2 - 1.0, 3);
			break;
		case -5:
			Y = 2.367*costh*sinth*pow(1.0 - 1.0*costh2, 2.5);
			break;
		case -4:
			Y = 0.001068*sinth*(5198.0*costh2 - 472.5)*pow(costh2 - 1.0, 2);
			break;
		case -3:
			Y = -0.005849*sinth*pow(1.0 - 1.0*costh2, 1.5)*(- 1732.0*costh2*costh + 472.5*costh);
			break;
		case -2:
			Y = -0.03509*sinth*(costh2 - 1.0)*(433.1*costh2*costh2 - 236.2*costh2 + 13.12);
			break;
		case -1:
			Y = 0.222*sinth*pow(1.0 - 1.0*costh2, 0.5)*(86.62*costh2*costh2*costh - 78.75*costh2*costh + 13.12*costh);
			break;
		case 0:
			Y = 14.68*costh2*costh2*costh2 - 20.02*costh2*costh2 + 6.675*costh2 - 0.3178;
			break;
		case 1:
			Y = 0.222*cosph*pow(1.0 - 1.0*costh2, 0.5)*(86.62*costh2*costh2*costh - 78.75*costh2*costh + 13.12*costh);
			break;
		case 2:
			Y = -0.03509*cosph*(costh2 - 1.0)*(433.1*costh2*costh2 - 236.2*costh2 + 13.12);
			break;
		case 3:
			Y = -0.005849*cosph*pow(1.0 - 1.0*costh2, 1.5)*(-1732.0*costh2*costh + 472.5*costh);
			break;
		case 4:
			Y = 0.001068*cosph*(5198.0*costh2 - 472.5)*pow(costh2 - 1.0, 2);
			break;
        case 5:
            Y = 2.367*costh*cosph*pow(1.0 - 1.0*costh2, 2.5);
            break;
        case 6:
            Y = -0.6832*cosph*pow(costh2 - 1.0, 3);
            break;
		}break;
    case 7:
        switch (m)
		{
		case -7:
			Y = 0.7072*sinth*pow(1.0 - 1.0*costh2, 3.5);
			break;
		case -6:
			Y = -2.646*costh*sinth*pow(costh2 - 1.0, 3);
			break;
		case -5:
			Y = 9.984e-5*sinth*pow(1.0 - 1.0*costh2, 2.5)*(67570.0*costh2 - 5198.0);
			break;
		case -4:
			Y = -0.000599*sinth*pow(costh2 - 1.0, 2)*(-22520.0*costh2*costh + 5198.0*costh);
			break;
		case -3:
			Y = 0.003974*sinth*pow(1.0 - 1.0*costh2, 1.5)*(5631.0*costh2*costh2 - 2599.0*costh2 + 118.1);
			break;
		case -2:
			Y = -0.0281*sinth*(costh2 - 1.0)*(1126.0*costh2*costh2*costh - 866.2*costh2*costh + 118.1*costh);
			break;
		case -1:
			Y = 0.2065*sinth*pow(1.0 - 1.0*costh2, 0.5)*(187.7*costh2*costh2*costh2 - 216.6*costh2*costh2 + 59.06*costh2 - 2.188);
			break;
		case 0:
			Y = 29.29*costh2*costh2*costh2*costh - 47.32*costh2*costh2*costh + 21.51*costh2*costh - 2.39*costh;
			break;
		case 1:
			Y = 0.2065*cosph*pow(1.0 - 1.0*costh2, 0.5)*(187.7*costh2*costh2*costh2 - 216.6*costh2*costh2 + 59.06*costh2 - 2.188);
			break;
		case 2:
			Y = -0.0281*cosph*(costh2 - 1.0)*(1126.0*costh2*costh2*costh - 866.2*costh2*costh + 118.1*costh);
			break;
		case 3:
			Y = 0.003974*cosph*pow(1.0 - 1.0*costh2, 1.5)*(5631.0*costh2*costh2 - 2599.0*costh2 + 118.1);
			break;
        case 4:
            Y = -0.000599*cosph*pow(costh2 - 1.0, 2)*(- 22520.0*costh2*costh + 5198.0*costh);
            break;
        case 5:
            Y = 9.984e-5*cosph*pow(1.0 - 1.0*costh2, 2.5)*(67570.0*costh2 - 5198.0);
            break;
        case 6:
            Y = -2.646*cosph*costh*pow(costh2 - 1.0, 3);
            break;
        case 7:
            Y = 0.7072*cosph*pow(1.0 - 1.0*costh2, 3.5);
            break;
		}break;
    case 8:
        switch (m)
		{
		case -8:
			Y = sinth*pow(costh2-1.0,4.0)*7.289266601746931E-1;
			break;
		case -7:
			Y = costh*sinth*pow(costh2*-1.0+1.0,7.0/2.0)*2.915706640698772;
			break;
		case -6:
			Y = sinth*(costh2*1.0135125E+6-6.75675E+4)*pow(costh2-1.0,3.0)*-7.878532816224526E-6;
			break;
		case -5:
			Y = sinth*pow(costh2*-1.0+1.0,5.0/2.0)*(costh*6.75675E+4-(costh2*costh)*3.378375E+5)*-5.105872826582925E-5;
			break;
		case -4:
			Y = sinth*pow(costh2-1.0,2.0)*(costh2*-3.378375E+4+(costh2*costh2)*8.4459375E+4+1.299375E+3)*3.681897256448963E-4;
			break;
		case -3:
			Y = sinth*pow(costh2*-1.0+1.0,3.0/2.0)*(costh*1.299375E+3-(costh*costh2)*1.126125E+4+(costh*costh2*costh2)*1.6891875E+4)*2.851985351334463E-3;
			break;
		case -2:
			Y = sinth*(costh2-1.0)*(costh2*6.496875E+2-(costh2*costh2)*2.8153125E+3+(costh2*costh2*costh2)*2.8153125E+3-1.96875E+1)*-2.316963852365461E-2;
			break;
		case -1:
			Y = sinth*sqrt(costh2*-1.0+1.0)*(costh*1.96875E+1-(costh*costh2)*2.165625E+2+(costh*costh2*costh2)*5.630625E+2-(costh*costh2*costh2*costh2)*4.021875E+2)*-1.938511038201796E-1;;
			break;
		case 0:
			Y = costh2*-1.144933081936324E+1+(costh2*costh2)*6.297131950652692E+1-(costh2*costh2*costh2)*1.091502871445846E+2+(costh2*costh2*costh2*costh2)*5.847336811327841E+1+3.180369672045344E-1;
			break;
		case 1:
			Y = cosph*sqrt(costh2*-1.0+1.0)*(costh*1.96875E+1-(costh*costh2)*2.165625E+2+(costh*costh2*costh2)*5.630625E+2-(costh*costh2*costh2*costh2)*4.021875E+2)*-1.938511038201796E-1;;
			break;
		case 2:
			Y = cosph*(costh2-1.0)*(costh2*6.496875E+2-(costh2*costh2)*2.8153125E+3+(costh2*costh2*costh2)*2.8153125E+3-1.96875E+1)*-2.316963852365461E-2;
			break;
        case 3:
            Y = cosph*pow(costh2*-1.0+1.0,3.0/2.0)*(costh*1.299375E+3-(costh*costh2)*1.126125E+4+(costh*costh2*costh2)*1.6891875E+4)*2.851985351334463E-3;
            break;
        case 4:
            Y = cosph*pow(costh2-1.0,2.0)*(costh2*-3.378375E+4+(costh2*costh2)*8.4459375E+4+1.299375E+3)*3.681897256448963E-4;
            break;
        case 5:
            Y = cosph*pow(costh2*-1.0+1.0,5.0/2.0)*(costh*6.75675E+4-(costh*costh2)*3.378375E+5)*-5.105872826582925E-5;
            break;
        case 6:
            Y = cosph*(costh2*1.0135125E+6-6.75675E+4)*pow(costh2-1.0,3.0)*-7.878532816224526E-6;
            break;
        case 7:
            Y = cosph*costh*pow(costh2*-1.0+1.0,7.0/2.0)*2.915706640698772;
            break;
        case 8:
            Y = cosph*pow(costh2-1.0,4.0)*7.289266601746931E-1;
            break;
		}break;
    case 9:
        switch (m)
		{
		case -9:
			Y = sinth*pow(costh2*-1.0+1.0,9.0/2.0)*7.489009518540115E-1;
			break;
		case -8:
			Y = costh*sinth*pow(costh2-1.0,4.0)*3.17731764895143;
			break;
		case -7:
			Y = sinth*pow(costh2*-1.0+1.0,7.0/2.0)*(costh2*1.72297125E+7-1.0135125E+6)*5.376406125665728E-7;
			break;
		case -6:
			Y = sinth*(costh*1.0135125E+6-(costh*costh2)*5.7432375E+6)*pow(costh2-1.0,3.0)*3.724883428715686E-6;
			break;
		case -5:
			Y = sinth*pow(costh2*-1.0+1.0,5.0/2.0)*(costh2*-5.0675625E+5+(costh2*costh2)*1.435809375E+6+1.6891875E+4)*2.885282297193648E-5;
			break;
		case -4:
			Y = sinth*pow(costh2-1.0,2.0)*(costh*1.6891875E+4-(costh*costh2)*1.6891875E+5+(costh*costh2*costh2)*2.87161875E+5)*2.414000363328839E-4;
			break;
		case -3:
			Y = sinth*pow(costh2*-1.0+1.0,3.0/2.0)*((costh2)*8.4459375E+3-(costh2*costh2)*4.22296875E+4+(costh2*costh2*costh2)*4.78603125E+4-2.165625E+2)*2.131987394015766E-3;
			break;
		case -2:
			Y = sinth*(costh2-1.0)*(costh*2.165625E+2-(costh*costh2)*2.8153125E+3+(costh*costh2*costh2)*8.4459375E+3-(costh*costh2*costh2*costh2)*6.8371875E+3)*1.953998722751749E-2;
			break;
		case -1:
			Y = sinth*sqrt(costh2*-1.0+1.0)*(costh2*-1.0828125E+2+(costh2*costh2)*7.03828125E+2-(costh2*costh2*costh2)*1.40765625E+3+(costh2*costh2*costh2*costh2)*8.546484375E+2+2.4609375)*1.833013280775049E-1;
			break;
		case 0:
			Y = costh*3.026024588281871-(costh*costh2)*4.438169396144804E+1+(costh*costh2*costh2)*1.730886064497754E+2-(costh*costh2*costh2*costh2)*2.472694377852604E+2+(costh*costh2*costh2*costh2*costh2)*1.167661233986728E+2;
			break;
		case 1:
			Y = cosph*sqrt(costh2*-1.0+1.0)*(costh2*-1.0828125E+2+(costh2*costh2)*7.03828125E+2-(costh2*costh2*costh2)*1.40765625E+3+(costh2*costh2*costh2*costh2)*8.546484375E+2+2.4609375)*1.833013280775049E-1;
			break;
        case 2:
            Y = cosph*(costh2-1.0)*(costh*2.165625E+2-(costh*costh2)*2.8153125E+3+(costh*costh2*costh2)*8.4459375E+3-(costh*costh2*costh2*costh2)*6.8371875E+3)*1.953998722751749E-2;
            break;
        case 3:
            Y = cosph*pow(costh2*-1.0+1.0,3.0/2.0)*(costh2*8.4459375E+3-(costh2*costh2)*4.22296875E+4+(costh2*costh2*costh2)*4.78603125E+4-2.165625E+2)*2.131987394015766E-3;
            break;
        case 4:
            Y = cosph*pow(costh2-1.0,2.0)*(costh*1.6891875E+4-(costh*costh2)*1.6891875E+5+(costh*costh2*costh2)*2.87161875E+5)*2.414000363328839E-4;
            break;
        case 5:
            Y = cosph*pow(costh2*-1.0+1.0,5.0/2.0)*(costh2*-5.0675625E+5+(costh2*costh2)*1.435809375E+6+1.6891875E+4)*2.885282297193648E-5;
            break;
        case 6:
            Y = cosph*(costh*1.0135125E+6-(costh*costh2)*5.7432375E+6)*pow(costh2-1.0,3.0)*3.724883428715686E-6;
            break;
        case 7:
            Y = cosph*pow(costh2*-1.0+1.0,7.0/2.0)*(costh2*1.72297125E+7-1.0135125E+6)*5.376406125665728E-7;
            break;
        case 8:
            Y = cosph*costh*pow(costh2-1.0,4.0)*3.17731764895143;
            break;
        case 9:
            Y = cosph*pow(costh2*-1.0+1.0,9.0/2.0)*7.489009518540115E-1;
            break;
		}break;
    case 10:
        switch (m)
		{
        case -10:
            Y = sinth*pow(costh*costh-1.0,5.0)*-7.673951182223391E-1;
            break;
		case -9:
			Y = costh*sinth*pow((costh*costh)*-1.0+1.0,9.0/2.0)*3.431895299894677;
			break;
		case -8:
			Y = sinth*((costh*costh)*3.273645375E+8-1.72297125E+7)*pow(costh*costh-1.0,4.0)*3.231202683857352E-8;
			break;
		case -7:
            Y = sinth*pow((costh*costh)*-1.0+1.0,7.0/2.0)*(costh*1.72297125E+7-(costh*costh*costh)*1.091215125E+8)*-2.374439349284684E-7;
			break;
		case -6:
            Y = sinth*pow(costh*costh-1.0,3.0)*((costh*costh)*-8.61485625E+6+(costh*costh*costh*costh)*2.7280378125E+7+2.53378125E+5)*-1.958012847746993E-6;
			break;
		case -5:
            Y = sinth*pow((costh*costh)*-1.0+1.0,5.0/2.0)*(costh*2.53378125E+5-(costh*costh*costh)*2.87161875E+6+(costh*costh*costh*costh*costh)*5.456075625E+6)*1.751299931351813E-5;
			break;
		case -4:
            Y = sinth*pow(costh*costh-1.0,2.0)*((costh*costh)*1.266890625E+5-(costh*costh*costh*costh)*7.179046875E+5+(costh*costh*costh*costh*costh*costh)*9.093459375E+5-2.8153125E+3)*1.661428994750302E-4;
			break;
		case -3:
            Y = sinth*pow((costh*costh)*-1.0+1.0,3.0/2.0)*(costh*2.8153125E+3-(costh*costh*costh)*4.22296875E+4+(costh*costh*costh*costh*costh)*1.435809375E+5-(costh*costh*costh*costh*costh*costh*costh)*1.299065625E+5)*-1.644730792108362E-3;
			break;
		case -2:
            Y = sinth*(costh*costh-1.0)*((costh*costh)*-1.40765625E+3+(costh*costh*costh*costh)*1.0557421875E+4-(costh*costh*costh*costh*costh*costh)*2.393015625E+4+(costh*costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+2.70703125E+1)*-1.67730288071084E-2;
			break;
		case -1:
            Y = sinth*sqrt((costh*costh)*-1.0+1.0)*(costh*2.70703125E+1-(costh*costh*costh)*4.6921875E+2+(costh*costh*costh*costh*costh)*2.111484375E+3-(costh*costh*costh*costh*costh*costh*costh)*3.41859375E+3+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.8042578125E+3)*1.743104285446861E-1;
			break;
		case 0:
            Y = (costh*costh)*1.749717715557199E+1-(costh*costh*costh*costh)*1.516422020150349E+2+(costh*costh*costh*costh*costh*costh)*4.549266060441732E+2-(costh*costh*costh*costh*costh*costh*costh*costh)*5.524108787681907E+2+pow(costh,1.0E+1)*2.332401488134637E+2-3.181304937370442E-1;
			break;
		case 1:
            Y = cosph*sqrt((costh*costh)*-1.0+1.0)*(costh*2.70703125E+1-(costh*costh*costh)*4.6921875E+2+(costh*costh*costh*costh*costh)*2.111484375E+3-(costh*costh*costh*costh*costh*costh*costh)*3.41859375E+3+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.8042578125E+3)*1.743104285446861E-1;
			break;
        case 2:
            Y = cosph*(costh*costh-1.0)*((costh*costh)*-1.40765625E+3+(costh*costh*costh*costh)*1.0557421875E+4-(costh*costh*costh*costh*costh*costh)*2.393015625E+4+(costh*costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+2.70703125E+1)*-1.67730288071084E-2;
            break;
        case 3:
            Y = cosph*pow((costh*costh)*-1.0+1.0,3.0/2.0)*(costh*2.8153125E+3-(costh*costh*costh)*4.22296875E+4+(costh*costh*costh*costh*costh)*1.435809375E+5-(costh*costh*costh*costh*costh*costh*costh)*1.299065625E+5)*-1.644730792108362E-3;
            break;
        case 4:
            Y = cosph*pow(costh*costh-1.0,2.0)*((costh*costh)*1.266890625E+5-(costh*costh*costh*costh)*7.179046875E+5+(costh*costh*costh*costh*costh*costh)*9.093459375E+5-2.8153125E+3)*1.661428994750302E-4;
            break;
        case 5:
            Y = cosph*pow((costh*costh)*-1.0+1.0,5.0/2.0)*(costh*2.53378125E+5-(costh*costh*costh)*2.87161875E+6+(costh*costh*costh*costh*costh)*5.456075625E+6)*1.751299931351813E-5;
            break;
        case 6:
            Y = cosph*pow(costh*costh-1.0,3.0)*((costh*costh)*-8.61485625E+6+(costh*costh*costh*costh)*2.7280378125E+7+2.53378125E+5)*-1.958012847746993E-6;
            break;
        case 7:
            Y = cosph*pow((costh*costh)*-1.0+1.0,7.0/2.0)*(costh*1.72297125E+7-(costh*costh*costh)*1.091215125E+8)*-2.374439349284684E-7;
            break;
        case 8:
            Y = cosph*((costh*costh)*3.273645375E+8-1.72297125E+7)*pow(costh*costh-1.0,4.0)*3.231202683857352E-8;
            break;
        case 9:
            Y = cosph*costh*pow((costh*costh)*-1.0+1.0,9.0/2.0)*3.431895299894677;
            break;
        case 10:
            Y = cosph*pow(costh*costh-1.0,5.0)*-7.673951182223391E-1;
            break;
		}break;
    case 11:
        switch (m)
		{
        case -11:
            Y = sinth*pow((costh*costh)*-1.0+1.0,1.1E+1/2.0)*7.846421057874977E-1;
            break;
        case -10:
            Y = costh*sinth*pow(costh*costh-1.0,5.0)*-3.68029769880377;
            break;
		case -9:
            Y = sinth*pow((costh*costh)*-1.0+1.0,9.0/2.0)*((costh*costh)*6.8746552875E+9-3.273645375E+8)*1.734709165873547E-9;
			break;
		case -8:
            Y = sinth*pow(costh*costh-1.0,4.0)*(costh*3.273645375E+8-(costh*costh*costh)*2.2915517625E+9)*-1.343699941990114E-8;
			break;
		case -7:
            Y = sinth*pow((costh*costh)*-1.0+1.0,7.0/2.0)*((costh*costh)*-1.6368226875E+8+(costh*costh*costh*costh)*5.72887940625E+8+4.307428125E+6)*1.171410451514688E-7;
			break;
		case -6:
            Y = sinth*pow(costh*costh-1.0,3.0)*(costh*4.307428125E+6-(costh*costh*costh)*5.456075625E+7+(costh*costh*costh*costh*costh)*1.14577588125E+8)*-1.111297530512201E-6;
			break;
		case -5:
            Y = sinth*pow((costh*costh)*-1.0+1.0,5.0/2.0)*((costh*costh)*2.1537140625E+6-(costh*costh*costh*costh)*1.36401890625E+7+(costh*costh*costh*costh*costh*costh)*1.90962646875E+7-4.22296875E+4)*1.122355489741045E-5;
			break;
		case -4:
            Y = sinth*pow(costh*costh-1.0,2.0)*(costh*4.22296875E+4-(costh*costh*costh)*7.179046875E+5+(costh*costh*costh*costh*costh)*2.7280378125E+6-(costh*costh*costh*costh*costh*costh*costh)*2.7280378125E+6)*-1.187789403385153E-4;
			break;
		case -3:
            Y = sinth*pow((costh*costh)*-1.0+1.0,3.0/2.0)*((costh*costh)*-2.111484375E+4+(costh*costh*costh*costh)*1.79476171875E+5-(costh*costh*costh*costh*costh*costh)*4.5467296875E+5+(costh*costh*costh*costh*costh*costh*costh*costh)*3.410047265625E+5+3.519140625E+2)*1.301158099600741E-3;
			break;
		case -2:
            Y = sinth*(costh*costh-1.0)*(costh*3.519140625E+2-(costh*costh*costh)*7.03828125E+3+(costh*costh*costh*costh*costh)*3.5895234375E+4-(costh*costh*costh*costh*costh*costh*costh)*6.495328125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*3.78894140625E+4)*-1.46054634441839E-2;
			break;
		case -1:
            Y = sinth*sqrt((costh*costh)*-1.0+1.0)*((costh*costh)*1.7595703125E+2-(costh*costh*costh*costh)*1.7595703125E+3+(costh*costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh*costh)*8.11916015625E+3+pow(costh,1.0E+1)*3.78894140625E+3-2.70703125)*1.665279049125274E-1;
			break;
		case 0:
            Y = costh*-3.662285987506039+(costh*costh*costh)*7.934952972922474E+1-(costh*costh*costh*costh*costh)*4.760971783753484E+2+(costh*costh*costh*costh*costh*costh*costh)*1.156236004628241E+3-(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.220471338216215E+3+pow(costh,1.1E+1)*4.65998147319071E+2;
			break;
		case 1:
            Y = cosph*sqrt((costh*costh)*-1.0+1.0)*((costh*costh)*1.7595703125E+2-(costh*costh*costh*costh)*1.7595703125E+3+(costh*costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh*costh)*8.11916015625E+3+pow(costh,1.0E+1)*3.78894140625E+3-2.70703125)*1.665279049125274E-1;
			break;
        case 2:
            Y = cosph*(costh*costh-1.0)*(costh*3.519140625E+2-(costh*costh*costh)*7.03828125E+3+(costh*costh*costh*costh*costh)*3.5895234375E+4-(costh*costh*costh*costh*costh*costh*costh)*6.495328125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*3.78894140625E+4)*-1.46054634441839E-2;
            break;
        case 3:
            Y = cosph*pow((costh*costh)*-1.0+1.0,3.0/2.0)*((costh*costh)*-2.111484375E+4+(costh*costh*costh*costh)*1.79476171875E+5-(costh*costh*costh*costh*costh*costh)*4.5467296875E+5+(costh*costh*costh*costh*costh*costh*costh*costh)*3.410047265625E+5+3.519140625E+2)*1.301158099600741E-3;
            break;
        case 4:
            Y = cosph*pow(costh*costh-1.0,2.0)*(costh*4.22296875E+4-(costh*costh*costh)*7.179046875E+5+(costh*costh*costh*costh*costh)*2.7280378125E+6-(costh*costh*costh*costh*costh*costh*costh)*2.7280378125E+6)*-1.187789403385153E-4;
            break;
        case 5:
            Y = cosph*pow((costh*costh)*-1.0+1.0,5.0/2.0)*((costh*costh)*2.1537140625E+6-(costh*costh*costh*costh)*1.36401890625E+7+(costh*costh*costh*costh*costh*costh)*1.90962646875E+7-4.22296875E+4)*1.122355489741045E-5;
            break;
        case 6:
            Y = cosph*pow(costh*costh-1.0,3.0)*(costh*4.307428125E+6-(costh*costh*costh)*5.456075625E+7+(costh*costh*costh*costh*costh)*1.14577588125E+8)*-1.111297530512201E-6;
            break;
        case 7:
            Y = cosph*pow((costh*costh)*-1.0+1.0,7.0/2.0)*((costh*costh)*-1.6368226875E+8+(costh*costh*costh*costh)*5.72887940625E+8+4.307428125E+6)*1.171410451514688E-7;
            break;
        case 8:
            Y = cosph*pow(costh*costh-1.0,4.0)*(costh*3.273645375E+8-(costh*costh*costh)*2.2915517625E+9)*-1.343699941990114E-8;
            break;
        case 9:
            Y = cosph*pow((costh*costh)*-1.0+1.0,9.0/2.0)*((costh*costh)*6.8746552875E+9-3.273645375E+8)*1.734709165873547E-9;
            break;
        case 10:
            Y = cosph*costh*pow(costh*costh-1.0,5.0)*-3.68029769880377;
            break;
        case 11:
            Y = cosph*pow((costh*costh)*-1.0+1.0,1.1E+1/2.0)*7.846421057874977E-1;
            break;
		}break;
    case 12:
        switch (m)
		{
        case -12:
            Y = sinth*pow(costh*costh-1.0,6.0)*8.00821995784645E-1;
            break;
        case -11:
            Y = costh*sinth*pow((costh*costh)*-1.0+1.0,1.1E+1/2.0)*3.923210528933851;
            break;
        case -10:
            Y = sinth*((costh*costh)*1.581170716125E+11-6.8746552875E+9)*pow(costh*costh-1.0,5.0)*-8.414179483959553E-11;
            break;
		case -9:
            Y = sinth*pow((costh*costh)*-1.0+1.0,9.0/2.0)*(costh*6.8746552875E+9-(costh*costh*costh)*5.27056905375E+10)*-6.83571172712202E-10;
			break;
		case -8:
            Y = sinth*pow(costh*costh-1.0,4.0)*((costh*costh)*-3.43732764375E+9+(costh*costh*costh*costh)*1.3176422634375E+10+8.1841134375E+7)*6.265033283689913E-9;
			break;
		case -7:
            Y = sinth*pow((costh*costh)*-1.0+1.0,7.0/2.0)*(costh*8.1841134375E+7-(costh*costh*costh)*1.14577588125E+9+(costh*costh*costh*costh*costh)*2.635284526875E+9)*6.26503328367365E-8;
			break;
		case -6:
            Y = sinth*pow(costh*costh-1.0,3.0)*((costh*costh)*4.09205671875E+7-(costh*costh*costh*costh)*2.864439703125E+8+(costh*costh*costh*costh*costh*costh)*4.392140878125E+8-7.179046875E+5)*-6.689225062143228E-7;
			break;
		case -5:
            Y = sinth*pow((costh*costh)*-1.0+1.0,5.0/2.0)*(costh*7.179046875E+5-(costh*costh*costh)*1.36401890625E+7+(costh*costh*costh*costh*costh)*5.72887940625E+7-(costh*costh*costh*costh*costh*costh*costh)*6.27448696875E+7)*-7.50863650966771E-6;
			break;
		case -4:
            Y = sinth*pow(costh*costh-1.0,2.0)*((costh*costh)*-3.5895234375E+5+(costh*costh*costh*costh)*3.410047265625E+6-(costh*costh*costh*costh*costh*costh)*9.54813234375E+6+(costh*costh*costh*costh*costh*costh*costh*costh)*7.8431087109375E+6+5.2787109375E+3)*8.756499656747962E-5;
			break;
		case -3:
            Y = sinth*pow((costh*costh)*-1.0+1.0,3.0/2.0)*(costh*5.2787109375E+3-(costh*costh*costh)*1.1965078125E+5+(costh*costh*costh*costh*costh)*6.82009453125E+5-(costh*costh*costh*costh*costh*costh*costh)*1.36401890625E+6+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*8.714565234375E+5)*1.050779958809755E-3;
			break;
		case -2:
            Y = sinth*(costh*costh-1.0)*((costh*costh)*2.63935546875E+3-(costh*costh*costh*costh)*2.99126953125E+4+(costh*costh*costh*costh*costh*costh)*1.136682421875E+5-(costh*costh*costh*costh*costh*costh*costh*costh)*1.7050236328125E+5+pow(costh,1.0E+1)*8.714565234375E+4-3.519140625E+1)*-1.286937365514973E-2;
			break;
		case -1:
            Y = sinth*sqrt((costh*costh)*-1.0+1.0)*(costh*3.519140625E+1-(costh*costh*costh)*8.7978515625E+2+(costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.894470703125E+4-pow(costh,1.1E+1)*7.92233203125E+3)*-1.597047270888652E-1;
			break;
		case 0:
            Y = (costh*costh)*-2.481828104582382E+1+(costh*costh*costh*costh)*3.102285130722448E+2-(costh*costh*costh*costh*costh*costh)*1.40636925926432E+3+(costh*costh*costh*costh*costh*costh*costh*costh)*2.862965992070735E+3-pow(costh,1.0E+1)*2.672101592600346E+3+pow(costh,1.2E+1)*9.311869186330587E+2+3.181830903313312E-1;
			break;
		case 1:
            Y = cosph*sqrt((costh*costh)*-1.0+1.0)*(costh*3.519140625E+1-(costh*costh*costh)*8.7978515625E+2+(costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.894470703125E+4-pow(costh,1.1E+1)*7.92233203125E+3)*-1.597047270888652E-1;
			break;
        case 2:
            Y = cosph*(costh*costh-1.0)*((costh*costh)*2.63935546875E+3-(costh*costh*costh*costh)*2.99126953125E+4+(costh*costh*costh*costh*costh*costh)*1.136682421875E+5-(costh*costh*costh*costh*costh*costh*costh*costh)*1.7050236328125E+5+pow(costh,1.0E+1)*8.714565234375E+4-3.519140625E+1)*-1.286937365514973E-2;
            break;
        case 3:
            Y = cosph*pow((costh*costh)*-1.0+1.0,3.0/2.0)*(costh*5.2787109375E+3-(costh*costh*costh)*1.1965078125E+5+(costh*costh*costh*costh*costh)*6.82009453125E+5-(costh*costh*costh*costh*costh*costh*costh)*1.36401890625E+6+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*8.714565234375E+5)*1.050779958809755E-3;
            break;
        case 4:
            Y = cosph*pow(costh*costh-1.0,2.0)*((costh*costh)*-3.5895234375E+5+(costh*costh*costh*costh)*3.410047265625E+6-(costh*costh*costh*costh*costh*costh)*9.54813234375E+6+(costh*costh*costh*costh*costh*costh*costh*costh)*7.8431087109375E+6+5.2787109375E+3)*8.756499656747962E-5;
            break;
        case 5:
            Y = cosph*pow((costh*costh)*-1.0+1.0,5.0/2.0)*(costh*7.179046875E+5-(costh*costh*costh)*1.36401890625E+7+(costh*costh*costh*costh*costh)*5.72887940625E+7-(costh*costh*costh*costh*costh*costh*costh)*6.27448696875E+7)*-7.50863650966771E-6;
            break;
        case 6:
            Y = cosph*pow(costh*costh-1.0,3.0)*((costh*costh)*4.09205671875E+7-(costh*costh*costh*costh)*2.864439703125E+8+(costh*costh*costh*costh*costh*costh)*4.392140878125E+8-7.179046875E+5)*-6.689225062143228E-7;
            break;
        case 7:
            Y = cosph*pow((costh*costh)*-1.0+1.0,7.0/2.0)*(costh*8.1841134375E+7-(costh*costh*costh)*1.14577588125E+9+(costh*costh*costh*costh*costh)*2.635284526875E+9)*6.26503328367365E-8;
            break;
        case 8:
            Y = cosph*pow(costh*costh-1.0,4.0)*((costh*costh)*-3.43732764375E+9+(costh*costh*costh*costh)*1.3176422634375E+10+8.1841134375E+7)*6.265033283689913E-9;
            break;
        case 9:
            Y = cosph*pow((costh*costh)*-1.0+1.0,9.0/2.0)*(costh*6.8746552875E+9-(costh*costh*costh)*5.27056905375E+10)*-6.83571172712202E-10;
            break;
        case 10:
            Y = cosph*((costh*costh)*1.581170716125E+11-6.8746552875E+9)*pow(costh*costh-1.0,5.0)*-8.414179483959553E-11;
            break;
        case 11:
            Y = cosph*costh*pow((costh*costh)*-1.0+1.0,1.1E+1/2.0)*3.923210528933851;
            break;
        case 12:
            Y = cosph*pow(costh*costh-1.0,6.0)*8.00821995784645E-1;
            break;
		}break;
    default: break;
	}

	return R*Y;
}

double ZernikeSphericalHarmonics(int l1, int n, int l2, int m, double xr, double yr, double zr, double r) {
    switch (l1) {
        case 0 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<0, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<0, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<0, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<0, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<0, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<0, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<0, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<0, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<0, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<0, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<0, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<0, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<0, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 1 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<1, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<1, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<1, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<1, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<1, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<1, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<1, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<1, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<1, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<1, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<1, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<1, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<1, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 2 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<2, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<2, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<2, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<2, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<2, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<2, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<2, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<2, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<2, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<2, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<2, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<2, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<2, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 3 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<3, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<3, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<3, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<3, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<3, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<3, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<3, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<3, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<3, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<3, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<3, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<3, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<3, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 4 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<4, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<4, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<4, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<4, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<4, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<4, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<4, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<4, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<4, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<4, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<4, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<4, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<4, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 5 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<5, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<5, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<5, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<5, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<5, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<5, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<5, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<5, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<5, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<5, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<5, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<5, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<5, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 6 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<6, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<6, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<6, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<6, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<6, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<6, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<6, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<6, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<6, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<6, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<6, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<6, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<6, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 7 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<7, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<7, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<7, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<7, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<7, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<7, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<7, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<7, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<7, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<7, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<7, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<7, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<7, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 8 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<8, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<8, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<8, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<8, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<8, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<8, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<8, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<8, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<8, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<8, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<8, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<8, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<8, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 9 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<9, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<9, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<9, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<9, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<9, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<9, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<9, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<9, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<9, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<9, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<9, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<9, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<9, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 10 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<10, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<10, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<10, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<10, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<10, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<10, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<10, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<10, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<10, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<10, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<10, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<10, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<10, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 11 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<11, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<11, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<11, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<11, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<11, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<11, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<11, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<11, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<11, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<11, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<11, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<11, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<11, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 12 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<12, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<12, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<12, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<12, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<12, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<12, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<12, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<12, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<12, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<12, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<12, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<12, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<12, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 13 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<13, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<13, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<13, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<13, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<13, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<13, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<13, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<13, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<13, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<13, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<13, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<13, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<13, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 14 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<14, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<14, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<14, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<14, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<14, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<14, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<14, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<14, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<14, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<14, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<14, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<14, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<14, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        case 15 : {
            switch (l2) {
            case 0: return ZernikeSphericalHarmonics<15, 0>(n, m, xr, yr, zr, r);
            case 1: return ZernikeSphericalHarmonics<15, 1>(n, m, xr, yr, zr, r);
            case 2: return ZernikeSphericalHarmonics<15, 2>(n, m, xr, yr, zr, r);
            case 3: return ZernikeSphericalHarmonics<15, 3>(n, m, xr, yr, zr, r);
            case 4: return ZernikeSphericalHarmonics<15, 4>(n, m, xr, yr, zr, r);
            case 5: return ZernikeSphericalHarmonics<15, 5>(n, m, xr, yr, zr, r);
            case 6: return ZernikeSphericalHarmonics<15, 6>(n, m, xr, yr, zr, r);
            case 7: return ZernikeSphericalHarmonics<15, 7>(n, m, xr, yr, zr, r);
            case 8: return ZernikeSphericalHarmonics<15, 8>(n, m, xr, yr, zr, r);
            case 9: return ZernikeSphericalHarmonics<15, 9>(n, m, xr, yr, zr, r);
            case 10: return ZernikeSphericalHarmonics<15, 10>(n, m, xr, yr, zr, r);
            case 11: return ZernikeSphericalHarmonics<15, 11>(n, m, xr, yr, zr, r);
            case 12: return ZernikeSphericalHarmonics<15, 12>(n, m, xr, yr, zr, r);
            default: break;
            }
        } break;
        default: break;
    }
    REPORT_ERROR(ERR_ARG_INCORRECT, "ZernikeSphericalHarmonics not supported for l1 = " + std::to_string(l1) + " and l2 = " + std::to_string(l2));
}

#ifdef NEVERDEFINED
double ALegendreSphericalHarmonics(int l, int m, double xr, double yr, double zr, double r)
{
	// General variables
	double rp=2.0*r-1.0;
	double rp2=rp*rp,xr2=xr*xr,yr2=yr*yr,zr2=zr*zr;
	double pol=sqrt(1.0-rp2);
	double pol2=pol*pol;

	// Associated Legendre polynomial
	double R=0.0;

	switch (l)
	{
	case 0:
		R=1.0/2.0;
		break;
	case 1:
		switch (m)
		{
		case -1:
			R=(1.0/4.0)*pol;
			break;
		case 0:
			R=(1.0/2.0)*pol;
			break;
		case 1:
			R=-(1.0/2.0)*pol;
			break;
		}break;
	case 2:
		switch (m)
		{
		case -2:
			R=(1.0/(2.0*24.0))*3.0*pol2;
			break;
		case -1:
			R=(1.0/12.0)*rp*pol;
			break;
		case 0:
			R=(1.0/4.0)*(3*rp2-1.0);
			break;
		case 1:
			R=-3.0*rp*pol;
			break;
		case 2:
			R=3.0*pol2;
		}break;
	case 3:
		switch (m)
		{
		case -3:
			R=(1.0/(2.0*720.0))*15.0*pol2*pol;
			break;
		case -2:
			R=(1.0/240.0)*15.0*rp*pol2;
			break;
		case -1:
			R=(1.0/24.0)*(3.0/2.0)*(5.0*rp2-1.0)*pol;
			break;
		case 0:
			R=(1.0/4.0)*(5.0*rp2*rp-3.0*rp);
			break;
		case 1:
			R=-(3.0/4.0)*(5.0*rp2-1.0)*pol;
			break;
		case 2:
			R=(15.0/2.0)*rp*pol2;
			break;
		case 3:
			R=-15.0*pol2*pol;
			break;
		}break;
	case 4:
		switch (m)
		{
		case -4:
			R=(1.0/(2.0*40320.0))*105.0*pol2*pol2;
			break;
		case -3:
			R=(1.0/(5040.0*2.0))*105.0*rp*pol2*pol;
			break;
		case -2:
			R=(1.0/(2.0*360.0))*(15.0/2.0)*(7.0*rp2-1.0)*pol2;
			break;
		case -1:
			R=(1.0/40.0)*(5.0/2.0)*(7.0*rp2*rp-3.0*rp)*pol;
			break;
		case 0:
			R=(1.0/16.0)*(35.0*rp2*rp2-30.0*rp2+3.0);
			break;
		case 1:
			R=-(5.0/4.0)*(7.0*rp2*rp-3*rp)*pol;
			break;
		case 2:
			R=(15.0/4.0)*(7.0*rp2-1)*pol2;
			break;
		case 3:
			R=-(105.0/2.0)*rp*pol2*pol;
			break;
		case 4:
			R=(105.0/2.0)*pol2*pol2;
		}break;
	}

	// Spherical harmonic
	double Y=0.0;

	switch (l)
	{
	case 0:
		Y = (1.0/2.0)*sqrt(1.0/PI);
		break;
	case 1:
		switch (m)
		{
		case -1:
			Y = sqrt(3.0/(4.0*PI))*yr;
			break;
		case 0:
			Y = sqrt(3.0/(4.0*PI))*zr;
			break;
		case 1:
			Y = sqrt(3.0/(4.0*PI))*xr;
			break;
		} break;
	case 2:
		switch (m)
		{
		case -2:
			Y = sqrt(15.0/(4.0*PI))*xr*yr;
			break;
		case -1:
			Y = sqrt(15.0/(4.0*PI))*zr*yr;
			break;
		case 0:
			Y = sqrt(5.0/(16.0*PI))*(-xr2-yr2+2.0*zr2);
			break;
		case 1:
			Y = sqrt(15.0/(4.0*PI))*xr*zr;
			break;
		case 2:
			Y = sqrt(15.0/(16.0*PI))*(xr2-yr2);
			break;
		} break;
	case 3:
		switch (m)
		{
		case -3:
			Y = sqrt(35.0/(16.0*2.0*PI))*yr*(3.0*xr2-yr2);
			break;
		case -2:
			Y = sqrt(105.0/(4.0*PI))*zr*yr*xr;
			break;
		case -1:
			Y = sqrt(21.0/(16.0*2.0*PI))*yr*(4.0*zr2-xr2-yr2);
			break;
		case 0:
			Y = sqrt(7.0/(16.0*PI))*zr*(2.0*zr2-3.0*xr2-3.0*yr2);
			break;
		case 1:
			Y = sqrt(21.0/(16.0*2.0*PI))*xr*(4.0*zr2-xr2-yr2);
			break;
		case 2:
			Y = sqrt(105.0/(16.0*PI))*zr*(xr2-yr2);
			break;
		case 3:
			Y = sqrt(35.0/(16.0*2.0*PI))*xr*(xr2-3.0*yr2);
			break;
		} break;
	case 4:
		switch (m)
		{
		case -4:
			Y = sqrt((35.0*9.0)/(16.0*PI))*yr*xr*(xr2-yr2);
			break;
		case -3:
			Y = sqrt((9.0*35.0)/(16.0*2.0*PI))*yr*zr*(3.0*xr2-yr2);
			break;
		case -2:
			Y = sqrt((9.0*5.0)/(16.0*PI))*yr*xr*(7.0*zr2-(xr2+yr2+zr2));
			break;
		case -1:
			Y = sqrt((9.0*5.0)/(16.0*2.0*PI))*yr*zr*(7.0*zr2-3.0*(xr2+yr2+zr2));
			break;
		case 0:
			Y = sqrt(9.0/(16.0*16.0*PI))*(35.0*zr2*zr2-30.0*zr2+3.0);
			break;
		case 1:
			Y = sqrt((9.0*5.0)/(16.0*2.0*PI))*xr*zr*(7.0*zr2-3.0*(xr2+yr2+zr2));
			break;
		case 2:
			Y = sqrt((9.0*5.0)/(8.0*8.0*PI))*(xr2-yr2)*(7.0*zr2-(xr2+yr2+zr2));
			break;
		case 3:
			Y = sqrt((9.0*35.0)/(16.0*2.0*PI))*xr*zr*(xr2-3.0*yr2);
			break;
		case 4:
			Y = sqrt((9.0*35.0)/(16.0*16.0*PI))*(xr2*(xr2-3.0*yr2)-yr2*(3.0*xr2-yr2));
			break;
		} break;
	}

	return R*Y;
}
#endif

void spherical_index2lnm(int idx, int &l1, int &n, int &l2, int &m, int max_l1)
{
    auto numR = static_cast<int>(std::floor((4+4*max_l1+std::pow(max_l1,2))/4));
    double aux_id = std::floor(idx-(idx/numR)*numR);
    l1 = static_cast<int>(std::floor((1.0+std::sqrt(1.0+4.0*aux_id))/2.0) + std::floor((2.0+2.0*std::sqrt(aux_id))/2.0) - 2.0);
    n = static_cast<int>(std::ceil((4.0*aux_id - l1*(l1+2.0))/2.0));
    l2 = static_cast<int>(std::floor(std::sqrt(std::floor(idx/numR))));
    m = static_cast<int>(std::floor(idx/numR)-l2*(l2+1));
}

int spherical_lnm2index(int l1, int n, int l2, int m, int max_l1)
{
    auto numR = static_cast<int>(std::floor((4+4*max_l1+std::pow(max_l1,2))/4));
    int id_SH = l2*(l2+1)+m;
    auto id_R = static_cast<int>(std::floor((2*n + l1*(l1 + 2))/4));
    int id_Z = id_SH*numR+id_R;
    return id_Z;
}

template<typename T>
void solveBySVD(const Matrix2D< T >& A, const Matrix1D< T >& b,
                  Matrix1D< double >& result, double tolerance)
{
    if (A.Xdim() == 0)
        REPORT_ERROR(ERR_MATRIX_EMPTY, "Solve: Matrix is empty");

    if (A.Xdim() != A.Ydim())
        REPORT_ERROR(ERR_MATRIX_SIZE, "Solve: Matrix is not squared");

    if (A.Xdim() != b.size())
        REPORT_ERROR(ERR_MATRIX_SIZE, "Solve: Different sizes of Matrix and Vector");

    if (b.isRow())
        REPORT_ERROR(ERR_MATRIX_DIM, "Solve: Not correct vector shape");

    // First perform de single value decomposition
    // Xmipp interface that calls to svdcmp of numerical recipes
    Matrix2D< double > u;
    Matrix2D< double > v;
    Matrix1D< double > w;
    svdcmp(A, u, w, v);

    // Here is checked if eigenvalues of the svd decomposition are acceptable
    // If a value is lower than tolerance, the it's zeroed, as this increases
    // the precision of the routine.
    for (int i = 0; i < w.size(); i++)
        if (w(i) < tolerance)
            w(i) = 0;

    // Set size of matrices
    result.resize(b.size());

    // Xmipp interface that calls to svdksb of numerical recipes
    Matrix1D< double > bd;
    typeCast(b, bd);
    svbksb(u, w, v, bd, result);
}

template<typename T>
void solve(const Matrix2D<T>& A, const Matrix2D<T>& b, Matrix2D<T>& result)
{
    if (A.Xdim() == 0)
        REPORT_ERROR(ERR_MATRIX_EMPTY, "Solve: Matrix is empty");

    if (A.Xdim() != A.Ydim())
        REPORT_ERROR(ERR_MATRIX_SIZE, "Solve: Matrix is not squared");

    if (A.Ydim() != b.Ydim())
        REPORT_ERROR(ERR_MATRIX_SIZE, "Solve: Different sizes of A and b");

    // Solve
    result = b;
    Matrix2D<T> Aux = A;
    gaussj(Aux.adaptForNumericalRecipes2(), Aux.Ydim(),
           result.adaptForNumericalRecipes2(), b.Xdim());
}

// explicit instantiation
template void solveBySVD<double>(Matrix2D<double> const&, Matrix1D<double> const&, Matrix1D<double>&, double);
template void solve<double>(Matrix2D<double> const&, Matrix2D<double> const&, Matrix2D<double>&);
