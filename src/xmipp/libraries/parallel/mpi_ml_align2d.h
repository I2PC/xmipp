/***************************************************************************
 * Authors:     J.M. de la Rosa Trevin (jmdelarosa@cnb.csic.es)
 *
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

#ifndef MPI_ML_ALIGN2D_H_
#define MPI_ML_ALIGN2D_H_

#include "parallel/xmipp_mpi.h"
#include "reconstruction/ml_align2d.h"
#include "reconstruction/mlf_align2d.h"


/**@defgroup MPI_ML MPI_ML
   @ingroup Programs */
//@{

/** Class to organize some useful MPI-functions for ML programs
 * It will also serve as base for those programs*/
template<typename T>
class MpiML2DBase
{
protected:
    MpiNode *node;
    bool created_node;
    //Reference to the program to be parallelized
    T* program;

public:
    /** Read arguments sequentially to avoid concurrency problems */
    void readMpi(int argc, char **argv) {
        if (node == nullptr)
        {
            node = new MpiNode(argc, argv);
            created_node = true;
        }
        //The following makes the asumption that 'this' also
        //inherits from an XmippProgram
        if (!node->isMaster())
            program->verbose = 0;
        // Read subsequently to avoid problems in restart procedure
        for (size_t proc = 0; proc < node->size; ++proc)
        {
            if (proc == node->rank)
                program->read(argc, (const char **)argv);
            node->barrierWait();
        }
        //Send "master" seed to slaves for same randomization
        MPI_Bcast(&program->seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    /** Default constructor */
    MpiML2DBase(T *prm);
    /** Constructor passing the MpiNode */
    MpiML2DBase(T *prm, MpiNode * node);

    MpiML2DBase(const MpiML2DBase &)=delete;
    MpiML2DBase(const MpiML2DBase &&)=delete;

    /** Destructor */
    ~MpiML2DBase() {
        if (created_node)
            delete node;
    }

    MpiML2DBase & operator =(const MpiML2DBase &)=delete;
    MpiML2DBase & operator =(const MpiML2DBase &&)=delete;

    /** This function is only valid for 2D ML programs*/
    void sendDocfile(const MultidimArray<double> &data);
}
;//end of class MpiML

/** Class to parallelize the ML 2D alignment program */
class MpiProgML2D: public ProgML2D, public MpiML2DBase<MpiProgML2D>
{

public:
    /** Default constructor */
    MpiProgML2D();
    /** Constructor passing the MpiNode */
    MpiProgML2D(MpiNode * node);
    /** Only take a part of images for process */
    void setNumberOfLocalImages();
    /** All mpi nodes should syncronize at this point
     * that's why the need of override the implementation.
     */
    void produceSideInfo2();
    /// Write model parameters
    void writeOutputFiles(const ModelML2D &model, OutputType outputType = OUT_FINAL);
    /// After normal ML2D expectation, data must be collected from nodes
    void expectation();
    /// Redefine endIteration for some synchronization
    void endIteration();
    //Just for debugging
    void printModel(const String &msg, const ModelML2D & model);
    //Redefine usage, only master should print
    virtual void usage(int verb = 0) const;

}
;//end of class MpiProgML2D


/** Class to parallelize the MLF 2D alignment program */
class MpiProgMLF2D: public ProgMLF2D, public MpiML2DBase<MpiProgMLF2D>
{
public:
    /** Default constructor */
    MpiProgMLF2D();
    /** Constructor passing the MpiNode */
    MpiProgMLF2D(MpiNode * node);
    /** All mpi nodes should syncronize at this point
     * that's why the need of override the implementation.
     */
    void produceSideInfo();
    void produceSideInfo2();
    /// Write model parameters
    void writeOutputFiles(const ModelML2D &model, OutputType outputType = OUT_FINAL);
    /// After normal ML2D expectation, data must be collected from nodes
    void expectation();
    /// Redefine endIteration for some synchronization
    void endIteration();
}
;//end of class MpiProgMLF2D

/** @} */
#endif /* MPI_ML_ALIGN2D_H_ */
