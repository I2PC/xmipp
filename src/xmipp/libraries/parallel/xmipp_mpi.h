 /*
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

#ifndef XMIPP_MPI_H_
#define XMIPP_MPI_H_

#include <mpi.h>
#include "core/xmipp_threads.h"
#include "core/xmipp_program.h"
#include "core/metadata_vec.h"
#include "core/metadata_db.h"

class FileName;

#define XMIPP_MPI_SIZE_T MPI_UNSIGNED_LONG

/** @defgroup MPI MPI
 *  @ingroup ParallelLibrary
 * @{
 */

/** Class to wrapp some MPI common calls in an work node.
*
*/
class MpiNode
{
public:

    //MPI_Comm *comm;
    size_t rank, size, active;//, activeNodes;
    MpiNode(int &argc, char **& argv);
    ~MpiNode();
    MpiNode(const MpiNode &)=delete;
    MpiNode & operator =(const MpiNode &)=delete;

    /** Check if the node is master */
    bool isMaster() const;

    /** Wait on a barrier for the other MPI nodes */
    void barrierWait();

    /** Gather metadatas */
    template <typename T> // T = MetaData*
    void gatherMetadatas(T &MD, const FileName &rootName);

    /** Update the MPI communicator to connect the currently active nodes */
//    void updateComm();

protected:
    /** Calculate the number of still active nodes */
    size_t getActiveNodes();

};

//mpi macros
#define TAG_WORK   0
#define TAG_STOP   1
#define TAG_WAIT   2

#define TAG_WORK_REQUEST 100
#define TAG_WORK_RESPONSE 101

/** This class is another implementation of ParallelTaskDistributor with MPI workers.
 * It extends from ThreadTaskDistributor and adds the MPI call
 * for making the distribution and extra locking mechanisms among
 * MPI nodes.
 */
class MpiTaskDistributor: public ThreadTaskDistributor
{
protected:
	std::shared_ptr<MpiNode> node;

    virtual bool distribute(size_t &first, size_t &last);

public:
    MpiTaskDistributor(size_t nTasks, size_t bSize, const std::shared_ptr<MpiNode> &node);
    /** All nodes wait until distribution is done.
     * In particular, the master node should wait for the distribution thread.
     */
    void wait();

private:
    /** Method that should be called in the master only.
     * It will listen for job requests from nodes, assign tasks and
     * sent the response back
     */
    bool distributeMaster();
    /** Workers should ask for jobs from master. */
    bool distributeSlaves(size_t &first, size_t &last);
}
;//end of class MpiTaskDistributor

/** Mutex on files.
 * This class extends threads mutex to also provide file locking.
 */
class MpiFileMutex: public Mutex
{
protected:
	std::shared_ptr<MpiNode> node;
    int lockFile;
    bool fileCreator;
public:
    /** Default constructor. */
    MpiFileMutex(const std::shared_ptr<MpiNode> &node);

    /** Destructor. */
    ~MpiFileMutex();

    /** Function to get the access to the mutex.
     * If the some thread has the mutex and other
     * ask to lock will be waiting until the first one
     * release the mutex
     */
    void lock();

    /** Function to release the mutex.
     * This allow the access to the mutex to other
     * threads that are waiting for it.
     */
    void unlock();

    char lockFilename[L_tmpnam];
}
;//end of class MpiFileMutex

/** This class represent an Xmipp MPI Program.
 *  It includes the basic MPI functionalities to the programs,
 *  like an mpinode, a mutex...
 *
 *  To be compatible with inheritance multiple, the  BaseXmippProgram
 *  must be declared with XmippProgramm as virtual.
 *
 * @code
 * class BaseProgram: public virtual XmippProgram {};
 * class BaseMpiProgram: public BaseProgram, public XmippMpiProgram {};
 * @endcode
 */
class XmippMpiProgram: public virtual XmippProgram
{
protected:
    /** Mpi node */
    std::shared_ptr<MpiNode> node;
    /** Number of Processors **/
    size_t nProcs;
    /** Number of independent MPI jobs **/
    size_t numberOfJobs;
    /** status after an MPI call */
    MPI_Status status;

    /** Provide a node when calling from another MPI program  */
    void setNode(const std::shared_ptr<MpiNode> & node);

public:
    /** Read MPI params from command line */
    void read(int argc, char **argv);
    /** Call the run function inside a try/catch block
    * sending an abort signal to the rest of mpi nodes.
    * */
    virtual int tryRun();
};

class MpiMetadataProgram: public XmippMpiProgram
{
protected:
    /** Divide the job in this number block with this number of images */
    int blockSize;
    std::vector<size_t> imgsId;
    MpiTaskDistributor *distributor=nullptr;
    size_t first, last;

public:
    /** Constructor */
    MpiMetadataProgram() {}
    MpiMetadataProgram(const MpiMetadataProgram &)=delete;
    MpiMetadataProgram(const MpiMetadataProgram &&)=delete;

    /** Destructor */
    ~MpiMetadataProgram();
    MpiMetadataProgram & operator=(const MpiMetadataProgram &)=delete;
    MpiMetadataProgram & operator=(const MpiMetadataProgram &&)=delete;

    /** Read arguments */
    void read(int argc, char **argv);

    void defineParams();
    void readParams();
    /** Create task distributor */
    void createTaskDistributor(MetaData &mdIn, size_t blockSize = 0);
    /** Preprocess */
    virtual void preProcess() { /* nothing to do */ };
    /** finishProcessing */
    virtual void finishProcessing() { /* nothing to do */ };
    /** Get task to process */
    bool getTaskToProcess(size_t &objId, size_t &objIndex);
};

/** Macro to define a simple MPI parallelization
 * of a program based on XmippMetaDataProgram */
template <typename BASE_CLASS>
class BasicMpiMetadataProgram : public BASE_CLASS, public MpiMetadataProgram {
protected:
  void defineParams() override {
    BASE_CLASS::defineParams();
    MpiMetadataProgram::defineParams();
  }

  void readParams() override {
    MpiMetadataProgram::readParams();
    BASE_CLASS::readParams();
  }

  void preProcess() override {
    BASE_CLASS::preProcess();
    MetaData &mdIn = *this->getInputMd();
    mdIn.addLabel(MDL_GATHER_ID);
    mdIn.fillLinear(MDL_GATHER_ID, 1, 1);
    createTaskDistributor(mdIn, blockSize);
  }

  void startProcessing() {
    if (node->rank == 1) {
      verbose = 1;
      BASE_CLASS::startProcessing();
    }
    node->barrierWait();
  }

  void showProgress() {
    if (node->rank == 1) {
      BASE_CLASS::time_bar_done = first + 1;
      BASE_CLASS::showProgress();
    }
  }

  bool getImageToProcess(size_t &objId, size_t &objIndex) override {
    return getTaskToProcess(objId, objIndex);
  }

  void finishProcessing() override {
    node->gatherMetadatas(this->getOutputMd(), BASE_CLASS::fn_out);
    MetaDataVec MDaux;
    MDaux.sort(this->getOutputMd(), MDL_GATHER_ID);
    MDaux.removeLabel(MDL_GATHER_ID);
    this->getOutputMd() = MDaux;
    if (node->isMaster())
      BASE_CLASS::finishProcessing();
  }

  void wait() { distributor->wait(); }
};

/** MPI Reduce with memory constraint.
 * MPI_Reduce may give a segmentation fault when sharing large objects
 */
void xmipp_MPI_Reduce(
    void* send_data,
    void* recv_data,
    size_t count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm communicator,
	size_t blockSize=1048576);

/** @} */
#endif /* XMIPP_MPI_H_ */
