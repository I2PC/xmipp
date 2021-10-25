#include <mex.h>
#include <core/xmipp_metadata_program.h>

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  /* check for proper number of arguments */
  if(nrhs!=1) 
    mexErrMsgTxt("1 input required.");
  if(nlhs!=2)
    mexErrMsgTxt("2 outputs are required.");
  
  /* Get parameters */
  char runDir[1024];
  mxGetString(prhs[0],runDir,mxGetN(prhs[0])+1);
  
  /* Read metadata */
  MetaDataVec md;
  md.read(((String)runDir)+"/structureFactor.xmd");
  int nSamples=(int)md.size();

  // Allocate output
  plhs[0]=mxCreateDoubleMatrix((mwSize)nSamples, (mwSize)1, mxREAL);
  plhs[1]=mxCreateDoubleMatrix((mwSize)nSamples, (mwSize)1, mxREAL);

  // Fill output
  std::vector<double> freq2 = md.getColumnValues<double>(MDL_RESOLUTION_FREQ2);
  std::vector<double> logStruct = md.getColumnValues<double>(MDL_RESOLUTION_LOG_STRUCTURE_FACTOR);
  memcpy(mxGetData(plhs[0]), &freq2[0], nSamples*sizeof(double));
  memcpy(mxGetData(plhs[1]), &logStruct[0], nSamples*sizeof(double));
}
