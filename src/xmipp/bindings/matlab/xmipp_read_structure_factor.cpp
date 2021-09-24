#include <mex.h>
#include <core/metadata_vec.h>

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
  double *ptrFreq2=mxGetPr(plhs[0]);
  plhs[1]=mxCreateDoubleMatrix((mwSize)nSamples, (mwSize)1, mxREAL);
  double *ptrLogStruct=mxGetPr(plhs[1]);

  // Fill output
  int i=0;
  FOR_ALL_OBJECTS_IN_METADATA(md)
  {
	  md.getValue(MDL_RESOLUTION_FREQ2,*ptrFreq2++,objId);
	  md.getValue(MDL_RESOLUTION_LOG_STRUCTURE_FACTOR,*ptrLogStruct++,objId);
  }
  
}
