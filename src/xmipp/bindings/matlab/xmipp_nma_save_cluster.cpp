#include <mex.h>
#include <core/xmipp_metadata_program.h>

/* the gateway function */
/* xmipp_nma_save_cluster(NMAdirectory,clusterName,inCluster) */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  /* check for proper number of arguments */
  if(nrhs!=3)
    mexErrMsgTxt("3 input required.");
  
  /* Get parameters */
  char nmaDir[1024];
  mxGetString(prhs[0],nmaDir,mxGetN(prhs[0])+1);
  char clusterName[256];
  mxGetString(prhs[1],clusterName,mxGetN(prhs[1])+1);
  double *ptrInCluster = mxGetPr(prhs[2]);
  
  /* Read images */
  MetaDataVec mdImages, mdImagesOut;
  mdImages.read(((String)nmaDir)+"/images.xmd");
  mdImages.removeDisabled();

  // Fill output
  for (auto& row : mdImages)
  {
	  if (*ptrInCluster!=0)
	  {
              mdImagesOut.addRow(row);
	  }
	  ptrInCluster++;
  }
  mdImagesOut.write(((String)nmaDir)+"/images_"+clusterName+".xmd");
}
