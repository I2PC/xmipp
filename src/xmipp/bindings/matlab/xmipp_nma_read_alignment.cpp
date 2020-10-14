#include <mex.h>
#include <core/metadata.h>

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  /* check for proper number of arguments */
  if(nrhs!=1) 
    mexErrMsgTxt("1 input required.");
  if(nlhs!=3)
    mexErrMsgTxt("3 outputs are required.");
  
  /* Get parameters */
  char nmaDir[1024];
  mxGetString(prhs[0],nmaDir,mxGetN(prhs[0])+1);
  
  /* Read images */
  MetaData mdImages;
  mdImages.read(((String)nmaDir)+"/images.xmd");
  if (!mdImages.containsLabel(MDL_NMA))
  {
	  // May be it did not finish, try to load the nmaDone.xmd
	  FileName fnDone=((String)nmaDir)+"/tmp/nmaDone.xmd";
	  if (fileExists(fnDone))
	  {
		  mdImages.read(fnDone);
		  if (!mdImages.containsLabel(MDL_NMA))
		     REPORT_ERROR(ERR_MD_MISSINGLABEL,"Cannot find NMA displacements in nmaDone.xmd");
		  // Write the deformations file because it does not exist yet
		  std::vector< std::vector<double> > nmaDisplacements;
		  mdImages.getColumnValues(MDL_NMA, nmaDisplacements);
		  std::ofstream fhOut;
		  fhOut.open((((String)nmaDir)+"/extra/deformations.txt").c_str());
		  size_t nImages=nmaDisplacements.size();
		  if (nImages>0)
		  {
			  size_t nDisplacements=nmaDisplacements[0].size();
			  for (size_t n=0; n<nImages; ++n)
			  {
				  for (size_t i=0; i<nDisplacements; ++i)
					  fhOut << nmaDisplacements[n][i] << " ";
				  fhOut << std::endl;
			  }
		  }
		  fhOut.close();
	  }
	  else
		  REPORT_ERROR(ERR_MD_MISSINGLABEL,"Cannot find NMA displacements");
  }
  mdImages.removeDisabled();
  int nImgs=(int)mdImages.size();

  /* Read modes */
  MetaData mdModes;
  mdModes.read(((String)nmaDir)+"/modes.xmd");
  mdModes.removeDisabled();
  int nModes=(int)mdModes.size();

  // Allocate output
  mwSize dims[1];
  dims[0]=(mwSize)nImgs;
  plhs[0]=mxCreateCellArray((mwSize)1, dims);
  plhs[1]=mxCreateDoubleMatrix((mwSize)nImgs, (mwSize)nModes, mxREAL);
  double *ptrNMADistplacements=mxGetPr(plhs[1]);
  plhs[2]=mxCreateDoubleMatrix((mwSize)nImgs, (mwSize)1, mxREAL);
  double *ptrCost=mxGetPr(plhs[2]);

  // Fill output
  int i=0;
  String fnImg;
  std::vector<double> lambda;
  FOR_ALL_OBJECTS_IN_METADATA(mdImages)
  {
	  mdImages.getValue(MDL_IMAGE,fnImg,__iter.objId);
	  mxSetCell(plhs[0], i, mxCreateString(fnImg.c_str()));
	  mdImages.getValue(MDL_NMA,lambda,__iter.objId);
	  for (int j=0; j<nModes; ++j)
		  ptrNMADistplacements[j*nImgs+i]=lambda[j]; // x*Ydim+y
	  mdImages.getValue(MDL_COST,*ptrCost,__iter.objId);
	  i++;
	  ptrCost++;
  }
}
