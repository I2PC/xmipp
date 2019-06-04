

#include "cuda_xmipp_utils.h"
#include "cuda_asserts.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include <nvml.h>

#include <time.h>
#include <sys/time.h>

struct pointwiseMult{
	int normFactor;
	cufftComplex *data;
};

template<typename T>
void TransformMatrix<T>::resize(myStreamHandle &myStream,
        size_t _Ndim, size_t _Xdim, size_t _Ydim, size_t _Zdim) {
    if (_Xdim*_Ydim*_Zdim*_Ndim==nzyxdim)
        return;

    clear();

    Xdim=_Xdim;
    Ydim=_Ydim;
    Zdim=_Zdim;
    Ndim=_Ndim;
    yxdim=_Ydim*_Xdim;
    zyxdim=yxdim*_Zdim;
    nzyxdim=zyxdim*_Ndim;
    gpuErrchk(cudaMalloc(&d_data,nzyxdim*sizeof(T)));
    gpuErrchk(cudaMallocHost(&h_data,nzyxdim*sizeof(T)));
    initializeIdentity(d_data, h_data, Ndim, myStream);
}

template void TransformMatrix<float>::resize(myStreamHandle &myStream,
        size_t _Ndim, size_t _Xdim, size_t _Ydim, size_t _Zdim);

template<typename T>
void GpuMultidimArrayAtGpu<T>::resize(size_t _Xdim, size_t _Ydim, size_t _Zdim, size_t _Ndim)
{
    if (_Xdim*_Ydim*_Zdim*_Ndim==nzyxdim){

        return;
    }

    clear();

    setDims(_Xdim, _Ydim, _Zdim, _Ndim);
    gpuErrchk(cudaMalloc(&d_data,nzyxdim*sizeof(T)));
}

void myStreamDestroy(void *ptr)
{
	cudaStream_t *streamPtr = (cudaStream_t *)ptr;
	cudaStreamDestroy(*streamPtr);
}

void myStreamCreate(myStreamHandle &myStream)
{
	cudaStream_t *streamPtr = new cudaStream_t;
	gpuErrchk(cudaStreamCreate(streamPtr));
	myStream.ptr = (void*)streamPtr;
	//printf("ptr %p\n", myStream.ptr);
	//printf("streamPtr %p\n", streamPtr);
}

void mycufftDestroy(void* ptr)
{
	cufftHandle *planPtr = (cufftHandle *)ptr;
	cufftDestroy(*planPtr);
	delete planPtr;
}

void calculateFFTPlanSize(mycufftHandle &myhandle){
	printf("calculateFFTPlanSize  myhandle.ptr: %p\n",myhandle.ptr);
	size_t ws2;
	cufftHandle *planFptr=(cufftHandle *)myhandle.ptr;
	cufftGetSize(*planFptr, &ws2);
	printf("calculateFFTPlanSize  size %i\n", (int)ws2);
}


void createPlanFFT(int Xdim, int Ydim, int Ndim, int Zdim, bool forward, cufftHandle *plan){

	int Xfdim=(Xdim/2)+1;

	int nr1[] = {Xdim};   // --- Size of the image in real space
	int nr2[] = {Ydim, Xdim};   // --- Size of the image in real space
	int nr3[] = {Zdim, Ydim, Xdim};   // --- Size of the image in real space

	int nf1[] = {Xfdim};   // --- Size of the Fourier transform
	int nf2[] = {Ydim, Xfdim};   // --- Size of the Fourier transform
	int nf3[] = {Zdim, Ydim, Xfdim};   // --- Size of the Fourier transform
	int *nr=NULL, *nf=NULL;
	int NRANK; // 1D, 2D or 3D FFTs
	if (Ydim==1 && Zdim==1)
	{
		NRANK=1;
		nr=nr1;
		nf=nf1;
	}
	else if (Zdim==1)
	{
		NRANK=2;
		nr=nr2;
		nf=nf2;
	}
	else
	{
		NRANK=3;
		nr=nr3;
		nf=nf3;
	}

	int rstride = 1;				// --- Distance between two successive input/output elements
	int fstride = 1;
	int rdist = Xdim*Ydim*Zdim;	    // --- Distance between batches
	int fdist = Xfdim*Ydim*Zdim;

	if(forward){
		gpuErrchkFFT(cufftPlanMany(plan, NRANK, nr, nr, rstride, rdist, nf, fstride, fdist, CUFFT_R2C, Ndim));
	}else{
		gpuErrchkFFT(cufftPlanMany(plan, NRANK, nr, nf, fstride, fdist, nr, rstride, rdist, CUFFT_C2R, Ndim));
	}

}

void createPlanFFTStream(int Xdim, int Ydim, int Ndim, int Zdim,
		bool forward, cufftHandle *plan, myStreamHandle &myStream){

	int Xfdim=(Xdim/2)+1;

	int nr1[] = {Xdim};   // --- Size of the image in real space
	int nr2[] = {Ydim, Xdim};   // --- Size of the image in real space
	int nr3[] = {Zdim, Ydim, Xdim};   // --- Size of the image in real space

	int nf1[] = {Xfdim};   // --- Size of the Fourier transform
	int nf2[] = {Ydim, Xfdim};   // --- Size of the Fourier transform
	int nf3[] = {Zdim, Ydim, Xfdim};   // --- Size of the Fourier transform
	int *nr=NULL, *nf=NULL;
	int NRANK; // 1D, 2D or 3D FFTs
	if (Ydim==1 && Zdim==1)
	{
		NRANK=1;
		nr=nr1;
		nf=nf1;
	}
	else if (Zdim==1)
	{
		NRANK=2;
		nr=nr2;
		nf=nf2;
	}
	else
	{
		NRANK=3;
		nr=nr3;
		nf=nf3;
	}

	int rstride = 1;				// --- Distance between two successive input/output elements
	int fstride = 1;
	int rdist = Xdim*Ydim*Zdim;	    // --- Distance between batches
	int fdist = Xfdim*Ydim*Zdim;

	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;
	if(forward){
		gpuErrchkFFT(cufftPlanMany(plan, NRANK, nr, nr, rstride, rdist, nf, fstride, fdist, CUFFT_R2C, Ndim));
		gpuErrchkFFT(cufftSetStream(*plan, *stream));
	}else{
		gpuErrchkFFT(cufftPlanMany(plan, NRANK, nr, nf, fstride, fdist, nr, rstride, rdist, CUFFT_C2R, Ndim));
		gpuErrchkFFT(cufftSetStream(*plan, *stream));
	}

}

template<typename T>
void TransformMatrix<T>::clear()
{
    if (d_data!=NULL)
        gpuFree((void*) d_data);
    if (h_data!=NULL)
        gpuErrchk(cudaFreeHost(h_data));
    Xdim=Ydim=Zdim=Ndim=yxdim=zyxdim=nzyxdim=0;
    d_data=NULL;
    h_data=NULL;
}
// explicit instantiation
template void TransformMatrix<float>::clear();

void gpuMalloc(void** d_data, size_t Nbytes)
{
	gpuErrchk(cudaMalloc(d_data, Nbytes));
}

void gpuFree(void* d_data)
{
	gpuErrchk(cudaFree(d_data));
}

void cpuMalloc(void** h_data, size_t Nbytes)
{
	gpuErrchk(cudaMallocHost(h_data, Nbytes));
}

void cpuFree(void* h_data)
{
	gpuErrchk(cudaFreeHost(h_data));
}

void initializeIdentity(float* d_data, float *h_data, int Ndim, myStreamHandle &myStream)
{
	//float *matrices = new float[Ndim*9];
	for(int i=0; i<Ndim; i++){
		float aux_matrix[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
		int offset=i*9;
		for (int j=0; j<9; j++)
			h_data[offset+j] = aux_matrix[j];
	}
	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;
	gpuErrchk(cudaMemcpyAsync((void*)d_data, h_data, Ndim*9*sizeof(float), cudaMemcpyHostToDevice, *stream));
	//delete []matrices;

}

/*void setTranslationMatrix(float* d_data, float posX, float posY, int n)
{
	float matrix[9] = {1, 0, posX, 0, 1, posY, 0, 0, 1};
	gpuErrchk(cudaMemcpy((void*)&d_data[n*9], &matrix[0], 9*sizeof(float), cudaMemcpyHostToDevice));
}*/

void setTranslationMatrix(float* d_data, float* posX, float* posY, int Ndim, myStreamHandle &myStream)
{
	float *matrices;
	gpuErrchk(cudaMallocHost((void**)&matrices, sizeof(float)*Ndim*9));

	for(int i=0; i<Ndim; i++){
		float aux_matrix[9] = {1, 0, -posX[i], 0, 1, -posY[i], 0, 0, 1};
		int offset=i*9;
		//memcpy(&matrices[offset], &aux_matrix, 9*sizeof(float));
		for (int j=0; j<9; j++)
			matrices[offset+j] = aux_matrix[j];
	}
	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;
	gpuErrchk(cudaMemcpyAsync((void*)d_data, matrices, Ndim*9*sizeof(float), cudaMemcpyHostToDevice, *stream));
	delete []matrices;
}

/*void setRotationMatrix(float* d_data, float ang, int n)
{
	float rad = (float)(ang*PI/180);
	float matrix[9] = {cosf(rad), -sinf(rad), 0, sinf(rad), cosf(rad), 0, 0, 0, 1};
	gpuErrchk(cudaMemcpy((void*)&d_data[n*9], &matrix[0], 9*sizeof(float), cudaMemcpyHostToDevice));
}*/
void setRotationMatrix(float* d_data, float* ang, int Ndim, myStreamHandle &myStream)
{

	float *rad_vector;
	gpuErrchk(cudaMallocHost((void**)&rad_vector, sizeof(float)*Ndim*9));

	for(int i=0; i<Ndim; i++){
		float rad = (float)(-ang[i]*M_PI/180);
		float matrix[9] = {cosf(rad), -sinf(rad), 0, sinf(rad), cosf(rad), 0, 0, 0, 1};
		int offset=i*9;
		for (int j=0; j<9; j++)
			rad_vector[offset+j] = matrix[j];
	}
	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;
	gpuErrchk(cudaMemcpyAsync((void*)d_data, rad_vector, Ndim*9*sizeof(float), cudaMemcpyHostToDevice, *stream));
	delete []rad_vector;
}

void gpuCopyFromCPUToGPU(void* data, void* d_data, size_t Nbytes)
{
	gpuErrchk(cudaMemcpy(d_data, data, Nbytes, cudaMemcpyHostToDevice));
}

void gpuCopyFromGPUToCPU(void* d_data, void* data, size_t Nbytes)
{
	gpuErrchk(cudaMemcpy(data, d_data, Nbytes, cudaMemcpyDeviceToHost));
}

void gpuCopyFromGPUToGPU(void* d_dataFrom, void* d_dataTo, size_t Nbytes)
{
	gpuErrchk(cudaMemcpy(d_dataTo, d_dataFrom, Nbytes, cudaMemcpyDeviceToDevice));
}

void gpuCopyFromCPUToGPUStream(void* data, void* d_data, size_t Nbytes, myStreamHandle &myStream)
{
	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;
	gpuErrchk(cudaMemcpyAsync(d_data, data, Nbytes, cudaMemcpyHostToDevice, *stream));

	//gpuErrchk(cudaStreamSynchronize(*stream));
}

void gpuCopyFromGPUToCPUStream(void* d_data, void* data, size_t Nbytes, myStreamHandle &myStream)
{
	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;
	gpuErrchk(cudaMemcpyAsync(data, d_data, Nbytes, cudaMemcpyDeviceToHost, *stream));

	gpuErrchk(cudaStreamSynchronize(*stream));
	//cudaDeviceSynchronize();
}

void gpuCopyFromGPUToGPUStream(void* d_dataFrom, void* d_dataTo, size_t Nbytes, myStreamHandle &myStream)
{
	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;
	gpuErrchk(cudaMemcpyAsync(d_dataTo, d_dataFrom, Nbytes, cudaMemcpyDeviceToDevice, *stream));
}

int gridFromBlock(int tasks, int Nthreads)
{
    int numBlk = tasks/Nthreads;
    if(tasks%Nthreads>0)
    	numBlk++;
    return numBlk;
}
void cuda_check_gpu_memory(float* data)
{
	size_t free_byte, total_byte;
	gpuErrchk(cudaMemGetInfo(&free_byte, &total_byte));

	float free_db = (float)free_byte;
	float total_db = (float)total_byte;
	float used_db = total_db - free_db;

	data[0]=total_db;
	data[1]=free_db;
	data[2]=used_db;
}

void cuda_check_gpu_properties(int* grid)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	grid[0] = prop.maxGridSize[0];
	grid[1] = prop.maxGridSize[1];
	grid[2] = prop.maxGridSize[2];
}


__device__ cufftComplex CB_pointwiseMultiplicationComplexKernelLoad(void *dataIn, size_t offset,
		void *callerInfo, void *sharedPtr)
{

	//printf("INSIDEEEE IFFT\n");
	pointwiseMult *myData = (pointwiseMult*)callerInfo;

	cufftComplex reference = ((cufftComplex*)dataIn)[offset];
	cufftComplex *mask = (cufftComplex*)myData->data;

	int normFactor = myData->normFactor;
	int indexM = offset%normFactor;

	float factor = 1.0f / normFactor;

	cufftComplex mulOut = cuCmulf((cuComplex)reference, (cuComplex)mask[indexM]);
	cufftComplex out;
	out.x = mulOut.x*factor;
	out.y = mulOut.y*factor;

	//if(offset>9000 && offset<9100)
	//	printf("offset %i, mask %f, data %f, mul %f, factor %f\n", offset, mask[indexM].x, reference.x, out.x, factor);

	return out;
}
__device__ cufftCallbackLoadC d_pointwiseMultiplicationComplexKernelLoad = CB_pointwiseMultiplicationComplexKernelLoad;



__device__ void CB_pointwiseMultiplicationComplexKernelStore(void *dataOut, size_t offset, cufftComplex element,
		void *callerInfo, void *sharedPtr)
{

	pointwiseMult *myData = (pointwiseMult*)callerInfo;

	cufftComplex *mask = myData->data;
	int normFactor = myData->normFactor;
	int indexM = offset%normFactor;

	float factor = 1.0f / normFactor;

	cufftComplex mulOut = cuCmulf((cuComplex)element, (cuComplex)mask[indexM]);
	cufftComplex out;
	out.x = mulOut.x*factor;
	out.y = mulOut.y*factor;
	((cufftComplex*)dataOut)[offset] = out;

}
__device__ cufftCallbackStoreC d_pointwiseMultiplicationComplexKernelStore = CB_pointwiseMultiplicationComplexKernelStore;


template float* loadToGPU<float>(const float* data, size_t items);
template std::complex<float>* loadToGPU<std::complex<float> >(const std::complex<float>* data, size_t items);
template<typename T>
T* loadToGPU(const T* data, size_t items) {
T* d_data;
size_t bytes = items * sizeof(T);
gpuErrchk(cudaMalloc(&d_data, bytes));
gpuErrchk(cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice));
return d_data;
}

template<>
void GpuMultidimArrayAtGpu<float>::fftStream(GpuMultidimArrayAtGpu< std::complex<float> > &fourierTransform,
		mycufftHandle &myhandle, myStreamHandle &myStream, bool useCallback,
		GpuMultidimArrayAtGpu< std::complex<float> > &dataMask)
{

	int Xfdim=(Xdim/2)+1;
	//if(fourierTransform.d_data==NULL)
	fourierTransform.resize(Xfdim,Ydim,Zdim,Ndim);
	//printf("FFT Xdim %i, Ydim %i, Ndim %i, Zdim %i \n", Xfdim, Ydim, Zdim, Ndim);

	int positionReal=0;
	int positionFFT=0;
	int NdimNew, auxNdim;
	NdimNew = Ndim;
	int aux=Ndim;

	auxNdim=Ndim;

	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;

	if(myhandle.ptr!=NULL)
		NdimNew = Ndim;

	while(aux>0){

		GpuMultidimArrayAtGpu<cufftReal> auxInFFT;
		GpuMultidimArrayAtGpu<cufftComplex> auxOutFFT;
		if(NdimNew!=Ndim){
			auxInFFT.resize(Xdim,Ydim,Zdim,NdimNew);
			gpuCopyFromGPUToGPUStream((cufftReal*)&d_data[positionReal], auxInFFT.d_data, Xdim*Ydim*Zdim*NdimNew*sizeof(cufftReal), myStream);
			auxOutFFT.resize(Xfdim,Ydim,Zdim,NdimNew);
		}

		cufftHandle *planFptr = new cufftHandle;
		cufftHandle *planAuxFptr = new cufftHandle;
		if(auxNdim!=NdimNew){
			createPlanFFTStream(Xdim, Ydim, NdimNew, Zdim, true, planAuxFptr, myStream);
		}else{
			if(myhandle.ptr == NULL){
				//printf("Creating plan FFT\n");
				createPlanFFTStream(Xdim, Ydim, NdimNew, Zdim, true, planFptr, myStream);
				myhandle.ptr = (void *)planFptr;
				planFptr=(cufftHandle *)myhandle.ptr;
			}else{
				planFptr=(cufftHandle *)myhandle.ptr;
			}
		}

		//AJ using callbacks
		if(useCallback){

			pointwiseMult *dataCB;
			gpuErrchk(cudaMallocHost((void **)&dataCB, sizeof(pointwiseMult)));
			dataCB->normFactor = fourierTransform.yxdim;
			dataCB->data = (cufftComplex*)dataMask.d_data;

			//printf("Using callbacks %i \n", dataCB->normFactor);
			//fflush(stdout);

			// Allocate device memory for parameters
			pointwiseMult *d_params;
			gpuErrchk(cudaMalloc((void **)&d_params, sizeof(pointwiseMult)));

			// Copy host memory to device
			gpuErrchk(cudaMemcpyAsync(d_params, dataCB, sizeof(pointwiseMult), cudaMemcpyHostToDevice, *stream));

			cufftCallbackStoreC h_pointwiseMultiplicationComplexKernel;
			gpuErrchk(cudaMemcpyFromSymbolAsync(&h_pointwiseMultiplicationComplexKernel,
												d_pointwiseMultiplicationComplexKernelStore,
												sizeof(h_pointwiseMultiplicationComplexKernel), 0, cudaMemcpyDeviceToHost, *stream));

			cufftResult status = cufftXtSetCallback(*planFptr,
									(void **)&h_pointwiseMultiplicationComplexKernel,
									CUFFT_CB_ST_COMPLEX,
									(void **)&d_params);
		}
		//END AJ

		if(auxNdim==NdimNew){
			if(NdimNew!=Ndim){
				gpuErrchkFFT(cufftExecR2C(*planFptr, auxInFFT.d_data, auxOutFFT.d_data));
			}else{
				gpuErrchkFFT(cufftExecR2C(*planFptr, (cufftReal*)&d_data[positionReal], (cufftComplex*)&fourierTransform.d_data[positionFFT]));
			}
		}else{
			if(NdimNew!=Ndim){
				gpuErrchkFFT(cufftExecR2C(*planAuxFptr, auxInFFT.d_data, auxOutFFT.d_data));
			}else{
				gpuErrchkFFT(cufftExecR2C(*planAuxFptr, (cufftReal*)&d_data[positionReal], (cufftComplex*)&fourierTransform.d_data[positionFFT]));
			}
		}


		if(NdimNew!=Ndim){
			gpuCopyFromGPUToGPUStream(auxOutFFT.d_data, (cufftComplex*)&fourierTransform.d_data[positionFFT], Xfdim*Ydim*Zdim*NdimNew*sizeof(cufftComplex), myStream);
			auxOutFFT.clear();
			auxInFFT.clear();
		}

		auxNdim=NdimNew;

		positionReal+=(NdimNew*Xdim*Ydim*Zdim);
		positionFFT+=(NdimNew*Xfdim*Ydim*Zdim);
		aux-=NdimNew;
		if(aux<NdimNew)
			NdimNew=aux;

		if(auxNdim!=NdimNew && NdimNew!=0)
			cufftDestroy(*planAuxFptr);


		gpuErrchk(cudaStreamSynchronize(*stream));

	}//AJ end while
}


template<>
template<>
void GpuMultidimArrayAtGpu<float>::fft(GpuMultidimArrayAtGpu< std::complex<float> > &fourierTransform, mycufftHandle &myhandle)
{

	int Xfdim=(Xdim/2)+1;
	fourierTransform.resize(Xfdim,Ydim,Zdim,Ndim);

	int positionReal=0;
	int positionFFT=0;
	size_t NdimNew, auxNdim;
	NdimNew = Ndim;
	size_t aux=Ndim;

	auxNdim=Ndim;

	if(myhandle.ptr!=NULL) {
		NdimNew = Ndim;
	}

	while(aux>0){

		GpuMultidimArrayAtGpu<cufftReal> auxInFFT;
		GpuMultidimArrayAtGpu<cufftComplex> auxOutFFT;
		if(NdimNew!=Ndim){
			auxInFFT.resize(Xdim,Ydim,Zdim,NdimNew);
			gpuCopyFromGPUToGPU((cufftReal*)&d_data[positionReal], auxInFFT.d_data, Xdim*Ydim*Zdim*NdimNew*sizeof(cufftReal));
			auxOutFFT.resize(Xfdim,Ydim,Zdim,NdimNew);
		}

		cufftHandle *planFptr = NULL;
		cufftHandle *planAuxFptr = NULL;
		if(auxNdim!=NdimNew){
			planAuxFptr = new cufftHandle;
			createPlanFFT(Xdim, Ydim, NdimNew, Zdim, true, planAuxFptr);
		}else{
			if(myhandle.ptr == NULL){
				myhandle.ptr = planFptr = new cufftHandle;
				createPlanFFT(Xdim, Ydim, NdimNew, Zdim, true, planFptr);
			}
			planFptr=(cufftHandle *)myhandle.ptr;
		}

		if(auxNdim==NdimNew){
			if(NdimNew!=Ndim){
				gpuErrchkFFT(cufftExecR2C(*planFptr, auxInFFT.d_data, auxOutFFT.d_data));
			}else{
				gpuErrchkFFT(cufftExecR2C(*planFptr, (cufftReal*)&d_data[positionReal], (cufftComplex*)&fourierTransform.d_data[positionFFT]));
			}
		}else{
			if(NdimNew!=Ndim){
				gpuErrchkFFT(cufftExecR2C(*planAuxFptr, auxInFFT.d_data, auxOutFFT.d_data));
			}else{
				gpuErrchkFFT(cufftExecR2C(*planAuxFptr, (cufftReal*)&d_data[positionReal], (cufftComplex*)&fourierTransform.d_data[positionFFT]));
			}
		}

		gpuErrchk(cudaDeviceSynchronize());

		if(NdimNew!=Ndim){
			gpuCopyFromGPUToGPU(auxOutFFT.d_data, (cufftComplex*)&fourierTransform.d_data[positionFFT], Xfdim*Ydim*Zdim*NdimNew*sizeof(cufftComplex));
			auxOutFFT.clear();
			auxInFFT.clear();
		}

		auxNdim=NdimNew;

		positionReal+=(NdimNew*Xdim*Ydim*Zdim);
		positionFFT+=(NdimNew*Xfdim*Ydim*Zdim);
		aux-=NdimNew;
		if(aux<NdimNew)
			NdimNew=aux;

		if (NULL != planAuxFptr) {
			cufftDestroy(*planAuxFptr); // destroy if created
		}

	}//AJ end while

}

template<>
template<>
void GpuMultidimArrayAtGpu< std::complex<float> >::ifftStream(GpuMultidimArrayAtGpu<float> &realSpace,
		mycufftHandle &myhandle, myStreamHandle &myStream, bool useCallback,
		GpuMultidimArrayAtGpu< std::complex<float> > &dataMask)
{

	int Xfdim=(realSpace.Xdim/2)+1;

	//printf("FFT Xdim %i, Ydim %i, Ndim %i, Zdim %i Xfdim %i \n", Xdim, Ydim, Zdim, Ndim, Xfdim);

	int positionReal=0;
	int positionFFT=0;
	int NdimNew, auxNdim;
	NdimNew = realSpace.Ndim;
	int aux=realSpace.Ndim;

	auxNdim=realSpace.Ndim;

	cudaStream_t *stream = (cudaStream_t*) myStream.ptr;

	if(myhandle.ptr!=NULL)
		NdimNew = Ndim;

	while(aux>0){

		GpuMultidimArrayAtGpu<cufftComplex> auxInFFT;
		GpuMultidimArrayAtGpu<cufftReal> auxOutFFT;
		if(NdimNew!=Ndim){
			auxInFFT.resize(Xfdim,realSpace.Ydim,realSpace.Zdim,NdimNew);
			gpuCopyFromGPUToGPUStream((cufftComplex*)&d_data[positionFFT], auxInFFT.d_data, Xfdim*realSpace.Ydim*realSpace.Zdim*NdimNew*sizeof(cufftComplex), myStream);
			auxOutFFT.resize(realSpace.Xdim,realSpace.Ydim,realSpace.Zdim, NdimNew);
		}

		cufftHandle *planBptr = new cufftHandle;
		cufftHandle *planAuxBptr = new cufftHandle;
		if(auxNdim!=NdimNew){
			createPlanFFTStream(Xdim, Ydim, NdimNew, Zdim, false, planAuxBptr, myStream);
		}else{
			if(myhandle.ptr == NULL){
				//printf("Creating plan IFFT\n");
				createPlanFFTStream(realSpace.Xdim, realSpace.Ydim, NdimNew, Zdim, false, planBptr, myStream);
				myhandle.ptr = (void *)planBptr;
				planBptr=(cufftHandle *)myhandle.ptr;
			}else{
				planBptr=(cufftHandle *)myhandle.ptr;
			}
		}

		//AJ using callbacks
		if(useCallback){
			//printf("Using callbacks\n");
			pointwiseMult *dataCB;
			gpuErrchk(cudaMallocHost((void **)&dataCB, sizeof(pointwiseMult)));
			dataCB->normFactor = dataMask.yxdim;
			dataCB->data = (cufftComplex*)dataMask.d_data;

			// Allocate device memory for parameters
			pointwiseMult *d_params;
			gpuErrchk(cudaMalloc((void **)&d_params, sizeof(pointwiseMult)));

			// Copy host memory to device
			gpuErrchk(cudaMemcpyAsync(d_params, dataCB, sizeof(pointwiseMult), cudaMemcpyHostToDevice, *stream));


			cufftCallbackLoadC h_pointwiseMultiplicationComplexKernel;
			gpuErrchk(cudaMemcpyFromSymbolAsync(&h_pointwiseMultiplicationComplexKernel,
												d_pointwiseMultiplicationComplexKernelLoad,
												sizeof(h_pointwiseMultiplicationComplexKernel), 0, cudaMemcpyDeviceToHost, *stream));

			cufftResult status = cufftXtSetCallback(*planBptr,
									(void **)&h_pointwiseMultiplicationComplexKernel,
									CUFFT_CB_LD_COMPLEX,
									(void **)&d_params);
		}
		//END AJ

		if(auxNdim==NdimNew){
			if(NdimNew!=Ndim){
				gpuErrchkFFT(cufftExecC2R(*planBptr, auxInFFT.d_data, auxOutFFT.d_data));
			}else{
				gpuErrchkFFT(cufftExecC2R(*planBptr, (cufftComplex *)&d_data[positionFFT], (cufftReal*)&realSpace.d_data[positionReal]));
			}
		}else{
			if(NdimNew!=Ndim){
				gpuErrchkFFT(cufftExecC2R(*planAuxBptr, auxInFFT.d_data, auxOutFFT.d_data));
			}else{
				gpuErrchkFFT(cufftExecC2R(*planAuxBptr, (cufftComplex *)&d_data[positionFFT], (cufftReal*)&realSpace.d_data[positionReal]));
			}
		}

		gpuErrchk(cudaStreamSynchronize(*stream));

		if(NdimNew!=Ndim){
			gpuCopyFromGPUToGPUStream(auxOutFFT.d_data, (cufftReal*)&realSpace.d_data[positionReal], realSpace.Xdim*realSpace.Ydim*realSpace.Zdim*NdimNew*sizeof(cufftReal), myStream);
			auxOutFFT.clear();
			auxInFFT.clear();
		}

		auxNdim=NdimNew;

		positionReal+=(NdimNew*realSpace.Xdim*realSpace.Ydim*realSpace.Zdim);
		positionFFT+=(NdimNew*Xfdim*realSpace.Ydim*realSpace.Zdim);
		aux-=NdimNew;
		if(aux<NdimNew)
			NdimNew=aux;

		if(auxNdim!=NdimNew && NdimNew!=0)
			cufftDestroy(*planAuxBptr);

	}//AJ end while

}

template<>
template<>
void GpuMultidimArrayAtGpu< std::complex<float> >::ifft(GpuMultidimArrayAtGpu<float> &realSpace, mycufftHandle &myhandle)
{

	int Xfdim=(realSpace.Xdim/2)+1;

	int positionReal=0;
	int positionFFT=0;
	size_t NdimNew, auxNdim;
	NdimNew = realSpace.Ndim;
	size_t aux=realSpace.Ndim;

	auxNdim=realSpace.Ndim;

	if(myhandle.ptr!=NULL)
		NdimNew = Ndim;

	while(aux>0){

		GpuMultidimArrayAtGpu<cufftComplex> auxInFFT;
		GpuMultidimArrayAtGpu<cufftReal> auxOutFFT;
		if(NdimNew!=Ndim){
			auxInFFT.resize(Xfdim,realSpace.Ydim,realSpace.Zdim,NdimNew);
			gpuCopyFromGPUToGPU((cufftComplex*)&d_data[positionFFT], auxInFFT.d_data, Xfdim*realSpace.Ydim*realSpace.Zdim*NdimNew*sizeof(cufftComplex));
			auxOutFFT.resize(realSpace.Xdim,realSpace.Ydim,realSpace.Zdim, NdimNew);
		}

		cufftHandle *planBptr = nullptr;
		cufftHandle *planAuxBptr = nullptr;
		if(auxNdim!=NdimNew){
		    planAuxBptr = new cufftHandle;
			createPlanFFT(Xdim, Ydim, NdimNew, Zdim, false, planAuxBptr);
		}else{
			if(nullptr == myhandle.ptr){
				myhandle.ptr = planBptr = new cufftHandle;
				createPlanFFT(realSpace.Xdim, realSpace.Ydim, NdimNew, Zdim, false, planBptr);
			}
            planBptr=(cufftHandle *)myhandle.ptr;
		}

		if(auxNdim==NdimNew){
			if(NdimNew!=Ndim){
				gpuErrchkFFT(cufftExecC2R(*planBptr, auxInFFT.d_data, auxOutFFT.d_data));
			}else{
				gpuErrchkFFT(cufftExecC2R(*planBptr, (cufftComplex *)&d_data[positionFFT], (cufftReal*)&realSpace.d_data[positionReal]));
			}
		}else{
			if(NdimNew!=Ndim){
				gpuErrchkFFT(cufftExecC2R(*planAuxBptr, auxInFFT.d_data, auxOutFFT.d_data));
			}else{
				gpuErrchkFFT(cufftExecC2R(*planAuxBptr, (cufftComplex *)&d_data[positionFFT], (cufftReal*)&realSpace.d_data[positionReal]));
			}
		}


		gpuErrchk(cudaDeviceSynchronize());

		if(NdimNew!=Ndim){
			gpuCopyFromGPUToGPU(auxOutFFT.d_data, (cufftReal*)&realSpace.d_data[positionReal], realSpace.Xdim*realSpace.Ydim*realSpace.Zdim*NdimNew*sizeof(cufftReal));
			auxOutFFT.clear();
			auxInFFT.clear();
		}

		auxNdim=NdimNew;

		positionReal+=(NdimNew*realSpace.Xdim*realSpace.Ydim*realSpace.Zdim);
		positionFFT+=(NdimNew*Xfdim*realSpace.Ydim*realSpace.Zdim);
		aux-=NdimNew;
		if(aux<NdimNew)
			NdimNew=aux;

		if(nullptr != planAuxBptr)
			cufftDestroy(*planAuxBptr); // destroy if created

	}//AJ end while

}
