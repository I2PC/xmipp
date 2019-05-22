
#ifndef CUDA_XMIPP_UTILS_H
#define CUDA_XMIPP_UTILS_H

#include <stdio.h>
#include <complex>

class myStreamHandle;
void mycufftDestroy(void *ptr);
void myStreamDestroy(void *ptr);
void myStreamCreate(myStreamHandle &myStream);
void gpuMalloc(void** d_data, size_t Nbytes);
void gpuFree(void* d_data);
void cpuMalloc(void** h_data, size_t Nbytes);
void cpuFree(void* h_data);
void initializeIdentity(float* d_data, float *h_data, int Ndim, myStreamHandle &myStream);
void setTranslationMatrix(float* d_data, float* posX, float* posY, int Ndim, myStreamHandle &myStream);
void setRotationMatrix(float* d_data, float *ang, int Ndim, myStreamHandle &myStream);
void gpuCopyFromGPUToGPUStream(void* d_dataFrom, void* d_dataTo, size_t Nbytes, myStreamHandle &myStream);
void gpuCopyFromCPUToGPUStream(void* data, void* d_data, size_t Nbytes, myStreamHandle &myStream);
void gpuCopyFromGPUToCPUStream(void* d_data, void* data, size_t Nbytes, myStreamHandle &myStream);
void gpuCopyFromGPUToGPU(void* d_dataFrom, void* d_dataTo, size_t Nbytes);
void gpuCopyFromCPUToGPU(void* data, void* d_data, size_t Nbytes);
void gpuCopyFromGPUToCPU(void* d_data, void* data, size_t Nbytes);
int gridFromBlock(int tasks, int Nthreads);

template<typename T>
T* loadToGPU(const T* data, size_t items);

void cuda_check_gpu_memory(float* data);
void cuda_check_gpu_properties(int* maxGridSize);

class mycufftHandle {
public:
	void *ptr;

	~mycufftHandle() { clear(); }

	mycufftHandle(){
			ptr=NULL;
	}

	void clear()
	{
		if (ptr!=NULL)
			mycufftDestroy(ptr);
		ptr=NULL;
	}


};

class myStreamHandle {
public:
	void *ptr;

	myStreamHandle(){
		ptr=NULL;
	}

	void clear()
	{
		if(ptr!=NULL)
			myStreamDestroy(ptr);
	}

};

class XmippDim3 {
public:
    size_t x;
    size_t y;
    size_t z;

	XmippDim3(size_t _x, size_t _y, size_t _z)
	{
		x=_x;
		y=_y;
		z=_z;
	}

	XmippDim3()
	{
		x=y=z=0;
	}
};

#define CONVERT2DIM3(d) (dim3((d).x,(d).y,(d).z))


template<typename T>
class TransformMatrix
{
public:
    size_t Xdim, Ydim, Zdim, Ndim, yxdim, zyxdim, nzyxdim;
    T* d_data;
	T* h_data;

    TransformMatrix()
    {
		Xdim=Ydim=Zdim=Ndim=yxdim=zyxdim=nzyxdim=0;
		d_data=NULL;
		h_data=NULL;
    }

    TransformMatrix(myStreamHandle &myStream, size_t _Ndim, size_t _Xdim=3, size_t _Ydim=3, size_t _Zdim=1)
    {
		Xdim=Ydim=Zdim=Ndim=yxdim=zyxdim=nzyxdim=0;
		d_data=NULL;
		h_data=NULL;
		resize(myStream,_Ndim, _Xdim, _Ydim, _Zdim);
    }

	template<typename T1>
	void resize(const TransformMatrix<T1>& array, myStreamHandle &myStream)
	{

		resize(myStream, array.Ndim, array.Xdim, array.Ydim, array.Zdim);
	}

	void resize(myStreamHandle &myStream, size_t _Ndim, size_t _Xdim=3, size_t _Ydim=3, size_t _Zdim=1);

	bool isEmpty()
	{
		return d_data==NULL && h_data==NULL;
	}

	void clear();

	~TransformMatrix()
	{
		clear();
	}

	void initialize(myStreamHandle &myStream)
	{
		initializeIdentity(d_data, h_data, Ndim, myStream);
	}

	void setTranslation(float* posX, float* posY, float *d_out_max, myStreamHandle &myStream)
	{
		/*for(int i=0; i<Ndim; i++)
			setTranslationMatrix(d_data, -posX[i], -posY[i], i);*/
		setTranslationMatrix(d_data, posX, posY, Ndim, myStream);
	}

	void setRotation(float* ang, myStreamHandle &myStream)
	{
		/*for(int i=0; i<Ndim; i++)
			setRotationMatrix(d_data, -ang[i], i);*/
		setRotationMatrix(d_data, ang, Ndim, myStream);
	}

	void copyMatrix(TransformMatrix<float> &lastMatrix, myStreamHandle &myStream)
	{
		if (lastMatrix.isEmpty())
			lastMatrix.resize(myStream, Ndim, 3, 3, 1);

		gpuCopyFromGPUToGPUStream(d_data, lastMatrix.d_data, nzyxdim*sizeof(float), myStream);
	}

	void copyMatrixToCpu(myStreamHandle &myStream)
	{
		gpuCopyFromGPUToCPUStream(d_data, h_data, nzyxdim*sizeof(float), myStream);
	}

	void copyOneMatrixToCpu(float* &matrixCpu, int idxCpu, int idxGpu, myStreamHandle &myStream)
	{
		gpuCopyFromGPUToCPUStream(&d_data[9*idxGpu], &matrixCpu[9*idxCpu], 9*sizeof(float), myStream);
	}


};


template<typename T>
class GpuMultidimArrayAtGpu
{
public:
	size_t Xdim, Ydim, Zdim, Ndim, yxdim, zyxdim, nzyxdim;
    T* d_data;

	GpuMultidimArrayAtGpu()
    {
		Xdim=Ydim=Zdim=Ndim=yxdim=zyxdim=nzyxdim=0;
		d_data=NULL;
    }

	GpuMultidimArrayAtGpu(size_t _Xdim, size_t _Ydim=1, size_t _Zdim=1, size_t _Ndim=1)
    {
		Xdim=Ydim=Zdim=Ndim=yxdim=zyxdim=nzyxdim=0;
		d_data=NULL;
		resize(_Xdim, _Ydim, _Zdim, _Ndim);
    }

	GpuMultidimArrayAtGpu(size_t _Xdim, size_t _Ydim, size_t _Zdim, size_t _Ndim, T* deviceData)
	{
		setDims(_Xdim, _Ydim, _Zdim, _Ndim);
		d_data = deviceData;
	}

	template<typename T1>
	void resize(const GpuMultidimArrayAtGpu<T1>& array)
	{

		resize(array.Xdim, array.Ydim, array.Zdim, array.Ndim);
	}

	void resize(size_t _Xdim, size_t _Ydim=1, size_t _Zdim=1, size_t _Ndim=1);

	bool isEmpty()
	{
		return d_data==NULL;
	}

	void clear()
	{
		if (d_data!=NULL){
			gpuFree((void*) d_data);

		}
		Xdim=Ydim=Zdim=Ndim=yxdim=zyxdim=nzyxdim=0;
		d_data=NULL;
	}

	~GpuMultidimArrayAtGpu()
	{
		clear();
	}

	void copyToGpu(T* data)
	{
		gpuCopyFromCPUToGPU((void *)data, (void *)d_data, nzyxdim*sizeof(T));
	}

	void copyToCpu(T* data)
	{
		gpuCopyFromGPUToCPU((void *)d_data, (void *)data, nzyxdim*sizeof(T));
	}

	void copyToGpuStream(T* data, myStreamHandle &myStream)
	{
		gpuCopyFromCPUToGPUStream((void *)data, (void *)d_data, nzyxdim*sizeof(T), myStream);
	}

	void fillImageToGpu(T* data, size_t n=0)
	{
		gpuCopyFromCPUToGPU((void *)data, (void *)&d_data[n*zyxdim], zyxdim*sizeof(T));
	}

	void fillImageToGpuStream(T* data, myStreamHandle &myStream, int n=0)
	{
		gpuCopyFromCPUToGPUStream((void *)data, (void *)&d_data[n*zyxdim], zyxdim*sizeof(T), myStream);
	}

	void copyGpuToGpu(GpuMultidimArrayAtGpu<T> &gpuArray)
	{
		if (gpuArray.isEmpty())
			gpuArray.resize(Xdim,Ydim,Zdim,Ndim);

		gpuCopyFromGPUToGPU(d_data, gpuArray.d_data, nzyxdim*sizeof(T));
	}

	void copyGpuToGpuStream(GpuMultidimArrayAtGpu<T> &gpuArray, myStreamHandle &myStream)
	{
		if (gpuArray.isEmpty())
			gpuArray.resize(Xdim,Ydim,Zdim,Ndim);

		gpuCopyFromGPUToGPUStream(d_data, gpuArray.d_data, nzyxdim*sizeof(T), myStream);
	}

	void calculateGridSize(const XmippDim3 &blockSize, XmippDim3 &gridSize) const
	{
		gridSize.x=gridFromBlock(Xdim,blockSize.x);
		gridSize.y=gridFromBlock(Ydim,blockSize.y);
		gridSize.z=gridFromBlock(Zdim,blockSize.z);
	}

	void calculateGridSizeVectorized(const XmippDim3 &blockSize, XmippDim3 &gridSize) const
	{
		gridSize.x=gridFromBlock(nzyxdim,blockSize.x);
		gridSize.y=1;
		gridSize.z=1;
	}

	template <typename T1>
	void fft(GpuMultidimArrayAtGpu<T1> &fourierTransform, mycufftHandle &myhandle);

	// RealSpace must already be resized
	template <typename T1>
	void ifft(GpuMultidimArrayAtGpu<T1> &realSpace, mycufftHandle &myhandle);

	template <typename T1>
	void fftStream(GpuMultidimArrayAtGpu<T1> &fourierTransform, mycufftHandle &myhandle, myStreamHandle &myStream,
			bool useCallback, GpuMultidimArrayAtGpu< std::complex<float> > &dataRef);

	// RealSpace must already be resized
	template <typename T1>
	void ifftStream(GpuMultidimArrayAtGpu<T1> &realSpace, mycufftHandle &myhandle, myStreamHandle &myStream,
			bool useCallback, GpuMultidimArrayAtGpu< std::complex<float> > &dataExp);


	void calculateMax(float *max_values, float *posX, float *posY, int fixPadding);


private:
	void setDims(size_t _Xdim, size_t _Ydim=1, size_t _Zdim=1, size_t _Ndim=1) {
		Xdim=_Xdim;
		Ydim=_Ydim;
		Zdim=_Zdim;
		Ndim=_Ndim;
		yxdim=(size_t)_Ydim*_Xdim;
		zyxdim=yxdim*_Zdim;
		nzyxdim=zyxdim*_Ndim;
	}
};

#endif
