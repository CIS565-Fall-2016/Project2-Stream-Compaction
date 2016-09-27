#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radixSort.h"
#include "efficient.h"

namespace StreamCompaction {
namespace RadixSort {
#define blockSize 128

int getNbit(int input, int nth){
	return (input >> nth) & 1;
}

// assume the input and output are bits
__global__ void computeE(int n, int * edata, const int * bdata){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		edata[index] = ~(bdata[index]);
		//if (index ==0){
		//	odata[index] = ~(0|idata[index]);
		//}
		//else {
		//	odata[index] = ~(idata[index-1]|idata[index]);
		//}
	}
}
__global__ void computeT(int n, int * tdata, const int * fdata, const int totalFalses){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n){
		tdata[index] = index - fdata[index] + totalFalses;
	}
}

__global__ void computeB(int n, int *bdata, const int *idata, int ith){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n){
		bdata[index] = getNbit(idata[index], ith);
	}
}

__global__ void computeD(int n, int *ddata, const int * bdata, const int *tdata, const int * fdata){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n){
		ddata[index] = bdata[index] ? tdata[index] : fdata[index];
	}
}

__global__ void scatter(int n, int *odata, int *idata, const int * ddata){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n){
		odata[index]= idata[ddata[index]];
	}
}
/**
 * radix sort 
 */
void sort(int n, int *odata, const int *idata, int msb) {
	dim3 numblocks(std::ceil((double) n / blockSize));

	int * idata_buff;
	int * idata_buff2;
	int * bdata_buff;
	int * edata_buff;
	int * fdata_buff;
	int * tdata_buff;
	int * ddata_buff;


	cudaMalloc((void**)&idata_buff,n*sizeof(int));
		checkCUDAError("cudaMalloc-idata_buff-  failed!");	
	cudaMalloc((void**)&idata_buff2,n*sizeof(int));
		checkCUDAError("cudaMalloc-idata_buff2-  failed!");	
	cudaMalloc((void**)&bdata_buff,n*sizeof(int));
		checkCUDAError("cudaMalloc-bdata_buff-  failed!");	
	cudaMalloc((void**)&edata_buff,n*sizeof(int));
		checkCUDAError("cudaMalloc-edata_buff-  failed!");	
	cudaMalloc((void**)&fdata_buff,n*sizeof(int));
		checkCUDAError("cudaMalloc-fdata_buff-  failed!");	
	cudaMalloc((void**)&tdata_buff,n*sizeof(int));
		checkCUDAError("cudaMalloc-tdata_buff-  failed!");	
	cudaMalloc((void**)&ddata_buff,n*sizeof(int));
		checkCUDAError("cudaMalloc-ddata_buff-  failed!");	

	/// CPU -->GPU
	cudaMemcpy(idata_buff,idata,n*sizeof(int),cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy-idata_buff-failed");

	for (int i=0; i< msb; i++){
		int totalFalses = 0;

		//find b array for each bit
		computeB<<<numblocks, blockSize>>>(n, bdata_buff, idata_buff, i);
		computeE<<<numblocks, blockSize>>>(n,  edata_buff, bdata_buff);
		StreamCompaction::Efficient::scan(n, fdata_buff, edata_buff);

		totalFalses = edata_buff[n-1] + fdata_buff[n-1];

		computeT<<<numblocks, blockSize>>>(n,  tdata_buff,  fdata_buff,  totalFalses);
		computeD<<<numblocks, blockSize>>>(n, ddata_buff, bdata_buff, tdata_buff, fdata_buff);


		//scatter darray for this bit
		scatter<<<numblocks, blockSize>>>(n, idata_buff2, ddata_buff, idata_buff);
		cudaMemcpy(idata_buff,idata_buff2,n*sizeof(int),cudaMemcpyDeviceToDevice);
		checkCUDAError("cudaMemcpy-idata_buff-failed");
	}

	//GPU --> CPU

	//free
	cudaFree(idata_buff);
	cudaFree(idata_buff2);
	cudaFree(bdata_buff);
	cudaFree(tdata_buff);
	cudaFree(fdata_buff);
	cudaFree(edata_buff);
	cudaFree(ddata_buff);
} 

}
}
