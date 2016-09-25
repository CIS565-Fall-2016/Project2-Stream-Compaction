#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

#define blockSize 128
//__global__
__global__ void scan(int offset, int n, int *odata, const int *idata) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >=n) return;

	if (index >= offset){
		odata[index] = idata[index] + idata[index-offset];
	}
	else{
		odata[index] = idata[index];
	}
}
__global__ void excludesiveShift(int n, int *odata, int *idata){
	int index = threadIdx.x + blockIdx.x* blockDim.x;
	if (index>=n) return;
	if (index>=1){
		odata[index]= idata[index-1] ;
	}
	else {
		odata[index]= 0;
	}
}
/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
 
	//dim3 numblocks(std::ceil((double) n / blockSize));
	dim3 numblocks((n + blockSize - 1) / blockSize);
	int* idata_buff;
	int* odata_buff;

	cudaMalloc((void**)&idata_buff, n*sizeof(int));
	checkCUDAError("cudaMalloc-idata_buff-  failed!");	
	cudaMalloc((void**)&odata_buff, n*sizeof(int));
	checkCUDAError("cudaMalloc-odata_buff-failed!");

	/// CPU -->GPU
	cudaMemcpy(idata_buff,idata,sizeof(int)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(odata_buff,idata,sizeof(int)*n,cudaMemcpyHostToDevice);

	 
	for (int level= 1; level <= ilog2ceil(n); level++) {
		int offset; 
		if (level==1){
			offset = 1;
		}
		else {
			offset = 2 << (level -2);
		}
		// for the given level, all threads read from idata_buff
		scan <<<numblocks, blockSize>>>(offset, n, odata_buff, idata_buff);
		//std::swap(idata_buff, odata_buff);
		// odata_buff --> idata_buff for next iteration
		cudaMemcpy(idata_buff, odata_buff, sizeof(int)*n, cudaMemcpyDeviceToDevice);
	}
	excludesiveShift<<<numblocks, blockSize>>>(n, odata_buff, idata_buff);

	//GPU --> CPU 	
	cudaMemcpy(odata, odata_buff, sizeof(int)*n, cudaMemcpyDeviceToHost);
	cudaFree(idata_buff);
	cudaFree(odata_buff);
}

}
}
