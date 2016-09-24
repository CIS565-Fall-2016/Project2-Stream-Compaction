#include <cuda.h>
#include <cuda_runtime.h> 
#include "common.h"
#include "radix.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Radix {

	//b array
	__global__ void kernTestTrueFalseOnRightKthBit(int n, int k, int* odata, const int* idata) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		if (((1 << k) & idata[index]) == (1<<k)) {
			odata[index] = 1;
		} else {
			odata[index] = 0;
		}

	}

	//e array(! operation)
	__global__ void kernNotOperatorOnArray(int n, int *odata, const int *idata) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= n) {
			return;
		}
		odata[index] = 1 - idata[index];
	}

	//t array
	__global__ void kernComputeTArray(int n, const int totalFalses, int *idata, int *odata) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= n) {
			return;
		}
		odata[index] = index - idata[index] + totalFalses;
	}

	//d array
	__global__ void kernComputeDArray(int n, int * dArray, const int *bArray, const int *fArray, const int *tArray) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= n) {
			return;
		}
		dArray[index] = (bArray[index] ? tArray[index] : fArray[index]);
	}

	//Reshuffle Index
	__global__ void kernReshuffle(int n, int* idata, int *odata, const int *dArray) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= n) {
			return;
		}
		odata[dArray[index]] = idata[index];
	}

	void RadixSort(int n, int* idata, int maxNum) {
		int *devIdata; int *devOdata;
		int *bArray;  int *eArray; int *fArray;	int *tArray; int *dArray;
		int realN = 0;

		//Where bugs come from......
		if (n & (n - 1) != 0){
			realN = 1 << (ilog2ceil(n));
		} else {
			realN = n;
		}

		cudaMalloc((void**)&devIdata, sizeof(int) * n);
		checkCUDAError("cudaMalloc radix devIdata array failed");

		cudaMalloc((void**)&devOdata, sizeof(int) * n);
		checkCUDAError("cudaMalloc radix devOdata array failed");

		cudaMalloc((void**)&bArray, sizeof(int) * n);
		checkCUDAError("cudaMalloc radix bArray failed");

		cudaMalloc((void**)&eArray, sizeof(int) * n);
		checkCUDAError("cudaMalloc radix eArray failed");

		//Remember do realN here......
		cudaMalloc((void**)&fArray, sizeof(int) * realN);
		cudaMemset(fArray, realN, 0);
		checkCUDAError("cudaMalloc radix fArray failed");

		cudaMalloc((void**)&tArray, sizeof(int) * n);
		checkCUDAError("cudaMalloc radix tArray failed");

		cudaMalloc((void**)&dArray, sizeof(int) * n);
		checkCUDAError("cudaMalloc radix dArray failed");

		int blockNum = (n + blockSize - 1) / blockSize;
		int digitNum = ilog2ceil(maxNum);

		cudaMemcpy(devIdata, idata, n*sizeof(int), cudaMemcpyHostToDevice);		

		for (int i = 0; i < digitNum; i++) {
			kernTestTrueFalseOnRightKthBit << < blockNum, blockSize >> >(n, i, bArray, devIdata);
			kernNotOperatorOnArray << < blockNum, blockSize >> >(n, eArray, bArray);

			//fArray could directly use efficient scan of eArray!
			cudaMemcpy(fArray, eArray, n*sizeof(int), cudaMemcpyDeviceToDevice);			
			StreamCompaction::Efficient::scanInDevice(realN, fArray);

			//Slides....
			int totalFalses;
			cudaMemcpy(&totalFalses, fArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			int lastElement;
			cudaMemcpy(&lastElement, eArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			totalFalses += lastElement;

			kernComputeTArray << < blockNum, blockSize >> >(n, totalFalses, fArray, tArray);
			kernComputeDArray << < blockNum, blockSize >> >(n, dArray, bArray, fArray, tArray);
			kernReshuffle << <blockNum, blockSize >> >(n, devIdata, devOdata, dArray);
			std::swap(devOdata, devIdata);
		}

		cudaMemcpy(idata, devIdata, n * sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(devIdata); cudaFree(devOdata); cudaFree(bArray); cudaFree(eArray);
		cudaFree(fArray); cudaFree(tArray); cudaFree(dArray);

	}


}
}
