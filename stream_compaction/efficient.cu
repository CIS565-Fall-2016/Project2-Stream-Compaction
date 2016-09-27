#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__
	__global__ void upSweep(int N, int d, int *idata) {
		int n = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (n >= N) {
			return;
		}
		int delta = 1 << d;
		int doubleDelta = 1 << (d + 1);
		if (n % doubleDelta == 0) {
			idata[n + doubleDelta - 1] += idata[n + delta - 1];
		}
	}

	__global__ void downSweep(int N, int d, int *idata) {
		int n = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (n >= N) {
			return;
		}
		int delta = 1 << d;
		int doubleDelta = 1 << (d + 1);
		if (n % doubleDelta == 0) {
			int temp = idata[n + delta - 1];
			idata[n + delta - 1] = idata[n + doubleDelta - 1];
			idata[n + doubleDelta - 1] += temp;
		}
	}

	void scanInDevice(int n, int *devData) {
		int blockNum = (n + blockSize - 1) / blockSize;
		for (int d = 0; d < ilog2ceil(n) - 1; d++) {
			upSweep << <blockNum, blockSize >> >(n, d, devData);
			checkCUDAError("upSweep not correct...");
		}
		//set last element to zero, refer to slides!
		int counter = 0;
		cudaMemcpy(&devData[n - 1], &counter, sizeof(int), cudaMemcpyHostToDevice);

		for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
			downSweep << <blockNum, blockSize >> >(n, d, devData);
			checkCUDAError("downSweep not correct...");
		}
	}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    // printf("TODO\n");
	int *temp;
	int realN;

	// google for bit operation
	// if n is not 2^a number
	if (n & (n - 1) != 0) { 
	// enlarge to be a 2^a number
		realN = 1 << (ilog2ceil(n));
		temp = (int*)malloc(realN * sizeof(int));
		memcpy(temp, idata, realN * sizeof(int));
	// update the new added elements to zero
		for (int j = n; j < realN; j++) {
			temp[j] = 0;
		}

	} else { // is 2^a
		//do nothing, realN is n
		realN = n;
		temp = (int*)malloc(realN * sizeof(int));
		memcpy(temp, idata, realN * sizeof(int));
	}

	int arraySize = realN * sizeof(int);
	int *devIdata;

	cudaMalloc((void**)&devIdata, arraySize);
	checkCUDAError("cudaMalloc devIdata failed");
	cudaMemcpy(devIdata, temp, arraySize, cudaMemcpyHostToDevice);

	//Add performance analysis
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	//call scanInDevice Function
	scanInDevice(realN, devIdata);

	//Add performance analysis
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float deltaTime;
	cudaEventElapsedTime(&deltaTime, start, end);
	printf("GPU Efficient Scan time is %f ms\n", deltaTime);
	cudaMemcpy(odata, devIdata, arraySize, cudaMemcpyDeviceToHost);
	cudaFree(devIdata);
	 
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    // TODO
    // return -1;
	int *temp;
	int realN;

	if (n & (n - 1) != 0) { // if size is not a power of 2
		// enlarge to be a 2^a number
		realN = 1 << (ilog2ceil(n));
		temp = (int*)malloc(realN * sizeof(int));
		memcpy(temp, idata, realN * sizeof(int));

		// update the new added elements to zero
		for (int j = n; j < realN; j++) {
			temp[j] = 0;
		}

	} else { // is 2^a
		//do nothing, realN is n
		realN = n;
		temp = (int*)malloc(realN * sizeof(int));
		memcpy(temp, idata, realN * sizeof(int));
	}

	int arraySize = realN * sizeof(int);
	int blockNum = (realN + blockSize - 1) / blockSize;

	int *devIdata;
	int *devOdata;
	int *devIndex;

	cudaMalloc((void**)&devIdata, arraySize);
	checkCUDAError("cudaMalloc devIdata failed");
	cudaMalloc((void**)&devOdata, arraySize);
	checkCUDAError("cudaMalloc devOdata failed");
	cudaMalloc((void**)&devIndex, arraySize); 
	checkCUDAError("cudaMalloc devIndex failed");
	
	cudaMemcpy(devIdata, temp, arraySize, cudaMemcpyHostToDevice);

	//Add performance analysis
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	StreamCompaction::Common::kernMapToBoolean << <blockNum, blockSize >> >(realN, devIndex, devIdata);
	int lastElem;
	cudaMemcpy(&lastElem, devIndex + realN - 1, sizeof(int), cudaMemcpyDeviceToHost);

	scanInDevice(realN, devIndex);
	int size;
	cudaMemcpy(&size, devIndex + realN - 1, sizeof(int), cudaMemcpyDeviceToHost);

	StreamCompaction::Common::kernScatter << <blockNum, blockSize >> >(realN, devOdata, devIdata, devIndex, devIndex);

	//Add performance analysis
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float deltaTime;
	cudaEventElapsedTime(&deltaTime, start, end);
	printf("GPU Efficient Compact time is %f ms\n", deltaTime);

	cudaMemcpy(odata, devOdata, arraySize, cudaMemcpyDeviceToHost);

	//exclusive scan
	if (lastElem == 1) {
		size++;
	}

	cudaFree(devIdata);
	cudaFree(devOdata);
	cudaFree(devIndex);

	return size;

}

}
}
