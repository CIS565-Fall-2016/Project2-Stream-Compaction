#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define DEBUG 1

void printArray3(int n, int *a, bool abridged = true) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15 && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%3d ", a[i]);
	}
	printf("]\n");
}

namespace StreamCompaction {
namespace Efficient {

__global__ void upSweep(int n, int depth, int *data)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= n)
		return;

	int k = index * powf(2, depth + 1) ;

	if ( k >= n )
		return;

	int myIndex = k + powf(2, depth + 1) - 1;
	int neighborIndex = k + powf(2, depth) - 1;

	data[myIndex] += data[neighborIndex];
}

__global__ void downSweep(int n, int depth, int *data)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= n)
		return;

	int k = index * powf(2, depth + 1);

	if (k >= n)
		return;

	// save left child
	int leftChildIndex = k + powf(2, depth) - 1;

	int temp = data[leftChildIndex];

	//set right child to this node's value
	int rightChildIndex = k + powf(2, depth + 1) - 1;
	data[leftChildIndex] = data[rightChildIndex];

	// set right child tp old left + this value
	data[rightChildIndex] += temp;

}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	dim3 numBlocks = (n + blocksize - 1) / blocksize;
	int rounded_n = powf(2, ilog2ceil(n));
	int * dev_data;

	cudaMalloc((void **)&dev_data, rounded_n * sizeof(int));
	checkCUDAError("cudaMalloc dev_data failed!");

	cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy idata to dev_data failed!");

	// Fill in remaining space with zeros. 
	// Not sure this matters actually matters, so long as the correct
	// data range is copied back at the end
	cudaMemset(&dev_data[n], 0, sizeof(int) * (rounded_n - n));
	checkCUDAError("cudaMemset dev_data failed!");

	//up sweep
	for (int depth = 0; depth < ilog2ceil(n); ++depth)
	{
#if DEBUG
		printf("--------------before upsweep %d-------------------\n", depth);
		cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
		printArray3(n, odata);
#endif
		upSweep << <numBlocks, blocksize >> >(rounded_n, depth, dev_data);
		checkCUDAError("upSweep Kernel failed!");
#if DEBUG
		printf("--------------after upsweep %d-------------------\n", depth);
		cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
		printArray3(n, odata);
		printf("-----------------------------------------------\n", depth);
#endif
	}

	//down-sweep
	cudaMemset(&dev_data[n-1], 0, sizeof(int) * (rounded_n - n + 1)); //pretty questionable
#if DEBUG
	int * testZeroes = (int *)malloc(sizeof (int) * rounded_n);
	printf("--------------test zeros-------------------\n");
	cudaMemcpy(testZeroes, dev_data, sizeof(int) * rounded_n, cudaMemcpyDeviceToHost);
	printArray3(rounded_n, testZeroes, false);
	free(testZeroes);
#endif

	printf("whats going on here? rounded_n = %d, n = %d\n", rounded_n, n);
	for (int depth = ilog2ceil(n) - 1; depth >= 0; --depth)
	{
#if DEBUG
		printf("--------------before downsweep %d-------------------\n", depth);
		cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
		printArray3(n, odata);
#endif
		downSweep << <numBlocks, blocksize >> >(rounded_n, depth, dev_data);
		checkCUDAError("downSweep Kernel failed!");
#if DEBUG
		printf("--------------after downsweep %d-------------------\n", depth);
		cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
		printArray3(n, odata);
		printf("-----------------------------------------------\n", depth);
#endif
	}


	//copy back only the relevant data
	cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy dev_data to odata failed!");

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
    return -1;
}

}
}
