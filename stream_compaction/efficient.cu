#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

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
	/* Safer way to do this would be to use cudaPointerGetAttributes to check if already on device*/

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
	checkCUDAError("cudaMemset dev_data 1 failed!");

	//up sweep
	for (int depth = 0; depth < ilog2ceil(n); ++depth)
	{
		upSweep << <numBlocks, blocksize >> >(rounded_n, depth, dev_data);
		checkCUDAError("upSweep Kernel failed!");
	}

	// place 0 at end of array
	cudaMemset(&dev_data[n-1], 0, sizeof(int) * (rounded_n - n + 1)); 
	checkCUDAError("cudaMemset dev_data 2 failed!");
	
	//down-sweep
	for (int depth = ilog2ceil(n) - 1; depth >= 0; --depth)
	{
		downSweep << <numBlocks, blocksize >> >(rounded_n, depth, dev_data);
		checkCUDAError("downSweep Kernel failed!");
	}


	//copy back only the relevant data
	cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy dev_data to odata failed!");

	cudaFree(dev_data);
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
	dim3 numBlocks = (n + blocksize - 1) / blocksize;

	//device pointers
	int * dev_idata;
	int * dev_odata;
	int * dev_bools;
	int * dev_indices;
	cudaMalloc((void **)&dev_idata, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_idata failed!");
	cudaMalloc((void **)&dev_odata, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_odata failed!");
	cudaMalloc((void **)&dev_bools, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_bools failed!");
	cudaMalloc((void **)&dev_indices, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_indices failed!");

	//host pointers
	int * bools = (int *)malloc(n * sizeof(int));
	int * scanResult = (int *)malloc(n * sizeof(int));

	//compute temp array of bools
	cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy idata to dev_idata failed!");
	Common::kernMapToBoolean << < numBlocks, blocksize >> >(n, dev_bools, dev_idata);
	checkCUDAError("kernMapToBoolean failed!");
	cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy dev_bools to bools failed!");

	
	//run exclusive scan on temp array
	scan(n, scanResult, bools);
	cudaMemcpy(dev_indices, scanResult, sizeof(int) * n, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy scanResult to dev_indices failed!");

	//scatter
	cudaMemset(dev_odata, 0, n * sizeof(int));
	checkCUDAError("cudaMemset dev_odata failed!");
	Common::kernScatter << < numBlocks, blocksize >> >(n, dev_odata,
		dev_idata, dev_bools, dev_indices);
	checkCUDAError("kernScatter failed!");

	// copy compacted array to output
	cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy dev_odata to odata failed!");
	
	int cnt = bools[n-1] ? scanResult[n-1] + 1 : scanResult[n-1];
	
	cudaFree(dev_bools);
	cudaFree(dev_idata);
	cudaFree(dev_odata);
	cudaFree(dev_indices);
	free(bools);
	free(scanResult);

	// return # valid blocks
	return cnt;

}

}
}
