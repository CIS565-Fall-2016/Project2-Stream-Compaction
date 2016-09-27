#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

const int BlockSize = StreamCompaction::Common::BlockSize;

static StreamCompaction::Common::Timer timer;

// work - efficient parallel scan 
// stride = 2^(d+1)
__global__ void kernUpSweep(int n, int stride, int *x)
{
	// overflow if using int for k
	long long k = threadIdx.x + blockDim.x * blockIdx.x;
	k *= stride;
	if (k >= n)
		return;

	int halfStride = (stride >> 1);
	x[k + stride - 1] += x[k + halfStride - 1];
}

// stride = 2^(d+1)
__global__ void kernDownSweep(int n, int stride, int *x)
{
	long long k = threadIdx.x + blockDim.x * blockIdx.x;
	k *= stride;
	if (k >= n)
		return;

	int halfStride = (stride >> 1);
	int t = x[k + halfStride - 1];
	x[k + halfStride - 1] = x[k + stride - 1];
	x[k + stride - 1] += t;
}

// helper function for set data[index] = 0
__global__ void kernSetZero(int index, int* data)
{
	data[index] = 0;
}

// exclusive scan data is pointer on device
void scan_device(int n, int *data)
{
	int maxD = ilog2ceil(n) - 1;

	// up
	int stride;

	for (int d = 0; d <= maxD; ++d)
	{
		stride = (1 << (d + 1));
		int blockNumber = (n / stride + BlockSize - 1) / BlockSize;
		kernUpSweep << <blockNumber, BlockSize >> >(n, stride, data);
	}
	checkCUDAError("1");

	// set last to zero !
	kernSetZero << <1, 1 >> >(n - 1, data);

	// down
	for (int d = maxD; d >= 0; d--)
	{
		stride = (1 << (d + 1));
		int blockNumber = (n / stride + BlockSize - 1) / BlockSize;
		kernDownSweep << <blockNumber, BlockSize >> >(n, stride, data);
	}

	checkCUDAError("2");

}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) 
{

	int bufferSize = (1 << ilog2ceil(n));
	int *dev_buffer;

	cudaMalloc((void**)&dev_buffer, sizeof(int)*bufferSize);
	checkCUDAError("cudaMalloc dev failed");

	cudaMemset(dev_buffer, 0, bufferSize*sizeof(int));

	cudaMemcpy(dev_buffer, idata, sizeof(int)*n, cudaMemcpyHostToDevice);


	timer.startGpuTimer();

	scan_device(bufferSize, dev_buffer);

	timer.stopGpuTimer();
	timer.printTimerInfo("Scan::GPU::Efficient = ", timer.getGpuElapsedTime());

	cudaMemcpy(odata, dev_buffer, sizeof(int)*n, cudaMemcpyDeviceToHost);

	cudaFree(dev_buffer);
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

	int bufferSize = (1 << ilog2ceil(n));

	int *dev_input, *dev_bools, *dev_indices, *dev_output;

	cudaMalloc((void**)&dev_input, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_input failed");

	cudaMalloc((void**)&dev_bools, bufferSize * sizeof(int));
	checkCUDAError("cudaMalloc dev_bools failed");

	cudaMalloc((void**)&dev_indices, bufferSize * sizeof(int));
	checkCUDAError("cudaMalloc dev_indices failed");

	cudaMalloc((void**)&dev_output, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_out failed");

	cudaMemcpy(dev_input, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemset(dev_bools, 0, bufferSize*sizeof(int));

	timer.startGpuTimer();

	// map to booleans 
	dim3 blocks((n + BlockSize - 1) / BlockSize);
	Common::kernMapToBoolean << <blocks, BlockSize >> >(n, dev_bools, dev_input);

	cudaMemcpy(dev_indices, dev_bools, bufferSize*sizeof(int), cudaMemcpyDeviceToDevice);
	
	// run scan
	scan_device(bufferSize, dev_indices);

	// scatter 
	Common::kernScatter << <blocks, BlockSize >> >(n, dev_output, dev_input, dev_bools, dev_indices);

	timer.stopGpuTimer();
	timer.printTimerInfo("Compact::GPU::Efficient = ", timer.getGpuElapsedTime());

	// get length
	int len;
	cudaMemcpy(&len, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

	if (idata[n - 1] != 0) // last element
	{
		len++;
	}

	
	// copy result to odata
	cudaMemcpy(odata, dev_output, len*sizeof(int), cudaMemcpyDeviceToHost);


	// free memory
	cudaFree(dev_input);
	cudaFree(dev_bools);
	cudaFree(dev_indices);
	cudaFree(dev_output);

    return len;
}

}
}
