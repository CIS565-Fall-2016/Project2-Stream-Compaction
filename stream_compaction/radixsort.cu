#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radixsort.h"
#include "efficient.h"

namespace StreamCompaction {
namespace RadixSort {
	
const int BlockSize = StreamCompaction::Common::BlockSize;

static StreamCompaction::Common::Timer timer;

__global__ void kernComputeBArray(int n, int k, int *out, int *in)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n)
		return;

	int bitK = (1 << k);
	out[index] = int( (in[index] & bitK) == bitK );
}

__global__ void kernComputeEArray(int n,int *out,int *in)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n)
		return;

	out[index] = 1 - in[index];
}

__global__ void kernComputeTArray(int n, const int totalFalses, int *out, int *in)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n)
		return;

	out[index] = index - in[index] + totalFalses;
}

__global__ void kernComputeDArray(int n, int * d, const int *b, const int *t, const int *f)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n)
		return;

	d[index] = (b[index] ? t[index] : f[index]);
}


__global__ void kernRearrangeInput(int n, int *out, int* in, const int *d)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n)
		return;

	out[d[index]] = in[index];
}

/**
 * radix sort
 */
void RadixSort(int n, int* data, int maxValue)
{
	int * dev_input = NULL, *dev_input2 = NULL;
	int * b = NULL, *e = NULL, *f = NULL, *t = NULL , *d = NULL;

	cudaMalloc((void**)&dev_input, sizeof(int)*n);
	checkCUDAError("cudaMalloc radix sort dev_input array failed");

	cudaMalloc((void**)&dev_input2, sizeof(int)*n);
	checkCUDAError("cudaMalloc radix sort dev_input2 array failed");

	cudaMalloc((void**)&b, sizeof(int)*n);
	checkCUDAError("cudaMalloc radix sort b array failed");

	cudaMalloc((void**)&e, sizeof(int)*n);
	checkCUDAError("cudaMalloc radix sort e array failed");

	cudaMalloc((void**)&f, sizeof(int)*n);
	checkCUDAError("cudaMalloc radix sort f array failed");

	cudaMalloc((void**)&t, sizeof(int)*n);
	checkCUDAError("cudaMalloc radix sort t array failed");

	cudaMalloc((void**)&d, sizeof(int)*n);
	checkCUDAError("cudaMalloc radix sort d array failed");

	// copy input data to device pointer
	cudaMemcpy(dev_input, data, n*sizeof(int), cudaMemcpyHostToDevice);

	timer.startGpuTimer();

	dim3 gridSize((n + BlockSize - 1) / BlockSize);

	int numDigitals = ilog2ceil(maxValue);

	for (int i = 0; i < numDigitals; i++)
	{
		// b array
		kernComputeBArray << < gridSize, BlockSize >> >(n, i, b, dev_input);
		
		// e array
		kernComputeEArray << < gridSize, BlockSize >> >(n, e, b);
		
		// f array
		cudaMemcpy(f, e, n*sizeof(int), cudaMemcpyDeviceToDevice);
		StreamCompaction::Efficient::scan_device(n, f);
		
		// totalfalses and t array 
		int totalFalses;
		cudaMemcpy(&totalFalses, f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
		int lastE;
		cudaMemcpy(&lastE, e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
		totalFalses += lastE;

		kernComputeTArray << < gridSize, BlockSize >> >(n, totalFalses, t, f);

		// d array
		kernComputeDArray << < gridSize, BlockSize >> >(n, d, b, t, f);

		// re-arrange dev_input
		kernRearrangeInput << <gridSize, BlockSize >> >(n, dev_input2, dev_input, d);
		std::swap(dev_input2, dev_input);

	}
	
	timer.stopGpuTimer();
	timer.printTimerInfo("GPU::Radix Sort = ",timer.getGpuElapsedTime());

	cudaMemcpy(data, dev_input, n*sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(dev_input);
	cudaFree(dev_input2);
	cudaFree(b);
	cudaFree(e);
	cudaFree(f);
	cudaFree(t);
	cudaFree(d);

}

}
}
