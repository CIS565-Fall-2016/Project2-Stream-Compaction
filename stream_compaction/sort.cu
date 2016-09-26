#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "naive.h"
#include "sort.h"

namespace StreamCompaction {
namespace Sort {

void printArray(int n, int *a, bool abridged = false) {
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


__global__ void kernGetBit(int n, int d, int *bits, int *nbits, const int *data) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}

	int s = (data[index] & (1 << d)) >> d;

	bits[index] = s;
	nbits[index] = 1 - s;
}


__global__ void kernScanTrue(int n, int *trues, const int *falses, const int *lastBit) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}

	trues[index] = index - falses[index] + falses[n - 1] + lastBit[n - 1];
}

__global__ void kernDestination(int n, int *dest, int *trues, int *falses, int* bits) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}

	dest[index] = bits[index] ? trues[index] : falses[index];
}


void radix(int n, const int k, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	// create device arrays
	int *dev_out;
	int *dev_in;
	int *dev_bits;
	int *dev_nbits;
	int *dev_true;
	int *dev_false;
	int *dev_dest;
	int *dev_all;

	cudaMalloc((void**)&dev_out, n*sizeof(int));
	cudaMalloc((void**)&dev_in, n*sizeof(int));
	cudaMalloc((void**)&dev_bits, n*sizeof(int));
	cudaMalloc((void**)&dev_nbits, n*sizeof(int));
	cudaMalloc((void**)&dev_true, n*sizeof(int));
	cudaMalloc((void**)&dev_false, n*sizeof(int));
	cudaMalloc((void**)&dev_dest, n*sizeof(int));
	cudaMalloc((void**)&dev_all, n*sizeof(int));

	cudaMemset(dev_all, 0, n);
	cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	for (int d = 0; d < k; d++) {
		kernGetBit << <fullBlocksPerGrid, blockSize >> >(n, d, dev_bits, dev_nbits, dev_in);
		cudaMemcpy(dev_false, dev_nbits, n*sizeof(int), cudaMemcpyDeviceToDevice);	
		StreamCompaction::Efficient::scan_dev(n, dev_false);
		kernScanTrue << <fullBlocksPerGrid, blockSize >> >(n, dev_true, dev_false, dev_nbits);
		kernDestination << <fullBlocksPerGrid, blockSize >> >(n, dev_dest, dev_true, dev_false, dev_bits);
		StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in, NULL, dev_dest);
		cudaMemcpy(dev_in, dev_out, n*sizeof(int), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(odata, dev_out, n*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_out);
	cudaFree(dev_in);
	cudaFree(dev_bits);
	cudaFree(dev_nbits);
	cudaFree(dev_true);
	cudaFree(dev_false);
	cudaFree(dev_dest);
	cudaFree(dev_all);
}


void radix_dev(int n, const int k, int *dev_out, int *dev_in) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	// create device arrays
	int *dev_bits;
	int *dev_nbits;
	int *dev_true;
	int *dev_false;
	int *dev_dest;
	int *dev_all;

	cudaMalloc((void**)&dev_bits, n*sizeof(int));
	cudaMalloc((void**)&dev_nbits, n*sizeof(int));
	cudaMalloc((void**)&dev_true, n*sizeof(int));
	cudaMalloc((void**)&dev_false, n*sizeof(int));
	cudaMalloc((void**)&dev_dest, n*sizeof(int));
	cudaMalloc((void**)&dev_all, n*sizeof(int));

	cudaMemset(dev_all, 0, n);

	for (int d = 0; d < k; d++) {
		kernGetBit << <fullBlocksPerGrid, blockSize >> >(n, d, dev_bits, dev_nbits, dev_in);
		cudaMemcpy(dev_false, dev_nbits, n*sizeof(int), cudaMemcpyDeviceToDevice);
		StreamCompaction::Efficient::scan_dev(n, dev_false);
		kernScanTrue << <fullBlocksPerGrid, blockSize >> >(n, dev_true, dev_false, dev_nbits);
		kernDestination << <fullBlocksPerGrid, blockSize >> >(n, dev_dest, dev_true, dev_false, dev_bits);
		StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in, NULL, dev_dest);
		cudaMemcpy(dev_in, dev_out, n*sizeof(int), cudaMemcpyDeviceToDevice);
	}

	cudaFree(dev_bits);
	cudaFree(dev_nbits);
	cudaFree(dev_true);
	cudaFree(dev_false);
	cudaFree(dev_dest);
	cudaFree(dev_all);
}


void TestSort(int n, const int k, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	int *dev_out;
	int *dev_in;
	cudaMalloc((void**)&dev_out, n*sizeof(int));
	cudaMalloc((void**)&dev_in, n*sizeof(int));
	cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 0; i < samp; i++) {
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		radix_dev(n, k, dev_out, dev_in);
		cudaThreadSynchronize(); // block until kernel is finished

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}

	cudaMemcpy(odata, dev_out, n*sizeof(int), cudaMemcpyDeviceToHost);
	printf("     %f\n", time);

}

}
}