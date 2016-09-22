#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
namespace Naive {

	__global__ void naiveScan(int n, int *odata, int *idata, int val) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}
		if (index >= val) {
			odata[index] = idata[index - val] + idata[index];
		}
		else {
			odata[index] = idata[index];
		}
	}

	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int *odata, const int *idata) {
		int *dev_odata;
		int *dev_idata;
		int flag = 1;
		int val;

		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

		cudaMalloc((void**)&dev_odata, n * sizeof(int));
		cudaMalloc((void**)&dev_idata, n * sizeof(int));

		cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);

		for (int d = 1; d <= ilog2ceil(n); d++) { //ilog2ceil(n)
			val = pow(2, d - 1);
			if (flag == 1)	{
				naiveScan << <fullBlocksPerGrid, blockSize >> >(n, dev_odata, dev_idata, val);
				flag = 0;
			}
			else {
				naiveScan << <fullBlocksPerGrid, blockSize >> >(n, dev_idata, dev_odata, val);
				flag = 1;
			}
			cudaThreadSynchronize();
		}

		odata[0] = 0;
		if (flag == 0) {
			cudaMemcpy(odata+1, dev_odata, (n-1)*sizeof(int), cudaMemcpyDeviceToHost);
		}
		else {
			cudaMemcpy(odata+1, dev_idata, (n-1)*sizeof(int), cudaMemcpyDeviceToHost);
		}

		cudaFree(dev_idata);
		cudaFree(dev_odata);
	}

}
}
