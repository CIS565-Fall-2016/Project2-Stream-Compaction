#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

namespace StreamCompaction {
namespace Common {

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int *bools, const int *idata)
{
	const int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
	if (0 <= iSelf && iSelf < n)
	{
		const bool elementIsValid = (idata[iSelf] != 0);
		bools[iSelf] = elementIsValid;
	}
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, int *odata,
        const int *idata, const int *bools, const int *indices)
{
	const int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
	if (0 <= iSelf && iSelf < n)
	{
		if (bools[iSelf] == 1)
		{
			const int dataIdx = indices[iSelf];
			odata[dataIdx] = idata[iSelf];
		}
	}
}

__global__ void kernScatter(int n, int *odata, const int *idata, const int *indices)
{
	const int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
	if (0 <= iSelf && iSelf < n)
	{
		const int dataIdx = indices[iSelf];
		odata[dataIdx] = idata[iSelf];
	}
}

__global__ void convertInclusiveToExclusiveScan(int N, int* inInclusiveScan, int* outExclusiveScan)
{
	const int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
	if (iSelf < N)
	{
		if (iSelf > 0)
		{
			outExclusiveScan[iSelf] = inInclusiveScan[iSelf - 1];
		}
		else
		{
			outExclusiveScan[iSelf] = 0;
		}
	}
}

}
}
