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
__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
    // TODO
	int parallelCount = threadIdx.x+blockIdx.x*blockDim.x;
	if(parallelCount<n)
	{
	    if(idata[parallelCount]==0)
	{
	    bools[parallelCount]=0;
	}
	else{
	    bools[parallelCount]=1;
	}
	}
	
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, int *odata,
        const int *idata, const int *bools, const int *indices) {
    // TODO
	int parallelCount = threadIdx.x+blockIdx.x*blockDim.x;
	if(bools[parallelCount]==1)
	{
	    odata[indices[parallelCount]]=idata[parallelCount];
	}
}

__global__ void inclusiveToExclusive(int n, int *idata, int *odata)
{
    int parallelCount = threadIdx.x+blockIdx.x*blockDim.x;
	if(parallelCount<n)
	{
	    if(parallelCount == 0)
		{
		    odata[0]=0;
			return;
		}

		odata[parallelCount] = idata[parallelCount-1];
	}
}

}
}
