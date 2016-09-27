#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`

	thrust::device_vector<int> devIdata(idata, idata + n);
	thrust::device_vector<int> devOdata(odata, odata + n);

	// example: for device_vectors dv_in and dv_out:
	// thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

	//Add performance analysis
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	thrust::exclusive_scan(devIdata.begin(), devIdata.end(), devOdata.begin());

	//Add performance analysis
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float deltaTime;
	cudaEventElapsedTime(&deltaTime, start, end);
	printf("GPU Thrust Scan time is %f ms\n", deltaTime);

	thrust::copy(devOdata.begin(), devOdata.end(), odata);
}

}
}
