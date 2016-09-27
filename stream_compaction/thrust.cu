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
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	thrust::device_vector<int> thrust_idata(idata, idata + n);
	thrust::device_vector<int> thrust_odata(odata, odata + n);
	thrust::exclusive_scan(thrust_idata.begin(), thrust_idata.end(), thrust_odata.begin());
	thrust::copy(thrust_odata.begin(), thrust_odata.end(), odata);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, end);
	printf("Thrust scan: %f ms\n", milliseconds);
}

}
}
