#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {


	static StreamCompaction::Common::Timer timer;

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

	thrust::host_vector<int> host_input(idata, idata + n);
	thrust::device_vector<int> dev_input = host_input;

	//thrust::host_vector<int> host_output(odata, odata + n);
	thrust::device_vector<int> dev_output(odata, odata + n);

	// what happened during thrust? GPU timer malfunctioning
	timer.startGpuTimer();
	// call
	thrust::exclusive_scan(dev_input.begin(), dev_input.end(), dev_output.begin());

	timer.stopGpuTimer();

	timer.printTimerInfo("Scan::Thrust = ", timer.getGpuElapsedTime());

	thrust::copy(dev_output.begin(), dev_output.end(), odata);
}

}
}
