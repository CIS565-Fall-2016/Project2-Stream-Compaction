#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"
#include <ctime>
//#include <chrono>

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
 
	//auto begin = std::chrono::high_resolution_clock::now();
	 
	thrust::exclusive_scan(idata , idata +n , odata);
	 
	//auto end = std::chrono::high_resolution_clock::now();
	//float ns = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	// float ns=0;
	//return ns;
}

}
}
