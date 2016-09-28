#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

double last_runtime;

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

    thrust::device_vector<int> devIn(idata, idata+n), devOut(n);
    double t1 = clock();
    thrust::exclusive_scan(devIn.begin(), devIn.end(), devOut.begin());
    double t2 = clock();
    last_runtime = 1.0E6 * (t2-t1) / CLOCKS_PER_SEC;

    thrust::copy(devOut.begin(), devOut.end(), odata);
}

}
}
