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

    //thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());


    thrust::host_vector<int> dv_in(idata, idata + n);
    thrust::host_vector<int> dv_out(odata, odata + n);

    thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

    //odata = &(dv_out.front()); 
    for (int i = 0; i < n; i++)
        odata[i] = dv_out[i];
}

}
}
