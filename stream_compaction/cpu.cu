#include "cpu.h"

#include "common.h"

#include <cstdio>
#include <vector>
#include <algorithm>

namespace StreamCompaction {
namespace CPU {

using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

void scan_implementation(int n, int *odata, const int *idata)
{
    if (n <= 0) { return; }

    odata[0] = 0;

    using std::size_t;
    for (size_t i = 1; i < n; i++)
    {
        odata[i] = odata[i - 1] + idata[i - 1];
    }
}

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) 
{
    // DONE
    timer().startCpuTimer();

    scan_implementation(n, odata, idata);

    timer().endCpuTimer();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) 
{
    //DONE
    timer().startCpuTimer();

    using std::size_t;
    size_t olength = 0;
    for (size_t i = 0; i < n; i++)
    {
        if (idata[i])
        {
            odata[olength] = idata[i];
            olength++;
        }
    }

    timer().endCpuTimer();
    
    return static_cast<int>(olength);
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) 
{
    // DONE
    // Run CPU scan
    using std::size_t;
    std::vector<int> scan_result(n, 0);

    timer().startCpuTimer();
    
    for (size_t i = 0; i < n; i++)
    {
        if (idata[i])
        {
            // also use odata as a temp buffer to save space
            odata[i] = 1;
        }
    }

    scan_implementation(n, scan_result.data(), odata);

    size_t olength = 0;
    for (size_t i = 0; i < n; i++)
    {
        if (idata[i])
        {
            odata[scan_result[i]] = idata[i];
            olength++;
        }
    }

    timer().endCpuTimer();

    return static_cast<int>(olength);
}

/**
* This just calls std::sort
*/
void stdSort(int* start, int* end)
{
    timer().startCpuTimer();
    std::sort(start, end);
    timer().endCpuTimer();
}

}
}
