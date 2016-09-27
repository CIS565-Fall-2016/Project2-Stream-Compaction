#include <cstdio>
#include "cpu.h"
#include "common.h"

namespace StreamCompaction {
namespace CPU {

	//static StreamCompaction::Common::Timer timer;

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    
	//timer.startCpuTimer();

	odata[0] = 0;
	for (int i = 1; i < n; ++i)
		odata[i] = odata[i - 1] + idata[i - 1];

	//timer.stopCpuTimer();
	//timer.printTimerInfo("Scan::CPU = ", timer.getCpuElapsedTime());

}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	
	//timer.startCpuTimer();

	int index = 0;
	for (int i = 0; i < n; ++i)
	{
		if (idata[i] != 0)
		{
			odata[index++] = idata[i];
		}
	}

	//timer.stopCpuTimer();
	//timer.printTimerInfo("StreamCompactWithoutScan::CPU = ", timer.getCpuElapsedTime());

	return index;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {

	int* inputScan = new int[n];
	int* outputScan = new int[n];

	//timer.startCpuTimer();

	for (int i = 0; i < n; ++i)
		inputScan[i] = (idata[i] == 0) ? 0 : 1;

	scan(n, outputScan, inputScan);

	int sum = 0;
	for (int i = 0; i < n; ++i)
	{
		if (idata[i] != 0)
		{
			odata[outputScan[i]] = idata[i];
			sum++;
		}
	}

	//timer.stopCpuTimer();
	//timer.printTimerInfo("StreamCompactWithScan::CPU = ",timer.getCpuElapsedTime());

	delete[] inputScan;
	delete[] outputScan;
	return sum;
}

}
}
