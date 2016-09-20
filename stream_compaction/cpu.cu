#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	odata[0] = 0;
	for (int i = 1; i < n; i++)
	{
		odata[i] = odata[i - 1] + idata[i - 1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	long j = 0;
	for (int i = 0; i < n; i++)
	{
		if (idata[i] != 0)
		{
			odata[j] = idata[i];
			j++;
		}
	}
	return j;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int *temporal;
	int *pscan;
	long j = 0;
	temporal = (int*)malloc(sizeof(int)*n);
	pscan = (int*)malloc(sizeof(int)*n);
	for (int i = 0; i < n; i++)
	{
		temporal[i] = idata[i] ? 1 : 0;
	}
	scan(n, pscan, temporal);
	for (int i = 0; i < n; i++)
	{
		if (temporal[i] == 1)
		{
			odata[pscan[i]] = idata[i];
			j++;
		}
	}
	return j;
}

}
}
