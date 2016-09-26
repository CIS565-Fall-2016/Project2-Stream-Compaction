#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO : finished
	if (n <= 0) return;
	odata[0] = idata[0];
	for (int i = 1; i < n; ++i)
	{
		odata[i] = odata[i - 1] + idata[i];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO : finished
	int num_remain = 0;
	for (int i = 0; i < n; ++i)
	{
		if (idata[i] != 0)
		{
			odata[num_remain++] = idata[i];
		}
	}
	return num_remain;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO : finished
	// map data to 1 and 0 for non-zero and zero.
	int *tmp_data = new int(n);
	for (int i = 0; i < n; ++i)
	{
		tmp_data[i] = idata[i] == 0 ? 0 : 1;
		//if (idata[i] == 0) tmp_data[i] = 0;
		//else tmp_data[i] = 1;
		//printf("%d is %d\n", i, tmp_data[i]);
	}

	// scan
	scan(n, odata, tmp_data);

	// scatter
	int num_remain = 0;
	for (int i = 0; i < n; ++i)
	{
		if (tmp_data[i] == 1)
		{
			odata[odata[i]-1] = idata[i];
			num_remain++;
		}
	}
	return num_remain;
}

}
}
