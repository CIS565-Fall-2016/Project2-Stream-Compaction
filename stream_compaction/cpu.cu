#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	odata[0] = 0;
	for (int i = 1; i < n; i++) {
		odata[i] = odata[i - 1] + idata[i - 1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int index = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			odata[index++] = idata[i];
		}
	}
    return index;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int *pre = new int[n];
	int n2 = 0;
	
	// Create 0 and 1 array
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			pre[i] = 1;
			n2++;
		}
		else {
			pre[i] = 0;
		}
	}

	// Scan
	int *o = new int[n];
	scan(n, o, pre);

	// Scatter
	for (int i = 0; i < n; i++) {
		if (pre[i] == 1) {
			odata[o[i]] = idata[i];
		}
	}
	
	delete[] pre;
	return n2;
}

}
}
