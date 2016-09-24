#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
	odata[0] = 0;

	// exclusive prefix sum
	for (int i = 1; i < n; i++) {
		odata[i] = odata[i - 1] + idata[i - 1];
	}

    // printf("TODO\n");
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
    // return -1;
	int counter = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			odata[counter++] = idata[i];			
		}
	}
	return counter;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */

/**
*  so helper function scatter here
**/

int scatter(int n, int *odata, const int *idata, const int *idataChanged, const int *exclusivePreSum){
	int counter = 0;
	for (int i = 0; i < n; i++) {
		if (idataChanged[i] == 1) {
			odata[exclusivePreSum[i]] = idata[i];
			counter++;
		}
	}
	return counter;
}

int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
    // return -1;
	int* idataChanged = new int[n];
	int* exclusivePreSum = new int[n];

	for (int i = 0; i < n; i++) {
		idataChanged[i] = (idata[i] == 0) ? 0 : 1;
	}

	//odataChanged is the exclusive prefix sum
	scan(n, exclusivePreSum, idataChanged);
	int counter = scatter(n, odata, idata, idataChanged, exclusivePreSum);
	delete[] idataChanged;
	delete[] exclusivePreSum;
	return counter;

}

}
}
