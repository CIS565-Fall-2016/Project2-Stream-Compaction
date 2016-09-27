#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
	namespace CPU {

		/**
		 * CPU scan (prefix sum).
		 */
		void scan(int n, int *odata, const int *idata) {
			// Initialize first value to 0
			odata[0] = 0;

			// Start loop at second element. The prefix sum should be sum of the 
			// previous elements in idata and odata
			for (int x = 1; x < n; x++){
				odata[x] = idata[x - 1] + odata[x - 1];
			}

		}

		/**
		 * CPU stream compaction without using the scan function.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithoutScan(int n, int *odata, const int *idata) {

			// Set initial count of elements to 0. Also start an index tracker for the output variable
			int numberOfNonZeroElements = 0;
			int outIdx = 0;

			// Loop over each element in the input array
			for (int x = 0; x < n; x++){
				if (idata[x] != 0){
					// If the value is nonzero, put into output array. Increment trackers as necessary.
					odata[outIdx] = idata[x];
					outIdx++;
					numberOfNonZeroElements++;
				}
			}

			return numberOfNonZeroElements;
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int *odata, const int *idata) {
			// Allocate a temporary array and set each entry to 1 if the corresponding entry
			// in idata should be kept, 0 otherwise.
			int *tempArray = new int[n];
			for (int x = 0; x < n; x++){
				if (idata[x] == 0) tempArray[x] = 0;
				else tempArray[x] = 1;
			}

			// Run scan on tempArray
			int *scanResults = new int[n];
			scan(n, scanResults, tempArray);

			// Scatter results into odata. Also keep track of number of elements added.
			int numberOfNonZeroElements = 0;
			for (int x = 0; x < n; x++){
				if (tempArray[x] == 1) {
					odata[scanResults[x]] = idata[x];
					numberOfNonZeroElements++;
				}
			}

			// Free memory for temporary arrays we created
			delete[] tempArray, scanResults;

			return numberOfNonZeroElements;
		}

	}
}
