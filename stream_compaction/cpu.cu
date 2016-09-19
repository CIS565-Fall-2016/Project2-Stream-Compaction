#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
  int total = 0;
  for (int i = 0; i < n; ++i) {
    int val = idata[i];
    odata[i] = total;
    total += val;
  }
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
  int idx = 0;
  for (int i = 0; i < n; ++i) {
    if (idata[i] != 0) odata[idx++] = idata[i];
  }
  return idx;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
  for (int i = 0; i < n; ++i) {
    odata[i] = idata[i] != 0 ? 1 : 0;
  }
  int last = odata[n - 1];
  scan(n, odata, odata);
  int count = odata[n - 1] + last;
  for (int i = 0; i < n; ++i) {
    odata[odata[i]] = idata[i];
  }
  return count;
}

}
}
