#pragma once

namespace StreamCompaction {
namespace Efficient {
    float scan(int n, int *odata, const int *idata, int blockSize = 128);

    int compact(int n, int *odata, const int *idata, double &time, int blockSize = 128);

	void radix_sort(int n, int *odata, const int *idata, int blockSize = 128);
}
}
