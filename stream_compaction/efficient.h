#pragma once

namespace StreamCompaction {
namespace Efficient {
	void scan(int n, int *odata, const int *idata, float* timeElapsedMs);

	int compact(int n, int *odata, const int *idata, float* timeElapsedMs);
}
}
