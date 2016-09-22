#pragma once

namespace StreamCompaction {
namespace Thrust {
#ifdef MEASURE_EXEC_TIME
    float scan(int n, int *odata, const int *idata);
#else
	void scan(int n, int *odata, const int *idata);
#endif
}
}
