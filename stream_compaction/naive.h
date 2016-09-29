#pragma once

namespace StreamCompaction {
namespace Naive {
	void scan(int n, int *odata, const int *idata);
	void scan_dev(int n, int *dev_out, int *dev_in);

	void TestScan(int n, int *odata, const int *idata);
}
}
