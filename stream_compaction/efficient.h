#pragma once

namespace StreamCompaction {
namespace Efficient {
	void scan(int n, int *odata, const int *idata);
	void scan_dev(int n, int *dev_data);

	int compact(int n, int *odata, const int *idata);
	int compact_dev(int n, int *dev_out, const int *dev_in);

	void TestScan(int n, int *odata, const int *idata);
	void TestCompact(int n, int *odata, const int *idata);
}
}
