#pragma once

namespace StreamCompaction {
namespace CPU {
    void scan(int n, int *odata, const int *idata);

    int compactWithoutScan(int n, int *odata, const int *idata);

    int compactWithScan(int n, int *odata, const int *idata);


	void TestScan(int n, int *odata, const int *idata);
	void TestCompact(int n, int *odata, const int *idata);
	void TestCompactWithoutScan(int n, int *odata, const int *idata);
}
}
