#pragma once

namespace StreamCompaction {
namespace Thrust {
	void scan(int n, int *odata, const int *idata);

	void TestScan(int n, int *odata, const int *idata); 
	void TestSortStable(int n, int *odata, const int *idata);
	void TestSortUnstable(int n, int *odata, const int *idata);

}
}
