#include <cstdio>
#include "stream_compaction\common.h"
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
	//int n = 8;
	//int idata[8] ={0,1,2,3,4,5,6,7};
	//int odata[8]={0};
	if (n <= 0)
		return;

	memcpy(odata, idata, n * sizeof(int));
	int nCeilLog = ilog2ceil(n);

	int nLength = 1 << nCeilLog;

	//for (int d = 0; d < nCeilLog; d++)
	//	for (int k = 0; k < nLength; k++)
	//	{
	//		int m = 1 << (d + 1);
	//		if (!(k % m))
	//			odata[k + m - 1] += odata[k + (m >> 1) - 1];
	//	}

	
	for (int d = 0; d < nCeilLog; d++)
	{
		int addTimes = 1 << (nCeilLog - 1 - d);
		for (int k = 0; k < addTimes; k++)
		{
			int m = (k + 1) * (1 << (d + 1));
			//printf("%d %d\n",m - 1, m - 1 - (1 << d));
			odata[m - 1] += odata[m - 1 - (1 << d)];
		}
	}

	//odata[nLength - 1] = 0;
	//for (int d = nCeilLog - 1; d >= 0; d--)
	//	for (int k = 0; k < nLength; k++)
	//	{
	//		int m = 1 << (d + 1);
	//		if (!(k % m))
	//		{
	//			int index1 = k + (m >> 1) - 1;
	//			int index2 = k + m - 1;
	//			int temp = odata[index1];
	//			odata[index1] = odata[index2];
	//			odata[index2] += temp;
	//		}
	//	}
		//for (int i = 0; i < 8; i++)
		//printf("%d ", odata[i]);
		//printf("\n");
	odata[nLength - 1] = 0;
	for (int d = nCeilLog - 1; d >= 0; d--)
	{
		int addTimes = 1 << (nCeilLog - 1 - d);
		for (int k = 0; k < addTimes; k++)
		{
			int m = (k + 1) * (1 << (d + 1));
			int index1 = m - 1 - (1 << d);
			int index2 = m - 1;
//			printf("%d %d\n", index1, index2);
			int temp = odata[index1];
			odata[index1] = odata[index2];
			odata[index2] += temp;
		}
	}
	//for (int i = 0; i < 8; i++)
	//	printf("%d ", odata[i]);
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
	if (n <= 0)
		return -1;
	int counter = 0;
	for (int i = 0; i < n; i++)
	{
		if (idata[i])
			odata[counter++] = idata[i];
	}
	return counter;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
	if (n <= 0)
		return -1;
	int counter = 0;
	
	for (int i = 0; i < n; i++)
		odata[i] = idata[i] ? 1 : 0;
	scan(n, odata, odata);
	for (int i = 0; i < n - 1; i++)
	{
		if (odata[i] != odata[i + 1])
			odata[counter++] = idata[i];
	}
	return counter;
}

}
}
