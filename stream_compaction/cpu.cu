#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    odata[0] = 0;
    printf("The output array is:\n");
    for(int tempCount = 1; tempCount < n; tempCount++)
    {
        odata[tempCount] = idata[tempCount-1] + odata[tempCount-1];
        printf("%5d",odata[tempCount]);
    }
	printf("\n");
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
	time_t start = clock();
    int countOut=0;
    for(int tempCount = 0; tempCount <n-1; tempCount++)
    {
        if(idata[tempCount]!=0)
        {
            odata[countOut]=idata[tempCount];
            countOut++;
        }
    }
	time_t end = clock();
	 printf("The running time is: %f ms. \n", double(end-start)*1000/CLOCKS_PER_SEC);
	return countOut;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
	time_t start = clock();
	int tempCount=0;
	int sumOutTemp=0;
	int *tempArray_1=&odata[2*n];
	int *tempArray_2=&odata[10*n];

	for (tempCount = 0; tempCount < n; tempCount++)
	{
		if (idata[tempCount] != 0)
		{
			tempArray_1[tempCount] = 1;
		}
		else
		{
			tempArray_1[tempCount] = 2;
		}
	}
	scan(n, tempArray_2, tempArray_1);

	for (tempCount = 0; tempCount < n-1; tempCount++)
	{
		if (tempArray_2[tempCount] != tempArray_2[tempCount + 1])
		{
			odata[sumOutTemp] = idata[tempCount];
			printf("%5d", odata[sumOutTemp]);
			sumOutTemp++;
		}
	}
	printf("\n");
	time_t end = clock();
	 printf("The running time is: %f ms. \n", double(end-start)*1000/CLOCKS_PER_SEC);
	return sumOutTemp++;
}

}
}
