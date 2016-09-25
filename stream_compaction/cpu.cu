#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    //printf("TODO\n");
	if (n<=0) return;
	for (int i=0; i<n; i++){
		if (i == 0)
			odata[i] = 0;
		else {
			odata[i] = odata[i-1] + idata[i-1];
		}
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO

	//int pid=0;
	int cnt=0;
	for (int i=0; i<n; i++){
		if (idata[i]!=0){
			odata[cnt]=idata[i];			 
			cnt++;
		}
	}
    return cnt;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int * tmp;
	tmp = new int [n];
	int * tmp2;
	tmp2 = new int [n];
	//Map the input array to an array of 0s and 1s
    for (int i =0; i<n; i++){
		tmp[i] = (idata[i]!=0);
	}
	//scan it,
	scan(n, tmp2, tmp);
	//scatter 
	int cnt=0;
	for (int i=0;i<n;i++){
		if (tmp2[i]){
			odata[tmp2[i]]=idata[i];
			cnt++;
		}
	}
	  
	delete[] tmp;
	delete[] tmp2;
    return cnt;

}

}
}
