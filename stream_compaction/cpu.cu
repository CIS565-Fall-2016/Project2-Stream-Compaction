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
    odata[0] = 0;
    for (int i = 1; i < n; i++)
    {
        odata[i] = odata[i-1] + idata[i-1];
    }
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
    int count = 0;
    odata[0] = 0;

    for (int i = 0; i < n; i++)
    {
        if (idata[i] != 0)
        {
            odata[count] = idata[i];
            count++;
        }
    }
    return count;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
    //return -1;
    
    int* mapdata = new int[n];
    int* scandata = new int[n];

    memset(mapdata, 0, n*sizeof(int));
    memset(scandata, 0, n*sizeof(int));
    

    for (int i = 0; i < n; i++)
    {
        if (idata[i] != 0)
            mapdata[i] = 1;
    }
    
    scan(n, scandata, mapdata);
    
    int count = 0;
    for (int i = 0; i < n; i++)
    {
        //odata[count] = mapdata[i] * idata[scandata[i]];
        //count += mapdata[i];
        if (mapdata[i] != 0)
        {
            odata[scandata[i]] = idata[i];
        }
    }
    count = scandata[n - 1] + mapdata[n-1];

    /*
    printf("\n%-10s", "idata: "); for (int i = 0; i < 20; i++)
       printf("%d ", idata[i]);
    printf("\n%-10s", "mapdata: "); for (int i = 0; i < 20; i++)
        printf("%d ", mapdata[i]);
    printf("\n%-10s", "scandata: "); for (int i = 0; i < 20; i++)
        printf("%d ", scandata[i]);
    printf("\n%-10s", "odata: "); for (int i = 0; i < 20; i++)
        printf("%d ", odata[i]);
    printf("\n");
    */

    delete[] mapdata;
    delete[] scandata;

    return count;

}

}
}
