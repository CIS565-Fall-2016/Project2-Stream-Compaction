#pragma once

#include <cstdlib>

template<typename T>
int cmpArrays(int n, T *a, T *b) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

void printDesc(const char *desc) {
    printf("==== %s ====\n", desc);
}

template<typename T>
void printCmpResult(int n, T *a, T *b) {
    printf("    %s \n",
            cmpArrays(n, a, b) ? "FAIL VALUE" : "passed");
}

template<typename T>
void printCmpLenResult(int n, int expN, T *a, T *b) {
    if (n != expN) {
        printf("    expected %d elements, got %d\n", expN, n);
    }
    printf("    %s \n",
            (n == -1 || n != expN) ? "FAIL COUNT" :
            cmpArrays(n, a, b) ? "FAIL VALUE" : "passed");
}

void zeroArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }
}

void genArray(int n, int *a, int maxval) {
    srand(0);

    for (int i = 0; i < n; i++) {
        a[i] = rand() % maxval;
    }
}

void genArraySort(int n, int *a, int maxval) {
	if (n == 8) {
		a[0] = 4;
		a[1] = 7;
		a[2] = 2;
		a[3] = 6;
		a[4] = 3;
		a[5] = 5;
		a[6] = 1;
		a[7] = 0;
	}
}

void printArray(int n, int *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
}

