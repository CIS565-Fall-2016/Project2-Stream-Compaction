CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Austin Eng
* Tested on: Windows 10, i7-4770K @ 3.50GHz 16GB, GTX 780 3072MB (Personal Computer)

## Output
```
****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  10   0 ]
==== cpu scan, power-of-two ====
Elapsed: 10.0046ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473618 205473628 ]
==== cpu scan, non-power-of-two ====
Elapsed: 9.0066ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473524 205473568 ]
    passed
==== naive scan, power-of-two ====
Elapsed: 9.708448ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473618 205473628 ]
    passed
==== naive scan, non-power-of-two ====
Elapsed: 9.713088ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
Elapsed: 4.019968ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473618 205473628 ]
    passed
==== work-efficient scan, non-power-of-two ====
Elapsed: 3.999136ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473524 205473568 ]
    passed
==== thrust scan, power-of-two ====
Elapsed: 0.906560ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473618 205473628 ]
    passed
==== thrust scan, non-power-of-two ====
Elapsed: 1.042912ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473524 205473568 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
Elapsed: 17.0074ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
Elapsed: 17.0071ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== cpu compact with scan ====
Elapsed: 6.0037ms
Elapsed: 31.0118ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
Elapsed: 5.496416ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== work-efficient compact, non-power-of-two ====
Elapsed: 5.449856ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
```