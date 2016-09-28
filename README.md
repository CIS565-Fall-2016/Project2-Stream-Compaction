CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Xiang Deng
* Tested on:  Windows 10-Home, i7-6700U @ 2.6GHz 16GB, GTX 1060 6GB (Personal Computer)

 
* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).

![](images/1.PNG)

![](images/7.PNG)

Based on the figure and data above, regarding scanning, I found the bottleneck for GPU outperforms the GPU is around the arraysize of 2^16, after which the GPU sigificantly speed up than the CPU.
The CPU shows its adavantage for small arraysize.

![](images/2.PNG)

![](images/3.PNG)

![](images/8.PNG)

Based on the figure and data above, regarding compacting, I found the bottleneck for GPU outperforms the GPU is between the arraysize of 2^16 and 2^20, after which the GPU sigificantly speed up than the CPU.
The CPU still shows its adavantage for small arraysize.

![](images/4.PNG)


* Optimization of blocksize:
Experiments was conducted on various blocksizes from 32 to 1024 with exponential growth. Typically we observed the optimizal value of block size (256) which best 
balance the optimal value of scan time as well as compact time for GPU. Since earlier we observed the array size of 2^16 is around the point of "bottleneck", we 
used this parameter for the tuning of the blocksize.

![](images/5.PNG)

![](images/6.PNG)

* Extra credits
* 1) I typically found arraysize of 2^16 or greater already makes the GPU outforms the GPU.
* 2) The radix sort was implemented and tested. The testing function (at the end of the main.cpp) generates array size of power of two and not power of two. In both cases,
we compare the sorting result with C++ built in sorting function to verify the correctness. It's correctness has been verified.

# Test output

```
****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  35   0 ]
==== cpu scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
time lapsed 0.131000 ms
==== cpu scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
time lapsed 0.130000 ms
==== naive scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
time lapsed 0.106496 ms
==== naive scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ...   0   0 ]
    passed
time lapsed 0.080352 ms
==== work-efficient scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
time lapsed 0.145792 ms
==== work-efficient scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
time lapsed 0.146432 ms
==== thrust scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
time lapsed 0.036000 ms
==== thrust scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
time lapsed 0.036000 ms

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
time lapsed 0.191000 ms
==== cpu compact without scan, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   1 ]
    passed
time lapsed 0.191000 ms
==== cpu compact with scan ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
time lapsed 0.368000 ms
==== work-efficient compact, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
time lapsed 0.585728 ms
==== work-efficient compact, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   1 ]
    passed
time lapsed 0.549632 ms

*****************************
** RADIX SORT TESTS **
*****************************
==== Array to be sorted power of 2 ====
    [  38 119  38  37  55 197 165  85  50  12  53 100 142 ...  85   0 ]
==== RADIX SORT POT ====
size of int is 32 bits
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 199 199 ]
==== C++ SORT POT ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 199 199 ]
    passed
==== Array to be sorted not power of 2 ====
    [  38 1719 1238 437 855 1797 365 285 450 612 1853 100 1142 ... 1085   0 ]
==== RADIX SORT NPOT ====
size of int is 32 bits
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 1999 1999 ]
==== C++ SORT NPOT ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 1999 1999 ]
    passed
```

## Note
### Modified files
CMakeList.txt : add radixSort.h and radixSort.cu, changed -arch=sm_20 to sm_61
Two files added: radixSort.h and radixSort.cu