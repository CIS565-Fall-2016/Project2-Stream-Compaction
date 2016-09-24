CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jian Ru
* Tested on: Windows 10, i7-4850 @ 2.30GHz 16GB, GT 750M 2GB (Personal)

### List of Features

* CPU scan and stream compaction
* Naive GPU scan
* Work-efficient GPU scan and stream compaction (used shared memory)
* Thrust scan
* GPU radix sort (added radix_sort.h and radix_sort.cu)
* Custom tests

### Performance Analysis

  [](img/sc_perf1.png)

* Optimizing Block Sizes
  * 128 or 256 turn out to be the best. Unless you use non-multiple of warp size,
  really small (like 32), or really big (like 1024) block size, the performance difference
  is not really that huge
  * The only exception is work-efficient scan. Since reducing block size increases the number
  of sub-sums and likely increases the number of recursions, it enjoys a larger block size
  (512 turns out to be optimal on my machine). Even if so, using 1024 turns out to be slower.
  It may due to reduced concurrent execution, since all the threads in a block need to be
  synchtronized at 3 different places in the kernel. Increasing block size means fewer threads
  can terminate and leave multiprocessor early. So it is likely to hurt performance.

* Array Size vs. Execution Time (ms)
  * Exclusive scan
  Array Size | Naive | Efficient | Thrust | CPU
  ---        | ---   | ---       | ---    | ---
  2^8  | 0.03 | 0.01 | 0.02 | 0.01
  2^15 | 0.19 | 0.06 | 0.52 | 0.01
  2^20 | 5.67 | 1.38 | 0.97 | 3.50
  2^25 | 225.00 | 43.40 | 18.14 | 89.33

  * Stream compaction
  Array Size | Efficient | CPU (no scan) | CPU (with scan)
  ---        | ---       | ---           | ---
  2^8  | 0.09 | 0.00 | 0.00
  2^15 | 0.16 | 0.09 | 0.20
  2^20 | 3.78 | 2.70 | 6.03
  2^25 | 107.73 | 109.55 | 194.05
  
  * Radix sort
  Array Size | Efficient | Thrust
  ---        | ---       | ---
  2^8  | 0.52 | 0.58
  2^15 | 4.37 | 0.34
  2^20 | 109.27 | 5.80
  2^25 | 3642.84 | 116.92

* Performance Bottlenecks
  * CUDA runtime APIs for memory manipulation (e.g. cudaMalloc, cudaMemcpy, cudaFree) are super expensive
  * Naive scan
    * Excessive global memory access
	* Too many kernel calls (each level of up/down-sweep require one kernel call)
  * Work-efficient scan
    * Bank conflicts at the beginning but have been resolved now
	* Algorithm not good enough (too much computation) when compared to thrust's implementation 
	bacause my scan kernel, on average, takes twice as much time to execute as thrust's kernel
    * Too many kernel calls when array size is large
	* In my implementation, if array size exceed 2 times block size, two more calls are
	required: one scan the sub-sums and another one scatter scanned sub-sums to corresponding
	blocks. If the size of sub-sum array is further greater than 2 times block size, another
	two extra calls are generated and this recursive behavior will go on as array size gets
	larger and larger.
	* Thrust's scan implementation stablizes at 3 kernel calls despite of the size of data.
	Thus even though my implementation is comparable or even slightly faster than thrust scan,
	it becomes much slower when array size becomes really large (2^25 for example)
  * Thrust scan
    * Additional cudaMalloc and cudaFree calls
	* Judging from the performance analysis timeline, thrust is doing cudaMalloc and cudaFree
	even if I pass in thrust::device_ptr<T>. This causes thrust scan become slower than my
	work-efficient scan when array size is small
  * Work-effcient stream compaction
    * My implementation is basically a wrapper on work-efficient scan plus two light-weight
	kernels, kernMapToBoolean and kernScatter. So the bottlenecks are the same as work efficient
	scan
  * Self-implemented radix sort
    * Excessive kernel calls
	* Scan kernel implementation not good enough when compared with thrust's
    * My implementation is super naive. I basically separate elements into 0 bin and 1 bin and
	do this for every bit. So if the type has 32 bits, there will be 3 * 32 kernel invocations
	or more when array size gets large.
	* On the other hand, judging from performance analysis timeline, thrust's implementation
	has 3 * 7 kernel invacations all the time and one of the kernel is super cheap (took less
	than 10 microseconds to execute on 2^25 array size). Moreover, even the two more expensive
	kernels run much faster than my scan kernel on large data size
	
* Sample Output

```

GeForce GT 750M [sm_30]
Array Size: 1048576
Test group size: 100
Note:
    1. Execution time is the average over @TEST_GROUP_SIZE times exections
    2. Runtime API memory operations were excluded from time measurement

****************
** RADIX SORT TESTS **
****************
    [   3   4   3   2   0   2   0   0   0   2   3   0   2 ...   1   0 ]
==== parallel radix sort, power-of-two ====
    Execution Time: 110.19ms
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...   4   4 ]
    passed
==== parallel radix sort, non power-of-two ====
    Execution Time: 109.44ms
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...   4   4 ]
    passed
==== thrust sort, power-of-two ====
    Execution Time: 5.64ms
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...   4   4 ]
    passed
==== thrust sort, non power-of-two ====
    Execution Time: 5.32ms
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...   4   4 ]
    passed

****************
** SCAN TESTS **
****************
==== cpu scan, power-of-two ====
    Execution Time: 2.36ms
    [   0   3   7  10  12  12  14  14  14  14  16  19  19 ... 2098448 2098449 ]
==== cpu scan, non-power-of-two ====
    Execution Time: 2.37ms
    [   0   3   7  10  12  12  14  14  14  14  16  19  19 ... 2098443 2098447 ]
    passed
==== naive scan, power-of-two ====
    Execution Time: 5.67ms
    [   0   3   7  10  12  12  14  14  14  14  16  19  19 ... 2098448 2098449 ]
    passed
==== naive scan, non-power-of-two ====
    Execution Time: 5.67ms
    [   0   3   7  10  12  12  14  14  14  14  16  19  19 ... 2098443 2098447 ]
    passed
==== work-efficient scan, power-of-two ====
    Execution Time: 1.39ms
    [   0   3   7  10  12  12  14  14  14  14  16  19  19 ... 2098448 2098449 ]
    passed
==== work-efficient scan, non-power-of-two ====
    Execution Time: 1.39ms
    [   0   3   7  10  12  12  14  14  14  14  16  19  19 ... 2098443 2098447 ]
    passed
==== thrust scan, power-of-two ====
    Execution Time: 1.02ms
    [   0   3   7  10  12  12  14  14  14  14  16  19  19 ... 2098448 2098449 ]
    passed
==== thrust scan, non-power-of-two ====
    Execution Time: 1.02ms
    [   0   3   7  10  12  12  14  14  14  14  16  19  19 ... 2098443 2098447 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
    Execution Time: 2.71ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
    Execution Time: 2.66ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== cpu compact with scan ====
    Execution Time: 5.87ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
    Execution Time: 3.89ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== work-efficient compact, non-power-of-two ====
    Execution Time: 3.86ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
Press any key to continue . . .
```
	
	