CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) Rony Edde (redde)
* Tested on: Windows 10, i7-6700k @ 4.00GHz 64GB, GTX 980M 8GB (Personal Laptop)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)


This is an implementation of scan and reduce on the CPU and GPU
for stream compaction.
There are 2 types of arrays to consider.  Powers of 2 and non
powers of 2 which are part of the test.

* Scan
  * The first method used is on the CPU where all elements are
  added using a for loop.  This works very well on small arrays.

  * The second method consists of using the GPU to compute the
  scan result.  This is a naive implementation where the first
  iteration copies elements left to right (exclusive), then 
  2^(i+1) for a depth of log2(n).  Each depth skips 2^(d-1).
  This is faster than the CPU version for large arrays.

  * The third method uses a double sweep.  An upsweep, followed by
  a downsweep using a balanced tree form.  Each sweeps takes 
  log2(n-1) iterations but the calls on the GPU are only taking
  place on multiples of 2^(d+1).  This should be fast because
  there are only O(n) adds for the up sweep and O(n) adds and
  O(n) swaps.

  * Thrust scan uses CUDA's thrust exclusive function which is 
  built in the CUDA library.

* Stream Compaction
  * The first implementation is on the CPU where a for loop looks
  for values greater than 0 and adds them to the new array while
  incrementing the count when a non zero value is found.

  * The second implementation uses the CPU but also uses the scan
  function to look up indices.

  * The third implementation uses the GPU to generate a [0 1] 
  mapped array which is then run into the scan GPU function
  and used as an index lookup for placing the elements.
  After the scan function all non zero elements will be 
  present wich will result in a compact array with no zeros.


* Thrust 
  * The implementation is mentionned in the Scan section.

* Radix sort
  * There are 2 versions of the Radix sort.  This first one runs
  on the CPU using a CPU version on scan.

  * The second version uses the GPU to run the scan.  It also uses
  a GPU function to determine the max number in the array.  This
  is used to determine how many loops are needed before we reach
  the maximum amount of decimals in the maximum number.
  Another GPU function is used to shift the scan from exclusive
  to inclusive which is needed for a correct radix sort.
  There are multiple scan functions that can be used, a few are
  benchmarked.

* Benchmarks
  * Running benhmarks on a range of 256 to 65536 with a power of 4
  increment gave the following results:


```

****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  26   0 ]
==== cpu scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6203 6229 ]
==== cpu scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6146 6190 ]
    passed
==== naive scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6203 6259 ]
    a[129] = 3191, b[129] = 3221
    FAIL VALUE
==== naive scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
    passed
==== work-efficient scan, non-power-of-two ====
    passed
==== thrust scan, power-of-two ====
    passed
==== thrust scan, non-power-of-two ====
    passed
==== cpu radix sort ====
    [   0   0   0   0   0   0   1   1   1   1   2   2   2 ...  49  49 ]
    passed
==== gpu radix sort cpu scan ====
    [   0   0   0   0   0   0   1   1   1   1   2   2   2 ...  49  49 ]
    passed
==== gpu radix sort naive scan ====
    [   0   0   0   0   0   0   1   1   1   1   2   2   2 ...  49  49 ]
    passed
==== gpu radix sort thrust scan ====
    [   0   0   0   0   0   0   1   1   1   1   2   2   2 ...  49  49 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   2 ]
    passed
==== cpu compact with scan ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
    expected 190 elements, got 0
    FAIL COUNT
==== work-efficient compact, non-power-of-two ====
    expected 189 elements, got -1
    FAIL COUNT

****************
** BENCHMARKS **
****************

** SIZE = 256 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  26   0 ]
==== cpu scan, power-of-two ====
Time:  0 microSeconds
==== naive scan, power-of-two ====
Time:  580 microSeconds
==== work-efficient scan, power-of-two ====
Time:  628 microSeconds
==== thrust scan, power-of-two ====
Time:  1 microSeconds
==== cpu radix sort ====
Time:  34 microSeconds
==== gpu radix sort cpu scan ====
Time:  925 microSeconds
==== gpu radix sort naive scan ====
Time:  1093 microSeconds
==== gpu radix sort thrust scan ====
Time:  903 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  0 microSeconds
==== cpu compact with scan ====
Time:  1 microSeconds
==== work-efficient compact, power-of-two ====
Time:  729 microSeconds

** SIZE = 512 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  11   0 ]
==== cpu scan, power-of-two ====
Time:  0 microSeconds
==== naive scan, power-of-two ====
Time:  540 microSeconds
==== work-efficient scan, power-of-two ====
Time:  578 microSeconds
==== thrust scan, power-of-two ====
Time:  1 microSeconds
==== cpu radix sort ====
Time:  66 microSeconds
==== gpu radix sort cpu scan ====
Time:  1593 microSeconds
==== gpu radix sort naive scan ====
Time:  2907 microSeconds
==== gpu radix sort thrust scan ====
Time:  1690 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  0 microSeconds
==== cpu compact with scan ====
Time:  3 microSeconds
==== work-efficient compact, power-of-two ====
Time:  718 microSeconds

** SIZE = 1024 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  40   0 ]
==== cpu scan, power-of-two ====
Time:  1 microSeconds
==== naive scan, power-of-two ====
Time:  550 microSeconds
==== work-efficient scan, power-of-two ====
Time:  527 microSeconds
==== thrust scan, power-of-two ====
Time:  8 microSeconds
==== cpu radix sort ====
Time:  137 microSeconds
==== gpu radix sort cpu scan ====
Time:  2025 microSeconds
==== gpu radix sort naive scan ====
Time:  2654 microSeconds
==== gpu radix sort thrust scan ====
Time:  1655 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  1 microSeconds
==== cpu compact with scan ====
Time:  5 microSeconds
==== work-efficient compact, power-of-two ====
Time:  775 microSeconds

** SIZE = 2048 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  32   0 ]
==== cpu scan, power-of-two ====
Time:  2 microSeconds
==== naive scan, power-of-two ====
Time:  579 microSeconds
==== work-efficient scan, power-of-two ====
Time:  612 microSeconds
==== thrust scan, power-of-two ====
Time:  2 microSeconds
==== cpu radix sort ====
Time:  264 microSeconds
==== gpu radix sort cpu scan ====
Time:  2095 microSeconds
==== gpu radix sort naive scan ====
Time:  2749 microSeconds
==== gpu radix sort thrust scan ====
Time:  1875 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  3 microSeconds
==== cpu compact with scan ====
Time:  10 microSeconds
==== work-efficient compact, power-of-two ====
Time:  734 microSeconds

** SIZE = 4096 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  24   0 ]
==== cpu scan, power-of-two ====
Time:  5 microSeconds
==== naive scan, power-of-two ====
Time:  538 microSeconds
==== work-efficient scan, power-of-two ====
Time:  574 microSeconds
==== thrust scan, power-of-two ====
Time:  6 microSeconds
==== cpu radix sort ====
Time:  533 microSeconds
==== gpu radix sort cpu scan ====
Time:  2041 microSeconds
==== gpu radix sort naive scan ====
Time:  2819 microSeconds
==== gpu radix sort thrust scan ====
Time:  1928 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  7 microSeconds
==== cpu compact with scan ====
Time:  23 microSeconds
==== work-efficient compact, power-of-two ====
Time:  781 microSeconds

** SIZE = 8192 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...   4   0 ]
==== cpu scan, power-of-two ====
Time:  11 microSeconds
==== naive scan, power-of-two ====
Time:  555 microSeconds
==== work-efficient scan, power-of-two ====
Time:  631 microSeconds
==== thrust scan, power-of-two ====
Time:  11 microSeconds
==== cpu radix sort ====
Time:  1062 microSeconds
==== gpu radix sort cpu scan ====
Time:  2257 microSeconds
==== gpu radix sort naive scan ====
Time:  3114 microSeconds
==== gpu radix sort thrust scan ====
Time:  2471 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  16 microSeconds
==== cpu compact with scan ====
Time:  47 microSeconds
==== work-efficient compact, power-of-two ====
Time:  825 microSeconds

** SIZE = 16384 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  26   0 ]
==== cpu scan, power-of-two ====
Time:  23 microSeconds
==== naive scan, power-of-two ====
Time:  554 microSeconds
==== work-efficient scan, power-of-two ====
Time:  619 microSeconds
==== thrust scan, power-of-two ====
Time:  21 microSeconds
==== cpu radix sort ====
Time:  2181 microSeconds
==== gpu radix sort cpu scan ====
Time:  3084 microSeconds
==== gpu radix sort naive scan ====
Time:  3642 microSeconds
==== gpu radix sort thrust scan ====
Time:  2595 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  34 microSeconds
==== cpu compact with scan ====
Time:  97 microSeconds
==== work-efficient compact, power-of-two ====
Time:  866 microSeconds

** SIZE = 32768 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...   7   0 ]
==== cpu scan, power-of-two ====
Time:  49 microSeconds
==== naive scan, power-of-two ====
Time:  612 microSeconds
==== work-efficient scan, power-of-two ====
Time:  895 microSeconds
==== thrust scan, power-of-two ====
Time:  44 microSeconds
==== cpu radix sort ====
Time:  4240 microSeconds
==== gpu radix sort cpu scan ====
Time:  4343 microSeconds
==== gpu radix sort naive scan ====
Time:  5006 microSeconds
==== gpu radix sort thrust scan ====
Time:  3910 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  69 microSeconds
==== cpu compact with scan ====
Time:  196 microSeconds
==== work-efficient compact, power-of-two ====
Time:  1046 microSeconds

** SIZE = 65536 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  35   0 ]
==== cpu scan, power-of-two ====
Time:  94 microSeconds
==== naive scan, power-of-two ====
Time:  704 microSeconds
==== work-efficient scan, power-of-two ====
Time:  917 microSeconds
==== thrust scan, power-of-two ====
Time:  90 microSeconds
==== cpu radix sort ====
Time:  8625 microSeconds
==== gpu radix sort cpu scan ====
Time:  6799 microSeconds
==== gpu radix sort naive scan ====
Time:  7905 microSeconds
==== gpu radix sort thrust scan ====
Time:  6677 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  140 microSeconds
==== cpu compact with scan ====
Time:  397 microSeconds
==== work-efficient compact, power-of-two ====
Time:  1730 microSeconds

** SIZE = 131072 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  10   0 ]
==== cpu scan, power-of-two ====
Time:  188 microSeconds
==== naive scan, power-of-two ====
Time:  885 microSeconds
==== work-efficient scan, power-of-two ====
Time:  1259 microSeconds
==== thrust scan, power-of-two ====
Time:  432 microSeconds
==== cpu radix sort ====
Time:  17407 microSeconds
==== gpu radix sort cpu scan ====
Time:  12398 microSeconds
==== gpu radix sort naive scan ====
Time:  14302 microSeconds
==== gpu radix sort thrust scan ====
Time:  11650 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  280 microSeconds
==== cpu compact with scan ====
Time:  1034 microSeconds
==== work-efficient compact, power-of-two ====
Time:  1204 microSeconds

** SIZE = 262144 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  42   0 ]
==== cpu scan, power-of-two ====
Time:  738 microSeconds
==== naive scan, power-of-two ====
Time:  15 microSeconds
==== work-efficient scan, power-of-two ====
Time:  25 microSeconds
==== thrust scan, power-of-two ====
Time:  766 microSeconds
==== cpu radix sort ====
Time:  35010 microSeconds
==== gpu radix sort cpu scan ====
Time:  2323 microSeconds
==== gpu radix sort naive scan ====
Time:  2052 microSeconds
==== gpu radix sort thrust scan ====
Time:  1959 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  590 microSeconds
==== cpu compact with scan ====
Time:  2427 microSeconds
==== work-efficient compact, power-of-two ====
Time:  523 microSeconds

** SIZE = 524288 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  42   0 ]
==== cpu scan, power-of-two ====
Time:  1415 microSeconds
==== naive scan, power-of-two ====
Time:  16 microSeconds
==== work-efficient scan, power-of-two ====
Time:  24 microSeconds
==== thrust scan, power-of-two ====
Time:  1762 microSeconds
==== cpu radix sort ====
Time:  69752 microSeconds
==== gpu radix sort cpu scan ====
Time:  4077 microSeconds
==== gpu radix sort naive scan ====
Time:  4009 microSeconds
==== gpu radix sort thrust scan ====
Time:  4086 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  1143 microSeconds
==== cpu compact with scan ====
Time:  4837 microSeconds
==== work-efficient compact, power-of-two ====
Time:  1044 microSeconds

** SIZE = 1048576 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...   6   0 ]
==== cpu scan, power-of-two ====
Time:  2821 microSeconds
==== naive scan, power-of-two ====
Time:  17 microSeconds
==== work-efficient scan, power-of-two ====
Time:  26 microSeconds
==== thrust scan, power-of-two ====
Time:  3631 microSeconds
==== cpu radix sort ====
Time:  139054 microSeconds
==== gpu radix sort cpu scan ====
Time:  8441 microSeconds
==== gpu radix sort naive scan ====
Time:  8830 microSeconds
==== gpu radix sort thrust scan ====
Time:  9037 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  2301 microSeconds
==== cpu compact with scan ====
Time:  8768 microSeconds
==== work-efficient compact, power-of-two ====
Time:  2365 microSeconds

** SIZE = 2097152 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...   2   0 ]
==== cpu scan, power-of-two ====
Time:  4628 microSeconds
==== naive scan, power-of-two ====
Time:  20 microSeconds
==== work-efficient scan, power-of-two ====
Time:  31 microSeconds
==== thrust scan, power-of-two ====
Time:  7244 microSeconds
==== cpu radix sort ====
Time:  277141 microSeconds
==== gpu radix sort cpu scan ====
Time:  17729 microSeconds
==== gpu radix sort naive scan ====
Time:  17591 microSeconds
==== gpu radix sort thrust scan ====
Time:  17553 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  4684 microSeconds
==== cpu compact with scan ====
Time:  17774 microSeconds
==== work-efficient compact, power-of-two ====
Time:  4784 microSeconds

** SIZE = 4194304 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  26   0 ]
==== cpu scan, power-of-two ====
Time:  6579 microSeconds
==== naive scan, power-of-two ====
Time:  18 microSeconds
==== work-efficient scan, power-of-two ====
Time:  30 microSeconds
==== thrust scan, power-of-two ====
Time:  15121 microSeconds
==== cpu radix sort ====
Time:  555419 microSeconds
==== gpu radix sort cpu scan ====
Time:  35741 microSeconds
==== gpu radix sort naive scan ====
Time:  35411 microSeconds
==== gpu radix sort thrust scan ====
Time:  35747 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  9555 microSeconds
==== cpu compact with scan ====
Time:  35407 microSeconds
==== work-efficient compact, power-of-two ====
Time:  9463 microSeconds

** SIZE = 8388608 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  10   0 ]
==== cpu scan, power-of-two ====
Time:  13667 microSeconds
==== naive scan, power-of-two ====
Time:  18 microSeconds
==== work-efficient scan, power-of-two ====
Time:  29 microSeconds
==== thrust scan, power-of-two ====
Time:  31478 microSeconds
==== cpu radix sort ====
Time:  1111762 microSeconds
==== gpu radix sort cpu scan ====
Time:  71012 microSeconds
==== gpu radix sort naive scan ====
Time:  70982 microSeconds
==== gpu radix sort thrust scan ====
Time:  71596 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  18627 microSeconds
==== cpu compact with scan ====
Time:  69722 microSeconds
==== work-efficient compact, power-of-two ====
Time:  18604 microSeconds

** SIZE = 16777216 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  42   0 ]
==== cpu scan, power-of-two ====
Time:  25969 microSeconds
==== naive scan, power-of-two ====
Time:  20 microSeconds
==== work-efficient scan, power-of-two ====
Time:  31 microSeconds
==== thrust scan, power-of-two ====
Time:  61397 microSeconds
==== cpu radix sort ====
Time:  2213557 microSeconds
==== gpu radix sort cpu scan ====
Time:  142111 microSeconds
==== gpu radix sort naive scan ====
Time:  142175 microSeconds
==== gpu radix sort thrust scan ====
Time:  142549 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  38071 microSeconds
==== cpu compact with scan ====
Time:  140700 microSeconds
==== work-efficient compact, power-of-two ====
Time:  38334 microSeconds
Press any key to continue . . .
```

* Large scale results:
![largescale](./images/bench1.png)

* Small scale results:
![largescale](./images/bench2.png)


The significance is obvious between the CPU and GPU
implementations.  When looking at small arrays, the
CPU seems to take an interesting advantage.  However
on large arrays, the CPU implementations scale very
poorly, hence the need to have 2 graph scales.
Radix sort suffers particularly when running on the 
CPU.  One pattern that emerges is that the naive
and efficient methods scale logarithmically.
The efficient version was a fraction slower and that
is due to interrupts between the up and downscan.
Every time upscan is called, memory has to be
allocated and copied, the same happens with downscan
This is shown in the last figure.  Despite that,
the loss is minumal.
Thrust scan does well and scales linearly albeit not
as good as the naive and efficient methods.
One thing that was interesting to note is that radix
benefitted more from the thrust scan than the other
GPU implementations.  Again possibly due to the 
multiple GPU functions.  The max number function is
run on the GPU and memory is copied back and forth.
One bottleneck that manifests itself in the graph
below is from the kernDecimalsMap function.
Not enough resources are used during this function
which created a bottleneck that is minimal but not
desired.  About 40% og the GPU is not used.
This is very obvious when looking at the following
timeline and the kernDecimalsMap function:
* Cuda memory calls frequency:
![timeline](./images/memoryswapping.png)


* DecimalsMap bottleneck:
![timeline](./images/bottleneck.png)





