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
    a[129] = 3191, b[129] = 3221
    FAIL VALUE
==== naive scan, non-power-of-two ====
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
    passed
==== work-efficient compact, non-power-of-two ====
    passed

****************
** BENCHMARKS **
****************

** SIZE = 256 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  26   0 ]
==== cpu scan, power-of-two ====
Time:  0 microSeconds
==== naive scan, power-of-two ====
Time:  146 microSeconds
==== work-efficient scan, power-of-two ====
Time:  166 microSeconds
==== thrust scan, power-of-two ====
Time:  0 microSeconds
==== cpu radix sort ====
Time:  36 microSeconds
==== gpu radix sort cpu scan ====
Time:  597 microSeconds
==== gpu radix sort naive scan ====
Time:  866 microSeconds
==== gpu radix sort thrust scan ====
Time:  436 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  0 microSeconds
==== cpu compact with scan ====
Time:  1 microSeconds
==== work-efficient compact, power-of-two ====
Time:  637 microSeconds

** SIZE = 4096 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  24   0 ]
==== cpu scan, power-of-two ====
Time:  5 microSeconds
==== naive scan, power-of-two ====
Time:  767 microSeconds
==== work-efficient scan, power-of-two ====
Time:  609 microSeconds
==== thrust scan, power-of-two ====
Time:  7 microSeconds
==== cpu radix sort ====
Time:  531 microSeconds
==== gpu radix sort cpu scan ====
Time:  1268 microSeconds
==== gpu radix sort naive scan ====
Time:  1700 microSeconds
==== gpu radix sort thrust scan ====
Time:  1143 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  7 microSeconds
==== cpu compact with scan ====
Time:  23 microSeconds
==== work-efficient compact, power-of-two ====
Time:  539 microSeconds

** SIZE = 65536 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  35   0 ]
==== cpu scan, power-of-two ====
Time:  94 microSeconds
==== naive scan, power-of-two ====
Time:  1072 microSeconds
==== work-efficient scan, power-of-two ====
Time:  833 microSeconds
==== thrust scan, power-of-two ====
Time:  102 microSeconds
==== cpu radix sort ====
Time:  8615 microSeconds
==== gpu radix sort cpu scan ====
Time:  5620 microSeconds
==== gpu radix sort naive scan ====
Time:  5810 microSeconds
==== gpu radix sort thrust scan ====
Time:  5537 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  145 microSeconds
==== cpu compact with scan ====
Time:  413 microSeconds
==== work-efficient compact, power-of-two ====
Time:  1451 microSeconds

** SIZE = 1048576 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...   6   0 ]
==== cpu scan, power-of-two ====
Time:  1521 microSeconds
==== naive scan, power-of-two ====
Time:  17 microSeconds
==== work-efficient scan, power-of-two ====
Time:  27 microSeconds
==== thrust scan, power-of-two ====
Time:  3638 microSeconds
==== cpu radix sort ====
Time:  137808 microSeconds
==== gpu radix sort cpu scan ====
Time:  8723 microSeconds
==== gpu radix sort naive scan ====
Time:  8787 microSeconds
==== gpu radix sort thrust scan ====
Time:  8568 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  2321 microSeconds
==== cpu compact with scan ====
Time:  10280 microSeconds
==== work-efficient compact, power-of-two ====
Time:  2295 microSeconds

** SIZE = 16777216 **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  42   0 ]
==== cpu scan, power-of-two ====
Time:  24411 microSeconds
==== naive scan, power-of-two ====
Time:  19 microSeconds
==== work-efficient scan, power-of-two ====
Time:  32 microSeconds
==== thrust scan, power-of-two ====
Time:  64703 microSeconds
==== cpu radix sort ====
Time:  2208906 microSeconds
==== gpu radix sort cpu scan ====
Time:  145916 microSeconds
==== gpu radix sort naive scan ====
Time:  145302 microSeconds
==== gpu radix sort thrust scan ====
Time:  144644 microSeconds
==== cpu compact without scan, power-of-two ====
Time:  37473 microSeconds
==== cpu compact with scan ====
Time:  167007 microSeconds
==== work-efficient compact, power-of-two ====
Time:  39963 microSeconds
Press any key to continue . . .
```

* Large scale results:
![largescale](../images/bench1.png)

* Small scale results:
![largescale](../images/bench2.png)


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
![timeline](../memoryswapping.png)


* DecimalsMap bottleneck:
![timeline](../bottleneck.png)





