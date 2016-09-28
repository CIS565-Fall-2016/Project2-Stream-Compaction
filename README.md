CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ottavio Hartman
* Tested on: Windows 7, i7-4790 @ 3.60GHz 16GB, Quadro K420 (Moore 102 Lab)

![Performance data](Capture.PNG)


### Summary
This project contains the implementation and tests for a __CPU Scan, CPU stream compaction, naive GPU scan, work-efficient GPU scan, and GPU stream compaction.__

### Performance analysis

As the graph above shows, the 128 and 256 block sizes worked best for the GPU scan and compact. 

Unfortunately, the CPU time showed `0`ns on all test cases, which either means that it is <`1`ns or that the timer is broken for some reason.
However, I'm surprised at the results of the GPU work-efficient scan; despite doing fewer additions it seems to take longer than the naive approach.
One reason this could be happening is because of how the data accessing in the work-efficient approach is less contiguos than in the naive approach.

I think one of the main bottlenecks is memory I/O. Since "scan" implements basically the simplest operation - addition - which is very fast on GPUs,
there must be a lot of time spent accessing and writing memory to the array. Despite the advantage of interleaving threads to hide stalls, the GPU seems
to just spend a lot of time waiting on memory.

```
****************
** SCAN TESTS **
****************
==== cpu scan, power-of-two ====
0ns
==== cpu scan, non-power-of-two ====
    passed
==== naive scan, power-of-two ====
Time: 0.043712
    passed
==== naive scan, non-power-of-two ====
Time: 0.044128
    passed
==== work-efficient scan, power-of-two ====
Scan Time: 0.083168
    passed
==== work-efficient scan, non-power-of-two ====
Scan Time: 0.079776
    a[0] = 0, b[0] = 3160
    FAIL VALUE
==== thrust scan, power-of-two ====
    passed
==== thrust scan, non-power-of-two ====
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
0ns
    passed
==== cpu compact without scan, non-power-of-two ====
    passed
==== cpu compact with scan ====
0ns
    passed
==== work-efficient compact, power-of-two ====
Scan Time: 0.079936
Compact Time: 0.362560
    passed
==== work-efficient compact, non-power-of-two ====
Scan Time: 0.078592
Compact Time: 0.361312
    expected 189 elements, got 0
    FAIL COUNT
	```