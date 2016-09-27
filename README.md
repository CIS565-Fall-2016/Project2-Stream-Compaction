CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Xiaomao Ding
* Tested on: Windows 8.1, i7-4700MQ @ 2.40GHz 8.00GB, GT 750M 2047MB (Personal Computer)

# Intro
The code in this repo implements stream compaction and scan algorithms on the GPU in CUDA as well as on the CPU in C++ for performance comparisons. The scan algorithm performs a parallel prefix sum on the GPU. For more information, read this [NVIDIA link](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html).

![Image of Prefix Sum](http://http.developer.nvidia.com/GPUGems3/elementLinks/39fig02.jpg)

<font size="8"> Image from NVIDIA </font>

# Performance Analysis
This section below discusses the performance of the algorithms in this repository.

### Optimal Block Size
Each GPU algorithm was tested using an array of 2^14 integers. The optimal block size was found to be 128-256 as shown below. All calculations following this section are done with block size 128. Performance was timed using CUDAEvents.

| Block Size    | Naive GPU scan (ms)  | Efficient GPU scan (ms)  | Efficient GPU Compaction (ms)|
| :------------- |-------------:| -----:|-----:|
| 64    | 0.124     | 0.527 |0.531 |
| 128   | 0.094     | 0.484 |0.412 |
| 256   | 0.095     | 0.473 |0.423 |
| 512   | 0.102     | 0.471 |0.454 |
| 1024  | 0.109     | 0.495 |0.487 |

![Plot of block size](https://github.com/xnieamo/Project2-Stream-Compaction/blob/master/images/blockSizePlot.png)

### Performance comparisons
This section describes the performance of the various implementations of scan and stream compaction in this repository. For some reason, I get a stack overflow error when trying to run the algorithms with greater than 2^16 array entries, so that is maximum array size presented here.

| Array Size     | CPU scan  | Naive GPU scan | Efficient GPU scan  | CPU Compact w/o scan | CPU compact w/ scan | Efficient GPU compact | Thrust |
|:------|-----------:|----------------:|---------------------:|----------------------:|---------------------:|-----------------------:|--------:|
| 2^12 | 0.015623  | 0.051032       | 0.298               | 0.0090072            | 0.0312529           | 0.263                 | 0.352  |
| 2^14 | 0.062499  | 0.0928         | 0.422               | 0.0468755            | 0.1716863           | 0.425                 | 0.502  |
| 2^16 | 0.2343767 | 0.342          | 1.15                | 0.250018             | 0.6718685           | 1.127                 | 1.325  |

![Plot of various runtimes](https://github.com/xnieamo/Project2-Stream-Compaction/blob/master/images/performanceChart.png)

Because we are implementing the work-efficient algorithm described in GPU Gems without any optimizations, it actually runs SLOWER! When looking at the NVIDIA NSight runtime analysis, it appears that the thrust implementation is using asynchronous memory transfer, which seems to allow the CPU to call functions while a kernel is running. Surprisingly, the thrust implementation is still slower than the efficient GPU implementation (runtime was taken from NSight analysis, discounting initial and final memcpy operations).

In the case of the work-efficient algorithm, one of the issues that affects runtime is the fact that many threads idle as the upsweep and downsweep progress. Aside from that, a main bottleneck in my implementation is memory transfer from host to device. In the stream compaction algorithm, there is a need to set the last index to 0. Instead of doing this via a kernel, I transfer back to host. This results in an expensive memory transfer and adds roughly 0.100 ms to the runtime. Another bottleneck that seems to take about as long as the calculation itself is the cudaLaunch function. The internet hasn't been helpful in telling me what this does, but I suspect that it is responsible for launching the grids or blocks on the GPU. If so, then changing the index to 0 on the GPU might save me 25% of my runtime!

With the naive GPU scan, there aren't really many addressable bottlenecks. The calculation just takes that long.

For the CPU implementation, I think for this particular project, the w/o scan compaction runs faster as it only needs to perform a single comparison operation per element. The w/ scan implementation adds a large amount of unnecessary calculations (on the CPU) which makes it run much slower. This shows that GPU and CPU algorithms and the way we should about implementing code on these machines differs by quite a lot!

### Program output
Finally, here is the output of the various tests to validate my implementations, using an array of 2^16 elements. They all pass, woohoo!

```

****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  35   0 ]
==== cpu scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
==== cpu scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
==== naive scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== naive scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== work-efficient scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
==== thrust scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== thrust scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   1 ]
    passed
==== cpu compact with scan ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
    passed
==== work-efficient compact, non-power-of-two ====
    passed
```
