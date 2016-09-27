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
