CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ruoyu Fan
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### Description

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

#### Things I have done

* Implemented __CPU scan__, __compaction__, __GPU naive scan__, __GPU work-efficient scan__, __GPU radix sort (extra)__, and compared my scan algorithms with thrust implemention

* In CPU compact scan part, I reused output buffer as a temp buffer to save space without having trouble of (imagined) access violation.

* I also wrote an __invlusive version__ of __work-efficient scan__ - because i misunderstood the requirement at first! The difference of the inclusive method is that it creates a buffer that is 1 element larger and swap the last(0) and and second last elements before downsweeping. Although I corrected my implemention to exclusive scan, the inclusive scan can still be called by passing ScanType::inclusive to scan_implenmention method in efficient.cu.

* __Radix sort__ assumes inputs are between [0, a_given_maximum) . I compared my radix sort with std::sort and thrust's unstable and stable sort.

* I added a helper class PerformanceTimer in common.h which is used to do performance measurement.


