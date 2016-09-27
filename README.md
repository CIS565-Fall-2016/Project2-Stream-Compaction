CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Kaixiang Miao
* Tested on: Windows 7, i7-3630QM @ 2.40GHz 8GB, GTX 660M 2GB (Lenovo Y580 laptop, personal computer)

## Screenshot

___

All tests passed.
![](./img/passed.jpg)

## Performance Analysis

___

### Scan Comparison

| length of input| CPU Scan (ms) | Naïve Scan (ms) | non-optimized efficient Scan (ms) | optimized efficient scan (ms) |thrust scan (ms)|
|----------------|---------------|-----------------|-----------------------------------|-------------------------------|----------------|
| 2^10           | 0             | 0.0418          | 0.0882                            | 0.0954                        |0               |
| 2^13           | 0             | 0.0761          | 0.1681                            | 0.1473                        |0               |
| 2^16           | 0             | 0.4851          | 0.9168                            | 0.3155                        |0               |
| 2^19           | 2.0001        | 3.5169          | 7.7984                            | 1.6069                        |1               |
| 2^22           | 33.0019       | 31.4348         | 61.9076                           | 11.3602                       |5.003           |

The performance of **Naive Boids**, **Uniform Boids** and **Coherent Uniform Boids** is measured by FPS. Different amounts of boids are considered. The results are as below.

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

