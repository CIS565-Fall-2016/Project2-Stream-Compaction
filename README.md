CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Michael Willett
* Tested on: Windows 10, I5-4690k @ 3.50GHz 8.00GB, GTX 750-TI 2GB (Personal Computer)

## Contents
1. [Introduction](#intro)
2. [Algorithms](#part1)
4. [Performance Analysis](#part3)
5. [Development Process](#part4)
6. [Build Instructions](#appendix)


<a name="intro"/>
## Introduction: Parallel Algorithms
This project explores introductory concepts of GPU paralization methods for simulating flocking behaviors
of simple particles known as boids. Boid motion is based off of three rules calculated from nearby particles:

1. *Cohesion* - Boids will move towards the center of mass of nearby boids
2. *Separation* - Boids will try to maintain a minimum distance from one another to avoid collision
3. *Alignment* - Boids in a group will try to align vector headings with those in the group

These simple rules with the appropriate tuning parameter set can lead to a surprisingly complex emergent 
behavior very similar to how schools of fish or flocks of birds move in nature, as seen below.


<a name="part1"/>
## Section 1: Scanning, Stream Compaction, and Sorting
The boids flocking simulation is naively calculated by comparing euclidean distance from the current
boid to every other boid in the simulation, and checking if the distance is within the desired range for
the rule being calculated (we use a smaller distance metric for calculating separation, otherwise the boids
never exhibit flocking behavior).

While computationally this results in the correct behavior, it can be wasteful in the number of comparison operations
since we only apply position and velocity updates if a boid has at least one other particle close to it. For smaller
particle counts, this method can achieve 60 fps on a cheap modern process, but scales very poorly as the number of 
comparisons increases. Detailed analysis is available in [Section 3: Performance Analysis](#part-3).

<a name="part2"/>
## Section 2: Performance Analysis

Code Correctness for Scan, Compact, and Sort Implementations:
> ****************
> ** SCAN TESTS **
> ****************
>     [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  35   0 ]

> ==== cpu scan, power-of-two ====

>     [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
> ==== cpu scan, non-power-of-two ====

>     passed
> ==== naive scan, power-of-two ====

>     [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
>     passed
> ==== naive scan, non-power-of-two ====

>     passed
> ==== work-efficient scan, power-of-two ====

>     [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
>     passed
> ==== work-efficient scan, non-power-of-two ====

>     passed
> ==== thrust scan, power-of-two ====

>     [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
>     passed
> ==== thrust scan, non-power-of-two ====

>     passed
> 
> *****************************
> ** STREAM COMPACTION TESTS **
> *****************************

>     [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   1   0 ]
> ==== cpu compact without scan, power-of-two ====

>     [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
>     passed
> ==== cpu compact without scan, non-power-of-two ====

>     [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   3 ]
>     passed
> ==== cpu compact with scan ====

>     [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
>     passed
> ==== work-efficient compact, power-of-two ====

>     passed
> ==== work-efficient compact, non-power-of-two ====

>     passed
> 
> **********************
> ** RADIX SORT TESTS **
> **********************

> ==== radix sort, power-of-two ====

>     [  38 7719 21238 2437 8855 11797 8365 32285 ]
>     [  38 2437 7719 8365 8855 11797 21238 32285 ]
>     passed
> ==== radix sort, non-power-of-two ====

>     [  38 7719 21238 2437 8855 11797 8365 ]
>     [  38 2437 7719 8365 8855 11797 21238 ]
>     passed

> 
> Sort passed 1000/1000 randomly generated verification tests.




<a name="part3"/>
## Section 3: Development Process
Development was fairly straight forward algorithmic implementation. Future work could be done in the work effecient 
implementations to better handle launching kernal functions at the high depth levels when only a couple of sums are
being calculated for the whole array. This should 

It was worth noting there was a bug in using std::pow for calculating the array index in each kernal invocation. For some
unknown reason, it consisantly produced erronius values at 2^11, or 2048. This is odd since variables were being cast to 
double precision before the computation, but the reverse cast was incorrect. This bug may be compiler specific as running
the index calculation on a separate machine result in accurate indexing in this range. Simply changing the code to use
bitshift operations cleared the error entirely. 



<a name="appendix"/>
## Appendix: Build Instructions

* `src/` contains the source code.

**CMake note:** Do not change any build settings or add any files to your
project directly (in Visual Studio, Nsight, etc.) Instead, edit the
`src/CMakeLists.txt` file. Any files you add must be added here. If you edit it,
just rebuild your VS/Nsight project to make it update itself.

#### Windows

1. In Git Bash, navigate to your cloned project directory.
2. Create a `build` directory: `mkdir build`
   * (This "out-of-source" build makes it easy to delete the `build` directory
     and try again if something goes wrong with the configuration.)
3. Navigate into that directory: `cd build`
4. Open the CMake GUI to configure the project:
   * `cmake-gui ..` or `"C:\Program Files (x86)\cmake\bin\cmake-gui.exe" ..`
     * Don't forget the `..` part!
   * Make sure that the "Source" directory is like
     `.../Project2-Stream-Compaction`.
   * Click *Configure*.  Select your version of Visual Studio, Win64.
     (**NOTE:** you must use Win64, as we don't provide libraries for Win32.)
   * If you see an error like `CUDA_SDK_ROOT_DIR-NOTFOUND`,
     set `CUDA_SDK_ROOT_DIR` to your CUDA install path. This will be something
     like: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5`
   * Click *Generate*.
5. If generation was successful, there should now be a Visual Studio solution
   (`.sln`) file in the `build` directory that you just created. Open this.
   (from the command line: `explorer *.sln`)
6. Build. (Note that there are Debug and Release configuration options.)
7. Run. Make sure you run the `cis565_` target (not `ALL_BUILD`) by
   right-clicking it and selecting "Set as StartUp Project".
   * If you have switchable graphics (NVIDIA Optimus), you may need to force
     your program to run with only the NVIDIA card. In NVIDIA Control Panel,
     under "Manage 3D Settings," set "Multi-display/Mixed GPU acceleration"
     to "Single display performance mode".

#### OS X & Linux

It is recommended that you use Nsight.

1. Open Nsight. Set the workspace to the one *containing* your cloned repo.
2. *File->Import...->General->Existing Projects Into Workspace*.
   * Select the Project 0 repository as the *root directory*.
3. Select the *cis565-* project in the Project Explorer. From the *Project*
   menu, select *Build All*.
   * For later use, note that you can select various Debug and Release build
     configurations under *Project->Build Configurations->Set Active...*.
4. If you see an error like `CUDA_SDK_ROOT_DIR-NOTFOUND`:
   * In a terminal, navigate to the build directory, then run: `cmake-gui ..`
   * Set `CUDA_SDK_ROOT_DIR` to your CUDA install path.
     This will be something like: `/usr/local/cuda`
   * Click *Configure*, then *Generate*.
5. Right click and *Refresh* the project.
6. From the *Run* menu, *Run*. Select "Local C/C++ Application" and the
   `cis565_` binary.
