
HessGPU = Hessian interest points detector + sift descriptor
------------------------------------------------------------

Source code
-----------

All modifications to implement Hessian interest points detector as
well as new features and improvements (topk, halfsift, type and
response export) were done directly into the SiftGPU code. We included
additional header file config.h that contains all defines related to
HessGPU.

Source code can be built in three versions (see config.h):

1) as Hessian interest point detector + new features
   =>
     use:             #define GPU_HESSIAN
     and comment out: #define GPU_SIFT_MODIFIED

2) as SIFT detector + new features
   =>
     use:             #define GPU_SIFT_MODIFIED
     and comment out: #define GPU_HESSIAN

3) as original SiftGPU without modifications  
   =>
     comment out:     #define GPU_SIFT_MODIFIED
     and comment out: #define GPU_HESSIAN

The usage of the library is the same as for the SiftGPU. Additionally,
there is application "hess" that allows you to run HessGPU directly
from command line and also shows possible ways to use library in your
projects.

Please send BUG REPORTS related to HessGPU to sloup@fel.cvut.cz.


Building with CMake
-------------------

Follow these steps to build HessGPU:

mkdir build
cd build
cmake ..
make

After successfull build you can install library and applications:

make install

