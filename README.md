# HessGPU - Implementation of the Hessian interest points detector with the SIFT descriptor on GPU

DESCRIPTION

We take a popular GPU implementation of SIFT – the de-facto standard in
fast interest point detectors – SiftGPU [2] and implement modifications
that according to recent research result in better performance in terms
of repeatability of the detected points.

The interest points found at local extrema of the Difference of Gaussians
(DoG) function in the original SIFT are replaced by the local extrema of
determinant of the Hessian matrix (a matrix of second-order partial
derivatives) of the intensity function. This yeilds interest points either
for blob-like (local maxima) or for saddle-like structures (local minima).
Experiments show that the extrema of the determinant of Hessian are more
repeatable and accurate than the extrema of the Difference of Gaussians,
and, thanks to the additional detection of saddle points, the object
coverage is generally also improved. The detection speed of Hessian is
similar to that of the SIFT.

This CUDA implementation of Hessian detector (shortly called HessGPU) is
a supplemental material for paper [1]. More information can be found here
http://dcgi.felk.cvut.cz/home/sloup/index.php?page=cvww2016_hessian.

Key Features:

* Interest points are found as local extrema of the determinant of the
  Hessian matrix (a matrix of second-order partial derivatives) of the
  intensity function.

* Selection of best K points (when ordered by magnitude of the determinant)
  is implemented in an early stage of the algorithm, before orientations are
  determined and descriptors computed.

* The feature type (saddle, dark, or white blob) is now part of the GPU code
  output. This is useful in followup matching – features of different types
  should not be considered for a correspondence.

* Reintroduced optional capability to compute orientations and descriptors
  only in <0,PI> range instead of <0,2*PI> by disregarding sign of the gradients
  involved, which is beneficial when matching images taken under significantly
  different illumination (day and night).

* The restriction that at each image location only at most two interest point
  orientations are detected was lifted.

* Faster feature list construction and descriptor normalization compared
  to SiftGPU.


REQUIREMENTS

HessGPU hardware and software requirements are the same as those for SiftGPU.
It requires a NVIDIA GPU that has a large graphic memory and supports CUDA
(at least CC 2.0 - Fermi architecture).


PROPER USE

If you use this code, please refer to [1].


REFERENCES

[1] Jaroslav Sloup, Michal Perďoch, Štěpán Obdržálek, Jiří Matas: Hessian
    Interest Points on GPU. In Proceedings of the 21st Computer Vision Winter
    Workshop (CVWW 2016), Slovenian Pattern Recognition Society, February 2016.
    http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/08.pdf

[2] Changchang Wu: SiftGPU - A GPU Implementation of Scale Invariant Feature
    Transform (SIFT). University of North Carolina at Chapel Hill, 2007.
    http://www.cs.unc.edu/~ccwu/siftgpu/">http://www.cs.unc.edu/~ccwu/siftgpu/
