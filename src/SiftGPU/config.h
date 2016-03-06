////////////////////////////////////////////////////////////////////////////
//	File:		config.h
//	Author:		Jaroslav Sloup
//	Description :	Basic defines related to the GPU Hessian (HessGPU).
//
//	Copyright (c) 2015 Czech Technical University in Prague
//	All Rights Reserved
//
//	Please send BUG REPORTS to sloup@fel.cvut.cz
//
////////////////////////////////////////////////////////////////////////////

#ifndef _CONFIG_H
#define _CONFIG_H

// timing stored in SiftGPU::_timing structure
enum {
  TIMINGS_LOAD_IMAGE = 0,          // 0
  TIMINGS_ALLOCATE_PYRAMID,        // 1
  TIMINGS_BUILD_PYRAMID,           // 2
  TIMINGS_DETECT_KEYPOINTS,        // 3
  TIMINGS_GENERATE_FEATURE_LIST,   // 4
  TIMINGS_COMPUTE_ORIENTATIONS,    // 5
  TIMINGS_MULTI_ORIENTATIONS,      // 6
  TIMINGS_DOWNLOAD_KEYPOINTS,      // 7
  TIMINGS_COMPUTE_DESCRIPTORS,     // 8
  TIMINGS_GENERATE_VBO,            // 9
  TIMINGS_FEATURES_REDUCTION,      // 10 - topk
  TIMINGS_TOTAL,                   // 11 - total time
  TIMINGS_COUNT
};

#define PI 3.14159265358979323846

// use determinant of hessian matrix instead of DoG for keypoints detection
#define GPU_HESSIAN

// SiftGPU modification to export response and allow topk selection (cuda version only)
//#define GPU_SIFT_MODIFIED

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  // HessGPU and modified SiftGPU specific defines

  // detected keypoint types
  enum {
    FEATURE_TYPE_DARK_BLOB    = 0,
    FEATURE_TYPE_BRIGHT_BLOB  = 1,
    FEATURE_TYPE_SADDLE_POINT = 2,
    FEATURE_TYPE_NONE         = 3
  };

  // uses compressed representation that allows to store up to four orientations per keypoint
  // and export additional info about keypoints (type + level)
  // position, scale, and orientation are stored in fixed point representation internally

  // enables version with less registers per thread requirements -> better gpu occupancy
  // original version consumes 80 registers per thread
  #define NORMALIZE_DESCRIPTOR_PER_WARP

  // requires at least CC 2.0 (Fermi)
  // works only with GPU_HESSIAN and GPU_SIFT_MODIFIED, but not with original SiftGPU
  // faster version of feature list generation that utilize atomic operations
  #define GENERATE_FEATURE_LIST_USING_ATOMICS

  // fixed point related defines
  #define FIXED_POINT_POSITION_PRECISION_BITS      10 // 24bits 14.10 (unsigned)
  #define FIXED_POINT_SCALE_PRECISION_BITS         8  // 16bits 8.8 (unsigned)
  #define FIXED_POINT_POSITION_MASK                0x00FFFFFFu // 24 bits => mask = (1<<numBits)-1
  #define FIXED_POINT_SCALE_MASK                   0x0000FFFFu // 16 bits
  #define FIXED_POINT_RESPONSE_MASK                0xFF000000u // 8 bits

  // float R --> fixed point F =>   F = (int)(R * (1<<N) + (R>=0 ? 0.5 : -0.5))
  #define FLOAT_TO_FIXED_POINT(fltValue, fracDigits) \
    (int)((fltValue) * (1<<(fracDigits)) + (((fltValue)>=0.0) ? 0.5 : -0.5))

  // fixed point F --> float R =>	R = (float)F / (1<<N)
  #define FIXED_POINT_TO_FLOAT(fpValue, fracDigits) \
    ((float)(fpValue) / (1<<(fracDigits)))

  #define MIN_VALUE  1.0e-30f

  // requires at least CC 2.0 (Fermi)
  // works only with GPU_HESSIAN and GPU_SIFT_MODIFIED, but not with original SiftGPU
  // selects top K keypoints with the highest absolute value of response
  #define TOP_K_SELECTION

  #ifdef TOP_K_SELECTION

	typedef struct TopKData {
		float         *keys;
		unsigned int  *indices;
		int           *levelFeaturesCount;
		int           *devLevelFeaturesCount;
		int            levelsCount;
		int            keypointsCount;
		int            keypointsCountAsPowerOfTwo; // the closest power of 2 - keypointsCount < keypointsCountAsPowerOf2 = 2^k;
		int            topKCountThreshold;
        unsigned int **scanBlockSums;              // prefix scan data
        unsigned int   numElementsAllocated;
        unsigned int   numLevelsAllocated;
	} TopKData;

  #endif // TOP_K_SELECTION

#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

#ifdef GPU_HESSIAN
  // HessGPU specific defines
#endif // GPU_HESSIAN

#ifdef GPU_SIFT_MODIFIED
  // modified SiftGPU specific defines
#endif // GPU_SIFT_MODIFIED

#endif // _CONFIG_H
