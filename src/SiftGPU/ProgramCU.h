////////////////////////////////////////////////////////////////////////////
//	File:		ProgramCU.h
//	Author:		Changchang Wu
//	Description :	interface for the ProgramCU classes.
//					It is basically a wrapper around all the CUDA kernels
//
//	Copyright (c) 2007 University of North Carolina at Chapel Hill
//	All Rights Reserved
//
//	Permission to use, copy, modify and distribute this software and its
//	documentation for educational, research and non-profit purposes, without
//	fee, and without a written agreement is hereby granted, provided that the
//	above copyright notice and the following paragraph appear in all copies.
//	
//	The University of North Carolina at Chapel Hill make no representations
//	about the suitability of this software for any purpose. It is provided
//	'as is' without express or implied warranty. 
//
//	Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////

#ifndef _PROGRAM_CU_H
#define _PROGRAM_CU_H
#if defined(CUDA_SIFTGPU_ENABLED)

#include "config.h"

class CuTexImage;

class ProgramCU
{
public:

    //GPU FUNCTIONS
	static void FinishCUDA();
	static int  CheckErrorCUDA(const char* location);
    static int  CheckCudaDevice(int device);

public:

    ////SIFTGPU FUNCTIONS
	static void CreateFilterKernel(float sigma, float* kernel, int& width);
	template<int KWIDTH> static void FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf);
	static void FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf, float sigma);

#ifdef GPU_HESSIAN
	static void ComputeHessian(CuTexImage* gus, CuTexImage* dog, CuTexImage* got, float norm);
#else
	static void ComputeDOG(CuTexImage* gus, CuTexImage* dog, CuTexImage* got);
#endif // GPU_HESSIAN

	static void ComputeKEY(CuTexImage* dog, CuTexImage* key
#if defined GPU_HESSIAN
        , CuTexImage* gus
#endif // GPU_HESSIAN
        , float Tdog, float Tedge
#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)
        , int *featureTexLen, int featureTexIdx
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)
    );

#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)
	static void GenerateList(CuTexImage* list, CuTexImage* hist, int *counter);
#else
	static void GenerateList(CuTexImage* list, CuTexImage* hist);
	static void InitHistogram(CuTexImage* key, CuTexImage* hist);
	static void ReduceHistogram(CuTexImage* hist1, CuTexImage* hist2);
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)
	static void ComputeOrientation(CuTexImage* list, CuTexImage* got, CuTexImage* key, float sigma, float sigma_step, int existing_keypoint);
	static void ComputeDescriptor(CuTexImage* list, CuTexImage* got, CuTexImage* dtex, int rect = 0, int stream = 0);

#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined TOP_K_SELECTION
	static void TopKInit(TopKData &data, int listSize, int countThreshold);
    static void TopKCopyData(CuTexImage* list, CuTexImage* key, TopKData &topKData, int offset);
    static void TopKSort(TopKData &topKData);
    static void TopKPrefixScan(TopKData &topKData);
    static void TopKGetLevelsFeatureNum(TopKData &topKData);
    static void TopKCompactLevelFeatures(CuTexImage *list, unsigned int oldLen, float **newLevelFeatures, unsigned int newLen, TopKData &topKData, unsigned int offset);
	static void TopKFinish(TopKData &data);
#endif // GPU_HESSIAN

#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)
	static void DetectionDataInit(int **featureTexLen, int len);
	static void DetectionDataDownload(int *dst, int *featureTexLen, int len);
    static void DetectionDataFinish(int **featureTexLen);
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)

    //data conversion
	static void SampleImageU(CuTexImage *dst, CuTexImage *src, int log_scale);
	static void SampleImageD(CuTexImage *dst, CuTexImage *src, int log_scale = 1); 
	static void ReduceToSingleChannel(CuTexImage* dst, CuTexImage* src, int convert_rgb);
    static void ConvertByteToFloat(CuTexImage*src, CuTexImage* dst);
    
    //visualization
	static void DisplayConvertDOG(CuTexImage* dog, CuTexImage* out);
	static void DisplayConvertGRD(CuTexImage* got, CuTexImage* out);
	static void DisplayConvertKEY(CuTexImage* key, CuTexImage* dog, CuTexImage* out);
	static void DisplayKeyPoint(CuTexImage* ftex, CuTexImage* out);
	static void DisplayKeyBox(CuTexImage* ftex, CuTexImage* out);
	
	//SIFTMATCH FUNCTIONS	
	static void MultiplyDescriptor(CuTexImage* tex1, CuTexImage* tex2, CuTexImage* texDot, CuTexImage* texCRT);
	static void MultiplyDescriptorG(CuTexImage* texDes1, CuTexImage* texDes2,
		CuTexImage* texLoc1, CuTexImage* texLoc2, CuTexImage* texDot, CuTexImage* texCRT,
		float H[3][3], float hdistmax, float F[3][3], float fdistmax);
	static void GetRowMatch(CuTexImage* texDot, CuTexImage* texMatch, float distmax, float ratiomax);
	static void GetColMatch(CuTexImage* texCRT, CuTexImage* texMatch, float distmax, float ratiomax);
};

#endif
#endif

