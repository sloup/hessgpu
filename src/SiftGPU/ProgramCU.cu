////////////////////////////////////////////////////////////////////////////
//	File:		ProgramCU.cu
//	Author:		Changchang Wu
//	Description : implementation of ProgramCU and all CUDA kernels
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

#if defined(CUDA_SIFTGPU_ENABLED)

#include "GL/glew.h"
#include "stdio.h"

#include <iostream>
#include <assert.h>
#include "CuTexImage.h"
#include "ProgramCU.h"
#include "GlobalUtil.h"

//----------------------------------------------------------------
//Begin SiftGPU setting section.
//////////////////////////////////////////////////////////
#define IMUL(X,Y) __mul24(X,Y)
//#define FDIV(X,Y) ((X)/(Y))
#define FDIV(X,Y) __fdividef(X,Y)

/////////////////////////////////////////////////////////
//filter kernel width range (don't change this)
#define KERNEL_MAX_WIDTH 33
#define KERNEL_MIN_WIDTH 5

//////////////////////////////////////////////////////////
//horizontal filter block size (32, 64, 128, 256, 512)
#define FILTERH_TILE_WIDTH 128
//thread block for vertical filter. FILTERV_BLOCK_WIDTH can be (4, 8 or 16)
#define FILTERV_BLOCK_WIDTH 16
#define FILTERV_BLOCK_HEIGHT 32
//The corresponding image patch for a thread block
#define FILTERV_PIXEL_PER_THREAD 4
#define FILTERV_TILE_WIDTH FILTERV_BLOCK_WIDTH
#define FILTERV_TILE_HEIGHT (FILTERV_PIXEL_PER_THREAD * FILTERV_BLOCK_HEIGHT)

//////////////////////////////////////////////////////////
//thread block size for computing Difference of Gaussian
#define DOG_BLOCK_LOG_DIMX 7
#define DOG_BLOCK_LOG_DIMY 0
#define DOG_BLOCK_DIMX (1 << DOG_BLOCK_LOG_DIMX)
#define DOG_BLOCK_DIMY (1 << DOG_BLOCK_LOG_DIMY)

//////////////////////////////////////////////////////////
//thread block size for keypoint detection
#define KEY_BLOCK_LOG_DIMX 5  // 3
#define KEY_BLOCK_LOG_DIMY 2  // 3
#define KEY_BLOCK_DIMX (1<<KEY_BLOCK_LOG_DIMX)
#define KEY_BLOCK_DIMY (1<<KEY_BLOCK_LOG_DIMY)
//make KEY_BLOCK_LOG_DIMX 4 will make the write coalesced..
//but it seems uncoalesced writes don't affect the speed

//////////////////////////////////////////////////////////
//thread block size for initializing list generation (64, 128, 256, 512 ...)
#define HIST_INIT_WIDTH 128
//thread block size for generating feature list (32, 64, 128, 256, 512, ...)
#define LISTGEN_BLOCK_DIM 128

/////////////////////////////////////////////////////////
//how many keypoint orientations to compute in a block
#define ORIENTATION_COMPUTE_PER_BLOCK 64
//how many keypoint descriptor to compute in a block (2, 4, 8, 16, 32)
#define DESCRIPTOR_COMPUTE_PER_BLOCK	4
#define DESCRIPTOR_COMPUTE_BLOCK_SIZE	(16 * DESCRIPTOR_COMPUTE_PER_BLOCK)

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  // block size for the keypoint descriptor normalization kernel 
  // it is assumed that one descriptor is processed by one warp 
  // -> have to be multiple of warp size (32)
  #define DESCRIPTOR_NORMALIZE_PER_BLOCK	128
#else
  //how many keypoint descriptor to normalized in a block (32, ...)
  #define DESCRIPTOR_NORMALIZE_PER_BLOCK	32
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

///////////////////////////////////////////
//Thread block size for visualization 
//(This doesn't affect the speed of computation)
#define BLOCK_LOG_DIM 4
#define BLOCK_DIM (1 << BLOCK_LOG_DIM)

//End SiftGPU setting section.
//----------------------------------------------------------------

__device__ __constant__ float d_kernel[KERNEL_MAX_WIDTH];

texture<float, 1, cudaReadModeElementType> texData;
texture<unsigned char, 1, cudaReadModeNormalizedFloat> texDataB;
texture<float2, 2, cudaReadModeElementType> texDataF2;
texture<float4, 1, cudaReadModeElementType> texDataF4;
texture<int4, 1, cudaReadModeElementType> texDataI4;
texture<int4, 1, cudaReadModeElementType> texDataList;

//template<int i>	 __device__ float Conv(float *data)		{    return Conv<i-1>(data) + data[i]*d_kernel[i];}
//template<>		__device__ float Conv<0>(float *data)	{    return data[0] * d_kernel[0];					}
  
//////////////////////////////////////////////////////////////
template<int FW> __global__ void FilterH( float* d_result, int width)
{
  const int HALF_WIDTH = FW >> 1;
  const int CACHE_WIDTH = FILTERH_TILE_WIDTH + FW -1;
  const int CACHE_COUNT = 2 + (CACHE_WIDTH - 2)/ FILTERH_TILE_WIDTH;

  __shared__ float data[CACHE_WIDTH];

  const int bcol = IMUL(blockIdx.x, FILTERH_TILE_WIDTH);
  const int col =  bcol + threadIdx.x;
  const int index_min = IMUL(blockIdx.y, width);
  const int index_max = index_min + width - 1;
  int src_index = index_min + bcol - HALF_WIDTH + threadIdx.x;
  int cache_index = threadIdx.x;
  float value = 0;

#pragma unroll
  for(int j = 0; j < CACHE_COUNT; ++j)
  {
    if(cache_index < CACHE_WIDTH)
    {
      int fetch_index = src_index < index_min? index_min : (src_index > index_max ? index_max : src_index);
      data[cache_index] = tex1Dfetch(texData,fetch_index);
      src_index += FILTERH_TILE_WIDTH;
      cache_index += FILTERH_TILE_WIDTH;
    }
  }

  __syncthreads(); 
  if(col >= width)
    return;

#pragma unroll
  for(int i = 0; i < FW; ++i)
  {
    value += (data[threadIdx.x + i]* d_kernel[i]);
  }
  // value = Conv<FW-1>(data + threadIdx.x);
  d_result[index_min + col] = value;
}

////////////////////////////////////////////////////////////////////
template<int  FW>  __global__ void FilterV(float* d_result, int width, int height)
{
  const int HALF_WIDTH = FW >> 1;
  const int CACHE_WIDTH = FW + FILTERV_TILE_HEIGHT - 1;
  const int TEMP = CACHE_WIDTH & 0xf;

  // add some extra space to avoid bank conflict
#if FILTERV_TILE_WIDTH == 16
  // make the stride 16 * n +/- 1
  const int EXTRA = (TEMP == 1 || TEMP == 0) ? 1 - TEMP : 15 - TEMP;
#elif FILTERV_TILE_WIDTH == 8
  // make the stride 16 * n +/- 2
  const int EXTRA = (TEMP == 2 || TEMP == 1 || TEMP == 0) ? 2 - TEMP : (TEMP == 15? 3 : 14 - TEMP);
#elif FILTERV_TILE_WIDTH == 4
  // make the stride 16 * n +/- 4
  const int EXTRA = (TEMP >=0 && TEMP <=4) ? 4 - TEMP : (TEMP > 12? 20 - TEMP : 12 - TEMP);
#else
#error
#endif

  const int CACHE_TRUE_WIDTH = CACHE_WIDTH + EXTRA;
  const int CACHE_COUNT = (CACHE_WIDTH + FILTERV_BLOCK_HEIGHT - 1) / FILTERV_BLOCK_HEIGHT;
  const int WRITE_COUNT = (FILTERV_TILE_HEIGHT + FILTERV_BLOCK_HEIGHT -1) / FILTERV_BLOCK_HEIGHT;

  __shared__ float data[CACHE_TRUE_WIDTH * FILTERV_TILE_WIDTH];

  const int row_block_first = IMUL(blockIdx.y, FILTERV_TILE_HEIGHT);
  const int col = IMUL(blockIdx.x, FILTERV_TILE_WIDTH) + threadIdx.x;
  const int row_first = row_block_first - HALF_WIDTH;
  const int data_index_max = IMUL(height - 1, width) + col;
  const int cache_col_start = threadIdx.y;	
  const int cache_row_start = IMUL(threadIdx.x, CACHE_TRUE_WIDTH);
  int cache_index = cache_col_start + cache_row_start;
  int data_index = IMUL(row_first + cache_col_start, width) + col;

  if(col < width) 
  {
#pragma unroll
    for(int i = 0; i < CACHE_COUNT; ++i)
    {
      if(cache_col_start < CACHE_WIDTH - i * FILTERV_BLOCK_HEIGHT) 
      {
        int fetch_index = data_index < col ? col : (data_index > data_index_max? data_index_max : data_index);
        data[cache_index + i * FILTERV_BLOCK_HEIGHT] = tex1Dfetch(texData,fetch_index);
        data_index += IMUL(FILTERV_BLOCK_HEIGHT, width);
      }
    }
  }
  __syncthreads();
	
  if(col >= width)
    return;

  int row = row_block_first + threadIdx.y;
  int index_start = cache_row_start + threadIdx.y;

#pragma unroll
  for(int i = 0; i < WRITE_COUNT; ++i, row += FILTERV_BLOCK_HEIGHT, index_start += FILTERV_BLOCK_HEIGHT)
  {
    if(row < height)
    {
      int index_dest = IMUL(row, width) + col;
      float value = 0;

#pragma unroll
      for(int i = 0; i < FW; ++i)
      {
        value += (data[index_start + i] * d_kernel[i]);
      }
      d_result[index_dest] = value;
    }
  }
}

template<int LOG_SCALE> __global__ void UpsampleKernel(float* d_result, int width)
{
  const int SCALE = (1 << LOG_SCALE), SCALE_MASK = (SCALE - 1);
  const float INV_SCALE = 1.0f / (float(SCALE));
  int col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;

  if(col >= width)
    return;

  int row = blockIdx.y >> LOG_SCALE;
  int index = row * width + col;
  int dst_row = blockIdx.y;
  int dst_idx = (width * dst_row + col) * SCALE;
  int helper = blockIdx.y & SCALE_MASK;

  if (helper)
  {
    float v11 = tex1Dfetch(texData, index);
    float v12 = tex1Dfetch(texData, index + 1);
    index += width;
    float v21 = tex1Dfetch(texData, index);
    float v22 = tex1Dfetch(texData, index + 1);

    float w1 = INV_SCALE * helper, w2 = 1.0 - w1;
    float v1 = (v21 * w1  + w2 * v11);
    float v2 = (v22 * w1  + w2 * v12);

    d_result[dst_idx] = v1;

#pragma unroll
    for(int i = 1; i < SCALE; ++i)
    {
      const float r2 = i * INV_SCALE;
      const float r1 = 1.0f - r2; 
      d_result[dst_idx +i] = v1 * r1 + v2 * r2;
    }
  }
  else
  {
    float v1 = tex1Dfetch(texData, index);
    float v2 = tex1Dfetch(texData, index + 1);

    d_result[dst_idx] = v1;

#pragma unroll
    for(int i = 1; i < SCALE; ++i)
    {
      const float r2 = i * INV_SCALE;
      const float r1 = 1.0f - r2; 
      d_result[dst_idx +i] = v1 * r1 + v2 * r2;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////
void ProgramCU::SampleImageU(CuTexImage *dst, CuTexImage *src, int log_scale)
{
  int width = src->GetImgWidth(), height = src->GetImgHeight();
  src->BindTexture(texData);

  dim3 grid((width +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH, height << log_scale);
  dim3 block(FILTERH_TILE_WIDTH);

  switch(log_scale)
  {
    case 1:
      UpsampleKernel<1> <<< grid, block>>> ((float*) dst->_cuData, width);
      break;
    case 2:
      UpsampleKernel<2> <<< grid, block>>> ((float*) dst->_cuData, width);
      break;
    case 3:
      UpsampleKernel<3> <<< grid, block>>> ((float*) dst->_cuData, width);
      break;
    default:
      break;
  }
}

template<int LOG_SCALE> __global__ void DownsampleKernel(float* d_result, int src_width, int dst_width)
{
  const int dst_col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;

  if(dst_col >= dst_width)
    return;

  const int src_col = min((dst_col << LOG_SCALE), (src_width - 1));
  const int dst_row = blockIdx.y; 
  const int src_row = blockIdx.y << LOG_SCALE;
  const int src_idx = IMUL(src_row, src_width) + src_col;
  const int dst_idx = IMUL(dst_width, dst_row) + dst_col;

  d_result[dst_idx] = tex1Dfetch(texData, src_idx);
}

__global__ void DownsampleKernel(float* d_result, int src_width, int dst_width, const int log_scale)
{
  const int dst_col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;

  if(dst_col >= dst_width)
    return;

  const int src_col = min((dst_col << log_scale), (src_width - 1));
  const int dst_row = blockIdx.y; 
  const int src_row = blockIdx.y << log_scale;
  const int src_idx = IMUL(src_row, src_width) + src_col;
  const int dst_idx = IMUL(dst_width, dst_row) + dst_col;

  d_result[dst_idx] = tex1Dfetch(texData, src_idx);
}

void ProgramCU::SampleImageD(CuTexImage *dst, CuTexImage *src, int log_scale)
{
  int src_width = src->GetImgWidth();
  int dst_width = dst->GetImgWidth();

  src->BindTexture(texData);
  dim3 grid((dst_width +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH, dst->GetImgHeight());
  dim3 block(FILTERH_TILE_WIDTH);

  switch(log_scale)
  {
    case 1:
      DownsampleKernel<1> <<< grid, block>>> ((float*) dst->_cuData, src_width, dst_width);
      break;
    case 2:
      DownsampleKernel<2> <<< grid, block>>> ((float*) dst->_cuData, src_width, dst_width);
      break;
    case 3:
      DownsampleKernel<3> <<< grid, block>>> ((float*) dst->_cuData, src_width, dst_width);
      break;
    default:
      DownsampleKernel    <<< grid, block>>> ((float*) dst->_cuData, src_width, dst_width, log_scale);
  }
}

__global__ void ChannelReduce_Kernel(float* d_result)
{
  int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;

  d_result[index] = tex1Dfetch(texData, index*4);
}

__global__ void ChannelReduce_Convert_Kernel(float* d_result)
{
  int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
  float4 rgba = tex1Dfetch(texDataF4, index);

  d_result[index] = 0.299f * rgba.x + 0.587f* rgba.y + 0.114f * rgba.z;
}

void ProgramCU::ReduceToSingleChannel(CuTexImage* dst, CuTexImage* src, int convert_rgb)
{
  int width = src->GetImgWidth();
  int height = dst->GetImgHeight() ;

  dim3 grid((width * height +  FILTERH_TILE_WIDTH - 1) / FILTERH_TILE_WIDTH);
  dim3 block(FILTERH_TILE_WIDTH);

  if(convert_rgb)
  {
    src->BindTexture(texDataF4);
    ChannelReduce_Convert_Kernel<<<grid, block>>>((float*)dst->_cuData);
  }
  else
  {
    src->BindTexture(texData);
    ChannelReduce_Kernel<<<grid, block>>>((float*)dst->_cuData);
  }
}

__global__ void ConvertByteToFloat_Kernel(float* d_result)
{
  int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;

  d_result[index] = tex1Dfetch(texDataB, index);
}

void ProgramCU::ConvertByteToFloat(CuTexImage*src, CuTexImage* dst)
{
  int width = src->GetImgWidth();
  int height = dst->GetImgHeight() ;
  dim3 grid((width * height +  FILTERH_TILE_WIDTH - 1) / FILTERH_TILE_WIDTH);
  dim3 block(FILTERH_TILE_WIDTH);

  src->BindTexture(texDataB);

  ConvertByteToFloat_Kernel<<<grid, block>>>((float*)dst->_cuData);
}

void ProgramCU::CreateFilterKernel(float sigma, float* kernel, int& width)
{
  int i, sz = int( ceil( GlobalUtil::_FilterWidthFactor * sigma -0.5) ); //
  width = 2*sz + 1;

  if(width > KERNEL_MAX_WIDTH)
  {
    //filter size truncation
    sz = KERNEL_MAX_WIDTH >> 1;
    width = KERNEL_MAX_WIDTH;
  }
  else if(width < KERNEL_MIN_WIDTH)
  {
    sz = KERNEL_MIN_WIDTH >> 1;
    width =KERNEL_MIN_WIDTH;
  }

  float rv = 1.0f / (sigma*sigma), v, ksum = 0; 

  // pre-compute filter
  for( i = -sz ; i <= sz ; ++i) 
  {
    kernel[i+sz] =  v = exp(-0.5f * i * i *rv) ;
    ksum += v;
  }

  //normalize the kernel
  rv = 1.0f / ksum;
  for(i = 0; i< width ;i++)
    kernel[i] *= rv;
}

template<int FW> void ProgramCU::FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf)
{
  int width = src->GetImgWidth();
  int height = src->GetImgHeight();

  //horizontal filtering
  src->BindTexture(texData);

  dim3 gridh((width +  FILTERH_TILE_WIDTH - 1)/FILTERH_TILE_WIDTH, height);
  dim3 blockh(FILTERH_TILE_WIDTH);

  FilterH<FW><<<gridh, blockh>>>((float*)buf->_cuData, width);

  CheckErrorCUDA("FilterH");

  ///vertical filtering
  buf->BindTexture(texData);

  dim3 gridv((width + FILTERV_TILE_WIDTH - 1)/FILTERV_TILE_WIDTH, (height + FILTERV_TILE_HEIGHT - 1)/FILTERV_TILE_HEIGHT);
  dim3 blockv(FILTERV_TILE_WIDTH, FILTERV_BLOCK_HEIGHT);

  FilterV<FW><<<gridv, blockv>>>((float*)dst->_cuData, width, height); 

  CheckErrorCUDA("FilterV");
}

//////////////////////////////////////////////////////////////////////
// tested on 2048x1500 image, the time on pyramid construction is
// OpenGL version : 18ms
// CUDA version: 28 ms
void ProgramCU::FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf, float sigma)
{
  float filter_kernel[KERNEL_MAX_WIDTH];
  int width;

  CreateFilterKernel(sigma, filter_kernel, width);
  cudaMemcpyToSymbol(d_kernel, filter_kernel, width * sizeof(float), 0, cudaMemcpyHostToDevice);

  switch(width)
  {
    case 5:		FilterImage< 5>(dst, src, buf);	break;
    case 7:		FilterImage< 7>(dst, src, buf);	break;
    case 9:		FilterImage< 9>(dst, src, buf);	break;
    case 11:	FilterImage<11>(dst, src, buf);	break;
    case 13:	FilterImage<13>(dst, src, buf);	break;
    case 15:	FilterImage<15>(dst, src, buf);	break;
    case 17:	FilterImage<17>(dst, src, buf);	break;
    case 19:	FilterImage<19>(dst, src, buf);	break;
    case 21:	FilterImage<21>(dst, src, buf);	break;
    case 23:	FilterImage<23>(dst, src, buf);	break;
    case 25:	FilterImage<25>(dst, src, buf);	break;
    case 27:	FilterImage<27>(dst, src, buf);	break;
    case 29:	FilterImage<29>(dst, src, buf);	break;
    case 31:	FilterImage<31>(dst, src, buf);	break;
    case 33:	FilterImage<33>(dst, src, buf);	break;
    default:	break;
  }
}

texture<float, 1, cudaReadModeElementType> texC;
texture<float, 1, cudaReadModeElementType> texP;
texture<float, 1, cudaReadModeElementType> texN;

#ifdef GPU_HESSIAN

texture<float, 1, cudaReadModeElementType> texG;

// compute 3x3 Hessian values from symmetric differences
#define COMPUTE_HESSIAN(tex, idx)                 \
  float v11 = tex1Dfetch(tex, idx - width - 1);   \
  float v12 = tex1Dfetch(tex, idx - width);       \
  float v13 = tex1Dfetch(tex, idx - width + 1);   \
                                                  \
  float v21 = tex1Dfetch(tex, idx - 1);           \
  float v22 = tex1Dfetch(tex, idx);               \
  float v23 = tex1Dfetch(tex, idx + 1);           \
                                                  \
  float v31 = tex1Dfetch(tex, idx + width - 1);   \
  float v32 = tex1Dfetch(tex, idx + width);       \
  float v33 = tex1Dfetch(tex, idx + width + 1);   \
                                                  \
  float Lxx = (v21 - 2.0f*v22 + v23);             \
  float Lyy = (v12 - 2.0f*v22 + v32);             \
  float Lxy = (v13 - v11 + v31 - v33) * 0.25f;    \


void __global__ ComputeHessian_Kernel(float *hessian, float2 *got, int width, int height, float norm)
{
  int row = (blockIdx.y << DOG_BLOCK_LOG_DIMY) + threadIdx.y;
  int col = (blockIdx.x << DOG_BLOCK_LOG_DIMX) + threadIdx.x;

  if((col < width) && (row < height)) 
  {
    int index = IMUL(row, width) + col;

    COMPUTE_HESSIAN(texC, index)

    // compute determinant of hessian matrix, normalize and write out
    hessian[index] = (Lxx*Lyy - Lxy*Lxy)*norm;

    // precompute gradient and rotation
    float dx = v23 - v21;
    float dy = v32 - v12;
    float gradient = 0.5f * sqrt(dx*dx  + dy*dy);
    float rot = ((gradient == 0.0f) ? 0.0f : atan2(dy, dx));

    got[index] = make_float2(gradient, rot);
  }
}

void __global__ ComputeHessian_Kernel(float *hessian, int width, int height, float norm)
{
  int row = (blockIdx.y << DOG_BLOCK_LOG_DIMY) + threadIdx.y;
  int col = (blockIdx.x << DOG_BLOCK_LOG_DIMX) + threadIdx.x;

  if((col < width) && (row < height)) 
  {
    int index = IMUL(row, width) + col;

    COMPUTE_HESSIAN(texC, index)

    // compute determinant of hessian matrix, normalize and write out
    hessian[index] = (Lxx*Lyy - Lxy*Lxy)*norm;
  }
}

void ProgramCU::ComputeHessian(CuTexImage* gus, CuTexImage* dog, CuTexImage* got, float norm)
{
  int width = gus->GetImgWidth();
  int height = gus->GetImgHeight();

  dim3 grid((width + DOG_BLOCK_DIMX - 1) / DOG_BLOCK_DIMX, (height + DOG_BLOCK_DIMY - 1) / DOG_BLOCK_DIMY);
  dim3 block(DOG_BLOCK_DIMX, DOG_BLOCK_DIMY);

  gus->BindTexture(texC);

  if(got->_cuData)
    ComputeHessian_Kernel<<<grid, block>>>((float*)dog->_cuData, (float2*)got->_cuData, width, height, norm*norm);
  else
    ComputeHessian_Kernel<<<grid, block>>>((float*)dog->_cuData, width, height, norm*norm);
}

#else

void __global__ ComputeDOG_Kernel(float* d_dog, float2* d_got, int width, int height)
{
  int row = (blockIdx.y << DOG_BLOCK_LOG_DIMY) + threadIdx.y;
  int col = (blockIdx.x << DOG_BLOCK_LOG_DIMX) + threadIdx.x;

  if(col < width && row < height) 
  {
    int index = IMUL(row, width) + col;
    float vp = tex1Dfetch(texP, index);
    float v = tex1Dfetch(texC, index);

    d_dog[index] = v - vp;

    float vxn = tex1Dfetch(texC, index + 1);
    float vxp = tex1Dfetch(texC, index - 1);
    float vyp = tex1Dfetch(texC, index - width);
    float vyn = tex1Dfetch(texC, index + width);
    float dx = vxn - vxp;
    float dy = vyn - vyp;
    float grd = 0.5f * sqrt(dx * dx  + dy * dy);
    float rot = (grd == 0.0f ? 0.0f : atan2(dy, dx));

    d_got[index] = make_float2(grd, rot);
  }
}

void __global__ ComputeDOG_Kernel(float* d_dog, int width, int height)
{
  int row = (blockIdx.y << DOG_BLOCK_LOG_DIMY) + threadIdx.y;
  int col = (blockIdx.x << DOG_BLOCK_LOG_DIMX) + threadIdx.x;

  if(col < width && row < height) 
  {
    int index = IMUL(row, width) + col;
    float vp = tex1Dfetch(texP, index);
    float v = tex1Dfetch(texC, index);
    d_dog[index] = v - vp;
  }
}

void ProgramCU::ComputeDOG(CuTexImage* gus, CuTexImage* dog, CuTexImage* got)
{
  int width = gus->GetImgWidth();
  int height = gus->GetImgHeight();
  dim3 grid((width + DOG_BLOCK_DIMX - 1)/ DOG_BLOCK_DIMX,  (height + DOG_BLOCK_DIMY - 1)/DOG_BLOCK_DIMY);
  dim3 block(DOG_BLOCK_DIMX, DOG_BLOCK_DIMY);

  gus->BindTexture(texC);
  (gus -1)->BindTexture(texP);

  if(got->_cuData)
    ComputeDOG_Kernel<<<grid, block>>>((float*) dog->_cuData, (float2*) got->_cuData, width, height);
  else
    ComputeDOG_Kernel<<<grid, block>>>((float*) dog->_cuData, width, height);
}

#endif // GPU_HESSIAN

#ifdef GPU_HESSIAN
  // GPU_HESSIAN: added test (response<0) and (response>0)
  #define READ_CMP_DOG_DATA(datai, tex, idx)  \
    datai[0] = tex1Dfetch(tex, idx - 1);      \
    datai[1] = tex1Dfetch(tex, idx);          \
    datai[2] = tex1Dfetch(tex, idx + 1);      \
    if(response > nmax)                       \
    {                                         \
      nmax = max(nmax, datai[0]);             \
      nmax = max(nmax, datai[1]);             \
      nmax = max(nmax, datai[2]);             \
      if((response < nmax) || (response < 0)) \
        goto key_finish;                      \
    }                                         \
    else                                      \
    {                                         \
      nmin = min(nmin, datai[0]);             \
      nmin = min(nmin, datai[1]);             \
      nmin = min(nmin, datai[2]);             \
      if((response > nmin) || (response > 0)) \
        goto key_finish;                      \
    }
#else
  #define READ_CMP_DOG_DATA(datai, tex, idx)  \
    datai[0] = tex1Dfetch(tex, idx - 1);      \
    datai[1] = tex1Dfetch(tex, idx);          \
    datai[2] = tex1Dfetch(tex, idx + 1);      \
    if(response > nmax)                       \
    {                                         \
      nmax = max(nmax, datai[0]);             \
      nmax = max(nmax, datai[1]);             \
      nmax = max(nmax, datai[2]);             \
      if(response < nmax)                     \
        goto key_finish;                      \
    }                                         \
    else                                      \
    {                                         \
      nmin = min(nmin, datai[0]);             \
      nmin = min(nmin, datai[1]);             \
      nmin = min(nmin, datai[2]);             \
      if(response > nmin)                     \
        goto key_finish;                      \
    }
#endif // GPU_HESSIAN

void __global__ ComputeKEY_Kernel(float4 *d_key, int width, int colmax, int rowmax, float dog_threshold0, float dog_threshold, float edge_threshold, int subpixel_localization
#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)
		  , int *featureTexLen
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)
	)
{
  int row = (blockIdx.y << KEY_BLOCK_LOG_DIMY) + threadIdx.y;
  int col = (blockIdx.x << KEY_BLOCK_LOG_DIMX) + threadIdx.x;

  float data[3][3];
  float datap[3][3];
  float datan[3][3];
  float response = 0.0f;
  int index = IMUL(row, width) + col;
  int idx[3] = {index - width, index, index + width};
  float nmax, nmin, result = 0.0f;
  float dx = 0, dy = 0, ds = 0;
  int in_image = 0;
  bool offset_test_passed = true;
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  unsigned short pointType = FEATURE_TYPE_NONE;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  if((row > 0) && (col > 0) && (row < rowmax) && (col < colmax))
  {
    in_image = 1;

    data[1][1] = response = tex1Dfetch(texC, idx[1]);
    if(fabs(response) <= dog_threshold0)
      goto key_finish;

    // fetch left and right neighbour
    data[1][0] = tex1Dfetch(texC, idx[1] - 1);
    data[1][2] = tex1Dfetch(texC, idx[1] + 1);
    nmax = max(data[1][0], data[1][2]);
    nmin = min(data[1][0], data[1][2]);

    if((response <= nmax) && (response >= nmin))
      goto key_finish;

    //if((response > nmax && response < 0 )|| (response < nmin && response > 0)) goto key_finish;
	// fetch values from the row above
    READ_CMP_DOG_DATA(data[0], texC, idx[0]);
	// fetch values from one the row below
    READ_CMP_DOG_DATA(data[2], texC, idx[2]);

    // edge supression
    float vx2 = response * 2.0f;
    float fxx = data[1][0] + data[1][2] - vx2;
    float fyy = data[0][1] + data[2][1] - vx2;
    float fxy = 0.25f * (data[2][2] + data[0][0] - data[2][0] - data[0][2]);
    float temp1 = fxx * fyy - fxy * fxy;
    float temp2 = (fxx + fyy) * (fxx + fyy);

    if((temp1 <= 0) || (temp2 > edge_threshold * temp1))
      goto key_finish; // local neighbourhood looks like an edge

    // read the previous level
    READ_CMP_DOG_DATA(datap[0], texP, idx[0]);
    READ_CMP_DOG_DATA(datap[1], texP, idx[1]);
    READ_CMP_DOG_DATA(datap[2], texP, idx[2]);

    // read the next level
    READ_CMP_DOG_DATA(datan[0], texN, idx[0]);
    READ_CMP_DOG_DATA(datan[1], texN, idx[1]);
    READ_CMP_DOG_DATA(datan[2], texN, idx[2]);

    if(subpixel_localization)
    {
      // subpixel localization
      float fx = 0.5f * (data[1][2] - data[1][0]);
      float fy = 0.5f * (data[2][1] - data[0][1]);
      float fs = 0.5f * (datan[1][1] - datap[1][1]);

      float fss = (datan[1][1] + datap[1][1] - vx2);
      float fxs = 0.25f * (datan[1][2] + datap[1][0] - datan[1][0] - datap[1][2]);
      float fys = 0.25f * (datan[2][1] + datap[0][1] - datan[0][1] - datap[2][1]);

      // need to solve dx, dy, ds;
      // |-fx|     | fxx fxy fxs |   |dx|
      // |-fy|  =  | fxy fyy fys | * |dy|
      // |-fs|     | fxs fys fss |   |ds|
      float4 A0 = (fxx > 0) ? make_float4(fxx, fxy, fxs, -fx) : make_float4(-fxx, -fxy, -fxs, fx);
      float4 A1 = (fxy > 0) ? make_float4(fxy, fyy, fys, -fy) : make_float4(-fxy, -fyy, -fys, fy);
      float4 A2 = (fxs > 0) ? make_float4(fxs, fys, fss, -fs) : make_float4(-fxs, -fys, -fss, fs);

      float maxa = max(max(A0.x, A1.x), A2.x);

      if(maxa >= 1e-10)
      {
        if(maxa == A1.x)
        {
          float4 TEMP = A1; A1 = A0; A0 = TEMP;
        }
        else if(maxa == A2.x)
        {
          float4 TEMP = A2; A2 = A0; A0 = TEMP;
        }
        A0.y /= A0.x;          A0.z /= A0.x;          A0.w /= A0.x;
        A1.y -= A1.x * A0.y;   A1.z -= A1.x * A0.z;   A1.w -= A1.x * A0.w;
        A2.y -= A2.x * A0.y;   A2.z -= A2.x * A0.z;   A2.w -= A2.x * A0.w;

        if(abs(A2.y) > abs(A1.y))
        {
          float4 TEMP = A2; A2 = A1; A1 = TEMP;
        }
        if(abs(A1.y) >= 1e-10) 
        {
          A1.z /= A1.y;          A1.w /= A1.y;
          A2.z -= A2.y * A1.z;   A2.w -= A2.y * A1.w;

          if(abs(A2.z) >= 1e-10) 
          {
            ds = A2.w / A2.z;
            dy = A1.w - ds * A1.z;
            dx = A0.w - ds * A0.z - dy * A0.y;

            response = data[1][1] + 0.5f * (dx * fx + dy * fy + ds * fs);

            offset_test_passed = (fabs(response) > dog_threshold) && (fabs(ds) < 1.0f) && (fabs(dx) < 1.0f) && (fabs(dy) < 1.0f);
          }
        }
      }
    }
 
    if(offset_test_passed)
#if defined GPU_HESSIAN
    {
      // find blob point type from Hessian matrix H, we know that:
      //   - if H is positive definite it is a DARK blob
      //   - if H is negative definite it is a BRIGHT blob
      //   - det H is negative it is a SADDLE point

      data[1][1] = tex1Dfetch(texG, idx[1]);
      data[1][0] = tex1Dfetch(texG, idx[1] - 1);
      data[1][2] = tex1Dfetch(texG, idx[1] + 1);

      if(response < 0)
      {
        pointType = FEATURE_TYPE_SADDLE_POINT;
      }
      else
      {
        // at this point we know that 2x2 determinant is positive
        // so only check the remaining 1x1 subdeterminant
        float Lxx = data[1][0] - 2*data[1][1] + data[1][2];

        pointType = (Lxx > 0) ? FEATURE_TYPE_DARK_BLOB : FEATURE_TYPE_BRIGHT_BLOB;
      }
    }
#elif defined GPU_SIFT_MODIFIED
      result = (response > nmax) ? FEATURE_TYPE_BRIGHT_BLOB : FEATURE_TYPE_DARK_BLOB;
#else
      result = (response > nmax) ? 1.0 : -1.0;
#endif // GPU_HESSIAN / GPU_SIFT_MODIFIED
  }

key_finish:

  if(in_image)
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  {
    // result: response 16b | 14b unused | 2b type
    unsigned int uspack = (((unsigned int)__float2half_rn(response)) << 16) | 0x00000004u | pointType;
	result = *((float *)(&uspack)); // __uint_as_float(uspack); // CUDA 7.5

    d_key[index] = make_float4(result, dx, dy, ds);
  }
#else
    d_key[index] = make_float4(result, dx, dy, ds);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)
  int count = __syncthreads_count(pointType != FEATURE_TYPE_NONE);

  if(threadIdx.x+threadIdx.y*blockDim.x == 0)
  {
	atomicAdd(featureTexLen, count);
  }
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)
}

void ProgramCU::ComputeKEY(CuTexImage* dog, CuTexImage* key
#if defined GPU_HESSIAN
    , CuTexImage* gus
#endif // GPU_HESSIAN
    , float Tdog, float Tedge
#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)
    , int *featureTexLen, int featureTexIdx
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)
)
{
  int width = dog->GetImgWidth();
  int height = dog->GetImgHeight();

  float Tdog1 = (GlobalUtil::_SubpixelLocalization ? 0.8f : 1.0f) * Tdog;

  CuTexImage *dogp = dog - 1;
  CuTexImage *dogn = dog + 1;

  dim3 grid((width + KEY_BLOCK_DIMX - 1)/KEY_BLOCK_DIMX, (height + KEY_BLOCK_DIMY - 1)/KEY_BLOCK_DIMY);
  dim3 block(KEY_BLOCK_DIMX, KEY_BLOCK_DIMY);

  dogp->BindTexture(texP);
  dog->BindTexture(texC);
  dogn->BindTexture(texN);

#if defined GPU_HESSIAN
  gus->BindTexture(texG);
#endif // GPU_HESSIAN

  Tedge = (Tedge+1)*(Tedge+1) / Tedge;

  ComputeKEY_Kernel<<<grid, block>>>((float4*)key->_cuData, width, width-1, height-1, Tdog1, Tdog, Tedge, GlobalUtil::_SubpixelLocalization
#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)
		  , featureTexLen + featureTexIdx
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)
	  );
}

#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)

#define GENERATE_LIST_BLOCK_DIMX 32
#define GENERATE_LIST_BLOCK_DIMY 4

__device__ int GFL_WarpScan(int val, volatile int *sData, int threadID)
{
  // pad each warp with zeros
  // int idx = 2*threadIdx.x - (threadIdx.x & (warpSize-1)); // 1D
  // int id = threadIdx.x + threadIdx.y*blockDim.x; // 2D
  // int idx = 2*id - (id & (warpSize-1));
  int idx = 2*threadID - (threadID & (warpSize-1));

  sData[idx] = 0;
  idx += warpSize;

  int t = sData[idx] = val;

  sData[idx] = t = t + sData[idx -  1];
  sData[idx] = t = t + sData[idx -  2];
  sData[idx] = t = t + sData[idx -  4];
  sData[idx] = t = t + sData[idx -  8];
  sData[idx] = t = t + sData[idx - 16];

  return sData[idx-1];
}

__device__ unsigned int GFL_LaneMaskLt(int threadID)
{
  // const unsigned int lane = threadIdx.x & (warpSize-1);                            // 1D block
  // const unsigned int lane = (threadIdx.x + threadIdx.y*blockDim.x) & (warpSize-1); // 2D block
  const unsigned int lane = threadID & (warpSize-1);

  return (1 << (lane)) - 1;
}

__device__ unsigned int GFL_WarpPrefixSums(bool p, int threadID)
{
  const unsigned int mask = GFL_LaneMaskLt(threadID);
  unsigned int b = __ballot(p);

  return __popc(b & mask);
}

__device__ int GFL_BlockBinaryPrefixSums(int x, int idx)
{
  extern __shared__ int sData[];

  //int idx = threadIdx.x + threadIdx.y*blockDim.x;

  // A. Compute exclusive prefix sums within each warp
  int warpPrefix = GFL_WarpPrefixSums(x, idx);

  // int idx = threadIdx.x; // 1D
  // int idx = threadIdx.x + threadIdx.y*blockDim.x; // 2D
  int warpIdx = idx / warpSize;
  int laneIdx = idx & (warpSize - 1);

  // B. The last thread of each warp stores inclusive
  // prefix sum to the warp’s index in shared memory
  if(laneIdx == warpSize - 1)
    sData[warpIdx] = warpPrefix + x;

  __syncthreads();

  // C. One warp scans the warp partial sums
  if(idx < warpSize)
    sData[idx] = GFL_WarpScan(sData[idx], sData, idx);

  __syncthreads();

  // D. Each thread adds prefix sums of warp partial
  // sums to its own intra-warp prefix sums
  return warpPrefix + sData[warpIdx];
}

void __global__ ListGen_Kernel(int4* d_list, int len, int width, int height, int *counter)
{
  int row = IMUL(blockIdx.y, GENERATE_LIST_BLOCK_DIMY) + threadIdx.y;
  int col = IMUL(blockIdx.x, GENERATE_LIST_BLOCK_DIMX) + threadIdx.x;

  // read the detected keypoint type -> flag=0 (type == FEATURE_TYPE_NONE) / 1 (otherwise)
  unsigned int flag = 0;

  if((row > 0) && (col > 0) && (row < height-1) && (col < width-1))
  {
    int index = IMUL(row, width) + col;
    float4 value = tex1Dfetch(texDataF4, index);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    // value: (response 16b | 14b unused | 2b type), dx, dy, ds
    // type = *((unsigned int *)(&offset.x)) & 0x00000003u;
	flag = ((*((unsigned int *)(&value.x)) & 0x00000003u) != FEATURE_TYPE_NONE) ? 1 : 0;
#else
    // value: (response 16b | 16b type), dx, dy, ds ... type = -1, 0, +1
    unsigned int resultUInt = *((unsigned int*)(&value.x)) & 0x0000FFFFu; // float_as_uint(value.x) & 0x0000FFFFu; // CUDA 7.5
    float result = __half2float(resultUInt);

    flag = (fabs(result) > 0.5f) ? 1 : 0; 	// flag = (result != 0.0f) ? 1 : 0;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  }

  int idxWithinBlock = IMUL(threadIdx.y, blockDim.x) + threadIdx.x;

  // compute prefix sums for each thread int the block 
  int blockPrefixSum = GFL_BlockBinaryPrefixSums(flag, idxWithinBlock);

  // allocate enough space in the feature list to store all features in this block
  // number of features can be obtained by __syncthreads_count() or it is given
  // by the prefix sum of the last thread in the block plus one (if the last thread
  // represents detected feature)
  __shared__ int blockStart; // index in the feature list of the first keypoint in the block

  // int count = __syncthreads_count(flag);
  // if(idxWitninBlock == 0)
  // {
  //   blockStart = atomicAdd(counter, count);
  // }
  // __syncthreads();

  if(idxWithinBlock == IMUL(blockDim.y, blockDim.x) - 1)
  {
    blockStart = atomicAdd(counter, blockPrefixSum + flag);
  }
  __syncthreads();

  // put detected keypoint into the feature list
  if(flag)
    d_list[blockStart + blockPrefixSum] = make_int4(col, row, 0, 0);
}

#else

void __global__ InitHist_Kernel(int4* hist, int ws, int wd, int height)
{
  int row = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
  int col = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;

  if((row < height) && (col < wd))
  {
    int hidx = IMUL(row, wd) + col;
    int scol = col << 2;
    int sidx = IMUL(row, ws) + scol;
    int v[4] = {0, 0, 0, 0};

    // each thread process 4 subsequent colums values in the same row
    if((row > 0) && (row < height-1))
    {
#pragma unroll
      for(int i = 0; i < 4 ; ++i, ++scol)
      {
        float4 temp = tex1Dfetch(texDataF4, sidx+i);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
        unsigned int featureType = *((unsigned int*)(&temp.x)) & 0x00000003u; // float_as_uint(temp.x) & 0x00000003u; // CUDA 7.5

        v[i] = ((scol < ws-1) && (scol > 0) && (featureType != FEATURE_TYPE_NONE)) ? 1 : 0;
#else
        v[i] = ((scol < ws-1) && (scol > 0) && (temp.x != 0.0)) ? 1 : 0;
#endif // GPU_SIFT_MODIFIED || GPU_HESSIAN
      }
    }
    hist[hidx] = make_int4(v[0], v[1], v[2], v[3]);
  }
}

void ProgramCU::InitHistogram(CuTexImage* key, CuTexImage* hist)
{
  int ws = key->GetImgWidth();
  int hs = key->GetImgHeight();
  int wd = hist->GetImgWidth();
  int hd = hist->GetImgHeight();
  dim3 grid((wd + HIST_INIT_WIDTH - 1) / HIST_INIT_WIDTH, hd);
  dim3 block(HIST_INIT_WIDTH, 1);

  key->BindTexture(texDataF4);

  InitHist_Kernel<<<grid, block>>>((int4*) hist->_cuData, ws, wd, hd);
}

void __global__ ReduceHist_Kernel(int4* d_hist, int ws, int wd, int height)
{
  int row = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
  int col = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;

  if((row < height) && (col < wd))
  {
    int hidx = IMUL(row, wd) + col;
    int scol = col << 2;
    int sidx = IMUL(row, ws) + scol;
    int v[4] = {0, 0, 0, 0};

#pragma unroll
    for(int i = 0; (i < 4) && (scol < ws); ++i, ++scol)
    {
      int4 temp = tex1Dfetch(texDataI4, sidx + i);
      v[i] = temp.x + temp.y + temp.z + temp.w;
    }

    d_hist[hidx] = make_int4(v[0], v[1], v[2], v[3]);
  }
}

void ProgramCU::ReduceHistogram(CuTexImage* hist1, CuTexImage* hist2)
{
  int ws = hist1->GetImgWidth();
  int hs = hist1->GetImgHeight();
  int wd = hist2->GetImgWidth();
  int hd = hist2->GetImgHeight();
  int temp = (int)floor(logf(float(wd * 2 / 3)) / logf(2.0f));
  const int wi = min(7, max(temp , 0));

  hist1->BindTexture(texDataI4);

  const int BW = 1 << wi;
  const int BH = 1 << (7 - wi);
  dim3 grid((wd  + BW - 1) / BW,  (hd + BH -1) / BH);
  dim3 block(BW, BH);

  ReduceHist_Kernel<<<grid, block>>>((int4*)hist2->_cuData, ws, wd, hd);
}

void __global__ ListGen_Kernel(int4* d_list, int len, int width)
{
  int idx1 = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  // GPU_HESSIAN fix
  if(idx1 >= len)
    return;
#endif // GPU_HESSIAN

  int4 pos = tex1Dfetch(texDataList, idx1);

  int idx2 = IMUL(pos.y, width) + pos.x;
  int4 temp = tex1Dfetch(texDataI4, idx2);

  int  sum1 = temp.x + temp.y;
  int  sum2 = sum1 + temp.z;

  pos.x <<= 2;
  if(pos.z >= sum2)
  {
    pos.x += 3;
    pos.z -= sum2;
  }
  else if(pos.z >= sum1)
  {
    pos.x += 2;
    pos.z -= sum1;
  }
  else if(pos.z >= temp.x)
  {
    pos.x += 1;
    pos.z -= temp.x;
  }

  d_list[idx1] = pos;
}

#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)

#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)

// input list (x, y) (x, y) ....
void ProgramCU::GenerateList(CuTexImage* list, CuTexImage* key, int *counter)
{
  int len = list->GetImgWidth();

  key->BindTexture(texDataF4);

  int width = key->GetImgWidth();
  int height = key->GetImgHeight();

  dim3 grid((width + GENERATE_LIST_BLOCK_DIMX - 1)/GENERATE_LIST_BLOCK_DIMX, (height + GENERATE_LIST_BLOCK_DIMY - 1)/GENERATE_LIST_BLOCK_DIMY);
  dim3 block(GENERATE_LIST_BLOCK_DIMX, GENERATE_LIST_BLOCK_DIMY);

  // shared memory is used to store warp scan data
  ListGen_Kernel<<<grid, block, 2*32*sizeof(int)>>>((int4*)list->_cuData, len, width, height, counter);
}

#else

// input list (x, y) (x, y) ....
void ProgramCU::GenerateList(CuTexImage* list, CuTexImage* hist)
{
  int len = list->GetImgWidth();

  list->BindTexture(texDataList);
  hist->BindTexture(texDataI4);

  dim3  grid((len + LISTGEN_BLOCK_DIM -1) / LISTGEN_BLOCK_DIM);
  dim3  block(LISTGEN_BLOCK_DIM);

  ListGen_Kernel<<<grid, block>>>((int4*)list->_cuData, len, hist->GetImgWidth());
}

#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)

void __global__ ComputeOrientation_Kernel(float4* d_list, int list_len, int width, int height, float sigma, float sigma_step, float gaussian_factor, float sample_factor,
										  int num_orientation, int existing_keypoint, int subpixel, int keepsign
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
                                          , bool doHalfSIFT
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
                                          )
{
  const float ten_degree_per_radius = 5.7295779513082320876798154814105;
  const float radius_per_ten_degrees = 1.0 / 5.7295779513082320876798154814105;
  int idx = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

  if(idx >= list_len)
    return;

  float4 key;

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  union {
    float        flt; // include response (16b), unused (14b), feature type (2b)
    unsigned int uint;
  } additionalData;

  additionalData.flt = 0.0f;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  if(existing_keypoint)
  {
    key = tex1Dfetch(texDataF4, idx);
	// read the data
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
 	// unpack scale, x, and y
	// later store just the strongest computed orientation

    // existing keypoint input
    //  key.x: response 8b H (cleared to zero) | x 24b-14.10
    //  key.y: response 8b L (cleared to zero) | y 24b-14.10
    //  key.z: 2b type | 14b unused | scale 16b-8.8 
    //  key.w: orientation (if any)

    // extract x position
    unsigned int tmpValue = *((unsigned int *)(&key.x)); // __float_as_uint(key.x); // CUDA 7.5

    tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
    key.x = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

    // extract y position
    tmpValue = *((unsigned int *)(&key.y)); // __float_as_uint(key.y); // CUDA 7.5

    tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
    key.y = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

    // extract scale
    tmpValue = *((unsigned int *)(&key.z)); // __float_as_uint(key.z); // CUDA 7.5

    tmpValue = tmpValue & FIXED_POINT_SCALE_MASK;
    key.z = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_SCALE_PRECISION_BITS);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  }
  else
  {
    int4 ikey = tex1Dfetch(texDataList, idx);
    key.x = ikey.x + 0.5f;
    key.y = ikey.y + 0.5f;
    key.z = sigma;

#if !defined GPU_HESSIAN && !defined GPU_SIFT_MODIFIED
    if(subpixel || keepsign)
    {
#endif // !GPU_HESSIAN && !GPU_SIFT_MODIFIED
      // offset: x(response 16b | 14b unused | 2b type), dx, dy, ds
      float4 offset = tex1Dfetch(texDataF4, IMUL(width, ikey.y) + ikey.x);

      if(subpixel)
      {
        key.x += offset.y;
        key.y += offset.z;
        key.z *= pow(sigma_step, offset.w);
      }
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
      additionalData.flt = offset.x;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

#if !defined GPU_HESSIAN && !defined GPU_SIFT_MODIFIED
      if(keepsign) // not supported for hessian
        key.z *= offset.x;
#endif // !GPU_HESSIAN && !GPU_SIFT_MODIFIED
#if !defined GPU_HESSIAN && !defined GPU_SIFT_MODIFIED
    }
#endif // !GPU_HESSIAN && !GPU_SIFT_MODIFIED
  }

  if(num_orientation == 0)
  {
    key.w = 0;

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    goto key_store_finish;
#else
    d_list[idx] = key;
    return;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  }

  float vote[37];
  float gsigma = key.z * gaussian_factor;
  float win = fabs(key.z) * sample_factor;
  float dist_threshold = win * win + 0.5;
  float factor = -0.5f / (gsigma * gsigma);
  float xmin = max(1.5f, floor(key.x - win) + 0.5f);
  float ymin = max(1.5f, floor(key.y - win) + 0.5f);
  float xmax = min(width - 1.5f, floor(key.x + win) + 0.5f);
  float ymax = min(height -1.5f, floor(key.y + win) + 0.5f);

#pragma unroll
  for(int i = 0; i < 36; ++i)
    vote[i] = 0.0f;

  for(float y = ymin; y <= ymax; y += 1.0f)
  {
    float dy = y - key.y;
    dy *= dy;

    for(float x = xmin; x <= xmax; x += 1.0f)
    {
      float dx = x - key.x;
      float sq_dist  = dx * dx + dy;

      if(sq_dist >= dist_threshold)
        continue;

      float2 got = tex2D(texDataF2, x, y);
      // float weight = got.x * exp(sq_dist * factor);
      // float fidx = floorf(got.y * ten_degree_per_radius);
      // int oidx = fidx;
      int oidx = (int)floorf(got.y * ten_degree_per_radius);

      if(oidx < 0)
        oidx += 36;
      vote[oidx] += got.x * expf(sq_dist * factor); // vote[oidx] += weight;
    }
  }

  // filter the vote
  const float one_third = 1.0 / 3.0;

#pragma unroll
  for(int i = 0; i < 6; ++i)
  {
    vote[36] = vote[0];
    float pre = vote[35];

#pragma unroll
    for(int j = 0; j < 36; ++j)
    {
      float temp = one_third * (pre + vote[j] + vote[j + 1]);
      pre = vote[j];
      vote[j] = temp;
    }
  }

  vote[36] = vote[0];

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  if(doHalfSIFT)
  {
#pragma unroll
    for(int i = 0; i < 18; i++)
    {
      vote[i] += vote[i+18];
      vote[i+18] = 0;
    }
  }

  int orientationsCount = 0;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  // just one orientation
  if((num_orientation == 1) || existing_keypoint)
  {
    int index_max = 0;
    float max_vote = vote[0];

#pragma unroll
    for(int i = 1; i < 36; ++i)
    {
      index_max = (vote[i] > max_vote) ? i : index_max;
      max_vote = max(max_vote, vote[i]);
    }

    float pre = vote[(index_max == 0) ? 35 : index_max - 1];
    float next = vote[index_max + 1];
    float weight = max_vote;
    float off =  0.5f * FDIV(next - pre, weight + weight - next - pre);

    key.w = radius_per_ten_degrees * (index_max + 0.5f + off);

#if !defined GPU_HESSIAN && !defined GPU_SIFT_MODIFIED
    d_list[idx] = key;	
#endif // !GPU_HESSIAN && !GPU_SIFT_MODIFIED
  }
  // multi-orientations allowed
  else
  {
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED

// up to 4 orientations may be stored
// max number of stored orientations depends on num_orientations parameter (1..4)
#define MAX_ORIENTATIONS 4

    // find the maximum value
    float max_vote = vote[0];

#pragma unroll
    for(int i = 1; i < 36; ++i)
      max_vote = max(max_vote, vote[i]);

    float vote_threshold = max_vote * 0.8f;
    float pre = vote[35];

    float max_vot[MAX_ORIENTATIONS+1];
    float max_rot[MAX_ORIENTATIONS+1];

#pragma unroll
    for(int i=0; i < 36; ++i)
    {
      float next = vote[i + 1];

      if((vote[i] > vote_threshold) && (vote[i] > pre) && (vote[i] > next)) // max from neighbours
      {
        float di = 0.5f * FDIV(next - pre, vote[i] + vote[i] - next - pre);
        float rot =  i + di + 0.5f;
        float weight = vote[i];

        int idx = orientationsCount;
        if(orientationsCount > 0)
        {
          // shift values
          while((idx > 0) && (max_vot[idx-1] < weight)) {
            max_vot[idx] = max_vot[idx-1];
            max_rot[idx] = max_rot[idx-1];
            idx--;
          }
        }
        // store maximum found
        max_vot[idx] = weight;
        max_rot[idx] = rot;

        if(orientationsCount < MAX_ORIENTATIONS)
          orientationsCount++;
      }
      pre = vote[i];
    }

    unsigned int packedOrientations = 0;

    // first 4 orientations (if exist)
    unsigned int maxCount = min(4, orientationsCount);
	int idx = 0;
    for(; idx < maxCount; idx++)
    {
      float orientation = max_rot[idx] / 36.0f; 
      if(orientation < 0)
        orientation += 1.0f;

      unsigned int uiOrientation = (unsigned int) floorf(orientation * 255.0f);
      packedOrientations = packedOrientations | (uiOrientation << 8*idx);
    }

    key.w = *((float *)(&packedOrientations)); // __uint_as_float(packedOrientations); // CUDA 7.5

#else
    float max_vote = vote[0];

#pragma unroll
    for(int i = 1; i < 36; ++i)
      max_vote = max(max_vote, vote[i]);

    float vote_threshold = max_vote * 0.8f;
    float pre = vote[35];
    float max_rot[2], max_vot[2] = {0, 0};
    int ocount = 0;

#pragma unroll
    for(int i=0; i < 36; ++i)
    {
      float next = vote[i + 1];

      if((vote[i] > vote_threshold) && (vote[i] > pre) && (vote[i] > next))
      {
        float di = 0.5f * FDIV(next - pre, vote[i] + vote[i] - next - pre);
        float rot = i + di + 0.5f;
        float weight = vote[i];
        ///
        if(weight > max_vot[1])
        {
          if(weight > max_vot[0])
          {
            max_vot[1] = max_vot[0];
            max_rot[1] = max_rot[0];
            max_vot[0] = weight;
            max_rot[0] = rot;
          }
          else
          {
            max_vot[1] = weight;
            max_rot[1] = rot;
          }
          ocount ++;
        }
      }
      pre = vote[i];
    }

    float fr1 = max_rot[0] / 36.0f; 
    if(fr1 < 0)
      fr1 += 1.0f;

    unsigned short us1 = (ocount == 0) ? 65535 : ((unsigned short)floor(fr1 * 65535.0f));
    unsigned short us2 = 65535;

    if(ocount > 1)
    {
      float fr2 = max_rot[1] / 36.0f;

      if(fr2 < 0)
        fr2 += 1.0f;

      us2 = (unsigned short) floor(fr2 * 65535.0f);
    }

    unsigned int uspack = (us2 << 16) | us1;
    key.w = *((float *)(&uspack)); // __uint_as_float(uspack); // CUDA 7.5

#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED


#if !defined GPU_HESSIAN && !defined GPU_SIFT_MODIFIED
    d_list[idx] = key;
#endif // !GPU_HESSIAN && !GPU_SIFT_MODIFIED
  }

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
key_store_finish:

  // input:
  //  additionalData: response 16b | 14b unused | 2b type 

  // output in the feature list d_list
  //  key.x: response 8b H | x 24b-14.10
  //  key.y: response 8b L | y 24b-14.10
  //  key.z: 2b type | 3b orientations count | 11b unused | scale 16b-8.8
  //  key.w: 8b orientation1 | 8b orientation2 | 8b orientation3 | 8b orientation4

  if(!existing_keypoint)
  {
    unsigned int posX = (unsigned int)FLOAT_TO_FIXED_POINT(key.x, FIXED_POINT_POSITION_PRECISION_BITS);
    posX = posX & FIXED_POINT_POSITION_MASK;
    unsigned int posY = (unsigned int)FLOAT_TO_FIXED_POINT(key.y, FIXED_POINT_POSITION_PRECISION_BITS);
    posY = posY & FIXED_POINT_POSITION_MASK;

    // store response
    posX = posX | (additionalData.uint & FIXED_POINT_RESPONSE_MASK);
    posY = posY | ((additionalData.uint << 8) & FIXED_POINT_RESPONSE_MASK);

    unsigned int scale = (unsigned int)(FLOAT_TO_FIXED_POINT(key.z, FIXED_POINT_SCALE_PRECISION_BITS));
    scale = scale & FIXED_POINT_SCALE_MASK;

    // type & orientations count (0 means single float value in key.w, otherwise we have to unpack 8b orientations into floats) 
    scale = scale | ((additionalData.uint & 0x00000003u) << 30) | ((orientationsCount & 0x00000007u) << 27);

    key.z = *((float *)(&scale)); // __uint_as_float(scale); // CUDA 7.5
    key.x = *((float *)(&posX));  // __uint_as_float(posX);  // CUDA 7.5
    key.y = *((float *)(&posY));  // __uint_as_float(posY);  // CUDA 7.5
	
	d_list[idx] = key;
  }
  else
  {
    // for existing keypoint just overwrite computed orientation and other components remain the same (x, y, and scale)
    // ReshapeFeatureListCPU() is not called when we have just one orientation
    d_list[idx].w = key.w;
  }

#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
}

void ProgramCU::ComputeOrientation(CuTexImage* list, CuTexImage* got, CuTexImage* key, float sigma, float sigma_step, int existing_keypoint)
{
  int len = list->GetImgWidth();
  if(len <= 0)
    return;

  int width = got->GetImgWidth();
  int height = got->GetImgHeight();

  if(existing_keypoint)
  {
    list->BindTexture(texDataF4);
  }
  else
  {
    list->BindTexture(texDataList);
#if !defined GPU_HESSIAN && !defined GPU_SIFT_MODIFIED
    if(GlobalUtil::_SubpixelLocalization)
#endif // !GPU_HESSIAN && !GPU_SIFT_MODIFIED
      key->BindTexture(texDataF4);
  }
  got->BindTexture2D(texDataF2);

  const int block_width = (len < ORIENTATION_COMPUTE_PER_BLOCK) ? 16 : ORIENTATION_COMPUTE_PER_BLOCK;
  dim3 grid((len + block_width -1) / block_width);
  dim3 block(block_width);

  ComputeOrientation_Kernel<<<grid, block>>>(
      (float4*) list->_cuData,
      len, width, height, sigma, sigma_step,
      GlobalUtil::_OrientationGaussianFactor,
      GlobalUtil::_OrientationGaussianFactor * GlobalUtil::_OrientationWindowFactor,
      GlobalUtil::_FixedOrientation ? 0 : GlobalUtil::_MaxOrientation,
      existing_keypoint, GlobalUtil::_SubpixelLocalization, GlobalUtil::_KeepExtremumSign
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
      , GlobalUtil::_HalfSIFT
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
      );

  ProgramCU::CheckErrorCUDA("ComputeOrientation");
}

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
template <bool DYNAMIC_INDEXING, bool HALF_SIFT>
#else
template <bool DYNAMIC_INDEXING>
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
void __global__ ComputeDescriptor_Kernel(float4* d_des, int num, int width, int height, float window_factor)
{
  const float rpi = 4.0 / PI;
  int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  int fidx = idx >> 4;

  if(fidx >= num)
    return;

  // fetch the feature
  float4 key = tex1Dfetch(texDataF4, fidx);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  // input in the feature list
  //  key.x: response 8b H | x 24b-14.10
  //  key.y: response 8b L | y 24b-14.10
  //  key.z: 2b type | 14b unused | scale 16b-8.8 
  //  key.w: orientation

  // extract x position
  unsigned int tmpValue = *((unsigned int *)(&key.x)); // __float_as_uint(key.x); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
  key.x = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

  // extract y position
  tmpValue = *((unsigned int *)(&key.y)); // __float_as_uint(key.y); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
  key.y = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

  // extract scale
  tmpValue = *((unsigned int *)(&key.z)); // __float_as_uint(key.z); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_SCALE_MASK;
  key.z = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_SCALE_PRECISION_BITS);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  int bidx = idx & 0xf;
  int ix = bidx & 0x3;
  int iy = bidx >> 2;
  float spt = fabs(key.z * window_factor);
  float s, c;
  
  __sincosf(key.w, &s, &c);

  float anglef = (key.w > PI) ? (key.w - (2.0 * PI)) : key.w;
  float cspt = c * spt;
  float sspt = s * spt;
  float crspt = c / spt;
  float srspt = s / spt;
  float2 offsetpt, pt;
  float xmin, ymin, xmax, ymax, bsz;

  offsetpt.x = ix - 1.5f;
  offsetpt.y = iy - 1.5f;
  pt.x = cspt * offsetpt.x - sspt * offsetpt.y + key.x;
  pt.y = cspt * offsetpt.y + sspt * offsetpt.x + key.y;
  bsz =  fabs(cspt) + fabs(sspt);
  xmin = max(1.5f, floor(pt.x - bsz) + 0.5f);
  ymin = max(1.5f, floor(pt.y - bsz) + 0.5f);
  xmax = min(width - 1.5f, floor(pt.x + bsz) + 0.5f);
  ymax = min(height - 1.5f, floor(pt.y + bsz) + 0.5f);

  float des[9];

#pragma unroll
  for(int i = 0; i < 9; ++i)
    des[i] = 0.0f;

  for(float y = ymin; y <= ymax; y += 1.0f)
  {
    for(float x = xmin; x <= xmax; x += 1.0f)
    {
      float dx = x - pt.x;
      float dy = y - pt.y;
      float nx = crspt * dx + srspt * dy;
      float ny = crspt * dy - srspt * dx;
      float nxn = fabs(nx);
      float nyn = fabs(ny);

      if((nxn < 1.0f) && (nyn < 1.0f))
      {
        float2 cc = tex2D(texDataF2, x, y);

        float dnx = nx + offsetpt.x;
        float dny = ny + offsetpt.y;
        float ww = expf(-0.125f * (dnx * dnx + dny * dny));
        float wx = 1.0 - nxn;
        float wy = 1.0 - nyn;
        float weight = ww * wx * wy * cc.x;
        float theta = (anglef - cc.y) * rpi;

        if(theta < 0)
          theta += 8.0f;

        float fo = floorf(theta);
        int fidx = fo;
        float weight1 = fo + 1.0f  - theta;
        float weight2 = theta - fo;

        if(DYNAMIC_INDEXING)
        {
          des[fidx] += (weight1 * weight);
          des[fidx + 1] += (weight2 * weight);
          // this dynamic indexing part might be slow
        }
        else
        {
#pragma unroll
          for(int k = 0; k < 8; ++k)
          {
            if(k == fidx) 
            {
              des[k] += (weight1 * weight);
              des[k+1] += (weight2 * weight);
            }
          }
        }
      }
    }
  }
  des[0] += des[8];

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  if(HALF_SIFT)
  {
    // half sift -> 4 directions only
    des[0] += des[4];
    des[1] += des[5];
    des[2] += des[6];
    des[3] += des[7];

    d_des[idx] = make_float4(des[0], des[1], des[2], des[3]);

    return;
  }
  else
  {
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  // full sift -> 8 directions
  int didx = idx << 1;

  d_des[didx] = make_float4(des[0], des[1], des[2], des[3]);
  d_des[didx+1] = make_float4(des[4], des[5], des[6], des[7]);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  }
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
}

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
template <bool DYNAMIC_INDEXING, bool HALF_SIFT>
#else
template <bool DYNAMIC_INDEXING>
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
void __global__ ComputeDescriptorRECT_Kernel(float4* d_des, int num, int width, int height, float window_factor)
{
  const float rpi = 4.0 / PI;
  int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  int fidx = idx >> 4;

  if(fidx >= num)
    return;

  // fetch the feature
  float4 key = tex1Dfetch(texDataF4, fidx);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  // input in the feature list
  //  key.x: response 8b H | x 24b-14.10
  //  key.y: response 8b L | y 24b-14.10
  //  key.z: 2b type | 14b unused | scale 16b-8.8 
  //  key.w: orientation1 orientation2

  // extract x position
  unsigned int tmpValue = *((unsigned int *)(&key.x)); // __float_as_uint(key.x); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
  key.x = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

  // extract y position
  tmpValue = *((unsigned int *)(&key.y)); // __float_as_uint(key.y); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
  key.y = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

  // extract scale
  tmpValue = *((unsigned int *)(&key.z)); // __float_as_uint(key.z); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_SCALE_MASK;
  key.z = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_SCALE_PRECISION_BITS);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  int bidx = idx & 0xf;
  int ix = bidx & 0x3;
  int iy = bidx >> 2;
  // float aspect_ratio = key.w / key.z;
  // float aspect_sq = aspect_ratio * aspect_ratio;
  float sptx = key.z * 0.25;
  float spty = key.w * 0.25;
  float xmin, ymin, xmax, ymax;
  float2 pt;

  pt.x = sptx * (ix + 0.5f) + key.x;
  pt.y = spty * (iy + 0.5f) + key.y;

  xmin = max(1.5f, floorf(pt.x - sptx) + 0.5f);
  ymin = max(1.5f, floorf(pt.y - spty) + 0.5f);
  xmax = min(width - 1.5f, floorf(pt.x + sptx) + 0.5f);
  ymax = min(height - 1.5f, floorf(pt.y + spty) + 0.5f);

  float des[9];

#pragma unroll
  for(int i =0; i < 9; ++i)
    des[i] = 0.0f;

  for(float y = ymin; y <= ymax; y += 1.0f)
  {
    for(float x = xmin; x <= xmax; x += 1.0f)
    {
      float nx = (x - pt.x) / sptx;
      float ny = (y - pt.y) / spty;
      float nxn = fabs(nx);
      float nyn = fabs(ny);

      if((nxn < 1.0f) && (nyn < 1.0f))
      {
        float2 cc = tex2D(texDataF2, x, y);

        float wx = 1.0 - nxn;
        float wy = 1.0 - nyn;
        float weight =  wx * wy * cc.x;
        float theta = (- cc.y) * rpi;

        if(theta < 0)
          theta += 8.0f;

        float fo = floorf(theta);
        int fidx = fo;
        float weight1 = fo + 1.0f  - theta;
        float weight2 = theta - fo;

        if(DYNAMIC_INDEXING)
        {
          des[fidx] += (weight1 * weight);
          des[fidx + 1] += (weight2 * weight);
          // this dynamic indexing part might be slow
        }
        else
        {
#pragma unroll
          for(int k = 0; k < 8; ++k)
          {
            if(k == fidx) 
            {
              des[k] += (weight1 * weight);
              des[k+1] += (weight2 * weight);
            }
          }
        }
      }
    }
  }
  des[0] += des[8];

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  if(HALF_SIFT)
  {
    // half sift -> 4 directions only
    des[0] += des[4];
    des[1] += des[5];
    des[2] += des[6];
    des[3] += des[7];

    d_des[idx] = make_float4(des[0], des[1], des[2], des[3]);

    return;
  }
  else
  {
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  // full sift -> 8 directions
  int didx = idx << 1;

  d_des[didx] = make_float4(des[0], des[1], des[2], des[3]);
  d_des[didx+1] = make_float4(des[4], des[5], des[6], des[7]);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  }
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
}

#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined NORMALIZE_DESCRIPTOR_PER_WARP

texture<float2, 1, cudaReadModeElementType> texDataH2;

__device__ float ND_WarpReduction(volatile float *sData, int idx)
{
  int warpIdx = idx & (warpSize-1); // index within warp

  // parallel reduction within warp - just half of the warp is active 
  if(warpIdx < 16)
  {
    sData[idx] += sData[idx + 16];
    sData[idx] += sData[idx +  8];
    sData[idx] += sData[idx +  4];
    sData[idx] += sData[idx +  2];
    sData[idx] += sData[idx +  1];
  }

  return sData[idx-warpIdx]; // first thread in the warp
}

// assumes the block size (128, 1, 1) and grid size (Bx, 1, 1)
void __global__ NormalizeDescriptor_Kernel(float4* d_des, int num)
{
  // size => numThreadsPerBlock * sizeof(float); numThreadsPerBlock must be multiple of 32
  extern __shared__ volatile float reductionCache[];

  // int globalIdx = threadIdx.x + IMUL(blockIdx.x + IMUL(blockIdx.y, gridDim.x), blockDim.x); // 2D grid
  int globalIdx = threadIdx.x + IMUL(blockIdx.x, blockDim.x);  // 1D grid
  int localIdx = threadIdx.x;

  while(globalIdx < 32*num)
  {
    // the vector is first normalized to unit length, thus adjusting for changing image contrast
    float4 temp = tex1Dfetch(texDataF4, globalIdx);
    float norm1 = (temp.x*temp.x + temp.y*temp.y + temp.z*temp.z + temp.w*temp.w);

    reductionCache[localIdx] = norm1;
    // __syncthreads(); // threads in warp are always sync
    norm1 = rsqrt(ND_WarpReduction(reductionCache, localIdx));

    // ... then all feature dimensions are thresholded to a maximum value of 0.2
    temp.x = min(0.2f, temp.x * norm1);
    temp.y = min(0.2f, temp.y * norm1);
    temp.z = min(0.2f, temp.z * norm1);
    temp.w = min(0.2f, temp.w * norm1);

    float norm2 = (temp.x*temp.x + temp.y*temp.y + temp.z*temp.z + temp.w*temp.w);
 
    reductionCache[localIdx] = norm2;
    // __syncthreads(); // threads in warp are always sync
    norm2 = rsqrt(ND_WarpReduction(reductionCache, localIdx));

    // ... and the vector is again normalized to unit length
    temp.x *= norm2;
    temp.y *= norm2;
    temp.z *= norm2;
    temp.w *= norm2;
    d_des[globalIdx] = temp;

    // move to the next descriptor, if there is any unprocessed
    globalIdx += IMUL(gridDim.x, blockDim.x);
  }
}

// assumes the block size (128, 1, 1) and grid size (Bx, 1, 1)
// version used for half sift
void __global__ NormalizeDescriptor_Kernel(float2* d_des, int num)
{
  // size => numThreadsPerBlock * sizeof(float); numThreadsPerBlock must be multiple of 32
  extern __shared__ volatile float reductionCache[];

  // int globalIdx = threadIdx.x + IMUL(blockIdx.x + IMUL(blockIdx.y, gridDim.x), blockDim.x); // 2D grid
  int globalIdx = threadIdx.x + IMUL(blockIdx.x, blockDim.x);  // 1D grid
  int localIdx = threadIdx.x;

  while(globalIdx < 32*num)
  {
    // the vector is first normalized to unit length, thus adjusting for changing image contrast
    float2 temp = tex1Dfetch(texDataH2, globalIdx);
    float norm1 = (temp.x*temp.x + temp.y*temp.y);

    reductionCache[localIdx] = norm1;
    // __syncthreads(); // threads in warp are always sync
    norm1 = rsqrt(ND_WarpReduction(reductionCache, localIdx));

    // ... then all feature dimensions are thresholded to a maximum value of 0.2
    temp.x = min(0.2f, temp.x * norm1);
    temp.y = min(0.2f, temp.y * norm1);

    float norm2 = (temp.x*temp.x + temp.y*temp.y);
 
    reductionCache[localIdx] = norm2;
    // __syncthreads(); // threads in warp are always sync
    norm2 = rsqrt(ND_WarpReduction(reductionCache, localIdx));

    // ... and the vector is again normalized to unit length
    temp.x *= norm2;
    temp.y *= norm2;
    d_des[globalIdx] = temp;

    // move to the next descriptor, if there is any unprocessed
    globalIdx += IMUL(gridDim.x, blockDim.x);
  }
}

#else
template<int DESCRIPTOR_SIZE, int SHIFT>
void __global__ NormalizeDescriptor_Kernel(float4* d_des, int num)
{
  float4 temp[DESCRIPTOR_SIZE];
  int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;

  if(idx >= num)
    return;

  int sidx = idx << SHIFT;
  float norm1 = 0;
  float norm2 = 0;

#pragma unroll
  // the vector is first normalized to unit length, thus adjusting for changing image contrast
  for(int i = 0; i < DESCRIPTOR_SIZE; ++i)
  {
    temp[i] = tex1Dfetch(texDataF4, sidx+i);
    norm1 += (temp[i].x*temp[i].x + temp[i].y*temp[i].y + temp[i].z*temp[i].z + temp[i].w*temp[i].w);
  }
  norm1 = rsqrt(norm1);

#pragma unroll
  // ... then all feature dimensions are thresholded to a maximum value of 0.2
  for(int i = 0; i < DESCRIPTOR_SIZE; ++i)
  {
    temp[i].x = min(0.2f, temp[i].x * norm1);
    temp[i].y = min(0.2f, temp[i].y * norm1);
    temp[i].z = min(0.2f, temp[i].z * norm1);
    temp[i].w = min(0.2f, temp[i].w * norm1);

    norm2 += (temp[i].x*temp[i].x + temp[i].y*temp[i].y + temp[i].z*temp[i].z + temp[i].w*temp[i].w);
  }
  norm2 = rsqrt(norm2);

#pragma unroll
  // ... and the vector is again normalized to unit length
  for(int i = 0; i < DESCRIPTOR_SIZE; ++i)
  {
    temp[i].x *= norm2;
    temp[i].y *= norm2;
    temp[i].z *= norm2;
    temp[i].w *= norm2;
    d_des[sidx + i] = temp[i];
  }
}
#endif // (GPU_HESSIAN || GPU_SIFT_MODIFIED) && NORMALIZE_DESCRIPTOR_PER_WARP

void ProgramCU::ComputeDescriptor(CuTexImage*list, CuTexImage* got, CuTexImage* dtex, int rect, int stream)
{
  int num = list->GetImgWidth();
  int width = got->GetImgWidth();
  int height = got->GetImgHeight();

  dtex->InitTexture(num * 128, 1, 1);
  got->BindTexture2D(texDataF2);
  list->BindTexture(texDataF4);

  int block_width = DESCRIPTOR_COMPUTE_BLOCK_SIZE;
  dim3 grid((num * 16 + block_width -1) / block_width);
  dim3 block(block_width);

  if(rect)
  {
    if(GlobalUtil::_UseDynamicIndexing)
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    {
      if(GlobalUtil::_HalfSIFT)
        ComputeDescriptorRECT_Kernel<true, true><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
      else
        ComputeDescriptorRECT_Kernel<true, false><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
    }
#else
      ComputeDescriptorRECT_Kernel<true><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
    else
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    {
      if(GlobalUtil::_HalfSIFT)
        ComputeDescriptorRECT_Kernel<false, true><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
      else
        ComputeDescriptorRECT_Kernel<false, false><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
    }
#else
      ComputeDescriptorRECT_Kernel<false><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  }
  else
  {
    if(GlobalUtil::_UseDynamicIndexing)
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    {
      if(GlobalUtil::_HalfSIFT)
        ComputeDescriptor_Kernel<true, true><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
      else
        ComputeDescriptor_Kernel<true, false><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
    }
#else
      ComputeDescriptor_Kernel<true><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
    else
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    {
      if(GlobalUtil::_HalfSIFT)
        ComputeDescriptor_Kernel<false, true><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
      else
        ComputeDescriptor_Kernel<false, false><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
    }
#else
      ComputeDescriptor_Kernel<false><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  }

  if(GlobalUtil::_NormalizedSIFT)
  {
#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined NORMALIZE_DESCRIPTOR_PER_WARP
	// 32 threads (one warp) normalize one descriptor -> one thread handles one float4
    // 1D block size (128, 1, 1) => four descriptors are normalized per block
    // 1D grid size (Bx, 1, 1)

    if(GlobalUtil::_HalfSIFT)
      dtex->BindTexture(texDataH2);
    else
      dtex->BindTexture(texDataF4);

    const int blockWidth = DESCRIPTOR_NORMALIZE_PER_BLOCK;
    int blocksInGrid = min(16384, (num*32 + blockWidth -1) / blockWidth);
    dim3 grid(blocksInGrid);
    dim3 block(blockWidth);

    if(GlobalUtil::_HalfSIFT)
      NormalizeDescriptor_Kernel<<<grid, block, blockWidth*sizeof(float)>>>((float2*) dtex->_cuData, num);
    else
      NormalizeDescriptor_Kernel<<<grid, block, blockWidth*sizeof(float)>>>((float4*) dtex->_cuData, num);
#else
    dtex->BindTexture(texDataF4);

    const int block_width = DESCRIPTOR_NORMALIZE_PER_BLOCK;
    dim3 grid((num + block_width -1) / block_width);
    dim3 block(block_width);

    NormalizeDescriptor_Kernel<32, 5><<<grid, block>>>((float4*) dtex->_cuData, num);
#endif // (GPU_HESSIAN || GPU_SIFT_MODIFIED) && NORMALIZE_DESCRIPTOR_PER_WARP
  }

  CheckErrorCUDA("ComputeDescriptor");
}

#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined TOP_K_SELECTION

// Enables maximum occupancy - bitonic sort
#define SHARED_SIZE_LIMIT 1024U // CC2.0
#define TOPK_BLOCK_SIZE    128  // 256

void ProgramCU::TopKInit(TopKData &data, int listSize, int countThreshold)
{
  data.keypointsCount = listSize;

  // the closest power of two higher than keypoints count
  data.keypointsCountAsPowerOfTwo = 1;
  while (data.keypointsCountAsPowerOfTwo < listSize)
    data.keypointsCountAsPowerOfTwo *= 2;

  // keypoints count have to be at least SHARED_SIZE_LIMIT (required by bitonic sort implementation)
  data.keypointsCountAsPowerOfTwo = max(data.keypointsCountAsPowerOfTwo, SHARED_SIZE_LIMIT);

  // allocate arrays to store responses and indices
  cudaMalloc(&data.keys, (data.keypointsCountAsPowerOfTwo+1) * sizeof(float));           // one extra element in both arrays is used by prefix scan
  cudaMalloc(&data.indices, (data.keypointsCountAsPowerOfTwo+1) * sizeof(unsigned int));

  data.topKCountThreshold = countThreshold;
}

void ProgramCU::TopKFinish(TopKData &data)
{
  // release all data
  cudaFree(data.keys);
  cudaFree(data.indices);

  cudaFree(data.devLevelFeaturesCount);
  delete[] data.levelFeaturesCount;
}

void __global__ CopyTopKData_Kernel(int width, float *keys, unsigned int *indices, int listLen, int offset, int keypointsCount)
{
  int idx = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

  if(idx >= listLen)
    return;

  float response = MIN_VALUE;

  if(idx+offset < keypointsCount)
  {
    // read keypoint index
    int4 ikey = tex1Dfetch(texDataList, idx);

    int keyIdx = IMUL(width, ikey.y) + ikey.x;
    // read keypoint additional info
    float4 key = tex1Dfetch(texDataF4, keyIdx);

 #if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    // keypoint info
    //  key.x: response 16b | 14b unused | 2b type
    //  key.y: x
    //  key.z: y 
    //  key.w: scale

    // extract response
    unsigned int value = *((unsigned int *)(&key.x)); // __float_as_uint(key.x); // CUDA 7.5

    value = (value & 0xFFFF0000u) >> 16;
    response = __half2float((unsigned short)value);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  }

  keys[idx+offset] = fabs(response);
  indices[idx+offset] = idx+offset;
}

void ProgramCU::TopKCopyData(CuTexImage* list, CuTexImage* key, TopKData &topKData, int offset)
{
  bool isActivePadding = ((list == NULL) && (key == NULL));    // clear the padding values?
  int len = isActivePadding ? (topKData.keypointsCountAsPowerOfTwo - offset) : list->GetImgWidth();

  if(len <= 0)
    return;

  int width = isActivePadding ? 0 : key->GetImgWidth();

  if(!isActivePadding)
  {
    list->BindTexture(texDataList);

    key->BindTexture(texDataF4);
  }

  int blockWidth = TOPK_BLOCK_SIZE;
  dim3 grid((len + blockWidth - 1) / blockWidth);
  dim3 block(blockWidth);

  CopyTopKData_Kernel<<<grid, block>>>(width, topKData.keys, topKData.indices, len, offset, topKData.keypointsCount);

  CheckErrorCUDA("CopyTopKData");
}

// Map to single instructions on G8x / G9x / G100
#define UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

__device__ inline void BMS_Comparator(float &keyA, unsigned int &valA, float &keyB, unsigned int &valB, unsigned int dir)
{
  union {
    unsigned int uintValue;
	float floatValue;
  };

  if ((keyA > keyB) == dir)
  {
    floatValue = keyA; keyA = keyB; keyB = floatValue;
    uintValue = valA; valA = valB; valB = uintValue;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void BitonicSortShared_Kernel(float *dstKey, unsigned int *dstVal, float *srcKey, unsigned int *srcVal, unsigned int arrayLength, unsigned int dir)
{
  // Shared memory storage for one or more short vectors
  __shared__ float sKey[SHARED_SIZE_LIMIT];
  __shared__ unsigned int sVal[SHARED_SIZE_LIMIT];

  // Offset to the beginning of subbatch and load data
  int offset = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x; 
  srcKey += offset;
  srcVal += offset;
  dstKey += offset;
  dstVal += offset;

  sKey[threadIdx.x +                       0] = srcKey[                      0];
  sVal[threadIdx.x +                       0] = srcVal[                      0];
  sKey[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = srcKey[(SHARED_SIZE_LIMIT / 2)];
  sVal[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = srcVal[(SHARED_SIZE_LIMIT / 2)];

  for (unsigned int size = 2; size < arrayLength; size <<= 1)
  {
    // bitonic merge
    unsigned int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

    for (unsigned int stride = size / 2; stride > 0; stride >>= 1)
    {
      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      BMS_Comparator(
        sKey[pos +      0], sVal[pos +      0],
        sKey[pos + stride], sVal[pos + stride],
        ddd
      );
    }
  }

  // ddd == dir for the last bitonic merge step
  {
    for (unsigned int stride = arrayLength / 2; stride > 0; stride >>= 1)
    {
      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      BMS_Comparator(
        sKey[pos +      0], sVal[pos +      0],
        sKey[pos + stride], sVal[pos + stride],
        dir
      );
    }
  }

  __syncthreads();

  dstKey[                      0] = sKey[threadIdx.x +                       0];
  dstVal[                      0] = sVal[threadIdx.x +                       0];
  dstKey[(SHARED_SIZE_LIMIT / 2)] = sKey[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  dstVal[(SHARED_SIZE_LIMIT / 2)] = sVal[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
// Bottom-level bitonic sort
// Almost the same as bitonicSortShared with the exception of even / odd subarrays being sorted in opposite directions
// Bitonic merge accepts both Ascending | descending or descending | ascending sorted pairs
__global__ void BitonicSortShared1_Kernel(float *dstKey, unsigned int *dstVal, float *srcKey, unsigned int *srcVal)
{
  // Shared memory storage for current subarray
  __shared__ float sKey[SHARED_SIZE_LIMIT];
  __shared__ unsigned int sVal[SHARED_SIZE_LIMIT];

  // Offset to the beginning of subarray and load data
  int offset = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x; 
  srcKey += offset;
  srcVal += offset;
  dstKey += offset;
  dstVal += offset;

  sKey[threadIdx.x +                       0] = srcKey[                      0];
  sVal[threadIdx.x +                       0] = srcVal[                      0];
  sKey[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = srcKey[(SHARED_SIZE_LIMIT / 2)];
  sVal[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = srcVal[(SHARED_SIZE_LIMIT / 2)];

  for (unsigned int size = 2; size < SHARED_SIZE_LIMIT; size <<= 1)
  {
    // Bitonic merge
    unsigned int ddd = (threadIdx.x & (size / 2)) != 0;

    for (unsigned int stride = size / 2; stride > 0; stride >>= 1)
    {
      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      BMS_Comparator(
        sKey[pos +      0], sVal[pos +      0],
        sKey[pos + stride], sVal[pos + stride],
        ddd
      );
    }
  }

  // odd / even arrays of SHARED_SIZE_LIMIT elements
  // sorted in opposite directions
  unsigned int ddd = blockIdx.x & 1;
  {
    for (unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
    {
      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      BMS_Comparator(
        sKey[pos +      0], sVal[pos +      0],
        sKey[pos + stride], sVal[pos + stride],
        ddd
      );
    }
  }

  __syncthreads();

  dstKey[                      0] = sKey[threadIdx.x +                       0];
  dstVal[                      0] = sVal[threadIdx.x +                       0];
  dstKey[(SHARED_SIZE_LIMIT / 2)] = sKey[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  dstVal[(SHARED_SIZE_LIMIT / 2)] = sVal[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

// Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void BitonicMergeGlobal_Kernel(float *dstKey, unsigned int *dstVal, float *srcKey, unsigned int *srcVal, unsigned int arrayLength, unsigned int size, unsigned int stride, unsigned int dir)
{
  unsigned int global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int comparatorI = global_comparatorI & (arrayLength / 2 - 1);

  // Bitonic merge
  unsigned int ddd = dir ^ ((comparatorI & (size / 2)) != 0);
  unsigned int pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

  float keyA = srcKey[pos + 0];
  unsigned int valA =  srcVal[pos + 0];
  float keyB = srcKey[pos + stride];
  unsigned int valB =  srcVal[pos + stride];

  BMS_Comparator(
    keyA, valA,
    keyB, valB,
    ddd
  );

  dstKey[pos +      0] = keyA;
  dstVal[pos +      0] = valA;
  dstKey[pos + stride] = keyB;
  dstVal[pos + stride] = valB;
}

// Combined bitonic merge steps for size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void BitonicMergeShared_Kernel(float *dstKey, unsigned int *dstVal, float *srcKey, unsigned int *srcVal, unsigned int arrayLength, unsigned int size, unsigned int dir)
{
  // Shared memory storage for current subarray
  __shared__ float sKey[SHARED_SIZE_LIMIT];
  __shared__ unsigned int sVal[SHARED_SIZE_LIMIT];

  int offset = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x; 
  srcKey += offset;
  srcVal += offset;
  dstKey += offset;
  dstVal += offset;

  sKey[threadIdx.x +                       0] = srcKey[                      0];
  sVal[threadIdx.x +                       0] = srcVal[                      0];
  sKey[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = srcKey[(SHARED_SIZE_LIMIT / 2)];
  sVal[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = srcVal[(SHARED_SIZE_LIMIT / 2)];

  // Bitonic merge
  unsigned int comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
  unsigned int ddd = dir ^ ((comparatorI & (size / 2)) != 0);

  for(unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
  {
    __syncthreads();

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    BMS_Comparator(
      sKey[pos +      0], sVal[pos +      0],
      sKey[pos + stride], sVal[pos + stride],
      ddd
    );
  }

  __syncthreads();

  dstKey[                      0] = sKey[threadIdx.x +                       0];
  dstVal[                      0] = sVal[threadIdx.x +                       0];
  dstKey[(SHARED_SIZE_LIMIT / 2)] = sKey[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  dstVal[(SHARED_SIZE_LIMIT / 2)] = sVal[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

unsigned int BMS_FactorOf2(unsigned int *log2L, unsigned int L)
{
  if (!L)
  {
    *log2L = 0;
    return 0;
  }
  else
  {
    for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);

    return L;
  }
}

///////////////////////////////////////////////////////////////////////////////
// bitonic sort is borrowed from the sorting networks sample which is a part of 
// the NVIDIA GPU Computing SDK (NVIDIA CUDA Code Samples)
unsigned int BMS_BitonicSort(float *dstKey, unsigned int *dstVal, float *srcKey, unsigned int *srcVal, unsigned int arrayLength, unsigned int dir)
{
  // Nothing to sort
  if(arrayLength < 2)
    return 0;

  // only power-of-two array lengths are supported by this implementation
  unsigned int log2L;
  unsigned int factorizationRemainder = BMS_FactorOf2(&log2L, arrayLength);
  assert(factorizationRemainder == 1);

  dir = (dir != 0);

  unsigned int blockCount = arrayLength / SHARED_SIZE_LIMIT;
  unsigned int threadCount = SHARED_SIZE_LIMIT / 2;

  if(arrayLength <= SHARED_SIZE_LIMIT)
  {
    assert(arrayLength % SHARED_SIZE_LIMIT == 0);
    BitonicSortShared_Kernel<<<blockCount, threadCount>>>(dstKey, dstVal, srcKey, srcVal, arrayLength, dir);
  }
  else
  {
    BitonicSortShared1_Kernel<<<blockCount, threadCount>>>(dstKey, dstVal, srcKey, srcVal);

    for(unsigned int size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
      for (unsigned stride = size / 2; stride > 0; stride >>= 1)
        if(stride >= SHARED_SIZE_LIMIT)
        {
          BitonicMergeGlobal_Kernel<<<arrayLength / 512, 256>>>(dstKey, dstVal, dstKey, dstVal, arrayLength, size, stride, dir);
        }
        else
        {
          BitonicMergeShared_Kernel<<<blockCount, threadCount>>>(dstKey, dstVal, dstKey, dstVal, arrayLength, size, dir);
          break;
        }
  }

  return threadCount;
}

#undef SHARED_SIZE_LIMIT

void ProgramCU::TopKSort(TopKData &topKData)
{
  // sort in descending order
  unsigned int threadsCount = BMS_BitonicSort(
    topKData.keys, topKData.indices,      // dst 
    topKData.keys, topKData.indices,      // src
    topKData.keypointsCountAsPowerOfTwo,  // array length - have to be power of 2
    0                                     // sort direction
  );
}

__global__ void MarkSelectedElements_Kernel(unsigned int *outFlags, unsigned int *inIndices, int len, int topK)
{
  int threadID = threadIdx.x + blockDim.x* blockIdx.x;

  if (threadID >= len)
    return;

  if(threadID < topK)
  {
    unsigned int index = inIndices[threadID];
    outFlags[index] = 1;
  } // 0 are not written because of cleared array to 0
}

/////////////////////
// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance is lower with ZERO_BANK_CONFLICTS enabled.
// It is provided as an example.
//#define ZERO_BANK_CONFLICTS 

// 16 banks on G80, 32 foc CC 2.X and 3.X
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

///////////////////////////////////////////////////////////////////////////////
// prefix scan is borrowed from the scan large array sample which is a part of 
// the NVIDIA GPU Computing SDK  
//
// Work-efficient compute implementation of scan, one thread per 2 elements
// Work-efficient: O(log(n)) steps, and O(n) adds.
// Also shared storage efficient: Uses n + n/NUM_BANKS shared memory -- no ping-ponging
// Also avoids most bank conflicts using single-element offsets every NUM_BANKS elements.
//
// In addition, If ZERO_BANK_CONFLICTS is defined, uses 
//     n + n/NUM_BANKS + n/(NUM_BANKS*NUM_BANKS) 
// shared memory. If ZERO_BANK_CONFLICTS is defined, avoids ALL bank conflicts using 
// single-element offsets every NUM_BANKS elements, plus additional single-element offsets 
// after every NUM_BANKS^2 elements.
//
// Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
// and Their Applications", or Prins and Chatterjee PRAM course notes:
// https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
// 
// This work-efficient version is based on the algorithm presented in Guy Blelloch's
// excellent paper "Prefix sums and their applications".
// http://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
//
// Pro: Work Efficient, very few bank conflicts (or zero if ZERO_BANK_CONFLICTS is defined)
// Con: More instructions to compute bank-conflict-free shared memory addressing,
// and slightly more shared memory storage used.
//

template <bool isNP2>
__device__ void PPS_LoadSharedChunkFromMem(unsigned int *sData, const unsigned int *inData, int n, int baseIndex, int& ai, int& bi, int& memAi, int& memBi, int& bankOffsetA, int& bankOffsetB)
{
  int threadID = threadIdx.x;
  memAi = baseIndex + threadIdx.x;
  memBi = memAi + blockDim.x;

  ai = threadID;
  bi = threadID + blockDim.x;

  // compute spacing to avoid bank conflicts
  bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  // cache the computational window in shared memory, pad values beyond n with zeros
  sData[ai + bankOffsetA] = inData[memAi]; 
    
  if(isNP2) // compile-time decision
  {
    sData[bi + bankOffsetB] = (bi < n) ? inData[memBi] : 0; 
  }
  else
  {
    sData[bi + bankOffsetB] = inData[memBi]; 
  }
}

template <bool isNP2>
__device__ void PPS_StoreSharedChunkToMem(unsigned int *outData, const unsigned int *sData, int n, int ai, int bi, int memAi, int memBi, int bankOffsetA, int bankOffsetB)
{
  __syncthreads();

  // write results to global memory
  outData[memAi] = sData[ai + bankOffsetA]; 
  if(isNP2) // compile-time decision
  {
    if(bi < n)
      outData[memBi] = sData[bi + bankOffsetB]; 
  }
  else
  {
    outData[memBi] = sData[bi + bankOffsetB]; 
  }
}

template <bool storeSum>
__device__ void PPS_ClearLastElement(unsigned int *sData, unsigned int *blockSums, int blockIndex)
{
  if(threadIdx.x == 0)
  {
    int index = (blockDim.x << 1) - 1;
    index += CONFLICT_FREE_OFFSET(index);
        
    if(storeSum) // compile-time decision
    {
      // write this block's total sum to the corresponding index in the blockSums array
      blockSums[blockIndex] = sData[index];
    }

    // zero the last element in the scan so it will propagate back to the front
    sData[index] = 0;
  }
}

__device__ unsigned int PPS_BuildSum(unsigned int *sData)
{
  unsigned int threadID = threadIdx.x;
  unsigned int stride = 1;
    
  // build the sum in place up the tree
  for(int d = blockDim.x; d > 0; d >>= 1)
  {
    __syncthreads();

    if(threadID < d)      
    {
      int i  = IMUL(IMUL(2, stride), threadID);
      int ai = i + stride - 1;
      int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      sData[bi] += sData[ai];
    }

    stride *= 2;
  }

  return stride;
}

__device__ void PPS_ScanRootToLeaves(unsigned int *sData, unsigned int stride)
{
  unsigned int threadID = threadIdx.x;

  // traverse down the tree building the scan in place
  for(int d = 1; d <= blockDim.x; d *= 2)
  {
    stride >>= 1;

    __syncthreads();

    if(threadID < d)
    {
      int i = IMUL(IMUL(2, stride), threadID);
      int ai = i + stride - 1;
      int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      unsigned int t = sData[ai];
      sData[ai] = sData[bi];
      sData[bi] += t;
    }
  }
}

template <bool storeSum>
__device__ void PPS_PrescanBlock(unsigned int *data, int blockIndex, unsigned int *blockSums)
{
  // build the sum in place up the tree
  int stride = PPS_BuildSum(data);

  PPS_ClearLastElement<storeSum>(data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
  // traverse down tree to build the scan 
  PPS_ScanRootToLeaves(data, stride);
}

template <bool storeSum, bool isNP2>
__global__ void PPS_Prescan_Kernel(unsigned int *outData, const unsigned int *inData, unsigned int *blockSums, int n, int blockIndex, int baseIndex)
{
  extern __shared__ unsigned int shData[];
  int ai, bi, memAi, memBi, bankOffsetA, bankOffsetB;

  // load data into shared memory
  PPS_LoadSharedChunkFromMem<isNP2>(shData, inData, n, (baseIndex == 0) ? IMUL(blockIdx.x, (blockDim.x << 1)) : baseIndex, ai, bi, memAi, memBi, bankOffsetA, bankOffsetB);
  // scan the data in each block
  PPS_PrescanBlock<storeSum>(shData, blockIndex, blockSums);
  // write results to device memory
  PPS_StoreSharedChunkToMem<isNP2>(outData, shData, n, ai, bi, memAi, memBi, bankOffsetA, bankOffsetB);
}

__global__ void PPS_UniformAdd_Kernel(unsigned int *data, unsigned int *uniforms, int n, int blockOffset, int baseIndex)
{
  __shared__ unsigned int uni;

  if(threadIdx.x == 0)
    uni = uniforms[blockIdx.x + blockOffset];
    
  unsigned int address = IMUL(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

  __syncthreads();
    
  // note two adds per thread
  data[address] += uni;
  data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}

inline bool PPS_IsPowerOfTwo(int n)
{
  return ((n&(n-1))==0) ;
}

inline int PPS_FloorPow2(int n)
{
#ifdef WIN32
  // method 2
  return 1 << (int)logb((float)n);
#else
  // method 1
  // float nf = (float)n;
  // return 1 << (((*(int*)&nf) >> 23) - 127); 
  int exp;
  frexp((float)n, &exp);
  return 1 << (exp - 1);
#endif
}

#define THREADS_PER_BLOCK 128

void PPS_PreallocBlockSums(TopKData &topKData, unsigned int maxNumElements)
{
  assert(topKData.numElementsAllocated == 0); // shouldn't be called 

  topKData.numElementsAllocated = maxNumElements;

  unsigned int blockSize = THREADS_PER_BLOCK; // max size of the thread blocks
  unsigned int numElements = maxNumElements;

  int level = 0;

  do
  {       
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    if(numBlocks > 1)
      level++;

    numElements = numBlocks;
  } while(numElements > 1);

  topKData.scanBlockSums = (unsigned int**) malloc(level * sizeof(unsigned int*));
  topKData.numLevelsAllocated = level;
    
  numElements = maxNumElements;
  level = 0;
    
  do
  {       
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    if (numBlocks > 1) 
      cudaMalloc((void**) &topKData.scanBlockSums[level++], numBlocks * sizeof(unsigned int));

    numElements = numBlocks;
  } while(numElements > 1);
}

void PPS_DeallocBlockSums(TopKData &topKData)
{
  for(unsigned int i = 0; i < topKData.numLevelsAllocated; i++)
    cudaFree(topKData.scanBlockSums[i]);
    
  free((void**)topKData.scanBlockSums);

  topKData.scanBlockSums = NULL;
  topKData.numElementsAllocated = 0;
  topKData.numLevelsAllocated = 0;
}

// prefix scan is borrowed from the scan large array sample which is a part of NVIDIA GPU Computing SDK  
void PPS_PrescanArrayRecursive(TopKData &topKData, unsigned int *outArray, const unsigned int *inArray, int numElements, int level)
{
  unsigned int blockSize = THREADS_PER_BLOCK; // max size of the thread blocks
  unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
  unsigned int numThreads;

  if(numBlocks > 1)
    numThreads = blockSize;
  else if(PPS_IsPowerOfTwo(numElements))
    numThreads = numElements / 2;
  else
    numThreads = PPS_FloorPow2(numElements);

  unsigned int numElementsPerBlock = numThreads * 2;

  // if this is a non-power-of-2 array, the last block will be non-full
  // compute the smallest power of 2 able to compute its scan.
  unsigned int numElementsLastBlock = numElements - (numBlocks-1) * numElementsPerBlock;
  unsigned int numThreadsLastBlock = max(1, numElementsLastBlock / 2);
  unsigned int np2LastBlock = 0;
  unsigned int sharedMemLastBlock = 0;
    
  if (numElementsLastBlock != numElementsPerBlock)
  {
    np2LastBlock = 1;

    if(!PPS_IsPowerOfTwo(numElementsLastBlock))
      numThreadsLastBlock = PPS_FloorPow2(numElementsLastBlock);    
        
    unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
    sharedMemLastBlock = sizeof(unsigned int) * (2 * numThreadsLastBlock + extraSpace);
  }

  // padding space is used to avoid shared memory bank conflicts
  unsigned int extraSpace = numElementsPerBlock / NUM_BANKS;
  unsigned int sharedMemSize = sizeof(unsigned int) * (numElementsPerBlock + extraSpace);

  // setup execution parameters
  // if NP2, we process the last block separately
  dim3 grid(max(1, numBlocks - np2LastBlock), 1, 1); 
  dim3 threads(numThreads, 1, 1);

  // execute the scan
  if(numBlocks > 1)
  {
    PPS_Prescan_Kernel<true, false><<<grid, threads, sharedMemSize>>>(outArray, inArray, topKData.scanBlockSums[level], numThreads*2, 0, 0);
    if(np2LastBlock)
      PPS_Prescan_Kernel<true, true><<<1, numThreadsLastBlock, sharedMemLastBlock>>>(outArray, inArray, topKData.scanBlockSums[level], numElementsLastBlock, numBlocks - 1, numElements - numElementsLastBlock);

    // After scanning all the sub-blocks, we are mostly done.  But now we need to take all of the last
	// values of the sub-blocks and scan those. This will give us a new value that must be sdded to each
	// block to get the final results.
    // recursive (CPU) call
    PPS_PrescanArrayRecursive(topKData, topKData.scanBlockSums[level], topKData.scanBlockSums[level], numBlocks, level+1);

    PPS_UniformAdd_Kernel<<<grid, threads>>>(outArray, topKData.scanBlockSums[level], numElements - numElementsLastBlock, 0, 0);
    if(np2LastBlock)
      PPS_UniformAdd_Kernel<<<1, numThreadsLastBlock>>>(outArray, topKData.scanBlockSums[level], numElementsLastBlock, numBlocks - 1, numElements - numElementsLastBlock);
  }
  else if(PPS_IsPowerOfTwo(numElements))
  {
    PPS_Prescan_Kernel<false, false><<<grid, threads, sharedMemSize>>>(outArray, inArray, 0, numThreads*2, 0, 0);
  }
  else
  {
    PPS_Prescan_Kernel<false, true><<<grid, threads, sharedMemSize>>>(outArray, inArray, 0, numElements, 0, 0);
  }
}

void ProgramCU::TopKPrefixScan(TopKData &topKData)
{
  if(topKData.keypointsCount <= 0)
    return;

  topKData.numElementsAllocated = 0;
  topKData.numLevelsAllocated = 0;

  unsigned int numElements = topKData.keypointsCountAsPowerOfTwo;

  unsigned int *devIdxs = (unsigned int *)topKData.keys;
  unsigned int *devData = topKData.indices;

  // clear keypoints flags to 0 
  cudaMemset(devIdxs, 0, (numElements+1)*sizeof(unsigned int));

  const int blocks = (numElements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // prepare input for prefix scan - mark selected keypoints by 1
  MarkSelectedElements_Kernel<<<blocks, THREADS_PER_BLOCK>>>(devIdxs, devData, numElements+1, topKData.topKCountThreshold);

  PPS_PreallocBlockSums(topKData, numElements);

  // run the prescan
  PPS_PrescanArrayRecursive(topKData, devIdxs, devIdxs, numElements, 0);

  PPS_DeallocBlockSums(topKData);    

  CheckErrorCUDA("TopKPrefixScan");
}

#undef THREADS_PER_BLOCK

void __global__ GetLevelsFeatureNum_Kernel(unsigned int *indices, int *levelFeatureNum, int levelsCount)
{
  int threadID = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

  if(threadID >= levelsCount)
    return;

  unsigned int index = levelFeatureNum[threadID];

  levelFeatureNum[threadID] = indices[index];
}

void ProgramCU::TopKGetLevelsFeatureNum(TopKData &topKData)
{
  if(topKData.levelsCount <= 0)
    return;

  cudaMalloc((void**)&(topKData.devLevelFeaturesCount), topKData.levelsCount*sizeof(unsigned int));
  cudaMemcpy(topKData.devLevelFeaturesCount, topKData.levelFeaturesCount, topKData.levelsCount*sizeof(unsigned int), cudaMemcpyHostToDevice);

  int blockWidth = TOPK_BLOCK_SIZE;
  dim3 grid((topKData.levelsCount + blockWidth - 1) / blockWidth);
  dim3 block(blockWidth);

  GetLevelsFeatureNum_Kernel<<<grid, block>>>((unsigned int *)topKData.keys, topKData.devLevelFeaturesCount, topKData.levelsCount);

  cudaMemcpy(topKData.levelFeaturesCount, topKData.devLevelFeaturesCount, topKData.levelsCount*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  CheckErrorCUDA("TopKGetLevelsFeatureNum");
}

void __global__ CompactLevelFeatures_Kernel(float4 *outFeatures, unsigned int *indices, unsigned int offset, unsigned int featuresCount)
{
  int threadID = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

  if(threadID >= featuresCount)
    return;

  unsigned int index = indices[offset + threadID];
  if(index == indices[offset + threadID + 1])
    return;

  unsigned int firstIndex = indices[offset];
  index -= firstIndex;

  float4 key = tex1Dfetch(texDataF4, threadID);

  outFeatures[index] = key;
}

void ProgramCU::TopKCompactLevelFeatures(CuTexImage *list, unsigned int oldLen, float **newLevelFeatures, unsigned int newLen, TopKData &topKData, unsigned int offset)
{
  if(newLen <= 0)
  {
    *newLevelFeatures = NULL;
    return;
  }

  cudaMalloc((void**)(newLevelFeatures), newLen*4*sizeof(float));

  list->BindTexture(texDataF4);

  int blockWidth = TOPK_BLOCK_SIZE;
  dim3 grid((oldLen + blockWidth - 1) / blockWidth);
  dim3 block(blockWidth);

  CompactLevelFeatures_Kernel<<<grid, block>>>((float4 *)(*newLevelFeatures), (unsigned int *)topKData.keys, offset, oldLen);

  CheckErrorCUDA("TopKCompactLevelFeatures");
}

#endif // (GPU_HESSIAN || GPU_SIFT_MODIFIED) && TOP_K_SELECTION

#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)

void ProgramCU::DetectionDataInit(int **featureTexLen, int len)
{
  cudaMalloc((void**)(featureTexLen), len*sizeof(int));
  cudaMemset(*featureTexLen, 0, len*sizeof(int));

  CheckErrorCUDA("ProgramCU::DetectionDataInit");
}

void ProgramCU::DetectionDataDownload(int *dst, int *featureTexLen, int len)
{
  cudaMemcpy(dst, featureTexLen, len*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemset(featureTexLen, 0, len*sizeof(int));

  CheckErrorCUDA("ProgramCU::DetectionDataDownload");
}

void ProgramCU::DetectionDataFinish(int **featureTexLen)
{
  cudaFree(*featureTexLen);
  *featureTexLen = NULL;

  CheckErrorCUDA("ProgramCU::DetectionDataFinish");
}

#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)


//////////////////////////////////////////////////////
void ProgramCU::FinishCUDA()
{
  cudaThreadSynchronize();
}

int ProgramCU::CheckErrorCUDA(const char* location)
{
  cudaError_t e = cudaGetLastError();

  if(e)
  {
    if(location)
      fprintf(stderr, "%s:\t",  location);
    fprintf(stderr, "%s\n",  cudaGetErrorString(e));
    // assert(0);
    return 1;
  }
  else {
    return 0; 
  }
}

void __global__ ConvertDOG_Kernel(float* d_result, int width, int height)
{
  int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
  int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;

  if((col < width) && (row < height))
  {
    int index = row * width  + col;
    float value = tex1Dfetch(texData, index);

    d_result[index] = ((col == 0) || (row == 0) || (col == width-1) || (row == height-1)) ? 0.5 : saturate(0.5+20.0*value);
  }
}

void ProgramCU::DisplayConvertDOG(CuTexImage* dog, CuTexImage* out)
{
  if(out->_cuData == NULL)
    return;

  int width = dog->GetImgWidth();
  int height = dog ->GetImgHeight();

  dog->BindTexture(texData);

  dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
  dim3 block(BLOCK_DIM, BLOCK_DIM);

  ConvertDOG_Kernel<<<grid, block>>>((float*) out->_cuData, width, height);
  ProgramCU::CheckErrorCUDA("DisplayConvertDOG");
}

void __global__ ConvertGRD_Kernel(float* d_result, int width, int height)
{
  int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
  int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;

  if((col < width) && (row < height))
  {
    int index = row * width  + col;
    float value = tex1Dfetch(texData, index << 1);

    d_result[index] = ((col == 0) || (row == 0) || (col == width-1) || (row == height-1)) ?	0.0 : saturate(5.0*value);
  }
}

void ProgramCU::DisplayConvertGRD(CuTexImage* got, CuTexImage* out)
{
  if(out->_cuData == NULL)
    return;

  int width = got->GetImgWidth();
  int height = got ->GetImgHeight();

  got->BindTexture(texData);

  dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
  dim3 block(BLOCK_DIM, BLOCK_DIM);

  ConvertGRD_Kernel<<<grid, block>>>((float*) out->_cuData, width, height);
  ProgramCU::CheckErrorCUDA("DisplayConvertGRD");
}

void __global__ ConvertKEY_Kernel(float4* d_result, int width, int height)
{
  int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
  int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;

  if((col < width) && (row < height))
  {
    int index = row * width + col;
    float4 key = tex1Dfetch(texDataF4, index);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    // input: make_float4(result, dx, dy, ds);   =>  dx, dy, ds ... subpixel localizations (otherwise zero)
    //  result = response 16b half float | 14b unused | 2b type
    //  type = FEATURE_TYPE_DARK_BLOB    = 0
    //         FEATURE_TYPE_BRIGHT_BLOB  = 1
    //         FEATURE_TYPE_SADDLE_POINT = 2
    //         FEATURE_TYPE_NONE         = 3

    // extract feature type
    unsigned int typeValue = *((unsigned int *)(&key.x)); // __float_as_uint(key.x); // CUDA7.5
    typeValue = typeValue & 0x00000003u;

  //  int is_key = (typeValue != FEATURE_TYPE_NONE);
#else
    int is_key = ((key.x == 1.0f) || (key.x == -1.0f));
#endif // GPU_SIFT_MODIFIED || GPU_HESSIAN

    int inside = (col > 0) && (row > 0) && (row < height-1) && (col < width-1);
    float value = inside ? saturate(0.5 + 20.0 * tex1Dfetch(texData, index)) : 0.5;

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    float4 result = make_float4(value, value, value, 0.0f);

    switch(typeValue)
    {
      case FEATURE_TYPE_DARK_BLOB:
        result = inside ? make_float4(1.0f, 0.0f, 0.0f, 1.0f) : result;
        break;
      case FEATURE_TYPE_BRIGHT_BLOB:
        result = inside ? make_float4(0.0f, 1.0f, 0.0f, 1.0f) : result;
        break;
      case FEATURE_TYPE_SADDLE_POINT:
        result = inside ? make_float4(0.0f, 0.0f, 1.0f, 1.0f) : result;
        break;
      case FEATURE_TYPE_NONE:
	  default:
        break;
    }

    d_result[index] = result;

    // if((typeValue != HESSIAN_FEATURE_TYPE_NONE) && inside)
    // {
    //   d_result[index-1] = result;
    //   d_result[index+1] = result;
    //   d_result[index-width] = result;
    //   d_result[index+width] = result;
    // }
#else
    d_result[index] = (is_key && inside) ? ((key.x > 0) ? make_float4(1.0f, 0.0f, 0.0f, 1.0f) : make_float4(0.0f, 1.0f, 0.0f, 1.0f)) : make_float4(value, value, value, 1.0f);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  }
}

void ProgramCU::DisplayConvertKEY(CuTexImage* key, CuTexImage* dog, CuTexImage* out)
{
  if(out->_cuData == NULL)
    return;

  int width = key->GetImgWidth();
  int height = key ->GetImgHeight();

  dog->BindTexture(texData);
  key->BindTexture(texDataF4);

  dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
  dim3 block(BLOCK_DIM, BLOCK_DIM);

  ConvertKEY_Kernel<<<grid, block>>>((float4*) out->_cuData, width, height);
}

void __global__ DisplayKeyPoint_Kernel(float4 * d_result, int num)
{
  int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;

  if(idx >= num)
    return;

  float4 key = tex1Dfetch(texDataF4, idx);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  // input in the feature list
  //  key.x: response 8b H | x 24b-14.10
  //  key.y: response 8b L | y 24b-14.10
  //  key.z: 2b type | 14b unused | scale 16b-8.8
  //  key.w: orientation

  // extract x position
  unsigned int tmpValue = *((unsigned int *)(&key.x)); // __float_as_uint(key.x); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
  key.x = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

  // extract y position
  tmpValue = *((unsigned int *)(&key.y)); // __float_as_uint(key.y); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
  key.y = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  d_result[idx] = make_float4(key.x, key.y, 0, 1.0f);
}

void ProgramCU::DisplayKeyPoint(CuTexImage* ftex, CuTexImage* out)
{
  int num = ftex->GetImgWidth();
  int block_width = 64;

  dim3 grid((num + block_width -1) /block_width);
  dim3 block(block_width);

  ftex->BindTexture(texDataF4);
  DisplayKeyPoint_Kernel<<<grid, block>>>((float4*) out->_cuData, num);

  ProgramCU::CheckErrorCUDA("DisplayKeyPoint");
}

void __global__ DisplayKeyBox_Kernel(float4* d_result, int num)
{
  int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;

  if(idx >= num)
    return;

  int kidx = idx / 10;
  int vidx = idx - IMUL(kidx , 10);

  // fetch feature/keypoint
  float4 key = tex1Dfetch(texDataF4, kidx);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  // input in the feature list
  //  key.x: response 8b H | x 24b-14.10
  //  key.y: response 8b L | y 24b-14.10
  //  key.z: 2b type | 14 unused | scale 16b-8.8
  //  key.w: orientation

  // extract x position
  unsigned int tmpValue = *((unsigned int *)(&key.x)); // __float_as_uint(key.x); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
  key.x = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

  // extract y position
  tmpValue = *((unsigned int *)(&key.y)); // __float_as_uint(key.y); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_POSITION_MASK;
  key.y = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_POSITION_PRECISION_BITS);

  // extract scale
  tmpValue = *((unsigned int *)(&key.z)); // __float_as_uint(key.z); // CUDA 7.5

  tmpValue = tmpValue & FIXED_POINT_SCALE_MASK;
  key.z = FIXED_POINT_TO_FLOAT(tmpValue, FIXED_POINT_SCALE_PRECISION_BITS);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  float sz = fabs(key.z * 3.0f);
  ///////////////////////
  float s, c;
  __sincosf(key.w, &s, &c);
  ///////////////////////
  float dx = (vidx == 0) ? 0 : (((vidx <= 4) || (vidx >= 9)) ? sz : -sz);
  float dy = (vidx <= 1) ? 0 : (((vidx <= 2) || (vidx >= 7)) ? -sz : sz);

  float4 pos;

  pos.x = key.x + c * dx - s * dy;
  pos.y = key.y + c * dy + s * dx;
  pos.z = 0;
  pos.w = 1.0f;

  d_result[idx] = pos;
}

void ProgramCU::DisplayKeyBox(CuTexImage* ftex, CuTexImage* out)
{
  int len = ftex->GetImgWidth();
  int block_width = 32;

  dim3 grid((len * 10 + block_width -1) / block_width);
  dim3 block(block_width);

  ftex->BindTexture(texDataF4);

  DisplayKeyBox_Kernel<<<grid, block>>>((float4*) out->_cuData, len * 10);
}

///////////////////////////////////////////////////////////////////
inline void CuTexImage::BindTexture(textureReference& texRef)
{
  cudaBindTexture(NULL, &texRef, _cuData, &texRef.channelDesc, _numBytes);
}

inline void CuTexImage::BindTexture2D(textureReference& texRef)
{
#if defined(SIFTGPU_ENABLE_LINEAR_TEX2D)
  if ((_imgWidth*_numChannel*sizeof(float)) % 32)
    std::cout<<"Warning: Row length should be multiply of 32 !"<<std::endl;

  cudaBindTexture2D(0, &texRef, _cuData, &texRef.channelDesc, _imgWidth, _imgHeight, _imgWidth*_numChannel*sizeof(float));
#else
  cudaChannelFormatDesc desc;
  cudaGetChannelDesc(&desc, _cuData2D);
  cudaBindTextureToArray(&texRef, _cuData2D, &desc);
#endif
}

int ProgramCU::CheckCudaDevice(int device)
{
  int count = 0, device_used;

  if((cudaGetDeviceCount(&count) != cudaSuccess) || (count <= 0))
  {
    ProgramCU::CheckErrorCUDA("CheckCudaDevice");
    return 0;
  }
  else if(count == 1)
  {
    cudaDeviceProp deviceProp;

    if((cudaGetDeviceProperties(&deviceProp, 0) != cudaSuccess) || ((deviceProp.major == 9999) && (deviceProp.minor == 9999)))
    {
      fprintf(stderr, "CheckCudaDevice: no device supporting CUDA.\n");
      return 0;
    }
    else
    {
      GlobalUtil::_MemCapGPU = deviceProp.totalGlobalMem / 1024;
      GlobalUtil::_texMaxDimGL = 32768;

      if(GlobalUtil::_verbose) 
        fprintf(stdout, "NOTE: changing maximum texture dimension to %d\n", GlobalUtil::_texMaxDimGL);
    }
  }

  if((device > 0) && (device < count))  
  {
    cudaSetDevice(device);
    CheckErrorCUDA("cudaSetDevice\n");
  }

  cudaGetDevice(&device_used);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  // we need CC 2.0 at least for feature list construction using atomics and topk selection
  cudaDeviceProp deviceProp;

  cudaGetDeviceProperties(&deviceProp, device_used);

  if(deviceProp.major < 2)
  {
    fprintf(stderr, "CheckCudaDevice: no device supporting CUDA CC 2.X or higher. Your device has just CC %d.%d.\n", deviceProp.major, deviceProp.minor);
    fprintf(stderr, "  Disable GENERATE_FEATURE_LIST_USING_ATOMICS and TOP_K_SELECTION in config.h and rebuild project.\n");
    return 0;
  }
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  if(device != device_used) 
    fprintf(stderr,  "\nERROR:   Cannot set device to %d\n"
        "\nWARNING: Use # %d device instead (out of %d)\n", device, device_used, count);
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////
// siftmatch funtions
//////////////////////////////////////////////////////////////////////////////////////////

#define MULT_TBLOCK_DIMX 128
#define MULT_TBLOCK_DIMY 1
#define MULT_BLOCK_DIMX (MULT_TBLOCK_DIMX)
#define MULT_BLOCK_DIMY (8 * MULT_TBLOCK_DIMY)


texture<uint4, 1, cudaReadModeElementType> texDes1;
texture<uint4, 1, cudaReadModeElementType> texDes2;

void __global__ MultiplyDescriptor_Kernel(int* d_result, int num1, int num2, int3* d_temp)
{
	int idx01 = (blockIdx.y  * MULT_BLOCK_DIMY),  idx02 = (blockIdx.x  * MULT_BLOCK_DIMX);

	int idx1 = idx01 + threadIdx.y, idx2 = idx02 + threadIdx.x;
	__shared__ int data1[17 * 2 * MULT_BLOCK_DIMY];
	int read_idx1 = idx01 * 8 +  threadIdx.x, read_idx2 = idx2 * 8;
	int col4 = threadIdx.x & 0x3, row4 = threadIdx.x >> 2;
	int cache_idx1 = IMUL(row4, 17) + (col4 << 2);

	///////////////////////////////////////////////////////////////
	//Load feature descriptors
	///////////////////////////////////////////////////////////////
#if MULT_BLOCK_DIMY == 16
	uint4 v = tex1Dfetch(texDes1, read_idx1);
	data1[cache_idx1]   = v.x;	data1[cache_idx1+1] = v.y;
	data1[cache_idx1+2] = v.z;	data1[cache_idx1+3] = v.w;
#elif MULT_BLOCK_DIMY == 8
	if(threadIdx.x < 64)
	{
		uint4 v = tex1Dfetch(texDes1, read_idx1);
		data1[cache_idx1]   = v.x;		data1[cache_idx1+1] = v.y;
		data1[cache_idx1+2] = v.z;		data1[cache_idx1+3] = v.w;
	}
#else
#error
#endif
	__syncthreads();

	///
	if(idx2 >= num2) return;
	///////////////////////////////////////////////////////////////////////////
	//compare descriptors

	int results[MULT_BLOCK_DIMY];
#pragma unroll
	for(int i = 0; i < MULT_BLOCK_DIMY; ++i) results[i] = 0;

#pragma unroll
	for(int i = 0; i < 8; ++i)
	{
		uint4 v = tex1Dfetch(texDes2, read_idx2 + i);
		unsigned char* p2 = (unsigned char*)(&v);
#pragma unroll
		for(int k = 0; k < MULT_BLOCK_DIMY; ++k)
		{
			unsigned char* p1 = (unsigned char*) (data1 + k * 34 + i *  4 + (i/4));
			results[k] += 	 ( IMUL(p1[0], p2[0])	+ IMUL(p1[1], p2[1])  
							 + IMUL(p1[2], p2[2])  	+ IMUL(p1[3], p2[3])  
							 + IMUL(p1[4], p2[4])  	+ IMUL(p1[5], p2[5])  
							 + IMUL(p1[6], p2[6])  	+ IMUL(p1[7], p2[7])  
							 + IMUL(p1[8], p2[8])  	+ IMUL(p1[9], p2[9])  
							 + IMUL(p1[10], p2[10])	+ IMUL(p1[11], p2[11])
							 + IMUL(p1[12], p2[12])	+ IMUL(p1[13], p2[13])
							 + IMUL(p1[14], p2[14])	+ IMUL(p1[15], p2[15]));
		}
	}

	int dst_idx = IMUL(idx1, num2)  + idx2;
	if(d_temp)
	{
		int3 cmp_result = make_int3(0, -1, 0);

#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1)
			{
				cmp_result = results[i] > cmp_result.x? 
				make_int3(results[i], idx1 + i, cmp_result.x) : 
				make_int3(cmp_result.x, cmp_result.y, max(cmp_result.z, results[i]));
				d_result[dst_idx + IMUL(i, num2)] = results[i];
			}
		}
		d_temp[ IMUL(blockIdx.y, num2) + idx2] = cmp_result; 
	}else
	{
#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1) d_result[dst_idx + IMUL(i, num2)] = results[i];
		}
	}

}


void ProgramCU::MultiplyDescriptor(CuTexImage* des1, CuTexImage* des2, CuTexImage* texDot, CuTexImage* texCRT)
{
	int num1 = des1->GetImgWidth() / 8;	
	int num2 = des2->GetImgWidth() / 8;
	dim3 grid(	(num2 + MULT_BLOCK_DIMX - 1)/ MULT_BLOCK_DIMX,
		(num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY);
	dim3 block(MULT_TBLOCK_DIMX, MULT_TBLOCK_DIMY);
	texDot->InitTexture( num2,num1);
	if(texCRT) texCRT->InitTexture(num2, (num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY, 32);
	des1->BindTexture(texDes1);
	des2->BindTexture(texDes2);

	MultiplyDescriptor_Kernel<<<grid, block>>>((int*)texDot->_cuData, num1, num2, 
												(texCRT? (int3*)texCRT->_cuData : NULL));
	ProgramCU::CheckErrorCUDA("MultiplyDescriptor");
}

texture<float, 1, cudaReadModeElementType> texLoc1;
texture<float2, 1, cudaReadModeElementType> texLoc2;
struct Matrix33{float mat[3][3];};



void __global__ MultiplyDescriptorG_Kernel(int* d_result, int num1, int num2, int3* d_temp,
										   Matrix33 H, float hdistmax, Matrix33 F, float fdistmax)
{
	int idx01 = (blockIdx.y  * MULT_BLOCK_DIMY);	
	int idx02 = (blockIdx.x  * MULT_BLOCK_DIMX);

	int idx1 = idx01 + threadIdx.y;	
	int idx2 = idx02 + threadIdx.x;
	__shared__ int data1[17 * 2 * MULT_BLOCK_DIMY];
	__shared__ float loc1[MULT_BLOCK_DIMY * 2];
	int read_idx1 = idx01 * 8 +  threadIdx.x ;
	int read_idx2 = idx2 * 8;
	int col4 = threadIdx.x & 0x3, row4 = threadIdx.x >> 2;
	int cache_idx1 = IMUL(row4, 17) + (col4 << 2);
#if MULT_BLOCK_DIMY == 16
	uint4 v = tex1Dfetch(texDes1, read_idx1);
	data1[cache_idx1]   = v.x;
	data1[cache_idx1+1] = v.y;
	data1[cache_idx1+2] = v.z;
	data1[cache_idx1+3] = v.w;
#elif MULT_BLOCK_DIMY == 8
	if(threadIdx.x < 64)
	{
		uint4 v = tex1Dfetch(texDes1, read_idx1);
		data1[cache_idx1]   = v.x;
		data1[cache_idx1+1] = v.y;
		data1[cache_idx1+2] = v.z;
		data1[cache_idx1+3] = v.w;
	}
#else
#error
#endif
	__syncthreads();
	if(threadIdx.x < MULT_BLOCK_DIMY * 2)
	{
		loc1[threadIdx.x] = tex1Dfetch(texLoc1, 2 * idx01 + threadIdx.x);
	}
	__syncthreads();
	if(idx2 >= num2) return;
	int results[MULT_BLOCK_DIMY];
	/////////////////////////////////////////////////////////////////////////////////////////////
	//geometric verification
	/////////////////////////////////////////////////////////////////////////////////////////////
	int good_count = 0;
	float2 loc2 = tex1Dfetch(texLoc2, idx2);
#pragma unroll
	for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
	{

		if(idx1 + i < num1)
		{
			float* loci = loc1 + i * 2;
			float locx = loci[0], locy = loci[1];
			//homography
			float x[3], diff[2];
			x[0] = H.mat[0][0] * locx + H.mat[0][1] * locy + H.mat[0][2];
			x[1] = H.mat[1][0] * locx + H.mat[1][1] * locy + H.mat[1][2];
			x[2] = H.mat[2][0] * locx + H.mat[2][1] * locy + H.mat[2][2];
			diff[0] = fabs(FDIV(x[0], x[2]) - loc2.x);
			diff[1] = fabs(FDIV(x[1], x[2]) - loc2.y);
			if(diff[0] < hdistmax && diff[1] < hdistmax)
			{
				//check fundamental matrix
				float fx1[3], ftx2[3], x2fx1, se; 
				fx1[0] = F.mat[0][0] * locx + F.mat[0][1] * locy + F.mat[0][2];
				fx1[1] = F.mat[1][0] * locx + F.mat[1][1] * locy + F.mat[1][2];
				fx1[2] = F.mat[2][0] * locx + F.mat[2][1] * locy + F.mat[2][2];

				ftx2[0] = F.mat[0][0] * loc2.x + F.mat[1][0] * loc2.y + F.mat[2][0];
				ftx2[1] = F.mat[0][1] * loc2.x + F.mat[1][1] * loc2.y + F.mat[2][1];
				//ftx2[2] = F.mat[0][2] * loc2.x + F.mat[1][2] * loc2.y + F.mat[2][2];

				x2fx1 = loc2.x * fx1[0]  + loc2.y * fx1[1] + fx1[2];
				se = FDIV(x2fx1 * x2fx1, fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1]);
				results[i] = se < fdistmax? 0: -262144;
			}else
			{
				results[i] = -262144;
			}
		}else
		{
			results[i] = -262144;
		}
		good_count += (results[i] >=0);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
	///compare feature descriptors anyway
	/////////////////////////////////////////////////////////////////////////////////////////////
	if(good_count > 0)
	{
#pragma unroll
		for(int i = 0; i < 8; ++i)
		{
			uint4 v = tex1Dfetch(texDes2, read_idx2 + i);
			unsigned char* p2 = (unsigned char*)(&v);
#pragma unroll
			for(int k = 0; k < MULT_BLOCK_DIMY; ++k)
			{
				unsigned char* p1 = (unsigned char*) (data1 + k * 34 + i *  4 + (i/4));
				results[k] += 	 ( IMUL(p1[0], p2[0])	+ IMUL(p1[1], p2[1])  
								 + IMUL(p1[2], p2[2])  	+ IMUL(p1[3], p2[3])  
								 + IMUL(p1[4], p2[4])  	+ IMUL(p1[5], p2[5])  
								 + IMUL(p1[6], p2[6])  	+ IMUL(p1[7], p2[7])  
								 + IMUL(p1[8], p2[8])  	+ IMUL(p1[9], p2[9])  
								 + IMUL(p1[10], p2[10])	+ IMUL(p1[11], p2[11])
								 + IMUL(p1[12], p2[12])	+ IMUL(p1[13], p2[13])
								 + IMUL(p1[14], p2[14])	+ IMUL(p1[15], p2[15]));
			}
		}
	}
	int dst_idx = IMUL(idx1, num2)  + idx2;
	if(d_temp)
	{
		int3 cmp_result = make_int3(0, -1, 0);
#pragma unroll
		for(int i= 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1)
			{
				cmp_result = results[i] > cmp_result.x? 
				make_int3(results[i], idx1 + i, cmp_result.x) : 
				make_int3(cmp_result.x, cmp_result.y, max(cmp_result.z, results[i]));
				d_result[dst_idx + IMUL(i, num2)] = max(results[i], 0);
			}else
			{
				break;
			}
		}
		d_temp[ IMUL(blockIdx.y, num2) + idx2] = cmp_result; 
	}else
	{
#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1) d_result[dst_idx + IMUL(i, num2)] = max(results[i], 0);
			else break;
		}
	}

}


void ProgramCU::MultiplyDescriptorG(CuTexImage* des1, CuTexImage* des2,
		CuTexImage* loc1, CuTexImage* loc2, CuTexImage* texDot, CuTexImage* texCRT,
		float H[3][3], float hdistmax, float F[3][3], float fdistmax)
{
	int num1 = des1->GetImgWidth() / 8;	
	int num2 = des2->GetImgWidth() / 8;
	Matrix33 MatF, MatH;
	//copy the matrix
	memcpy(MatF.mat, F, 9 * sizeof(float));
	memcpy(MatH.mat, H, 9 * sizeof(float));
	//thread blocks
	dim3 grid(	(num2 + MULT_BLOCK_DIMX - 1)/ MULT_BLOCK_DIMX,
		(num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY);
	dim3 block(MULT_TBLOCK_DIMX, MULT_TBLOCK_DIMY);
	//intermediate results
	texDot->InitTexture( num2,num1);
	if(texCRT) texCRT->InitTexture( num2, (num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY, 3);
	loc1->BindTexture(texLoc1);	
	loc2->BindTexture(texLoc2);
	des1->BindTexture(texDes1);	
	des2->BindTexture(texDes2);
	MultiplyDescriptorG_Kernel<<<grid, block>>>((int*)texDot->_cuData, num1, num2, 
												(texCRT? (int3*)texCRT->_cuData : NULL),
												MatH, hdistmax, MatF, fdistmax);
}


texture<int,  1, cudaReadModeElementType> texDOT;

#define ROWMATCH_BLOCK_WIDTH 32
#define ROWMATCH_BLOCK_HEIGHT 1

void __global__  RowMatch_Kernel(int*d_dot, int* d_result, int num2, float distmax, float ratiomax)
{
#if ROWMATCH_BLOCK_HEIGHT == 1
	__shared__ int dotmax[ROWMATCH_BLOCK_WIDTH];
	__shared__ int dotnxt[ROWMATCH_BLOCK_WIDTH];
	__shared__ int dotidx[ROWMATCH_BLOCK_WIDTH];
	int	row = blockIdx.y;
#else
	__shared__ int x_dotmax[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];
	__shared__ int x_dotnxt[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];
	__shared__ int x_dotidx[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];
	int*	dotmax = x_dotmax[threadIdx.y];
	int*	dotnxt = x_dotnxt[threadIdx.y];
	int*	dotidx = x_dotidx[threadIdx.y];
	int row = IMUL(blockIdx.y, ROWMATCH_BLOCK_HEIGHT) + threadIdx.y;
#endif

	int base_address = IMUL(row , num2);
	int t_dotmax = 0, t_dotnxt = 0, t_dotidx = -1;
	for(int i = 0; i < num2; i += ROWMATCH_BLOCK_WIDTH)
	{
		if(threadIdx.x + i < num2)
		{
			int v = tex1Dfetch(texDOT, base_address + threadIdx.x + i);//d_dot[base_address + threadIdx.x + i];//
			bool test = v > t_dotmax;
			t_dotnxt = test? t_dotmax : max(t_dotnxt, v);
			t_dotidx = test? (threadIdx.x + i) : t_dotidx;
			t_dotmax = test? v: t_dotmax;
		}
		__syncthreads();
	}
	dotmax[threadIdx.x] = t_dotmax;
	dotnxt[threadIdx.x] = t_dotnxt;
	dotidx[threadIdx.x] = t_dotidx;
	__syncthreads();
	
#pragma unroll
	for(int step = ROWMATCH_BLOCK_WIDTH/2; step >0; step /= 2)
	{
		if(threadIdx.x < step)
		{
			int v1 = dotmax[threadIdx.x], v2 = dotmax[threadIdx.x + step];
			bool test =  v2 > v1;
			dotnxt[threadIdx.x] = test? max(v1, dotnxt[threadIdx.x + step]) :max(dotnxt[threadIdx.x], v2);
			dotidx[threadIdx.x] = test? dotidx[threadIdx.x + step] : dotidx[threadIdx.x];
			dotmax[threadIdx.x] = test? v2 : v1;
		}
		__syncthreads();
	}
	if(threadIdx.x == 0)
	{
		float dist =  acos(min(dotmax[0] * 0.000003814697265625f, 1.0));
		float distn = acos(min(dotnxt[0] * 0.000003814697265625f, 1.0));
		//float ratio = dist / distn;
		d_result[row] = (dist < distmax) && (dist < distn * ratiomax) ? dotidx[0] : -1;//?  : -1;
	}

}


void ProgramCU::GetRowMatch(CuTexImage* texDot, CuTexImage* texMatch, float distmax, float ratiomax)
{
	int num1 = texDot->GetImgHeight();
	int num2 = texDot->GetImgWidth();
	dim3 grid(1, num1/ROWMATCH_BLOCK_HEIGHT);
	dim3 block(ROWMATCH_BLOCK_WIDTH, ROWMATCH_BLOCK_HEIGHT);
	texDot->BindTexture(texDOT);
	RowMatch_Kernel<<<grid, block>>>((int*)texDot->_cuData,
		(int*)texMatch->_cuData, num2, distmax, ratiomax);
}

#define COLMATCH_BLOCK_WIDTH 32

//texture<int3,  1, cudaReadModeElementType> texCT;

void __global__  ColMatch_Kernel(int3*d_crt, int* d_result, int height, int num2, float distmax, float ratiomax)
{
	int col = COLMATCH_BLOCK_WIDTH * blockIdx.x + threadIdx.x;
	if(col >= num2) return;
	int3 result = d_crt[col];//tex1Dfetch(texCT, col);
	int read_idx = col + num2;
	for(int i = 1; i < height; ++i, read_idx += num2)
	{
		int3 temp = d_crt[read_idx];//tex1Dfetch(texCT, read_idx);
		result = result.x < temp.x?
			make_int3(temp.x, temp.y, max(result.x, temp.z)) :
			make_int3(result.x, result.y, max(result.z, temp.x));
	}

	float dist =  acos(min(result.x * 0.000003814697265625f, 1.0));
	float distn = acos(min(result.z * 0.000003814697265625f, 1.0));
		//float ratio = dist / distn;
	d_result[col] = (dist < distmax) && (dist < distn * ratiomax) ? result.y : -1;//?  : -1;

}

void ProgramCU::GetColMatch(CuTexImage* texCRT, CuTexImage* texMatch, float distmax, float ratiomax)
{
	int height = texCRT->GetImgHeight();
	int num2 = texCRT->GetImgWidth();
	//texCRT->BindTexture(texCT);
    dim3 grid((num2 + COLMATCH_BLOCK_WIDTH -1) / COLMATCH_BLOCK_WIDTH);
    dim3 block(COLMATCH_BLOCK_WIDTH);
	ColMatch_Kernel<<<grid, block>>>((int3*)texCRT->_cuData, (int*) texMatch->_cuData, height, num2, distmax, ratiomax);
}

#endif
