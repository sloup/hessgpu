////////////////////////////////////////////////////////////////////////////
//	File:		PyramidCU.cpp
//	Author:		Changchang Wu
//	Description : implementation of the PyramidCU class.
//				CUDA-based implementation of SiftPyramid
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
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <math.h>
using namespace std;

#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "CuTexImage.h" 
#include "SiftGPU.h"
#include "SiftPyramid.h"
#include "ProgramCU.h"
#include "PyramidCU.h"
#ifdef GPU_HESSIAN
#include "ShaderMan.h"
#endif // GPU_HESSIAN

//#include "imdebug/imdebuggl.h"
//#pragma comment (lib, "../lib/imdebug.lib")



#define USE_TIMING()                                       \
	double t, t0, tt;
#define OCTAVE_START()                                     \
	if(GlobalUtil::_timingO)                               \
	{                                                      \
		t = t0 = CLOCK();                                  \
		cout << "#" << octave+_down_sample_factor << "\t"; \
	}
#define LEVEL_FINISH()                                     \
	if(GlobalUtil::_timingL)                               \
	{                                                      \
		ProgramCU::FinishCUDA();                           \
		tt = CLOCK();                                      \
        cout << (tt-t) << "ms \t";                         \
		t = CLOCK();                                       \
	}
#define OCTAVE_FINISH()	                                   \
	if(GlobalUtil::_timingO)                               \
		cout << "|\t" << (CLOCK()-t0) << "ms" << endl;


PyramidCU::PyramidCU(SiftParam& sp) : SiftPyramid(sp)
{
	_allPyramid = NULL;
	_histoPyramidTex = NULL;
	_featureTex = NULL;
	_descriptorTex = NULL;
	_orientationTex = NULL;
	_bufferPBO = 0;
    _bufferTEX = NULL;
	_inputTex = new CuTexImage();

    /////////////////////////
    InitializeContext();
}

PyramidCU::~PyramidCU()
{
	DestroyPerLevelData();
	DestroySharedData();
	DestroyPyramidData();

	if(_inputTex)
		delete _inputTex;
    if(_bufferPBO)
		glDeleteBuffers(1, &_bufferPBO);
    if(_bufferTEX)
		delete _bufferTEX;
}

void PyramidCU::InitializeContext()
{
    GlobalUtil::InitGLParam(1);
    GlobalUtil::_GoodOpenGL = max(GlobalUtil::_GoodOpenGL, 1);

#ifdef GPU_HESSIAN
	if (GlobalUtil::_UseSiftGPUEX == 1)
	  ShaderMan::InitShaderMan(param);
#endif // GPU_HESSIAN
}

void PyramidCU::InitPyramid(int w, int h, int ds)
{
	int wp, hp;
	int toobig = 0;

	if(ds == 0)
	{
		//
		TruncateWidth(w);
		////
		_down_sample_factor = 0;
		if(GlobalUtil::_octave_min_default >= 0)
		{
			wp = w >> _octave_min_default;
			hp = h >> _octave_min_default;
		}
		else
		{
			//can't upsample by more than 8
			_octave_min_default = max(-3, _octave_min_default);
			//
			wp = w << (-_octave_min_default);
			hp = h << (-_octave_min_default);
		}
		_octave_min = _octave_min_default;
	}
	else
	{
		//must use 0 as _octave_min; 
		_octave_min = 0;
		_down_sample_factor = ds;
		w >>= ds;
		h >>= ds;

		TruncateWidth(w);

		wp = w;
		hp = h;
	}

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
	if((wp > GlobalUtil::_texMaxDim) || (hp > GlobalUtil::_texMaxDim))
	{
		if(GlobalUtil::_AutoImageDownScaling)
		{
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

			while((wp > GlobalUtil::_texMaxDim) || (hp > GlobalUtil::_texMaxDim))
			{
				_octave_min ++;
				wp >>= 1;
				hp >>= 1;
				toobig = 1;
			}

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
		}
		else
		{
			std::cerr << "PyramidCU::InitPyramid(): Max texture dimension exceeded. Try to use -ads option to auto-downsamping input image to fit max texture dimension." << std::endl; 
			exit(EXIT_FAILURE);
		}
	}
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

	while((GlobalUtil::_MemCapGPU > 0) && GlobalUtil::_FitMemoryCap && ((wp >_pyramid_width) || (hp > _pyramid_height)) && 
		(max(max(wp, hp), max(_pyramid_width, _pyramid_height)) > 1024 * sqrt(GlobalUtil::_MemCapGPU / 110.0)))
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 2;
	}

	if(toobig && GlobalUtil::_verbose && (_octave_min > 0))
	{
		std::cout<<((toobig == 2) ? "[**SKIP OCTAVES**]:\tExceeding Memory Cap (-nomc)\n" :
					"[**SKIP OCTAVES**]:\tReaching the dimension limit(-maxd)!\n");
	}
	//ResizePyramid(wp, hp);
	if((wp == _pyramid_width) && (hp == _pyramid_height) && _allocated)
	{
		FitPyramid(wp, hp);
	}
	else if(GlobalUtil::_ForceTightPyramid || (_allocated == 0))
	{
		ResizePyramid(wp, hp);
	}
	else if( (wp > _pyramid_width) || (hp > _pyramid_height))
	{
		ResizePyramid(max(wp, _pyramid_width), max(hp, _pyramid_height));
		if((wp < _pyramid_width) || (hp < _pyramid_height))
			FitPyramid(wp, hp);
	}
	else
	{
		//try use the pyramid allocated for large image on small input images
		FitPyramid(wp, hp);
	}
}

void PyramidCU::ResizePyramid(int w, int h)
{
	unsigned int totalkb = 0;

	if((_pyramid_width == w) && (_pyramid_height == h) && _allocated)
		return;

	if((w > GlobalUtil::_texMaxDim) || (h > GlobalUtil::_texMaxDim))
		return;

	if(GlobalUtil::_verbose && GlobalUtil::_timingS)
		std::cout << "[Allocate Pyramid]:\t" << w << "x" << h << endl;

	//first octave does not change
	_pyramid_octave_first = 0;
	
	//compute # of octaves

	int input_sz = min(w, h);
	_pyramid_width = w;
	_pyramid_height = h;

	//reset to preset parameters

	int _octave_num_new  = GlobalUtil::_octave_num_default;

	if(_octave_num_new < 1) 
	{
		_octave_num_new = (int) floor (log ( double(input_sz))/log(2.0)) - 3 ;
		if(_octave_num_new < 1)
			_octave_num_new = 1;
	}

	if(_pyramid_octave_num != _octave_num_new)
	{
		//destroy the original pyramid if the # of octave changes
		if(_octave_num > 0)
		{
			DestroyPerLevelData();
			DestroyPyramidData();
		}
		_pyramid_octave_num = _octave_num_new;
	}

	_octave_num = _pyramid_octave_num;

	int noct = _octave_num;
	int nlev = param._level_num;

	//	//initialize the pyramid
	if(_allPyramid == NULL)
		_allPyramid = new CuTexImage[ noct* nlev * DATA_NUM];

	CuTexImage *gus = GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	CuTexImage *dog = GetBaseLevel(_octave_min, DATA_DOG);
	CuTexImage *got = GetBaseLevel(_octave_min, DATA_GRAD);
	CuTexImage *key = GetBaseLevel(_octave_min, DATA_KEYPOINT);

	////////////there could be "out of memory" happening during the allocation

	for(int octave = 0; octave<noct; octave++)
	{
		int wa = ((w + 3) / 4) * 4;

#ifdef GPU_HESSIAN
		totalkb += ((nlev *8 -12)* (wa * h) * 4 / 1024);
#else
		totalkb += ((nlev *8 -19)* (wa * h) * 4 / 1024);
#endif // GPU_HESSIAN
		for(int level=0; level<nlev; level++, gus++, dog++, got++, key++)
		{
			gus->InitTexture(wa, h); //nlev
#ifdef GPU_HESSIAN
			dog->InitTexture(wa, h);  //nlev
			if((level >= 1) && (level <= param._dog_level_num))
			{
				got->InitTexture(wa, h, 2); //2 * (nlev - 2) = 2 * nlev - 4
				got->InitTexture2D();

				key->InitTexture(wa, h, 4); // nlev - 2 ; 4 * nlev - 8
			}				
#else
			if(level == 0)
				continue;
			dog->InitTexture(wa, h);  //nlev -1
			if(	level >= 1 && level < 1 + param._dog_level_num)
			{
				got->InitTexture(wa, h, 2); //2 * nlev - 6
				got->InitTexture2D();
			}
			if(level > 1 && level < nlev -1)
				key->InitTexture(wa, h, 4); // nlev -3 ; 4 * nlev - 12
#endif // GPU_HESSIAN
		}
		w >>= 1;
		h >>= 1;
	}

	totalkb += ResizeFeatureStorage();

	if(ProgramCU::CheckErrorCUDA("ResizePyramid"))
		SetFailStatus(); 

    //if(GetSucessStatus())
	_allocated = 1;

	if(GlobalUtil::_verbose && GlobalUtil::_timingS)
		std::cout << "[Allocate Pyramid]:\t" << (totalkb/1024) << "MB\n";
}

void PyramidCU::FitPyramid(int w, int h)
{
	_pyramid_octave_first = 0;

	_octave_num  = GlobalUtil::_octave_num_default;

	int _octave_num_max = max(1, (int) floor (log ( double(min(w, h)))/log(2.0))  -3 );

	if((_octave_num < 1) || (_octave_num > _octave_num_max)) 
	{
		_octave_num = _octave_num_max;
	}

	int pw = _pyramid_width >> 1;
	int ph = _pyramid_height >> 1;

	while((_pyramid_octave_first+_octave_num < _pyramid_octave_num) && (pw >= w) && (ph >= h))
	{
		_pyramid_octave_first++;
		pw >>= 1;
		ph >>= 1;
	}

	//////////////////
	int nlev = param._level_num;
	CuTexImage *gus = GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	CuTexImage *dog = GetBaseLevel(_octave_min, DATA_DOG);
	CuTexImage *got = GetBaseLevel(_octave_min, DATA_GRAD);
	CuTexImage *key = GetBaseLevel(_octave_min, DATA_KEYPOINT);

	for(int octave = 0; octave < _octave_num; octave++)
	{
		int wa = ((w + 3) / 4) * 4;

		for(int level = 0; level < nlev; level++, gus++, dog++, got++, key++)
		{
			gus->InitTexture(wa, h); //nlev
#ifdef GPU_HESSIAN
			dog->InitTexture(wa, h);  //nlev
			if((level >= 1) && (level <= param._dog_level_num))
			{
				got->InitTexture(wa, h, 2); //nlev - 2; 2 * nlev - 4
				got->InitTexture2D();

				key->InitTexture(wa, h, 4); // nlev -2 ; 4 * nlev - 8
			}
#else
			if(level == 0)
				continue;
			dog->InitTexture(wa, h);  //nlev -1
			if(	level >= 1 && level < 1 + param._dog_level_num)
			{
				got->InitTexture(wa, h, 2); //2 * nlev - 6
				got->InitTexture2D();
			}
			if(level > 1 && level < nlev -1)
				key->InitTexture(wa, h, 4); // nlev -3 ; 4 * nlev - 12
#endif // GPU_HESSIAN
		}
		w >>= 1;
		h >>= 1;
	}
}

int PyramidCU::CheckCudaDevice(int device)
{
  return ProgramCU::CheckCudaDevice(device);
}

void PyramidCU::SetLevelFeatureNum(int idx, int fcount)
{
	_featureTex[idx].InitTexture(fcount, 1, 4);
	_levelFeatureNum[idx] = fcount;
}

int PyramidCU::ResizeFeatureStorage()
{
	int totalkb = 0;

	if(_levelFeatureNum == NULL)
		_levelFeatureNum = new int[_octave_num * param._dog_level_num];

	std::fill(_levelFeatureNum, _levelFeatureNum+_octave_num * param._dog_level_num, 0);

	int wmax = GetBaseLevel(_octave_min)->GetImgWidth();
	int hmax = GetBaseLevel(_octave_min)->GetImgHeight();
	int whmax = max(wmax, hmax);

	//
	int num = (int)ceil(log(double(whmax))/log(4.0));

	if( _hpLevelNum != num)
	{
		_hpLevelNum = num;
		if(_histoPyramidTex )
			delete [] _histoPyramidTex;
		_histoPyramidTex = new CuTexImage[_hpLevelNum];
	}

	for(int i = 0, w = 1; i < _hpLevelNum; i++)
	{
		_histoPyramidTex[i].InitTexture(w, whmax, 4);
		w <<= 2;
	}

	// (4 ^ (_hpLevelNum) -1 / 3) pixels
	totalkb += (((1 << (2 * _hpLevelNum)) -1) / 3 * 16 / 1024);

	//initialize the feature texture
	int n = _octave_num * param._dog_level_num;

	if(_featureTex == NULL)
		_featureTex = new CuTexImage[n];

	if((GlobalUtil::_MaxOrientation > 1) && (GlobalUtil::_OrientationPack2 == 0) && (_orientationTex == NULL))
		_orientationTex = new CuTexImage[n];

	int idx = 0;

	for(int octave = 0; octave < _octave_num; octave++)
	{
		CuTexImage *tex = GetBaseLevel(octave+_octave_min);
		int fmax = int(tex->GetImgWidth() * tex->GetImgHeight()*GlobalUtil::_MaxFeaturePercent);

		if(fmax > GlobalUtil::_MaxLevelFeatureNum)
			fmax = GlobalUtil::_MaxLevelFeatureNum;
		else if(fmax < 32)
			fmax = 32;	//give it at least a space of 32 feature

#ifdef GPU_HESSIAN
		for(int level = 1; level <= param._dog_level_num; level++, idx++)
#else
		for(int level = 0; level < param._dog_level_num; level++, idx++)
#endif // GPU_HESSIAN
		{
			_featureTex[idx].InitTexture(fmax, 1, 4);
			totalkb += fmax * 16 / 1024;

			if((GlobalUtil::_MaxOrientation > 1) && (GlobalUtil::_OrientationPack2 == 0))
			{
				_orientationTex[idx].InitTexture(fmax, 1, 4);
				totalkb += fmax * 16 /1024;
			}
		}
	}

	//this just need be initialized once
	if(_descriptorTex == NULL)
	{
		//initialize feature texture pyramid
		int fmax = _featureTex->GetImgWidth();
		_descriptorTex = new CuTexImage;
		totalkb += (fmax / 2);
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
        int descriptorSize = (GlobalUtil::_HalfSIFT) ? 64 : 128;
		_descriptorTex->InitTexture(fmax * descriptorSize, 1, 1);
#else
		_descriptorTex->InitTexture(fmax * 128, 1, 1);
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
	}
	else
	{
		totalkb += _descriptorTex->GetDataSize()/1024;
	}
	return totalkb;
}

void PyramidCU::GetFeatureDescriptors() 
{
  // descriptors...
  float *pd = &_descriptor_buffer[0];
  vector<float> descriptor_buffer2;

  // use another buffer if we need to re-order the descriptors
  if(_keypoint_index.size() > 0)
  {
    descriptor_buffer2.resize(_descriptor_buffer.size());
    pd = &descriptor_buffer2[0];
  }

  CuTexImage *got, *ftex = _featureTex;
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  int descriptorSize = (GlobalUtil::_HalfSIFT) ? 64 : 128;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

  for(int octave = 0, idx = 0; octave < _octave_num; octave++)
  {
    got = GetBaseLevel(octave + _octave_min, DATA_GRAD) + 1;

#ifdef GPU_HESSIAN
    for(int level = 1; level <= param._dog_level_num; level++, ftex++, idx++, got++)
#else
    for(int level = 0; level < param._dog_level_num; level++, ftex++, idx++, got++)
#endif // GPU_HESSIAN
    {
      if(_levelFeatureNum[idx] == 0)
        continue;

      ProgramCU::ComputeDescriptor(ftex, got, _descriptorTex, IsUsingRectDescription()); // process

      _descriptorTex->CopyToHost(pd); // readback descriptor

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
      pd += descriptorSize*_levelFeatureNum[idx];
#else
      pd += 128*_levelFeatureNum[idx];
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
    }
  }

  if(GlobalUtil::_timingS)
    ProgramCU::FinishCUDA();

  if(_keypoint_index.size() > 0)
  {
    // put the descriptor back to the original order for keypoint list.
    for(int i = 0; i < _featureNum; ++i)
    {
      int index = _keypoint_index[i];
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
      memcpy(&_descriptor_buffer[index*descriptorSize], &descriptor_buffer2[i*descriptorSize], descriptorSize * sizeof(float));
#else
      memcpy(&_descriptor_buffer[index*128], &descriptor_buffer2[i*128], 128 * sizeof(float));
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
    }
  }

  if(ProgramCU::CheckErrorCUDA("PyramidCU::GetFeatureDescriptors"))
    SetFailStatus();
}

void PyramidCU::GenerateFeatureListTex()
{
  vector<float> list;
  int idx = 0;
  const double twopi = 2.0 * PI;
  float sigma_half_step = powf(2.0f, 0.5f / param._dog_level_num);
  float octave_sigma = (_octave_min >= 0) ? float(1<<_octave_min) : 1.0f/(1<<(-_octave_min));
  float offset = GlobalUtil::_LoweOrigin ? 0.0f : 0.5f; 
  //int cc = 0;

  if(_down_sample_factor > 0)
    octave_sigma *= float(1<<_down_sample_factor); 

  _keypoint_index.resize(0); // should already be 0

  for(int octave = 0; octave < _octave_num; octave++, octave_sigma*=2.0f)
  {
#ifdef GPU_HESSIAN
    for(int level = 1; level <= param._dog_level_num; level++, idx++)
#else
    for(int level = 0; level < param._dog_level_num; level++, idx++)
#endif // GPU_HESSIAN
    {
      list.resize(0);

#ifdef GPU_HESSIAN
      float level_sigma = param.GetLevelSigma(level + param._level_min) * octave_sigma;
#else
      float level_sigma = param.GetLevelSigma(level + param._level_min + 1) * octave_sigma;
#endif // GPU_HESSIAN

      float sigma_min = level_sigma / sigma_half_step;
      float sigma_max = level_sigma * sigma_half_step;

      int fcount = 0 ;

      for(int k = 0; k < _featureNum; k++)
      {
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
        SiftKeypoint *key = (SiftKeypoint *)&_keypoint_buffer[k*SIFT_KEYPOINT_ITEMS];

        float sigmak = key->s; 

        if(IsUsingRectDescription())
          sigmak = min(key->s, key->o) / 12.0f; 

#ifdef GPU_HESSIAN
        if( ((sigmak >= sigma_min) && (sigmak < sigma_max))
             || ((sigmak < sigma_min) && (octave == 0) && (level == 1))                                      // first level of the first octave
             || ((sigmak > sigma_max) && (octave == _octave_num -1) && (level == param._dog_level_num))      // last level of the last octave
           )
#else
        if( ((sigmak >= sigma_min) && (sigmak < sigma_max))
             || ((sigmak < sigma_min) && (octave == 0) && (level == 0))
             || ((sigmak > sigma_max) && (octave == _octave_num -1) && (level == param._dog_level_num - 1))
           )
#endif // GPU_HESSIAN
        {
          float fX, fY;
          float fScale, fOrientation;

          fX = (key->x - offset) / octave_sigma + 0.5f;
          fY = (key->y - offset) / octave_sigma + 0.5f;
          if(IsUsingRectDescription())
          {
            fScale = key->s / octave_sigma;
            fOrientation = key->o / octave_sigma;
          }
          else
          {
            fScale = key->s / octave_sigma;
            fOrientation = (float)fmod(twopi-key->o, twopi);
          }

          // pack data into fixed point
          // input:
          //   fx, fY, fScale, fOrientation
          // output in the feature list d_list
          //   key.x: response 8b H | x 24b-14.10
          //   key.y: response 8b L | y 24b-14.10
          //   key.z: 2b type | 14b unused | scale 16b-8.8 
          //   key.w: orientation

          unsigned int posX = (unsigned int)FLOAT_TO_FIXED_POINT(fX, FIXED_POINT_POSITION_PRECISION_BITS);
          posX = posX & FIXED_POINT_POSITION_MASK;
          unsigned int posY = (unsigned int)FLOAT_TO_FIXED_POINT(fY, FIXED_POINT_POSITION_PRECISION_BITS);
          posY = posY & FIXED_POINT_POSITION_MASK;

		  // response is not used since we already have keypoints -> bits are cleared
		  // responseUShort = halfFloat(key->response);
          // posX = posX | (responseUShort & FIXED_POINT_RESPONSE_MASK);
          // posY = posY | ((responseUShort << 8) & FIXED_POINT_RESPONSE_MASK);

          *((unsigned int *)&fX) = posX;
          *((unsigned int *)&fY) = posY;

          unsigned int scale = (unsigned int)(FLOAT_TO_FIXED_POINT(fScale, FIXED_POINT_SCALE_PRECISION_BITS));
          scale = scale & FIXED_POINT_SCALE_MASK;

          // type is not important
		  // scale = scale | ((key->level & 0x00000003u) << 30);

          *((unsigned int *)&fScale) = scale;

          //if(key->level != (level-1)+octave*param._dog_level_num)
          //    cc++;

          // add this keypoint to the list
          list.push_back(fX);
          list.push_back(fY);
          list.push_back(fScale);
          list.push_back(fOrientation);
#else
        float *key = &_keypoint_buffer[k*4];

        float sigmak = key[2]; 

        if(IsUsingRectDescription())
          sigmak = min(key[2], key[3]) / 12.0f; 

        if( ((sigmak >= sigma_min) && (sigmak < sigma_max))
             || ((sigmak < sigma_min) && (octave == 0) && (level == 0))
             || ((sigmak > sigma_max) && (octave == _octave_num -1) && (level == param._dog_level_num - 1))
           )
        {
          // add this keypoint to the list
          list.push_back((key[0] - offset) / octave_sigma + 0.5f);
          list.push_back((key[1] - offset) / octave_sigma + 0.5f);
          if(IsUsingRectDescription())
          {
            list.push_back(key[2] / octave_sigma);
            list.push_back(key[3] / octave_sigma);
          }
          else
          {
            list.push_back(key[2] / octave_sigma);
            list.push_back((float)fmod(twopi-key[3], twopi));
          }
#endif // GPU_HESSIAN || GPU_SIFT_MODIIFED

          fcount ++;
          // save the index of keypoints
          _keypoint_index.push_back(k);
        }
      }

      _levelFeatureNum[idx] = fcount;

      if(fcount == 0)
        continue;

      CuTexImage *ftex = _featureTex + idx;

      SetLevelFeatureNum(idx, fcount);
      ftex->CopyFromHost(&list[0]);
    }
  }

  if(GlobalUtil::_verbose)
  {
    // std::cout << "wrongly classified: " << cc << std::endl;
    std::cout << "#Features:\t" << _featureNum << "\n";
  }
}

void PyramidCU::ReshapeFeatureListCPU() 
{
  int szmax = 0, sz;
  int n = param._dog_level_num*_octave_num;

  for(int i = 0; i < n; i++) 
  {
    sz = _levelFeatureNum[i];
    if(sz > szmax)
      szmax = sz;
  }

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  float *buffer = new float[szmax*4 + szmax*4*6];
#else
  float *buffer = new float[szmax*16];
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  float *buffer1 = buffer;
  float *buffer2 = buffer + szmax*4;

  _featureNum = 0;

#ifdef NO_DUPLICATE_DOWNLOAD
  const double twopi = 2.0 * PI;
  _keypoint_buffer.resize(0);

  float octave_sigma = (_octave_min >= 0) ? float(1<<_octave_min) : 1.0f/(1<<(-_octave_min));
  if(_down_sample_factor > 0)
    octave_sigma *= float(1<<_down_sample_factor);

  float offset = GlobalUtil::_LoweOrigin ? 0 : 0.5f;
#endif

  for(int i = 0; i < n; i++)
  {
    if(_levelFeatureNum[i] == 0)
      continue;

    _featureTex[i].CopyToHost(buffer1);

    int fcount = 0;
    float *src = buffer1;
    float *dst = buffer2;
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    const static double factor = 2.0 * PI / 255.0;
#else
    const static double factor = 2.0 * PI / 65535.0;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

    for(int j = 0; j < _levelFeatureNum[i]; j++, src+=4)
    {
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED 

      unsigned int *key = (unsigned int *)src;

      //  key.x: response 8b H | x 24b-14.10
      //  key.y: response 8b L | y 24b-14.10
      //  key.z: 2b type | 3b orientations count | 11b unused | scale 16b-8.8
      //  key.w: 8b orientation1 | 8b orientation2 | 8b orientation3 | 8b orientation4

      unsigned int count = (key[2] >> 27) & 0x00000007u;

      if(count != 0) // if count is zero then key.w contains directly the unpacked orientation
      {
        for(unsigned int idx=0; idx<count; idx++)
        {
          unsigned int orientation = (key[3] >> 8*idx) & 0x000000FFu;

          dst[0] = src[0];
          dst[1] = src[1];
          dst[2] = src[2];
          dst[3] = float(factor * orientation);

          fcount++;
          dst += 4;
		}
	  }
#else
      unsigned short *orientations = (unsigned short *) (&src[3]);

      if(orientations[0] != 65535)
      {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = float(factor * orientations[0]);

        fcount++;
        dst += 4;

        if((orientations[1] != 65535) && (orientations[1] != orientations[0]))
        {
          dst[0] = src[0];
          dst[1] = src[1];
          dst[2] = src[2];
          dst[3] = float(factor * orientations[1]);

          fcount++;
          dst += 4;
        }
      }
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
    }

    // texture size
    SetLevelFeatureNum(i, fcount);
    _featureTex[i].CopyFromHost(buffer2);

    if(fcount == 0)
      continue;

#ifdef NO_DUPLICATE_DOWNLOAD

    float oss = octave_sigma * (1 << (i / param._dog_level_num));

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    _keypoint_buffer.resize((_featureNum + fcount) * SIFT_KEYPOINT_ITEMS);

    float *ds = &_keypoint_buffer[_featureNum * SIFT_KEYPOINT_ITEMS];
#else
    _keypoint_buffer.resize((_featureNum + fcount) * 4);

    float *ds = &_keypoint_buffer[_featureNum * 4];
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
    float *fs = buffer2;

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
    for(int k = 0;  k < fcount; k++, ds+=SIFT_KEYPOINT_ITEMS, fs+=4)
#else
    for(int k = 0;  k < fcount; k++, ds+=4, fs+=4)
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
    {
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
      // input from the level feature list fs
      //  fs[0]: response 8b H | x 24b-14.10
      //  fs[1]: response 8b L | y 24b-14.10
      //  fs[2]: 2b type | 14b unused | scale 16b-8.8 
      //  fs[3]: orientation

      // output to the keypoint buffer ds
      //  ds[0]: position X
      //  ds[1]: position Y
      //  ds[2]: scale
      //  ds[3]: orientation
      //  ds[4]: response
      //  ds[5]: level 16b | unused 14b| type 2b

      // extract x position
      unsigned int xValue = *((unsigned int *)(fs+0));
      unsigned int responseH = xValue & FIXED_POINT_RESPONSE_MASK;
      xValue = xValue & FIXED_POINT_POSITION_MASK;
      float posX = FIXED_POINT_TO_FLOAT(xValue, FIXED_POINT_POSITION_PRECISION_BITS);

      // extract y position
      unsigned int yValue = *((unsigned int *)(fs+1));
      unsigned int responseL = yValue & FIXED_POINT_RESPONSE_MASK;
      yValue = yValue & FIXED_POINT_POSITION_MASK;
      float posY = FIXED_POINT_TO_FLOAT(yValue, FIXED_POINT_POSITION_PRECISION_BITS);

      // extract response
      unsigned short usResponse = (unsigned short)((responseH >> 16) | (responseL >> 24));
      unsigned int intResponse = half2float(usResponse);
      float response = *((float *)(&intResponse));

      // extract scale
      unsigned int scaleValue = *((unsigned int *)(fs+2));
      unsigned int type = (scaleValue & 0xC0000000u) >> 30;
      scaleValue = scaleValue & FIXED_POINT_SCALE_MASK;
      float scale = FIXED_POINT_TO_FLOAT(scaleValue, FIXED_POINT_SCALE_PRECISION_BITS);

      posX = oss*(posX-0.5f) + offset;      // x
      posY = oss*(posY-0.5f) + offset;      // y
      scale = oss*scale;                    // scale

      unsigned int level = i;
      //unsigned int level = (i % param._dog_level_num) + 1;
	  //unsigned int octave = i / param._dog_level_num;
      //level = octave*param._level_num + level;

      // store computed values
      ds[0] = posX;
      ds[1] = posY;
      ds[2] = scale;
      ds[3] = (float)fmod(twopi-fs[3], twopi);   // orientation, mirrored
      ds[4] = response;
      ((unsigned short *)(ds+5))[0] = (unsigned short)level;
      ((unsigned short *)(ds+5))[1] = (unsigned short)type;
#else
      ds[0] = oss*(fs[0]-0.5f) + offset;       // x
      ds[1] = oss*(fs[1]-0.5f) + offset;       // y
      ds[2] = oss*fs[2];                       // scale
      ds[3] = (float)fmod(twopi-fs[3], twopi); // orientation, mirrored
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
    }
#endif // NO_DUPLICATE_DOWNLOAD
    _featureNum += fcount;
  }

  delete[] buffer;

  if(GlobalUtil::_verbose)
  {
    std::cout<<"#Features MO:\t"<<_featureNum<<endl;
  }
}

void PyramidCU::GenerateFeatureDisplayVBO() 
{
  // it is weried that this part is very slow.
  // use a big VBO to save all the SIFT box vertices
  int nvbo = _octave_num * param._dog_level_num;

  if(_featureDisplayVBO == NULL)
  {
    // initialize the vbos
    _featureDisplayVBO = new GLuint[nvbo];
    _featurePointVBO = new GLuint[nvbo];

    glGenBuffers(nvbo, _featureDisplayVBO);
    glGenBuffers(nvbo, _featurePointVBO);
  }

  for(int i = 0; i < nvbo; i++)
  {
    if(_levelFeatureNum[i] <= 0)
      continue;

    CuTexImage *ftex  = _featureTex + i;
    CuTexImage texPBO1(_levelFeatureNum[i]*10, 1, 4, _featureDisplayVBO[i]);
    CuTexImage texPBO2(_levelFeatureNum[i], 1, 4, _featurePointVBO[i]);

    ProgramCU::DisplayKeyBox(ftex, &texPBO1);
    ProgramCU::DisplayKeyPoint(ftex, &texPBO2);	
  }
}

void PyramidCU::DestroySharedData() 
{
  // histogram reduction
  if(_histoPyramidTex)
  {
    delete[] _histoPyramidTex;
    _hpLevelNum = 0;
    _histoPyramidTex = NULL;
  }

  // descriptor storage shared by all levels
  if(_descriptorTex)
  {
    delete _descriptorTex;
    _descriptorTex = NULL;
  }

  // cpu reduction buffer.
  if(_histo_buffer)
  {
    delete[] _histo_buffer;
    _histo_buffer = 0;
  }
}

void PyramidCU::DestroyPerLevelData() 
{
  // integers vector to store the feature numbers.
  if(_levelFeatureNum)
  {
    delete [] _levelFeatureNum;
    _levelFeatureNum = NULL;
  }
  // texture used to store features
  if(_featureTex)
  {
    delete [] _featureTex;
    _featureTex = NULL;
  }
  // texture used for multi-orientation 
  if(_orientationTex)
  {
    delete [] _orientationTex;
    _orientationTex = NULL;
  }

  int no = _octave_num * param._dog_level_num;

  // two sets of vbos used to display the features
  if(_featureDisplayVBO)
  {
    glDeleteBuffers(no, _featureDisplayVBO);
    delete [] _featureDisplayVBO;
    _featureDisplayVBO = NULL;
  }
  if( _featurePointVBO)
  {
    glDeleteBuffers(no, _featurePointVBO);

    delete [] _featurePointVBO;
    _featurePointVBO = NULL;
  }
}

void PyramidCU::DestroyPyramidData()
{
  if(_allPyramid)
  {
    delete [] _allPyramid;
    _allPyramid = NULL;
  }
}

void PyramidCU::DownloadKeypoints() 
{
  const double twopi = 2.0 * PI;

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
//    _keypoint_buffer.resize((_featureNum + fcount) * SIFT_KEYPOINT_ITEMS);
//std::cout << "size: " << _keypoint_buffer.size() << " should be: " << _featureNum*SIFT_KEYPOINT_ITEMS << std::endl;
#endif

  float *buffer = &_keypoint_buffer[0];
  vector<float> keypoint_buffer2;

  // use a different keypoint buffer when processing with an existing features list without orientation information. 
  if(_keypoint_index.size() > 0)
  {
    keypoint_buffer2.resize(_keypoint_buffer.size());
    buffer = &keypoint_buffer2[0];
  }

  float *p = buffer, *ps;
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  float *pd;
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
  CuTexImage *ftex = _featureTex;
  /////////////////////
  float octave_sigma = (_octave_min >= 0) ? float(1<<_octave_min) : 1.0f/(1<<(-_octave_min));

  if(_down_sample_factor > 0)
    octave_sigma *= float(1<<_down_sample_factor);

  float offset = GlobalUtil::_LoweOrigin ? 0.0f : 0.5f;
  /////////////////////
  int idx = 0;

  for(int octave = 0; octave < _octave_num; octave++, octave_sigma *= 2.0f)
  {
#ifdef GPU_HESSIAN
    for(int level = 1; level <= param._dog_level_num; level++, idx++, ftex++)
#else
    for(int level = 0; level < param._dog_level_num; level++, idx++, ftex++)
#endif // GPU_HESSIAN
    {
      if(_levelFeatureNum[idx] > 0)
      {
        ftex->CopyToHost(ps = p);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED

        // we start from the end since we are using the same array
        pd = p + (_levelFeatureNum[idx]-1) * SIFT_KEYPOINT_ITEMS;
        ps = p + (_levelFeatureNum[idx]-1) * 4;

        for(int k = _levelFeatureNum[idx]-1; k >= 0; k--, ps-=4, pd-=SIFT_KEYPOINT_ITEMS)
        {
          // input from the level feature list ps
          //  ps[0]: response 8b H | x 24b-14.10
          //  ps[1]: response 8b L | y 24b-14.10
          //  ps[2]: 2b type | 14b unused | scale 16b-8.8 
          //  ps[3]: orientation

          // output to the keypoint buffer pd
          //  pd[0]: position X
          //  pd[1]: position Y
          //  pd[2]: scale
          //  pd[3]: orientation
          //  pd[4]: response
          //  pd[5]: level 16b | unused 14b | type 2b

          // extract x position
          unsigned int xValue = *((unsigned int *)(ps+0));
          unsigned int responseH = xValue & FIXED_POINT_RESPONSE_MASK;
          xValue = xValue & FIXED_POINT_POSITION_MASK;
          float posX = FIXED_POINT_TO_FLOAT(xValue, FIXED_POINT_POSITION_PRECISION_BITS);

          // extract y position
          unsigned int yValue = *((unsigned int *)(ps+1));
          unsigned int responseL = yValue & FIXED_POINT_RESPONSE_MASK;
          yValue = yValue & FIXED_POINT_POSITION_MASK;
          float posY = FIXED_POINT_TO_FLOAT(yValue, FIXED_POINT_POSITION_PRECISION_BITS);

          // extract response
          unsigned short usResponse = (unsigned short)((responseH >> 16) | (responseL >> 24));
          unsigned int intResponse = half2float(usResponse);
          float response = *((float *)(&intResponse));

          // extract scale
          unsigned int scaleValue = *((unsigned int *)(ps+2));
          unsigned int type = (scaleValue & 0xC0000000u) >> 30;
          scaleValue = scaleValue & FIXED_POINT_SCALE_MASK;
          float scale = FIXED_POINT_TO_FLOAT(scaleValue, FIXED_POINT_SCALE_PRECISION_BITS);

          posX = octave_sigma*(posX-0.5f) + offset;      // x
          posY = octave_sigma*(posY-0.5f) + offset;      // y
          scale = octave_sigma*scale;                    // scale

#ifdef GPU_HESSIAN
	      unsigned short levelUShort = (unsigned short)(octave*param._dog_level_num + level-1);
#else
	      unsigned short levelUShort = (unsigned short)(octave*param._dog_level_num + level);
#endif // GPU_HESSIAN

          // store computed values
          pd[0] = posX;
          pd[1] = posY;
          pd[2] = scale;
          pd[3] = (float)fmod(twopi-ps[3], twopi);   // orientation, mirrored
          pd[4] = response;
          ((unsigned short *)(pd+5))[0] = levelUShort;
          ((unsigned short *)(pd+5))[1] = (unsigned short)type;
		}

        p += SIFT_KEYPOINT_ITEMS * _levelFeatureNum[idx];
#else
        for(int k = 0; k < _levelFeatureNum[idx]; k++, ps+=4)
        {
           ps[0] = octave_sigma*(ps[0]-0.5f) + offset;  // x
           ps[1] = octave_sigma*(ps[1]-0.5f) + offset;  // y
           ps[2] = octave_sigma*ps[2];                  // scale
           ps[3] = (float)fmod(twopi-ps[3], twopi);     // orientation, mirrored
        }

        p += 4 * _levelFeatureNum[idx];
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
      }
    }
  }

  // put the feature into their original order for existing keypoint 
  if(_keypoint_index.size() > 0)
  {
    for(int i = 0; i < _featureNum; ++i)
    {
      int index = _keypoint_index[i];
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
      memcpy(&_keypoint_buffer[index*SIFT_KEYPOINT_ITEMS], &keypoint_buffer2[i*SIFT_KEYPOINT_ITEMS], SIFT_KEYPOINT_ITEMS * sizeof(float));
#else
      memcpy(&_keypoint_buffer[index*4], &keypoint_buffer2[i*4], 4 * sizeof(float));
#endif
    }
  }
}

void PyramidCU::GenerateFeatureListCPU()
{
  // no cpu version provided
  GenerateFeatureList();
}

#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED)

void PyramidCU::GenerateFeatureList(int octave, int level)
{
#ifdef GPU_HESSIAN
  int idx = octave * param._dog_level_num + (level-1);
#else
  int idx = octave * param._dog_level_num + level;
#endif // GPU_HESSIAN

  CuTexImage *ftex = _featureTex + idx;

#ifdef GPU_HESSIAN
  CuTexImage *tex = GetBaseLevel(_octave_min + octave, DATA_KEYPOINT) + 1 + (level-1);
#else
  CuTexImage *tex = GetBaseLevel(_octave_min + octave, DATA_KEYPOINT) + 2 + level;
#endif // GPU_HESSIAN

  int fcount = _featureTexLen[idx];
  SetLevelFeatureNum(idx, fcount);

  // build the feature list
  if(fcount > 0)
  {
    _featureNum += fcount;
    ProgramCU::GenerateList(_featureTex + idx, tex, _devFeatureTexLen+idx);
  }
}
#else
void PyramidCU::GenerateFeatureList(int octave, int level, int reduction_count, vector<int>& hbuffer)
{
  int fcount = 0;
#ifdef GPU_HESSIAN
  int idx = octave * param._dog_level_num + (level-1);
#else
  int idx = octave * param._dog_level_num + level;
#endif // GPU_HESSIAN
  int hist_level_num = _hpLevelNum - _pyramid_octave_first / 2;

  CuTexImage *ftex = _featureTex + idx;
  CuTexImage *htex = _histoPyramidTex + hist_level_num -1;

#ifdef GPU_HESSIAN
  CuTexImage *tex = GetBaseLevel(_octave_min + octave, DATA_KEYPOINT) + 1 + (level-1);
#else
  CuTexImage *tex = GetBaseLevel(_octave_min + octave, DATA_KEYPOINT) + 2 + level;
  //CuTexImage *got = GetBaseLevel(_octave_min + octave, DATA_GRAD) + 2 + level;
#endif // GPU_HESSIAN

  ProgramCU::InitHistogram(tex, htex);

  for(int k = 0; k < reduction_count-1; k++, htex--)
  {
    ProgramCU::ReduceHistogram(htex, htex-1);	
  }

  // htex has the row reduction result
  int len = htex->GetImgHeight() * 4;
  hbuffer.resize(len);
  ProgramCU::FinishCUDA();
  htex->CopyToHost(&hbuffer[0]);

  ////TO DO: track the error found here..
  for(int ii = 0; ii < len; ++ii)
  {
    if(!(hbuffer[ii] >= 0))
      hbuffer[ii] = 0;
  }//?

  for(int ii = 0; ii < len; ++ii)
    fcount += hbuffer[ii];

  SetLevelFeatureNum(idx, fcount);

  // build the feature list
  if(fcount > 0)
  {
    _featureNum += fcount;
    _keypoint_buffer.resize(fcount * 4);
    //vector<int> ikbuf(fcount*4);
    int *ibuf = (int*) (&_keypoint_buffer[0]);

    for(int ii = 0; ii < len; ++ii)
    {
      int x = ii%4, y = ii / 4;

      for(int jj = 0; jj < hbuffer[ii]; ++jj, ibuf+=4)
      {
        ibuf[0] = x;
        ibuf[1] = y;
        ibuf[2] = jj;
        ibuf[3] = 0;
      }
    }
    _featureTex[idx].CopyFromHost(&_keypoint_buffer[0]);

    ////////////////////////////////////////////
    ProgramCU::GenerateList(_featureTex + idx, ++htex);
    for(int k = 2; k < reduction_count; k++)
    {
      ProgramCU::GenerateList(_featureTex + idx, ++htex);
    }
  }
}
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && (GPU_HESSIAN || GPU_SIFT_MODIFIED)

void PyramidCU::GenerateFeatureList()
{
  double t1, t2;
  int reduction_count;
  int ocount = 0;
  int reverse = (GlobalUtil::_TruncateMethod == 1);

  vector<int> hbuffer;
  _featureNum = 0;

  // for(int i = 0, idx = 0; i < _octave_num; i++)
  FOR_EACH_OCTAVE(octave, reverse)
  {
#ifdef GPU_HESSIAN
    CuTexImage *tex = GetBaseLevel(_octave_min + octave, DATA_KEYPOINT) + 1;
#else
    CuTexImage *tex = GetBaseLevel(_octave_min + octave, DATA_KEYPOINT) + 2;
#endif // GPU_HESSIAN
    reduction_count = FitHistogramPyramid(tex);

    if(GlobalUtil::_timingO)
    {
      t1 = CLOCK(); 
      ocount = 0;
      std::cout << "#" << octave+_octave_min + _down_sample_factor << ":\t";
    }

    // for(int j = 0; j < param._dog_level_num; j++, idx++)
    FOR_EACH_LEVEL(level, reverse)
    {
#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined TOP_K_SELECTION
      if( ((GlobalUtil::_TruncateMethod == TRUNCATE_METHOD_KEEP_HIGHEST_LEVELS_1) || (GlobalUtil::_TruncateMethod == TRUNCATE_METHOD_KEEP_LOWEST_LEVELS) )
		  && (GlobalUtil::_FeatureCountThreshold > 0) && (_featureNum > GlobalUtil::_FeatureCountThreshold))
#else
      if(GlobalUtil::_TruncateMethod && (GlobalUtil::_FeatureCountThreshold > 0) && (_featureNum > GlobalUtil::_FeatureCountThreshold))
#endif // (GPU_HESSIAN || GPU_SIFT_MODIFIED) && TOP_K_SELECTION
        continue;

#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined GENERATE_FEATURE_LIST_USING_ATOMICS
      GenerateFeatureList(octave, level);
#else
      GenerateFeatureList(octave, level, reduction_count, hbuffer);
#endif // (GPU_HESSIAN || GPU_SIFT_MODIFIED) && GENERATE_FEATURE_LIST_USING_ATOMICS

      if(GlobalUtil::_timingO)
      {
#ifdef GPU_HESSIAN
        int idx = octave * param._dog_level_num + (level-1);
#else
        int idx = octave * param._dog_level_num + level;
#endif // GPU_HESSIAN
        ocount += _levelFeatureNum[idx];
        std::cout<< _levelFeatureNum[idx] <<"\t";
      }
    }

    if(GlobalUtil::_timingO)
    {	
      t2 = CLOCK(); 
      std::cout << "| \t" << int(ocount) << " :\t(" << (t2 - t1) << "ms)\n";
    }
  }

#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined GENERATE_FEATURE_LIST_USING_ATOMICS
    if(_featureTexLen != NULL)
    {
      delete[] _featureTexLen;
      _featureTexLen = NULL;
    }

    ProgramCU::DetectionDataFinish(&_devFeatureTexLen);
#endif // (GPU_HESSIAN || GPU_SIFT_MODIFIED) && GENERATE_FEATURE_LIST_USING_ATOMICS

  CopyGradientTex();

  if(GlobalUtil::_timingS)
    ProgramCU::FinishCUDA();

  if(GlobalUtil::_verbose)
  {
    std::cout << "#Features:\t" << _featureNum << "\n";
  }

  if(ProgramCU::CheckErrorCUDA("PyramidCU::GenerateFeatureList"))
    SetFailStatus();
}

GLTexImage* PyramidCU::GetLevelTexture(int octave, int level)
{
  return GetLevelTexture(octave, level, DATA_GAUSSIAN);
}

GLTexImage* PyramidCU::ConvertTexCU2GL(CuTexImage* tex, int dataName)
{
  GLenum format = GL_LUMINANCE;
  int convert_done = 1;

  if(_bufferPBO == 0)
    glGenBuffers(1, &_bufferPBO);

  if(_bufferTEX == NULL)
    _bufferTEX = new GLTexImage;

  switch(dataName)
  {
    case DATA_GAUSSIAN:
    {
      convert_done = tex->CopyToPBO(_bufferPBO);
      break;
    }
    case DATA_DOG:
    {
      CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 1, _bufferPBO);

      if((texPBO._cuData == 0) || (tex->_cuData == NULL))
        convert_done = 0;
      else
        ProgramCU::DisplayConvertDOG(tex, &texPBO);
      break;
    }
    case DATA_GRAD:
    {
      CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 1, _bufferPBO);

      if((texPBO._cuData == 0) || (tex->_cuData == NULL))
        convert_done = 0;
      else
        ProgramCU::DisplayConvertGRD(tex, &texPBO);
      break;
    }
    case DATA_KEYPOINT:
    {
      CuTexImage *dog = tex - param._level_num * _pyramid_octave_num;

      format = GL_RGBA;
      CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 4, _bufferPBO);

      if((texPBO._cuData == 0) || (tex->_cuData == NULL))
        convert_done = 0;
      else
        ProgramCU::DisplayConvertKEY(tex, dog, &texPBO);
      break;
    }
    default:
      convert_done = 0;
      break;
  }

  if(convert_done)
  {
    _bufferTEX->InitTexture(max(_bufferTEX->GetTexWidth(), tex->GetImgWidth()), max(_bufferTEX->GetTexHeight(), tex->GetImgHeight()));
    _bufferTEX->CopyFromPBO(_bufferPBO, tex->GetImgWidth(), tex->GetImgHeight(), format);
  }
  else {
    _bufferTEX->SetImageSize(0, 0);
  }

  return _bufferTEX;
}

GLTexImage* PyramidCU::GetLevelTexture(int octave, int level, int dataName) 
{
  CuTexImage* tex = GetBaseLevel(octave, dataName) + (level - param._level_min);
  //CuTexImage* gus = GetBaseLevel(octave, DATA_GAUSSIAN) + (level - param._level_min); 
  return ConvertTexCU2GL(tex, dataName);
}

void PyramidCU::ConvertInputToCU(GLTexInput* input)
{
  int ws = input->GetImgWidth();
  int hs = input->GetImgHeight();

  TruncateWidth(ws);

  // copy the input image to pixel buffer object
  if(input->_pixel_data)
  {
    _inputTex->InitTexture(ws, hs, 1);
    _inputTex->CopyFromHost(input->_pixel_data); 
  }
  else
  {
    if(_bufferPBO == 0)
      glGenBuffers(1, &_bufferPBO);

    if(input->_rgb_converted && input->CopyToPBO(_bufferPBO, ws, hs, GL_LUMINANCE))
    {
      _inputTex->InitTexture(ws, hs, 1);
      _inputTex->CopyFromPBO(ws, hs, _bufferPBO); 
    }
    else if(input->CopyToPBO(_bufferPBO, ws, hs))
    {
      CuTexImage texPBO(ws, hs, 4, _bufferPBO);
      _inputTex->InitTexture(ws, hs, 1);
      ProgramCU::ReduceToSingleChannel(_inputTex, &texPBO, !input->_rgb_converted);
    }
    else
    {
      std::cerr << "Unable To Convert Intput\n";
    }
  }
}

void PyramidCU::BuildPyramid(GLTexInput *input)
{
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
  if(GlobalUtil::_verbose)
  {
    std::cout << "# octaves:\t" << _octave_num << endl;
    std::cout << "# min octave:\t" << _octave_min << endl;
    std::cout << "# levels:\t" << param._level_num << endl;
    std::cout << "# min level:\t" << param._level_min << endl;
  }
#endif // GPU_HESSIAN

  USE_TIMING();
	
  for (int octave = _octave_min; octave < _octave_min + _octave_num; octave++)
  {
    float *filter_sigma = param._sigma;

    CuTexImage *tex = GetBaseLevel(octave);
#ifdef GPU_HESSIAN
    CuTexImage *buf = GetBaseLevel(octave, DATA_KEYPOINT) + 1;
#else
    CuTexImage *buf = GetBaseLevel(octave, DATA_KEYPOINT) + 2;
#endif // GPU_HESSIAN

    OCTAVE_START();

    if( octave == _octave_min )
    {
      ConvertInputToCU(input);

      if(octave == 0)
      {
        ProgramCU::FilterImage(tex, _inputTex, buf, param.GetInitialSmoothSigma(_octave_min + _down_sample_factor));
      }
      else
      {
        if(octave < 0)
          ProgramCU::SampleImageU(tex, _inputTex, -octave);
        else
          ProgramCU::SampleImageD(tex, _inputTex, octave);

        ProgramCU::FilterImage(tex, tex, buf, param.GetInitialSmoothSigma(_octave_min + _down_sample_factor));
      }
    }
    else
    {
      ProgramCU::SampleImageD(tex, GetBaseLevel(octave - 1) + param._level_ds - param._level_min);

      if(param._sigma_skip1 > 0)
      {
        ProgramCU::FilterImage(tex, tex, buf, param._sigma_skip1);
      }
    }
    LEVEL_FINISH();

    for(int level = param._level_min+1; level <= param._level_max; level++, tex++, filter_sigma++)
    {
      // filtering
      ProgramCU::FilterImage(tex+1, tex, buf, *filter_sigma);

      LEVEL_FINISH();
    }

    OCTAVE_FINISH();
  }

  if(GlobalUtil::_timingS)
    ProgramCU::FinishCUDA();

  if(ProgramCU::CheckErrorCUDA("PyramidCU::BuildPyramid"))
    SetFailStatus();
}

void PyramidCU::DetectKeypointsEX()
{
  int octave, level;
  double t0, t, ts, t1, t2;

  if(GlobalUtil::_timingS && GlobalUtil::_verbose)
    ts = CLOCK();

#ifdef GPU_HESSIAN
  //float octaveSigma = (_octave_min >= 0) ? float(1<<_octave_min) : 1.0f/(1<<(-_octave_min));

  //if(_down_sample_factor > 0)
  //	octaveSigma *= float(1<<_down_sample_factor); 

  float octaveSigma = 1.0f;

  for(octave = _octave_min; octave < _octave_min+_octave_num; octave++/*, octaveSigma *= 2.0f*/)
  {
    CuTexImage *gus = GetBaseLevel(octave);
    CuTexImage *hessian = GetBaseLevel(octave, DATA_DOG);
    CuTexImage *got = GetBaseLevel(octave, DATA_GRAD);

    // compute hessian response, gradient and rotation
    for(level = param._level_min; level <= param._level_max; level++, gus++, hessian++, got++)
    {
      float levelSigma = param.GetLevelSigma(level) * octaveSigma;

      // input: gus
      // output: hessian, gradient+orientation
      ProgramCU::ComputeHessian(gus, hessian, got, levelSigma*levelSigma);
    }
  }
#else
  for(octave = _octave_min; octave < _octave_min+_octave_num; octave++)
  {
    CuTexImage *gus = GetBaseLevel(octave) + 1;
    CuTexImage *dog = GetBaseLevel(octave, DATA_DOG) + 1;
    CuTexImage *got = GetBaseLevel(octave, DATA_GRAD) + 1;
    // compute the gradient
    for(level = param._level_min+1; level <= param._level_max; level++, gus++, dog++, got++)
    {
      // input: gus and gus -1
      // output: gradient, dog, orientation
      ProgramCU::ComputeDOG(gus, dog, got);
    }
  }
#endif // GPU_HESSIAN

  if(GlobalUtil::_timingS && GlobalUtil::_verbose)
  {
    ProgramCU::FinishCUDA();
    t1 = CLOCK();
  }

#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined GENERATE_FEATURE_LIST_USING_ATOMICS
  int featureTexCount = _octave_num * param._dog_level_num;
  _featureTexLen = new int[featureTexCount];

  ProgramCU::DetectionDataInit(&_devFeatureTexLen, featureTexCount);
#endif // (GPU_HESSIAN || GPU_SIFT_MODIFED) && GENERATE_FEATURE_LIST_USING_ATOMICS

  for (octave = _octave_min; octave < _octave_min+_octave_num; octave++)
  {
    if(GlobalUtil::_timingO)
    {
      t0 = CLOCK();
      std::cout << "#" << (octave + _down_sample_factor) << "\t";
    }

#ifdef GPU_HESSIAN

    CuTexImage *gus = GetBaseLevel(octave) + 1;
    CuTexImage *hessian = GetBaseLevel(octave, DATA_DOG) + 1;
    CuTexImage *key = GetBaseLevel(octave, DATA_KEYPOINT) + 1;

    for(level = param._level_min+1; level < param._level_max; level++, hessian++, key++, gus++)
    {
      if(GlobalUtil::_timingL)
        t = CLOCK();

      // input: hessian -1, hessian, hessian + 1
      // output: key
	  ProgramCU::ComputeKEY(hessian, key, gus , param._dog_threshold, param._edge_threshold
#ifdef GENERATE_FEATURE_LIST_USING_ATOMICS
         , _devFeatureTexLen, (octave-_octave_min) * param._dog_level_num + level-param._level_min-1
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS
      );

      if(GlobalUtil::_timingL)
      {
        std::cout << (CLOCK()-t) << "ms\t";
      }
    }
#else
    CuTexImage *dog = GetBaseLevel(octave, DATA_DOG) + 2;
    CuTexImage *key = GetBaseLevel(octave, DATA_KEYPOINT) + 2;

    for(level = param._level_min+2; level < param._level_max; level++, dog++, key++)
    {
      if(GlobalUtil::_timingL)
        t = CLOCK();

      // input, dog, dog + 1, dog -1
      // output, key
      ProgramCU::ComputeKEY(dog, key, param._dog_threshold, param._edge_threshold
#if defined GENERATE_FEATURE_LIST_USING_ATOMICS && defined GPU_SIFT_MODIFIED
         , _devFeatureTexLen, (octave-_octave_min) * param._dog_level_num + level-param._level_min-2
#endif // GENERATE_FEATURE_LIST_USING_ATOMICS && GPU_SIFT_MODIFIED
      );

      if(GlobalUtil::_timingL)
      {
        std::cout << (CLOCK()-t) << "ms\t";
      }
    }
#endif // GPU_HESSIAN

    if(GlobalUtil::_timingO)
    {
      std::cout << "|\t" << (CLOCK()-t0) << "ms\n";
    }
  }

#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined GENERATE_FEATURE_LIST_USING_ATOMICS
  ProgramCU::FinishCUDA();
  ProgramCU::DetectionDataDownload(_featureTexLen, _devFeatureTexLen, featureTexCount);
#endif // (GPU_HESSIAN || GPU_SIFT_MODIFIED) && GENERATE_FEATURE_LIST_USING_ATOMICS

  if(GlobalUtil::_timingS)
  {
    ProgramCU::FinishCUDA();

    if(GlobalUtil::_verbose) 
    {	
      t2 = CLOCK();
      std::cout <<"<Gradient, DOG  >\t" << (t1-ts) << "ms\n"
                <<"<Get Keypoints  >\t" << (t2-t1) << "ms\n";
    }
  }
}

void PyramidCU::CopyGradientTex()
{
  double ts, t1;

  if(GlobalUtil::_timingS && GlobalUtil::_verbose)
    ts = CLOCK();

  for(int octave = 0, idx = 0; octave < _octave_num; octave++)
  {
    CuTexImage *got = GetBaseLevel(octave + _octave_min, DATA_GRAD) +  1;

    //compute the gradient
#ifdef GPU_HESSIAN
    for(int level = 1; level <= param._dog_level_num; level++, got++, idx++)
#else
    for(int level = 0; level < param._dog_level_num; level++, got++, idx++)
#endif // GPU_HESSIAN
    {
      if(_levelFeatureNum[idx] > 0)
        got->CopyToTexture2D();
    }
  }

  if(GlobalUtil::_timingS)
  {
    ProgramCU::FinishCUDA();

    if(GlobalUtil::_verbose)
    {
      t1 = CLOCK();
      std::cout << "<Copy Grad/Orientation>\t" << (t1-ts) << "ms\n";
    }
  }
}

void PyramidCU::ComputeGradient()
{
  double ts, t1;

  if(GlobalUtil::_timingS && GlobalUtil::_verbose)
    ts = CLOCK();

#ifdef GPU_HESSIAN
  //float octaveSigma = (_octave_min >= 0) ? float(1<<_octave_min) : 1.0f/(1<<(-_octave_min));

  //if(_down_sample_factor > 0)
  //  octaveSigma *= float(1<<_down_sample_factor); 

  float octaveSigma = 1.0f;

  for(int octave = _octave_min; octave < _octave_min+_octave_num; octave++/*, octaveSigma *= 2.0f*/)
  {
    CuTexImage *gus = GetBaseLevel(octave);
    CuTexImage *hessian = GetBaseLevel(octave, DATA_DOG);
    CuTexImage *got = GetBaseLevel(octave, DATA_GRAD);

    //compute the gradient
    for(int level = param._level_min; level <= param._level_max; level++, gus++, hessian++, got++)
    {
      float levelSigma = param.GetLevelSigma(level) * octaveSigma;
      //std::cout<<"PyramidCU::ComputeGradient():\t level: "<<level<<" sigma: "<<levelSigma<<endl;

      ProgramCU::ComputeHessian(gus, hessian, got, levelSigma*levelSigma);
    }
  }
#else
  for(int octave = _octave_min; octave < _octave_min + _octave_num; octave++)
  {
    CuTexImage *gus = GetBaseLevel(octave) + 1;
    CuTexImage *dog = GetBaseLevel(octave, DATA_DOG) + 1;
    CuTexImage *got = GetBaseLevel(octave, DATA_GRAD) + 1;

    //compute the gradient
    for(int level = 0; level < param._dog_level_num; level++, gus++, dog++, got++)
    {
      ProgramCU::ComputeDOG(gus, dog, got);
    }
  }
#endif // GPU_HESSIAN

  if(GlobalUtil::_timingS)
  {
    ProgramCU::FinishCUDA();
    if(GlobalUtil::_verbose)
    {
      t1 = CLOCK();
      std::cout <<"<Gradient, DOG  >\t"<<(t1-ts)<<"ms\n";
    }
  }
}

int PyramidCU::FitHistogramPyramid(CuTexImage* tex)
{
  int hist_level_num = _hpLevelNum - _pyramid_octave_first / 2; 
  CuTexImage *htex = _histoPyramidTex + hist_level_num - 1;

  int w = (tex->GetImgWidth() + 2) >> 2;
  int h = tex->GetImgHeight();
  int count = 0;

  for(int k = 0; k < hist_level_num; k++, htex--)
  {
    //htex->SetImageSize(w, h);	
    htex->InitTexture(w, h, 4); 
    ++count;

    if(w == 1)
      break;

    w = (w + 3) >> 2;
  }
  return count;
}

void PyramidCU::GetFeatureOrientations() 
{
  CuTexImage *ftex = _featureTex;
  int *count = _levelFeatureNum;

  float sigma;
  float sigma_step = powf(2.0f, 1.0f/param._dog_level_num);

  for(int octave = 0; octave < _octave_num; octave++)
  {
    CuTexImage *got = GetBaseLevel(octave + _octave_min, DATA_GRAD) + 1;

#ifdef GPU_HESSIAN
    CuTexImage *key = GetBaseLevel(octave + _octave_min, DATA_KEYPOINT) + 1;

	for(int level = 1; level <= param._dog_level_num; level++, ftex++, count++, got++, key++)
#else
    CuTexImage *key = GetBaseLevel(octave + _octave_min, DATA_KEYPOINT) + 2;

    for(int level = 0; level < param._dog_level_num; level++, ftex++, count++, got++, key++)
#endif // GPU_HESSIAN
    {
      if(*count <= 0)
        continue;

      //if(ftex->GetImgWidth() < *count) ftex->InitTexture(*count, 1, 4);

#ifdef GPU_HESSIAN
      sigma = param.GetLevelSigma(level+param._level_min);
#else
      sigma = param.GetLevelSigma(level+param._level_min+1);
#endif // GPU_HESSIAN

      ProgramCU::ComputeOrientation(ftex, got, key, sigma, sigma_step, _existing_keypoints);
    }
  }

  if(GlobalUtil::_timingS)
    ProgramCU::FinishCUDA();

  if(ProgramCU::CheckErrorCUDA("PyramidCU::GetFeatureOrientations"))
    SetFailStatus();
}

void PyramidCU::GetSimplifiedOrientation() 
{
  // no simplified orientation
  GetFeatureOrientations();
}

CuTexImage* PyramidCU::GetBaseLevel(int octave, int dataName)
{
  if((octave <_octave_min) || (octave > _octave_min+_octave_num))
    return NULL;

  int offset = (_pyramid_octave_first + octave - _octave_min) * param._level_num;
  int num = param._level_num * _pyramid_octave_num;

  if (dataName == DATA_ROT)
    dataName = DATA_GRAD;

  return _allPyramid + num * dataName + offset;
}

#if (defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED) && defined TOP_K_SELECTION

void PyramidCU::SelectTopK()
{
  if(_existing_keypoints)
    return;

  if((_featureNum < GlobalUtil::_FeatureCountThreshold) && (GlobalUtil::_FeatureCountThreshold > 0))
    return;

  TopKData topKData;

  topKData.keys = NULL;
  topKData.indices = NULL;

  int listSize = _featureNum;

  ProgramCU::TopKInit(topKData, listSize, GlobalUtil::_FeatureCountThreshold);

  // copy responses and indices into separate arrays for sorting
  CuTexImage *ftex = _featureTex;
  int *count = _levelFeatureNum;
  int offset = 0;

  for(int octave = 0; octave < _octave_num; octave++)
  {
#ifdef GPU_SIFT_MODIFIED
    CuTexImage *key = GetBaseLevel(_octave_min + octave, DATA_KEYPOINT) + 2;

	for(int level = 0; level < param._dog_level_num; level++, ftex++, key++, count++)
#else
    CuTexImage *key = GetBaseLevel(octave + _octave_min, DATA_KEYPOINT) + 1;

	for(int level = 1; level <= param._dog_level_num; level++, ftex++, key++, count++)
#endif
    {
      if(*count <= 0)
        continue;

      // copy responses and indices of features from ftex into topKData arrays
      ProgramCU::TopKCopyData(ftex, key, topKData, offset);

      offset += *count;
    }
  }

  // set the padding elements (response to FLT_MAX, indices to their position in array)
  ProgramCU::TopKCopyData(NULL, NULL, topKData, offset);

  // sort responses
  ProgramCU::TopKSort(topKData);

  // compute prefix sum
  ProgramCU::TopKPrefixScan(topKData);

  // reduce number of features for each level separately
  topKData.levelsCount = _octave_num * param._dog_level_num;
  topKData.levelFeaturesCount = new int[topKData.levelsCount+1];
  int *oldLevelFeaturesStart = new int[topKData.levelsCount+1];

  // exclusive prefix scan
  topKData.levelFeaturesCount[0] = _levelFeatureNum[0];
  oldLevelFeaturesStart[0] = 0;
  for(int i=1; i<topKData.levelsCount; i++)
  {
    topKData.levelFeaturesCount[i] = _levelFeatureNum[i] + topKData.levelFeaturesCount[i-1];
    oldLevelFeaturesStart[i] = _levelFeatureNum[i-1] + oldLevelFeaturesStart[i-1];
  }
  ProgramCU::TopKGetLevelsFeatureNum(topKData);

  unsigned int *levelsLen = new unsigned int[topKData.levelsCount];

  for(int i=0; i<topKData.levelsCount; i++)
  {
    levelsLen[i] = (i > 0) ? (topKData.levelFeaturesCount[i] - topKData.levelFeaturesCount[i-1]) : topKData.levelFeaturesCount[i];
  }

  float *newLevelFeatures = NULL;
  ftex = _featureTex;
  unsigned int newFeaturesCount = 0;

  int idx = 0;
  for(int octave = 0; octave < _octave_num; octave++)
  {
    for(int level = 1; level <= param._dog_level_num; level++, ftex++, idx++)
    {
	  newFeaturesCount += levelsLen[idx];

      // the same number of features
      if(_levelFeatureNum[idx] == levelsLen[idx])
        continue;

      ProgramCU::TopKCompactLevelFeatures(ftex, _levelFeatureNum[idx], &newLevelFeatures, levelsLen[idx], topKData, oldLevelFeaturesStart[idx]);

      ftex->ReInitTexture(levelsLen[idx], 1, 4, newLevelFeatures);
      _levelFeatureNum[idx] = levelsLen[idx];
    }
  }

  _featureNum = newFeaturesCount;

  delete[] oldLevelFeaturesStart;
  delete[] levelsLen;

  ProgramCU::TopKFinish(topKData);

  if(GlobalUtil::_timingS)
    ProgramCU::FinishCUDA();
}

#endif // GPU_HESSIAN && TOP_K_SELECTION

#endif

