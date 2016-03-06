
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string.h>

#ifdef _WIN32
  // dll import definition for win32
  #define SIFTGPU_DLL
#endif

#include "../SiftGPU/SiftGPU.h"

#ifdef _WIN32
  #define snprintf _snprintf
#endif

#define SPEED_TEST_NUM_ITERATIONS  10
// enable to display speed test iteration timings
// #define SHOW_TEST_MODE_ITERATION_TIMING

int main(int argc, char **argv)
{
  // create a SiftGPU instance
  SiftGPU sift;

  // parse command line parameters
  sift.ParseParam(argc, argv);

  // overwrite selected processing parameters
  char *localargv[] = { "-cuda", "0", "-nogl", "-v", "1"};
  // -v 1, only print out # feature and overall time
  sift.ParseParam(5, localargv);

  if(sift.GetImageCount() < 1)
  {
    std::cout << "gpuhess -i <list of image names> | -il <file with image names> [-time] [sift params list]" << std::endl;
    std::cout << std::endl;
    std::cout << "-time                enable generation of file with timming and suppress output" << std::endl;
    std::cout << "-speed               speed test - average of 10 runs" << std::endl;
    std::cout << "                     (except warnings and errors)" << std::endl;
    std::cout << "[sift params list]   use option -h to get a list of these params" << std::endl;

    return EXIT_FAILURE;
  }

  bool exportTimings = false;
  bool saveOutput = true;
  bool speedTest = false;

  // search for "-time", "-speed", and "-o" option
  for(int i=1; i<argc; i++)
  {
    if (strcmp(argv[i], "-time") == 0)
    {
      exportTimings = true;
    }
    else if (strcmp(argv[i], "-speed") == 0)
    {
      speedTest = true;
    }
    else if (strcmp(argv[i], "-o") == 0)
    {
	  saveOutput = (sift.GetImageCount() > 1) ? true : false;
    }
  }

  // create an OpenGL context for computation
  int support = sift.CreateContextGL();
  // call VerfifyContexGL instead if using your own GL context
  // int support = sift.VerifyContextGL();

  if(support != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    return EXIT_FAILURE;

  char outSuffix[] = ".sift";
  char timingsSuffix[] = ".timings";

  if (exportTimings)
    sift.SetVerbose(-2); // little trick to disable all output but keep the timing

  double accTimes[12];

  for(int idx=0; idx<sift.GetImageCount(); idx++)
  {
    // process an image, and save ASCII format SIFT files
    if(sift.RunSIFT(idx))
    {

      if(speedTest)
      {
	    // save first timings
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
        for(int i=0; i<TIMINGS_COUNT; i++)
          accTimes[i] = sift._timing[i];
#else
        for(int i=0; i<10; i++)
          accTimes[i] = sift._timing[i];
        accTimes[10] = 0;
        accTimes[TIMINGS_TOTAL] = sift._timing[10];
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

#ifdef SHOW_TEST_MODE_ITERATION_TIMING
        std::cout
            << "Iteration 0 timing: " 
            << sift._timing[TIMINGS_LOAD_IMAGE] << ", " // Load input image [ms]
            << sift._timing[TIMINGS_ALLOCATE_PYRAMID] << ", " // Initialize pyramid [ms]
            << sift._timing[TIMINGS_BUILD_PYRAMID] << ", " // Build pyramid [ms]
            << sift._timing[TIMINGS_DETECT_KEYPOINTS] << ", " // Detect keypoints [ms]
            << sift._timing[TIMINGS_GENERATE_FEATURE_LIST] << ", " // Generate feature list [ms]
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
            << sift._timing[TIMINGS_FEATURES_REDUCTION] << ", " // Feature reduction (topk)
#else
            << "0.0" << ", " // Feature reduction (topk)
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
            << sift._timing[TIMINGS_COMPUTE_ORIENTATIONS] << ", " // Compute feature orientations [ms]
            << sift._timing[TIMINGS_MULTI_ORIENTATIONS] << ", " // Generate multi-orientations feature list [ms]
            << sift._timing[TIMINGS_DOWNLOAD_KEYPOINTS] << ", " // Download keypoints [ms]
            << sift._timing[TIMINGS_COMPUTE_DESCRIPTORS] << ", " // Get descriptors [ms]
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
            << sift._timing[TIMINGS_TOTAL]        // Total time [ms]
#else
            << sift._timing[TIMINGS_FEATURES_REDUCTION]        // Total time [ms]
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
            << std::endl;
#endif // SHOW_TEST_MODE_ITERATION_TIMING

        for(int iter=1; iter<SPEED_TEST_NUM_ITERATIONS; iter++)
        {
          sift.RunSIFT(idx);

#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
          for(int i=0; i<11; i++)
            accTimes[i] += sift._timing[i];
          // total time: image load + initialize pyramid is included only for the first call of RunSIFT
          accTimes[TIMINGS_TOTAL] += sift._timing[TIMINGS_TOTAL] + accTimes[TIMINGS_LOAD_IMAGE] + accTimes[TIMINGS_ALLOCATE_PYRAMID];
#else
          for(int i=0; i<10; i++)
            accTimes[i] += sift._timing[i];
          accTimes[10] = 0;
          accTimes[TIMINGS_TOTAL] += sift._timing[10] + accTimes[TIMINGS_LOAD_IMAGE] + accTimes[TIMINGS_ALLOCATE_PYRAMID];
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED

#ifdef SHOW_TEST_MODE_ITERATION_TIMING
          std::cout
            << "Iteration " << iter << " timing: " 
            << sift._timing[TIMINGS_LOAD_IMAGE] << ", "            // Load input image [ms]
            << sift._timing[TIMINGS_ALLOCATE_PYRAMID] << ", "      // Initialize pyramid [ms]
            << sift._timing[TIMINGS_BUILD_PYRAMID] << ", "         // Build pyramid [ms]
            << sift._timing[TIMINGS_DETECT_KEYPOINTS] << ", "      // Detect keypoints [ms]
            << sift._timing[TIMINGS_GENERATE_FEATURE_LIST] << ", " // Generate feature list [ms]
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
            << sift._timing[TIMINGS_FEATURES_REDUCTION] << ", "    // Feature reduction (topk)
#else
            << "0.0" << ", "                                       // Feature reduction (topk)
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
            << sift._timing[TIMINGS_COMPUTE_ORIENTATIONS] << ", "  // Compute feature orientations [ms]
            << sift._timing[TIMINGS_MULTI_ORIENTATIONS] << ", "    // Generate multi-orientations feature list [ms]
            << sift._timing[TIMINGS_DOWNLOAD_KEYPOINTS] << ", "    // Download keypoints [ms]
            << sift._timing[TIMINGS_COMPUTE_DESCRIPTORS] << ", "   // Get descriptors [ms]
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
            << sift._timing[TIMINGS_TOTAL]+accTimes[TIMINGS_LOAD_IMAGE]+accTimes[TIMINGS_ALLOCATE_PYRAMID]                 // Total time [ms]
#else
            << sift._timing[TIMINGS_FEATURES_REDUCTION]+accTimes[TIMINGS_LOAD_IMAGE]+accTimes[TIMINGS_ALLOCATE_PYRAMID]        // Total time [ms]
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
            << std::endl;
#endif // SHOW_TEST_MODE_ITERATION_TIMING
	    }

        for(int i=2; i<12; i++) // image load and pyramid allocation is done once
          accTimes[i] /= SPEED_TEST_NUM_ITERATIONS;

#ifdef SHOW_TEST_MODE_ITERATION_TIMING
        std::cout
            << "Final timing: >>> " 
            << accTimes[TIMINGS_LOAD_IMAGE] << ", "            // Load input image [ms]
            << accTimes[TIMINGS_ALLOCATE_PYRAMID] << ", "      // Initialize pyramid [ms]
            << accTimes[TIMINGS_BUILD_PYRAMID] << ", "         // Build pyramid [ms]
            << accTimes[TIMINGS_DETECT_KEYPOINTS] << ", "      // Detect keypoints [ms]
            << accTimes[TIMINGS_GENERATE_FEATURE_LIST] << ", " // Generate feature list [ms]
            << accTimes[TIMINGS_FEATURES_REDUCTION] << ", "    // Feature reduction (topk)
            << accTimes[TIMINGS_COMPUTE_ORIENTATIONS] << ", "  // Compute feature orientations [ms]
            << accTimes[TIMINGS_MULTI_ORIENTATIONS] << ", "    // Generate multi-orientations feature list [ms]
            << accTimes[TIMINGS_DOWNLOAD_KEYPOINTS] << ", "    // Download keypoints [ms]
            << accTimes[TIMINGS_COMPUTE_DESCRIPTORS] << ", "   // Get descriptors [ms]
            << accTimes[TIMINGS_TOTAL]                         // Total time [ms]
            << std::endl;
#endif // SHOW_TEST_MODE_ITERATION_TIMING

      }

      const char *imgName = sift.GetCurrentImagePath();

      size_t len = strlen(imgName) + strlen(outSuffix) + 1;
      char *outFileName = new char[len];

      snprintf(outFileName, len, "%s%s", imgName, outSuffix);
      outFileName[len-1] = 0;

      if(saveOutput)
        sift.SaveSIFT(outFileName);


/* ======================= sample - using own keypoints list =======================
  // simple example showing how to use own keypoints to compute orientation and descriptors

  // you can get the feature vector and store it yourself
  int num = sift.GetFeatureNum(); //get feature count
  // allocate memory for readback
  // std::vector<float> descriptors(128*num);
  std::vector<SiftGPU::SiftKeypoint> keys(num);
  // read back keypoints and normalized descriptors
  // specify NULL if you don’t need keypionts or descriptors
  // sift.GetFeatureVector(&keys[0], NULL);
  sift.GetFeatureVector(&keys[0], &descriptors[0]);

  // when we use existing keypoints list than we can compute the orientation and descriptor only
  // what is computed depends on parameters of the methods called below (version 1 x version 2)

  // specify the keypoints for next image to siftgpu

  // version 1
  // assumes that keypoint have position and scale and we will run sift on different image
  // if keypoints have also orientation then set the 3rd parameter to 1 (orientation computation
  // will be skipped), otherwise set it to 0 (orientation will be computed)
  // sift.SetKeypointList(keys.size(), &keys[0]);
  // sift.RunSIFT(idx); // RunSIFT on your image data

  // version 2
  // we will re-run SIFT with different keypoints
  // use sift.RunSIFT(keys.size(), &keys[0]) to skip filtering
  // runs on the same image but with different keypoints, filtering is skipped when we are running
  // on the same image, ie. pyramid is not build again
  sift.RunSIFT(keys.size(), &keys[0]);

  char *outFileName2 = new char[len+2];
  snprintf(outFileName2, len+2, "%s%s.n", imgName, outSuffix);
  outFileName2[len+2-1] = 0;

  if(saveOutput)
    sift.SaveSIFT(outFileName2);
======================= sample - using own keypoints list ======================= */

      // save timmings
      if(exportTimings)
      {
        len = strlen(imgName) + strlen(timingsSuffix) + 1;
        char *timingsFileName = new char[len];

        snprintf(timingsFileName, len, "%s%s", imgName, timingsSuffix);
        timingsFileName[len-1] = 0;

        std::ofstream out(timingsFileName);
        out.flags(std::ios::fixed);

        if(speedTest)
        {
          out
              << std::setprecision(2) << accTimes[TIMINGS_LOAD_IMAGE] << ", "            // Load input image [ms]
              << std::setprecision(2) << accTimes[TIMINGS_ALLOCATE_PYRAMID] << ", "      // Initialize pyramid [ms]
              << std::setprecision(2) << accTimes[TIMINGS_BUILD_PYRAMID] << ", "         // Build pyramid [ms]
              << std::setprecision(2) << accTimes[TIMINGS_DETECT_KEYPOINTS] << ", "      // Detect keypoints [ms]
              << std::setprecision(2) << accTimes[TIMINGS_GENERATE_FEATURE_LIST] << ", " // Generate feature list [ms]
              << std::setprecision(2) << accTimes[TIMINGS_FEATURES_REDUCTION] << ", "    // Feature reduction (topk)
              << std::setprecision(2) << accTimes[TIMINGS_COMPUTE_ORIENTATIONS] << ", "  // Compute feature orientations [ms]
              << std::setprecision(2) << accTimes[TIMINGS_MULTI_ORIENTATIONS] << ", "    // Generate multi-orientations feature list [ms]
              << std::setprecision(2) << accTimes[TIMINGS_DOWNLOAD_KEYPOINTS] << ", "    // Download keypoints [ms]
              << std::setprecision(2) << accTimes[TIMINGS_COMPUTE_DESCRIPTORS] << ", "   // Get descriptors [ms]
              // << std::setprecision(2) << accTimes[TIMINGS_GENERATE_VBO] << ", "       // Gen. Display VBO [ms]
              << std::setprecision(2) << accTimes[TIMINGS_TOTAL]                         // Total time [ms]
              << std::setprecision(2) << std::endl; 
        }
        else
        {
          out
              << sift._timing[TIMINGS_LOAD_IMAGE] << ", "            // Load input image [ms]
              << sift._timing[TIMINGS_ALLOCATE_PYRAMID] << ", "      // Initialize pyramid [ms]
              << sift._timing[TIMINGS_BUILD_PYRAMID] << ", "         // Build pyramid [ms]
              << sift._timing[TIMINGS_DETECT_KEYPOINTS] << ", "      // Detect keypoints [ms]
              << sift._timing[TIMINGS_GENERATE_FEATURE_LIST] << ", " // Generate feature list [ms]
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
              << sift._timing[TIMINGS_FEATURES_REDUCTION] << ", "    // Feature reduction (topk)
#else
              << "0.0" << ", "                                       // Feature reduction (topk)
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
              << sift._timing[TIMINGS_COMPUTE_ORIENTATIONS] << ", " // Compute feature orientations [ms]
              << sift._timing[TIMINGS_MULTI_ORIENTATIONS] << ", "   // Generate multi-orientations feature list [ms]
              << sift._timing[TIMINGS_DOWNLOAD_KEYPOINTS] << ", "   // Download keypoints [ms]
              << sift._timing[TIMINGS_COMPUTE_DESCRIPTORS] << ", "  // Get descriptors [ms]
              // << sift._timing[TIMINGS_GENERATE_VBO] << ", "      // Gen. Display VBO [ms]
#if defined GPU_HESSIAN || defined GPU_SIFT_MODIFIED
              << sift._timing[TIMINGS_TOTAL]                        // Total time [ms]
#else
              << sift._timing[TIMINGS_FEATURES_REDUCTION]           // Total time [ms]
#endif // GPU_HESSIAN || GPU_SIFT_MODIFIED
              << std::endl;
        }
      }
    }
  }

  return EXIT_SUCCESS;
}
