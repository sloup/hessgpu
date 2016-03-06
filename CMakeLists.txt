CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(SIFTGPU C CXX)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

#SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -Wno-deprecated -fPIC")
#SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wno-write-strings -Wno-unused-result -Wno-deprecated -fPIC")

# Build options
#option (SIFTGPU_BUILD_HESSIAN "Build HessGPU (otherwise modified SiftGPU will be built)" ON)

# packages                                                                                                                                                                                                 
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(DevIL REQUIRED)

# Target library name siftgpu
SET(TARGET_LIB_NAME "siftgpu")

SET(SIFTGPU_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

ADD_SUBDIRECTORY(src)
