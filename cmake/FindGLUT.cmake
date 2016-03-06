# - try to find glut library and include files
#  GLUT_INCLUDE_DIR, where to find GL/glut.h, etc.
#  GLUT_LIBRARIES, the libraries to link against
#  GLUT_FOUND, If false, do not try to use GLUT.
# Also defined, but not for general use are:
#  GLUT_glut_LIBRARY = the full path to the glut library.
#  GLUT_Xmu_LIBRARY  = the full path to the Xmu library.
#  GLUT_Xi_LIBRARY   = the full path to the Xi Library.

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

if (EXISTS $ENV{GLUT})
	set (GLUT_ROOT_PATH $ENV{GLUT} CACHE PATH "Path to glut")
else (EXISTS $ENV{GLUT})
	message("WARNING: GLUT enviroment variable is not set")
	set (GLUT_ROOT_PATH "" CACHE PATH "Path to glut")
endif (EXISTS $ENV{GLUT})

IF (WIN32)
  FIND_PATH( GLUT_INCLUDE_DIR NAMES GL/glut.h 
    PATHS
    ${GLUT_ROOT_PATH}/include
    ${PROJECT_SOURCE_DIR}/include
    DOC "The directory where GL/glut.h resides" )

	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		SET (GLUT_LIB_NAME glut64)
	else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		SET (GLUT_LIB_NAME glut32 glut freeglut)
	endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	
  FIND_LIBRARY( GLUT_glut_LIBRARY NAMES ${GLUT_LIB_NAME}
    PATHS
    ${OPENGL_LIBRARY_DIR}
    ${GLUT_ROOT_PATH}/lib
    ${GLUT_ROOT_PATH}/Release
    ${PROJECT_SOURCE_DIR}/bin
    ${PROJECT_SOURCE_DIR}/lib
    DOC "The GLUT library" )
	
	add_definitions(-DGLUT_NO_LIB_PRAGMA)
ELSE (WIN32)
  
  IF (APPLE)
    # These values for Apple could probably do with improvement.
    FIND_PATH( GLUT_INCLUDE_DIR glut.h
      /System/Library/Frameworks/GLUT.framework/Versions/A/Headers
      ${OPENGL_LIBRARY_DIR}
      )
    SET(GLUT_glut_LIBRARY "-framework GLUT" CACHE STRING "GLUT library for OSX") 
    SET(GLUT_cocoa_LIBRARY "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
  ELSE (APPLE)
    
    FIND_PATH( GLUT_INCLUDE_DIR GL/glut.h
      /usr/include
      /usr/openwin/share/include
      /usr/openwin/include
      /opt/graphics/OpenGL/include
      /opt/graphics/OpenGL/contrib/libglut
      )
  
    FIND_LIBRARY( GLUT_glut_LIBRARY glut
      /usr/lib64
      /usr/lib
      /usr/openwin/lib
      )
  ENDIF (APPLE)
  
ENDIF (WIN32)

SET( GLUT_FOUND "NO" )
IF(GLUT_INCLUDE_DIR)
  IF(GLUT_glut_LIBRARY)
    # Is -lXi and -lXmu required on all platforms that have it?
    # If not, we need some way to figure out what platform we are on.
    SET( GLUT_LIBRARIES
      ${GLUT_glut_LIBRARY}
      ${GLUT_Xmu_LIBRARY}
      ${GLUT_Xi_LIBRARY} 
      ${GLUT_cocoa_LIBRARY}
      )
    SET( GLUT_FOUND "YES" )
    
    #The following deprecated settings are for backwards compatibility with CMake1.4
    SET (GLUT_LIBRARY ${GLUT_LIBRARIES})
    SET (GLUT_INCLUDE_PATH ${GLUT_INCLUDE_DIR})
    
  ENDIF(GLUT_glut_LIBRARY)
ENDIF(GLUT_INCLUDE_DIR)
