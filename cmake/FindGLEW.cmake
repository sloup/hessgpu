#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARIES
# 

IF (WIN32)
	FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
		$ENV{PROGRAMFILES}/GLEW/include
		${PROJECT_SOURCE_DIR}/include
		DOC "The directory where GL/glew.h resides")
		
	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		SET (GLEW_LIB_NAME glew64 glew64s)
	else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		SET (GLEW_LIB_NAME glew32 glew32s)
	endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

	FIND_LIBRARY( GLEW_LIBRARY
		NAMES ${GLEW_LIB_NAME}
#		NAMES glew GLEW glew64 glew64s
		PATHS
		$ENV{PROGRAMFILES}/GLEW/lib
		${PROJECT_SOURCE_DIR}/bin
		${PROJECT_SOURCE_DIR}/lib
		DOC "The GLEW library")
ELSE (WIN32)
	FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where GL/glew.h resides")
	FIND_LIBRARY( GLEW_LIBRARY
		NAMES GLEW glew
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The GLEW library")
ENDIF (WIN32)

SET (GLEW_LIBRARIES
	${GLEW_LIBRARY}
)

IF (GLEW_INCLUDE_PATH)
	SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (GLEW_INCLUDE_PATH)
	SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLEW_FOUND )
