
# Try to find DevIL library and include path.
# Once done this will define
#
# DevIL_FOUND
# DevIL_INCLUDE_DIR
# DevIL_LIBRARIES
# DevIL_BINARIES
# 

SET (DevIL_BINARIES "")

IF (WIN32)
	FIND_PATH(DevIL_INCLUDE_DIR IL/il.h
		$ENV{PROGRAMFILES}/DevIL/include
		${PROJECT_SOURCE_DIR}/include
		DOC "The directory where il.h resides"
	)
			
	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		SET (DevIL_LIB_NAME DevIL64)
	else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		SET (DevIL_LIB_NAME DevIL)
	endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

	FIND_LIBRARY(DevIL_LIBRARY
		NAMES ${DevIL_LIB_NAME}
		PATHS
		$ENV{PROGRAMFILES}/DevIL/lib
		${PROJECT_SOURCE_DIR}/lib
		DOC "The DevIL library"
	)
	
	SET (DevIL_BINARIES 
		${PROJECT_SOURCE_DIR}/bin/DevIL.dll
	)
ELSE (WIN32)
	FIND_PATH(DevIL_INCLUDE_DIR IL/il.h
		/usr/local/include/
		/usr/include
	)

	SET (DevIL_LIB_PATHS
		/usr/local/lib/
		/usr/lib/
		/usr/lib32/
	)

	FIND_LIBRARY(DevIL_LIBRARY
		NAMES IL
		${DevIL_LIB_PATHS}
		DOC "The DevIL library"
	)
ENDIF (WIN32)

SET (DevIL_LIBRARIES
	${DevIL_LIBRARY}
)

IF (DevIL_INCLUDE_DIR)
	SET(DevIL_FOUND 1 CACHE STRING "Set to 1 if DevIL is found, 0 otherwise")
ELSE (DevIL_INCLUDE_DIR)
	SET(DevIL_FOUND 0 CACHE STRING "Set to 1 if DevIL is found, 0 otherwise")
ENDIF (DevIL_INCLUDE_DIR)

MARK_AS_ADVANCED( DevIL_FOUND )
