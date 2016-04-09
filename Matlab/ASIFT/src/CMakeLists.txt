CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Author : Pierre Moulon
# Date : December 2010.
# README :
#          The following cmake works for windows.
#	Openmp is accessible only on the professionnal version of Visual Studio.
# 	In order to use OPENMP in visual your have to add the preprocessor _OPENMP
# 	and enable OPENMP library in C/C++/Language.
PROJECT(libASIFT)

FIND_PACKAGE(OpenMP) 
if (OPENMP_FOUND) 
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

IF(MSVC)
   ADD_DEFINITIONS(/arch:SSE2)
ENDIF(MSVC)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)


set(SOURCES
	numerics1.cpp frot.cpp splines.cpp fproj.cpp
	library.cpp flimage.cpp filter.cpp
	demo_lib_sift.cpp compute_asift_keypoints.cpp
	
	computeH.cpp homography.cpp matrix.cpp 
    numerics.cpp rodrigues.cpp vector.cpp
	)
	
SET( LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/../lib/ )
add_library(libASIFT ${SOURCES})
