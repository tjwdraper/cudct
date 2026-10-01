cmake_minimum_required(VERSION 3.16)

project(cudct LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 61)

option(BUILD_FFTW "Build FFTW-based implementation" ON)
option(BUILD_MATLAB "Build Matlab EXecutable (Requires Matlab)" ON)
option(BUILD_OCTAVE "Build Octave Executable (Requires GNU Octave)" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(BUILD_FFTW)
    find_package(FFTW REQUIRED)
endif()

if (BUILD_OCTAVE)
    find_package(Octave REQUIRED)
endif()

if (BUILD_MATLAB)
    find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)

    if (NOT Matlab_FOUND)
        message(WARNING "Matlab not found.")
        set(BUILD_MATLAB OFF)
    endif()
endif()

add_subdirectory(mex)




# # ============================================================
# # Check dependencies
# # ============================================================

# # Check for FFTW library
# set(FFTW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/fftw")

# find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h HINTS "${FFTW_ROOT}")

# find_library(FFTW3_LIBRARY NAMES libfftw3-3 HINTS "${FFTW_ROOT}")
# find_library(FFTW3F_LIBRARY NAMES libfftw3f-3 HINTS "${FFTW_ROOT}")

# if(NOT FFTW3_INCLUDE_DIR)
#     message(FATAL_ERROR "Could not find fftw3.h")
# endif()

# if(NOT FFTW3_LIBRARY)
#     message(FATAL_ERROR "Could not find libfftw3-3.lib")
# endif()

# if(NOT FFTW3F_LIBRARY)
#     message(FATAL_ERROR "Could not find libfftw3f-3.lib")
# endif()

# message(STATUS "FFTW3 include: ${FFTW3_INCLUDE_DIR}")
# message(STATUS "FFTW3 library: ${FFTW3_LIBRARY}")
# message(STATUS "FFTW3F library: ${FFTW3F_LIBRARY}")

# Check for mkoctfile compiler
# find_program(MKOCTFILE_EXECUTABLE NAMES mkoctfile)

# if(MKOCTFILE_EXECUTABLE)
#     message(STATUS "Found mkoctfile: ${MKOCTFILE_EXECUTABLE}")
# else()
#     set(BUILD_OCTAVE OFF)
#     message(WARNING "mkoctfile not found. Octave executable will not be built.")
# endif()

# # Check for mex compiler
# find_package(Matlab QUIET COMPONENTS MEX_COMPILER)

# if (Matlab_MEX_COMPILER)
#     message(STATUS "Found mex: ${Matlab_MEX_COMPILER}")
# else()
#     set(BUILD_MATLAB OFF)
#     message(WARNING "mex not found. Matlab executable will not be built.")
# endif()

# # Check for nvcc compiler and cufft/cudart libraries
# find_program(NVCC_EXECUTABLE nvcc)
# find_library(CUFFT_LIBRARY cufft)
# find_library(CUDART_LIBRARY cudart)

# message(STATUS "FFTW3_INCLUDE_DIR = ${FFTW3_INCLUDE_DIR}")
# message(STATUS "FFTW3_LIBRARY     = ${FFTW3_LIBRARY}")
# message(STATUS "FFTW3F_LIBRARY    = ${FFTW3F_LIBRARY}")

# # ============================================================
# # Compilation
# # ============================================================

# # Compile the FFTW implementation
# if (BUILD_FFTW)
#     execute_process(
#         COMMAND ${MKOCTFILE_EXECUTABLE} -p OCTINCLUDEDIR
#         OUTPUT_VARIABLE OCTAVE_INCLUDE_DIR
#         OUTPUT_STRIP_TRAILING_WHITESPACE
#     )

#     if (EXISTS "${OCTAVE_INCLUDE_DIR}/mex.h")
#         message(STATUS "Octave include directory: ${OCTAVE_INCLUDE_DIR}")
#         message(STATUS "Octave MEX: ENABLED")

#         add_custom_target(octave_fftw ALL
#             COMMAND
#                 ${MKOCTFILE_EXECUTABLE}
#                 --mex
#                 -O3
#                 -I${PROJECT_SOURCE_DIR}/include
#                 -L${FFTW_ROOT}
#                 -lfftw3
#                 -lfftw3f
#                 -o
#                 ${PROJECT_SOURCE_DIR}/mex/octave_fftw.mex
#                 ${PROJECT_SOURCE_DIR}/src/wrapper_fftw.cpp

#             WORKING_DIRECTORY 
#                 ${PROJECT_SOURCE_DIR}
#         )
#     else()
#         message(STATUS "Octave include directory not found.")
#     endif()
# endif()