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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(BUILD_FFTW)
    find_package(FFTW REQUIRED)
endif()

if (BUILD_MATLAB)
    find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)

    if (NOT Matlab_FOUND)
        message(WARNING "Matlab not found.")
        set(BUILD_MATLAB OFF)
    endif()
endif()

add_subdirectory(mex)