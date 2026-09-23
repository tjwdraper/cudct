cmake_minimum_required(VERSION 3.16)

project(CUDCT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

###################################################################################################
## Mex Setup
###################################################################################################
find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)

message(STATUS "MATLAB_ROOT: ${Matlab_ROOT_DIR}")
message(STATUS "MATLAB_VERSION: ${Matlab_VERSION_STRING}")

###################################################################################################
## FFTW Setup
###################################################################################################
set(FFTW_ROOT "${PROJECT_SOURCE_DIR}/fftw3")

add_library(fftw3 SHARED IMPORTED)

set_target_properties(fftw3 PROPERTIES
    IMPORTED_IMPLIB
        "${FFTW_ROOT}/libfftw3-3.lib"

    IMPORTED_LOCATION
        "${FFTW_ROOT}/libfftw3-3.dll"

    INTERFACE_INCLUDE_DIRECTORIES
        "${FFTW_ROOT}"
)

###################################################################################################
## Compile MEX Function
###################################################################################################

# 2d DCT
matlab_add_mex(
    NAME matlab_dct2
    SRC
        src/wrapper_dct2.cpp
    LINK_TO
        fftw3
)

add_custom_command(TARGET matlab_dct2 POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:matlab_dct2>"
        "${PROJECT_SOURCE_DIR}/mex/$<TARGET_FILE_NAME:matlab_dct2>"

    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${FFTW_ROOT}/libfftw3-3.dll"
        "${PROJECT_SOURCE_DIR}/mex/libfftw3-3.dll"
)

#3d DCT
matlab_add_mex(
    NAME matlab_dct3
    SRC
        src/wrapper_dct3.cpp
    LINK_TO
        fftw3
)

add_custom_command(TARGET matlab_dct3 POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:matlab_dct3>"
        "${PROJECT_SOURCE_DIR}/mex/$<TARGET_FILE_NAME:matlab_dct3>"

    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${FFTW_ROOT}/libfftw3-3.dll"
        "${PROJECT_SOURCE_DIR}/mex/libfftw3-3.dll"
)