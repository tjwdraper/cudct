# Build MEX file of FFTW-based implementation for Octave
if(BUILD_OCTAVE AND BUILD_FFTW)
    add_custom_target(octave_fftw ALL
        COMMAND
            ${MKOCTFILE_EXECUTABLE}
            --mex
            -O3
            -L${FFTW_ROOT}
            -lfftw3
            -lfftw3f
            -o  
            ${PROJECT_SOURCE_DIR}/mex/octave/octave_fftw.mex
            ${PROJECT_SOURCE_DIR}/src/wrapper_fftw.cpp

        WORKING_DIRECTORY
            ${PROJECT_SOURCE_DIR}
    )
endif()



# Build MEX file of CUDA-based implementation for Octave
# if(BUILD_OCTAVE)
#     add_custom_target(octave_cuda ALL
#         COMMAND
#             ${MKOCTFILE_EXECUTABLE}
#             --mex
#             -O3
#             "-I${CUDA_INCLUDE_DIRS}"
#             "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
#             -I${PROJECT_SOURCE_DIR}
#             -L${PROJECT_SOURCE_DIR}/mex
#             -ltransformer
#             "-Wl,-rpath=mex"
#             -o  
#             ${PROJECT_SOURCE_DIR}/mex/octave_cudct.mex
#             ${PROJECT_SOURCE_DIR}/src/wrapper_cudct.cpp

#         WORKING_DIRECTORY
#             ${PROJECT_SOURCE_DIR}
#     )
# endif()
