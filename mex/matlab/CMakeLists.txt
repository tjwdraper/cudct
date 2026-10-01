# Build MEX file of FFTW-based implementation for Matlab
if(BUILD_MATLAB AND BUILD_FFTW)
    matlab_add_mex(
        NAME 
            matlab_fftw

        SRC
            ${PROJECT_SOURCE_DIR}/src/wrapper_fftw.cpp

        LINK_TO
            FFTW::FFTW3
            FFTW::FFTW3F
    )

    add_custom_command(TARGET matlab_fftw POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:matlab_fftw>
            ${PROJECT_SOURCE_DIR}/mex/matlab/$<TARGET_FILE_NAME:matlab_fftw>
    )  
endif()

if (BUILD_MATLAB)
    matlab_add_mex(
        NAME 
            matlab_cudct

        SRC
            ${PROJECT_SOURCE_DIR}/src/wrapper_cudct.cpp

        LINK_TO
            transformer
    )

    add_custom_command(TARGET matlab_cudct POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:matlab_cudct>
            ${PROJECT_SOURCE_DIR}/mex/matlab/$<TARGET_FILE_NAME:matlab_cudct>
    )   
endif()

if (BUILD_MATLAB)
    matlab_add_mex(
        NAME
            matlab_cudct_static

        SRC
            ${PROJECT_SOURCE_DIR}/src/wrapper_cudct_static.cpp

        LINK_TO
            transformer
    )

    add_custom_command(TARGET matlab_cudct_static POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:matlab_cudct_static>
            ${PROJECT_SOURCE_DIR}/mex/matlab/$<TARGET_FILE_NAME:matlab_cudct_static>
    )
endif()

if (BUILD_MATLAB)
    matlab_add_mex(
        NAME
            matlab_cudct_gpuarray

        SRC
            ${PROJECT_SOURCE_DIR}/src/wrapper_cudct_gpuarray.cpp

        LINK_TO
            transformer
            "${Matlab_ROOT_DIR}/extern/lib/win64/microsoft/gpu.lib"
    )

    target_include_directories(matlab_cudct_gpuarray
        PRIVATE
            "${Matlab_ROOT_DIR}/toolbox/distcomp/gpu/extern/include"
    )

    add_custom_command(TARGET matlab_cudct_gpuarray POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:matlab_cudct_gpuarray>
            ${PROJECT_SOURCE_DIR}/mex/matlab/$<TARGET_FILE_NAME:matlab_cudct_gpuarray>
    )
endif()