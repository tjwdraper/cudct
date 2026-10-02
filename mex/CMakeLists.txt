# Compile the CUDA implementation into a shared library
find_package(CUDAToolkit REQUIRED)

add_library(cudct SHARED
    ${PROJECT_SOURCE_DIR}/src/cudct.cu
)

target_include_directories(cudct
    PUBLIC
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(cudct
    PUBLIC
        CUDA::cudart
        CUDA::cufft
)

set_target_properties(cudct
    PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

add_custom_command(TARGET cudct POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${PROJECT_SOURCE_DIR}/mex/matlab

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:cudct>
        ${PROJECT_SOURCE_DIR}/mex/matlab/$<TARGET_FILE_NAME:cudct>
)

# Add subdirectories for building MEX files for Matlab from the cudct library
add_subdirectory(matlab)