# Compile the CUDA implementation into a shared library
find_package(CUDAToolkit REQUIRED)

add_library(transformer SHARED
    ${PROJECT_SOURCE_DIR}/src/transformer.cu
)

target_include_directories(transformer
    PUBLIC
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(transformer
    PUBLIC
        CUDA::cudart
        CUDA::cufft
)

set_target_properties(transformer
    PROPERTIES
        # LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/mex
        # RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/mex
        # ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/mex

        # RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/mex
        # LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/mex
        # ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/mex

        # RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/mex
        # LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/mex
        # ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/mex

        WINDOWS_EXPORT_ALL_SYMBOLS ON

        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

add_custom_command(TARGET transformer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${PROJECT_SOURCE_DIR}/mex/matlab

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:transformer>
        ${PROJECT_SOURCE_DIR}/mex/matlab/$<TARGET_FILE_NAME:transformer>
)

# Add subdirectories for building MEX files for Octave and Matlab from the transformer library
add_subdirectory(octave)
add_subdirectory(matlab)