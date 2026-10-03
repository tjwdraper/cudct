find_package(CUDAToolkit REQUIRED)

add_executable(benchmark_cudct
    ${CMAKE_SOURCE_DIR}/src/benchmark_cudct.cu
)

target_link_libraries(benchmark_cudct
    PRIVATE
        cudct
        CUDA::cudart
)

add_custom_command(TARGET benchmark_cudct POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:benchmark_cudct>
        ${PROJECT_SOURCE_DIR}/tests/cuda/$<TARGET_FILE_NAME:benchmark_cudct>
)