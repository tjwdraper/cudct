add_executable(benchmark_fftw
    ${CMAKE_SOURCE_DIR}/src/benchmark_fftw.cpp
)

target_link_libraries(benchmark_fftw
    PRIVATE
        FFTW::FFTW3
)

add_custom_command(TARGET benchmark_fftw POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:benchmark_fftw>
        ${PROJECT_SOURCE_DIR}/tests/cpp/$<TARGET_FILE_NAME:benchmark_fftw>
)