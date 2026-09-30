set(FFTW_ROOT "" CACHE PATH "Path to FFTW installation directory")

if(NOT FFTW_ROOT)
    set(FFTW_ROOT "${CMAKE_SOURCE_DIR}/fftw")
endif()

find_path(FFTW_INCLUDE_DIR NAMES fftw3.h HINTS "${FFTW_ROOT}")
find_library(FFTW3_LIBRARY NAMES fftw3 libfftw3-3 HINTS "${FFTW_ROOT}")
find_library(FFTW3F_LIBRARY NAMES fftw3f libfftw3f-3 HINTS "${FFTW_ROOT}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    FFTW
    REQUIRED_VARS
        FFTW_INCLUDE_DIR
        FFTW3_LIBRARY
        FFTW3F_LIBRARY
)

if (FFTW_FOUND)
    add_library(FFTW::FFTW3 UNKNOWN IMPORTED)
    set_target_properties(FFTW::FFTW3 PROPERTIES
        IMPORTED_LOCATION "${FFTW3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}"
    )
    
    add_library(FFTW::FFTW3F UNKNOWN IMPORTED)
    set_target_properties(FFTW::FFTW3F PROPERTIES
        IMPORTED_LOCATION "${FFTW3F_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}"
    )
else()
    message(WARNING "FFTW library not found. Please add FFTW to system path or create the FFTW folder in the working directory.") 
    set(BUILD_FFTW OFF)
endif()