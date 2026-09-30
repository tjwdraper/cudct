find_program(MKOCTFILE_EXECUTABLE NAMES mkoctfile)

# Use the mkoctfile executable to get the Octave include directory
if(MKOCTFILE_EXECUTABLE)
    execute_process(
        COMMAND
            ${MKOCTFILE_EXECUTABLE}
            -p
            OCTINCLUDEDIR
            OUTPUT_VARIABLE
            OCTAVE_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    Octave
    REQUIRED_VARS
        MKOCTFILE_EXECUTABLE
        OCTAVE_INCLUDE_DIR
)

if (NOT Octave_FOUND)
    message(WARNING "Octave not found. Please add the location of mkoctfile to system path or create the Octave folder in the working directory.")
    set(BUILD_OCTAVE OFF)
endif()