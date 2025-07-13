# --- freeimpala_mpi_async Executable ---
# Build rules for the freeimpala_mpi_async executable (requires MPI)

# Add source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main_mpi_async.cpp
)

# Optionally include shared source files from src/ if they exist
file(GLOB SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.cpp)
if(SHARED_SOURCES)
    list(APPEND SOURCES ${SHARED_SOURCES})
endif()

# Create executable
add_executable(freeimpala_mpi_async ${SOURCES})

# Set output directory for freeimpala_mpi_async
set_target_properties(freeimpala_mpi_async PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Link the MPI library directly
target_link_libraries(freeimpala_mpi_async
    PRIVATE
    # argparse # this line is not needed anymore
    MPI::MPI_CXX
)

# Include directories (including MPI headers)
target_include_directories(freeimpala_mpi_async
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor
    ${MPI_CXX_INCLUDE_DIRS}
)

# Add the -DUSE_MPI compile definition ONLY for freeimpala_mpi_async
target_compile_definitions(freeimpala_mpi_async
    PRIVATE
    USE_MPI
)

# Install the MPI executable
install(TARGETS freeimpala_mpi_async DESTINATION bin)