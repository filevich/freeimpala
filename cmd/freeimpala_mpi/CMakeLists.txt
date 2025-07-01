# --- freeimpala_mpi Executable ---
# Build rules for the freeimpala_mpi executable (requires MPI)

# Add source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main_mpi.cpp
)

# Optionally include shared source files from src/ if they exist
file(GLOB SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.cpp)
if(SHARED_SOURCES)
    list(APPEND SOURCES ${SHARED_SOURCES})
endif()

# Create executable
add_executable(freeimpala_mpi ${SOURCES})

# Set output directory for freeimpala_mpi
set_target_properties(freeimpala_mpi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Link the MPI library directly
target_link_libraries(freeimpala_mpi
    PRIVATE
    argparse
    MPI::MPI_CXX
)

# Include directories (including MPI headers)
target_include_directories(freeimpala_mpi
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${MPI_CXX_INCLUDE_DIRS}
)

# Install the MPI executable
install(TARGETS freeimpala_mpi DESTINATION bin)