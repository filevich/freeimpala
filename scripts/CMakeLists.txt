# --- MPI Executable ---
# This file is included only if MPI is found

# Add the MPI-based executable from its source file
add_executable(freeimpala_mpi ${CMAKE_CURRENT_SOURCE_DIR}/../src/main_mpi.cpp)

# Link the MPI libraries to the executable
target_link_libraries(freeimpala_mpi
    PRIVATE
    MPI::MPI_CXX
)

# Include directories
target_include_directories(freeimpala_mpi
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Install the MPI executable
install(TARGETS freeimpala_mpi DESTINATION bin)