cmake_minimum_required(VERSION 3.10)
project(freeimpala CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")

# Add static linking flags (supported only by unix)
if (APPLE)
    message(STATUS "Configuring for macOS (AppleClang with libc++)")
    # Don't try static linking or -static-libstdc++/libgcc
elseif (UNIX AND NOT CMAKE_HOST_APPLE)
    message(STATUS "Configuring for Linux")
    # Only do this on Linux/GCC where it's valid
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
endif()

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Download argparse header-only library
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

# --- Multi-Thread Executable ---
# Add source files
set(SOURCES
    src/main.cpp
)

# Create executable
add_executable(freeimpala ${SOURCES})

# Link libraries
target_link_libraries(freeimpala
    PRIVATE
    argparse
)

# Include directories
target_include_directories(freeimpala
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Install target
install(TARGETS freeimpala DESTINATION bin)


# --- MPI Executable ---
# Find MPI and define the second executable if found
find_package(MPI REQUIRED)

if (MPI_CXX_FOUND)
    message(STATUS "MPI C++ bindings found. Building 'freeimpala_mpi'.")

    # Add the MPI-based executable from its source file
    add_executable(freeimpala_mpi src/main_mpi.cpp)

    # Link the MPI libraries to the executable.
    # The modern CMake approach with `MPI::MPI_CXX` also handles include directories automatically.
    target_link_libraries(freeimpala_mpi PRIVATE MPI::MPI_CXX)

    # You can still add your project's specific include directories if needed.
    target_include_directories(freeimpala_mpi
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Install the new executable alongside the original one
    install(TARGETS freeimpala_mpi DESTINATION bin)

else()
    message(WARNING "MPI C++ bindings not found. The 'freeimpala_mpi' target will not be built.")
endif()
