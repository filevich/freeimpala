cmake_minimum_required(VERSION 3.10)
project(freeimpala CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")

# Add include directories (shared across all executables)
include_directories(
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

# Add subdirectories for executables
add_subdirectory(cmd/freeimpala)

# Find MPI and include freeimpala_mpi subdirectory if found
find_package(MPI)
if (MPI_CXX_FOUND)
    message(STATUS "MPI C++ bindings found. Including freeimpala_mpi subdirectory.")
    add_subdirectory(cmd/freeimpala_mpi)
else()
    message(WARNING "MPI C++ bindings not found. Skipping 'freeimpala_mpi' target.")
endif()