cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(freeimpala LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -pthread)

# — Fetch header-only libs —
include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG        v3.2
)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.10.0
)
FetchContent_MakeAvailable(argparse spdlog)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# — Find MPI —
find_package(MPI COMPONENTS CXX QUIET)
if(MPI_CXX_FOUND)
  message(STATUS "MPI found")
else()
  message(WARNING "MPI not found; MPI targets will be skipped")
endif()

# — Find LibTorch —
find_package(Torch QUIET)
if(Torch_FOUND)
  message(STATUS "LibTorch found")
else()
  message(WARNING "LibTorch not found; libtorch_bench will be skipped")
endif()

# — Find Paho MQTT C (pkg‑config style) —
find_path(PAHO_MQTT_C_INCLUDE_DIR
  NAMES MQTTClient.h
  HINTS $ENV{CMAKE_PREFIX_PATH}/include /usr/local/include /usr/include
)
find_library(PAHO_MQTT3C_LIBRARY
  NAMES paho-mqtt3c
  HINTS $ENV{CMAKE_PREFIX_PATH}/lib /usr/local/lib /usr/lib
)
if(NOT PAHO_MQTT_C_INCLUDE_DIR OR NOT PAHO_MQTT3C_LIBRARY)
  message(FATAL_ERROR "Paho MQTT C not found. Install paho-mqtt-c via brew/apt/spack.")
endif()
add_library(PahoMqttC::paho-mqtt3c UNKNOWN IMPORTED)
set_target_properties(PahoMqttC::paho-mqtt3c PROPERTIES
  IMPORTED_LOCATION           ${PAHO_MQTT3C_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${PAHO_MQTT_C_INCLUDE_DIR}
)

# — Add your binaries —
add_subdirectory(cmd/freeimpala)

if(MPI_CXX_FOUND)
  add_subdirectory(cmd/freeimpala_mpi_sync)
  add_subdirectory(cmd/freeimpala_mpi_async)
  add_subdirectory(cmd/freeimpala_mpi_async_pool)
endif()

if(Torch_FOUND)
  add_subdirectory(cmd/libtorch_bench)
endif()
