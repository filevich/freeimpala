# `${workspaceFolder}/cmd/freeimpala_mpi_async/CMakeLists.txt`

file(GLOB SHARED_SOURCES ../../src/*.cpp)

add_executable(freeimpala_mpi_async main.cpp ${SHARED_SOURCES})

target_link_libraries(freeimpala_mpi_async
  PRIVATE
    MPI::MPI_CXX
    argparse::argparse
    spdlog::spdlog_header_only
)

target_compile_definitions(freeimpala_mpi_async
  PRIVATE USE_MPI
)

install(TARGETS freeimpala_mpi_async DESTINATION bin)
