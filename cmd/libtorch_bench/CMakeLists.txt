# `${workspaceFolder}/cmd/libtorch_bench/CMakeLists.txt`

file(GLOB SHARED_SOURCES ../../src/*.cpp)

add_executable(libtorch_bench main.cpp ${SHARED_SOURCES})

target_link_libraries(libtorch_bench
  PRIVATE
    "${TORCH_LIBRARIES}"
    argparse::argparse                # header-only, from FetchContent
    spdlog::spdlog_header_only        # header-only, from FetchContent
)

target_include_directories(libtorch_bench
  PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

set_property(TARGET libtorch_bench PROPERTY CXX_STANDARD 17)
install(TARGETS libtorch_bench DESTINATION bin)
