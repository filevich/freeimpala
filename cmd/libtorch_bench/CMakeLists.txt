# --- libtorch_bench Executable ---
# Build rules for the libtorch_bench executable

# LibTorch should already be found by parent CMakeLists.txt
# But we need to apply the Torch flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# Optionally include shared source files from src/ if they exist
file(GLOB SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.cpp)
if(SHARED_SOURCES)
    list(APPEND SOURCES ${SHARED_SOURCES})
endif()

# Create executable
add_executable(libtorch_bench ${SOURCES})

# Set output directory for libtorch_bench
set_target_properties(libtorch_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Link libraries (LibTorch)
target_link_libraries(libtorch_bench
    PRIVATE
    "${TORCH_LIBRARIES}"
)

# Include directories
target_include_directories(libtorch_bench
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor
    ${TORCH_INCLUDE_DIRS}
)

# Set C++ standard for this target
set_property(TARGET libtorch_bench PROPERTY CXX_STANDARD 17)

# Install target
install(TARGETS libtorch_bench DESTINATION bin)