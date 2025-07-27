# --- freeimpala Executable ---
# Build rules for the freeimpala executable

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
add_executable(freeimpala ${SOURCES})

# Set output directory for freeimpala
set_target_properties(freeimpala PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Link libraries (only argparse, no MPI)
target_link_libraries(freeimpala
    PRIVATE
    # argparse # this line is not needed anymore
    mqtt_c
)

# Include directories
target_include_directories(freeimpala
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor
)

# Install target
install(TARGETS freeimpala DESTINATION bin)