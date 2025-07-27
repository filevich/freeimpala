# Collect all C source files from mqtt_c
file(GLOB MQTT_C_SOURCES "*.c")
file(GLOB MQTT_C_HEADERS "*.h")

# Create a static library
add_library(mqtt_c STATIC ${MQTT_C_SOURCES})

# Set include directories for the library
target_include_directories(mqtt_c
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# If mqtt_c requires C99 or specific C standard
set_target_properties(mqtt_c PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)
