# `${workspaceFolder}/cmd/mqtt_example/CMakeLists.txt`

# mqtt_example executable
file(GLOB SHARED_SOURCES ../../src/*.cpp)

add_executable(mqtt_example
  main.cpp
  ${SHARED_SOURCES}
)

target_link_libraries(mqtt_example
  PRIVATE
    argparse::argparse                # header-only, from FetchContent
    spdlog::spdlog_header_only        # header-only, from FetchContent
    PahoMqttC::paho-mqtt3c            # imported C lib target
)

target_include_directories(mqtt_example
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../../include
    ${PAHO_MQTT_C_INCLUDE_DIR}
)

install(TARGETS mqtt_example DESTINATION bin)
