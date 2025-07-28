# freeimpala executable
file(GLOB SHARED_SOURCES ../../src/*.cpp)

add_executable(freeimpala
  main.cpp
  ${SHARED_SOURCES}
)

target_link_libraries(freeimpala
  PRIVATE
    argparse::argparse                # header-only, from FetchContent
    spdlog::spdlog_header_only        # header-only, from FetchContent
    PahoMqttC::paho-mqtt3c            # imported C lib target
)

target_include_directories(freeimpala
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../../include
)
install(TARGETS freeimpala DESTINATION bin)
