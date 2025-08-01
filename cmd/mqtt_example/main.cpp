#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>
#include <ctime>
#include <cmath>
#include <random>
#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include "signals/mqtt_broker.h"
#include "signals/simple_serializer.h"

struct WeatherData {
    int temperature;
    std::string location;
    float wind;

    WeatherData() : temperature(0), wind(0.0f) {}
    WeatherData(int temp, const std::string& loc, float w)
        : temperature(temp), location(loc), wind(w) {}

    // Method to serialize the current object
    std::string serialize() const {
        return std::to_string(temperature) + "|" +
               location + "|" +
               std::to_string(wind);
    }

    // Method to update the current object by deserializing a string
    // Returns a reference to the modified object for chaining
    WeatherData& deserialize(const std::string& str) {
        size_t pos1 = str.find('|');
        size_t pos2 = str.find('|', pos1 + 1);

        if (pos1 == std::string::npos || pos2 == std::string::npos) {
            throw std::runtime_error("Invalid weather data format");
        }
        
        // Update the member variables of the current object
        temperature = std::stoi(str.substr(0, pos1));
        location = str.substr(pos1 + 1, pos2 - pos1 - 1);
        wind = std::stof(str.substr(pos2 + 1));

        return *this; // Return a reference to the modified object
    }
};

// Structure to hold all command-line parameters
struct ProgramParams {
    std::string broker;
    std::string log_level;
};

// Setup argument parser with all parameters
void setupArgumentParser(argparse::ArgumentParser& program) {
    program.add_description("Parallel consumer-producer system for game simulation");

    program.add_argument("--broker")
        .help("MQTT Broker")
        .default_value(std::string("tcp://localhost:1883"));
    
    // Log level parameter with restricted choices
    program.add_argument("--log-level")
        .help("Set the logging level")
        .default_value("info")
        .choices("trace", "debug", "info", "warn", "error", "critical", "off");
}

// Parse command line arguments and extract parameters
bool parseParameters(
    int argc,
    char** argv,
    ProgramParams& params
) {
    argparse::ArgumentParser program("mqtt_example");
    setupArgumentParser(program);
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        spdlog::error(err.what());
        std::cerr << program;
        return false;
    }

    params.log_level = program.get<std::string>("--log-level");
    params.broker = program.get<std::string>("--broker");

    return true;
}

// Validate parameters
bool validateParameters(const ProgramParams& params) {
    return true;
}

int publishRandomMessages(std::string broker_addr, std::string topic = "demo/topic") {
    try {
        // Create MQTT broker instance
        signals::MqttBroker broker(broker_addr, "freeimpala_learner");
        
        // Connect to broker
        if (!broker.connect()) {
            return EXIT_FAILURE;
        }
        
        // Random number generators
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> temp_dist(-20, 40);
        std::uniform_real_distribution<> wind_dist(0.0, 50.0);
        
        std::vector<std::string> locations = {"Miami", "New York", "London", "Tokyo", "Sydney"};
        std::uniform_int_distribution<> loc_dist(0, locations.size() - 1);
        
        // Publish 10 random messages
        for (int i = 1; i <= 10; ++i) {
            // Generate random weather data
            WeatherData weather(
                temp_dist(gen),
                locations[loc_dist(gen)],
                static_cast<float>(wind_dist(gen))
            );

            // Serialize the data
            std::vector<std::pair<std::string, std::string>> fields = {
                {"temperature", std::to_string(weather.temperature)},
                {"location", weather.location},
                {"wind", std::to_string(weather.wind)}
            };
            std::string payload = SimpleSerializer::serialize(fields);

            spdlog::info("Publishing to topic '{}' (using SimpleSerializer): {}", topic, payload);
            if (!broker.publish(topic, payload)) {
                spdlog::error("Failed to publish message {}", i);
                // Continue with next message
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        
        spdlog::info("All messages sent");
        // Destructor automatically handles disconnection and cleanup
        
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return 0;
}

int subscribeExample(std::string broker_addr) {
    try {
        signals::MqttBroker broker(broker_addr, "subscriber_client");
        
        // Set up message handler - this will be called automatically
        // whenever messages arrive on subscribed topics
        broker.setMessageHandler([](const std::string& topic, const std::string& message) {
            spdlog::info("Received message on topic '{}': {}", topic, message);
            if (topic == "weather/serialied") {
                WeatherData data;
                data.deserialize(message);
                spdlog::info("Parsed weather data - Temperature: {}°C, Location: {}, Wind: {} m/s", 
                             data.temperature, data.location, data.wind);
            }
        });
        
        if (!broker.connect()) {
            spdlog::error("Failed to connect");
            return EXIT_FAILURE;
        }
        
        // Subscribe to topic
        if (!broker.subscribe("#")) { // "#" is a wildcard for "everything"
            spdlog::error("Failed to subscribe");
            return EXIT_FAILURE;
        }
        
        std::cout << "Listening for messages... Messages will arrive automatically via callbacks!" << std::endl;
        std::cout << "Messages on topic `weather` will be parsed using its custom decoder" << std::endl;
        std::cout << "No explicit loop needed - just keep the main thread alive." << std::endl;
        std::cout << "Press Enter to exit..." << std::endl;

        // tip: to test the sub use the command `docker run --rm --network=host eclipse-mosquitto mosquitto_pub -h 0.0.0.0 -p 1883 -t "weather/serialied" -m `
        // followed by some of these examples:
        // "32|Sydney|36.599323"
        // "-5|Sydney|46.730946"
        // "-14|New York|10.132723"
        // "-11|Sydney|24.367514"
        // "13|Sydney|42.314491"
        // "34|Tokyo|49.792366"
        
        // Just keep the main thread alive - that's all we need!
        // The MQTT library handles everything else in background threads
        std::cin.get();
        
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
    }
    return 0;
}

int main(int argc, char** argv) {
    ProgramParams params;

    // Parse and validate parameters
    if (!parseParameters(argc, argv, params)) {
        return 1;
    }
    
    if (!validateParameters(params)) {
        return 1;
    }

    spdlog::info("Using params.broker={}", params.broker);

    publishRandomMessages(params.broker);
    subscribeExample(params.broker);

    return 0; 
}