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
#include "freeimpala/learner.h"
#include "freeimpala/agent.h"
#include "freeimpala/utils.h"
#include "WeatherData.h"
// Include the Paho MQTT C++ client library
#include <MQTTClient.h>

// Structure to hold all command-line parameters
struct ProgramParams {
    size_t num_players;
    size_t total_iterations;
    size_t entry_size;
    size_t buffer_capacity;
    size_t batch_size;
    size_t learner_time;
    size_t checkpoint_freq;
    std::string checkpoint_location;
    std::string starting_model;
    size_t num_agents;
    size_t game_steps;
    size_t agent_time;
    std::string metrics_file;
    unsigned int seed;
    std::string log_level;
    std::string broker;
};

// Setup argument parser with all parameters
void setupArgumentParser(argparse::ArgumentParser& program) {
    program.add_description("Parallel consumer-producer system for game simulation");

    program.add_argument("--broker")
        .help("MQTT Broker")
        .default_value(std::string("localhost"));

    // General parameters
    program.add_argument("-p", "--players")
        .help("Number of players")
        .default_value(2)
        .scan<'i', int>();
        
    program.add_argument("-T", "--iterations")
        .help("Total number of iterations")
        .default_value(100)
        .scan<'i', int>();
    
    program.add_argument("-S", "--entry-size")
        .help("Size of each buffer entry (in terms of 1024-byte elements)")
        .default_value(100)
        .scan<'i', int>();
    
    // Learner parameters
    program.add_argument("-B", "--buffer-capacity")
        .help("Capacity of each shared buffer")
        .default_value(10)
        .scan<'i', int>();
        
    program.add_argument("-M", "--batch-size")
        .help("Number of entries to process in each batch")
        .default_value(5)
        .scan<'i', int>();
        
    program.add_argument("--learner-time")
        .help("Simulated training time for the learner (in ms)")
        .default_value(500)
        .scan<'i', int>();
        
    program.add_argument("-c", "--checkpoint-freq")
        .help("Checkpoint frequency (in iterations)")
        .default_value(10)
        .scan<'i', int>();
        
    program.add_argument("-l", "--checkpoint-location")
        .help("Location to store and load checkpoint files")
        .default_value(std::string("/tmp/freeimpala_checkpoints"));
        
    program.add_argument("-m", "--starting-model")
        .help("Starting model location")
        .default_value(std::string(""));
    
    // Agent parameters
    program.add_argument("-a", "--agents")
        .help("Number of agent processes")
        .default_value(4)
        .scan<'i', int>();
        
    program.add_argument("--game-steps")
        .help("Number of steps in each game simulation")
        .default_value(100)
        .scan<'i', int>();
    
    program.add_argument("--agent-time")
        .help("Simulated game play time for agents (in ms)")
        .default_value(200)
        .scan<'i', int>();
        
    program.add_argument("--metrics-file")
        .help("File to save performance metrics (CSV)")
        .default_value(std::string(""));

    program.add_argument("--seed")
        .help("Seed for random number generation")
        .default_value(static_cast<unsigned int>(std::time(nullptr)))
        .scan<'u', unsigned int>();
    
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
    argparse::ArgumentParser program("freeimpala");
    setupArgumentParser(program);
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        spdlog::error(err.what());
        std::cerr << program;
        return false;
    }

    params.num_players = program.get<int>("--players");
    params.total_iterations = program.get<int>("--iterations");
    params.entry_size = program.get<int>("--entry-size");
    params.buffer_capacity = program.get<int>("--buffer-capacity");
    params.batch_size = program.get<int>("--batch-size");
    params.learner_time = program.get<int>("--learner-time");
    params.checkpoint_freq = program.get<int>("--checkpoint-freq");
    params.checkpoint_location = program.get<std::string>("--checkpoint-location");
    params.starting_model = program.get<std::string>("--starting-model");
    params.num_agents = program.get<int>("--agents");
    params.game_steps = program.get<int>("--game-steps");
    params.agent_time = program.get<int>("--agent-time");
    params.metrics_file = program.get<std::string>("--metrics-file");
    params.seed = program.get<unsigned int>("--seed");
    params.log_level = program.get<std::string>("--log-level");
    params.broker = program.get<std::string>("--broker");

    return true;
}

// Validate parameters
bool validateParameters(const ProgramParams& params) {
    if (params.batch_size > params.buffer_capacity) {
        spdlog::error("Batch size must be less than buffer capacity");
        return false;
    }
    
    if (params.game_steps > params.entry_size) {
        spdlog::error("Game steps must be less than or equal to entry size");
        return false;
    }
    
    return true;
}

// Setup and start the learner
std::unique_ptr<Learner> setupLearner(const ProgramParams& params) {
    spdlog::info("Creating learner");

    // Correctly calculate learner iterations using floating-point division
    size_t learner_iterations = ceil((params.num_agents * params.total_iterations) / params.batch_size);

    // Create the Learner on the heap and wrap it in a std::unique_ptr.
    auto learner = std::make_unique<Learner>(
        params.num_players,
        params.buffer_capacity,
        params.entry_size,
        params.batch_size,
        params.learner_time,
        params.checkpoint_freq,
        params.checkpoint_location,
        params.starting_model,
        learner_iterations
    );

    spdlog::info("Starting learner");
    
    // Start learner
    learner->start();

    return learner;
}

// Setup and start agents
std::vector<std::thread> setupAgents(
    const ProgramParams& params, 
    std::vector<std::shared_ptr<Agent>>& agents,
    Learner& learner
) {
    spdlog::info("Creating and starting {} agents", params.num_agents);

    std::vector<std::thread> agent_threads;
    auto shared_buffers = learner.getSharedBuffers();
    auto model_manager = learner.getModelManager();

    for (size_t a = 0; a < params.num_agents; a++) {
        auto agent = std::make_shared<Agent>(
            a,
            params.num_players,
            params.entry_size,
            params.game_steps,
            params.agent_time,
            params.total_iterations,
            shared_buffers,
            model_manager
        );
        
        agents.push_back(agent);
        agent_threads.emplace_back([agent] { agent->run(); });
    }

    return agent_threads;
}

// Cleanup and finalize execution
void cleanup(
    const ProgramParams& params,
    Learner& learner, 
    std::vector<std::thread>& agent_threads
) {
    spdlog::info("Waiting for agents to complete");

    for (auto& thread : agent_threads) {
        thread.join();
    }

    spdlog::info("Stopping learner");
    learner.stop();

    auto metrics = MetricsTracker::getInstance();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    metrics->stop();
    
    metrics->printMetricsSummary();
    
    if (!params.metrics_file.empty()) {
        metrics->saveMetricsToCSV(params.metrics_file);
        spdlog::info("Metrics saved to {}", params.metrics_file);
    }

    spdlog::info("Execution completed successfully");
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

    std::cout << "Using params.broker=" << params.broker << std::endl;

    // Define the MQTT broker address and topic
    const std::string SERVER_ADDRESS = "tcp://" + params.broker + ":1883";
    const std::string CLIENT_ID = "freeimpala_learner";
    const std::string TOPIC = "demo/topic";
    const int QOS = 1; // Quality of Service (0: at most once, 1: at least once, 2: exactly once)

    // Declare a Paho MQTT C client handle
    MQTTClient client;
    // Declare connection options structure and initialize it with default values
    MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
    // Declare message structure and initialize it with default values
    MQTTClient_message pubmsg = MQTTClient_message_initializer;
    // Declare a delivery token to track message delivery
    MQTTClient_deliveryToken token;
    int rc; // Return code for Paho MQTT C functions

    // Random number generation setup
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the random number generator
    std::uniform_int_distribution<> distrib(1, 1000); // Distribution for random message content

    // Create the MQTT client instance
    // Arguments: client handle, server URI (C-string), client ID (C-string), persistence type, persistence context
    // For this example, we use MQTTCLIENT_PERSISTENCE_NONE as we don't need message persistence
    rc = MQTTClient_create(&client, SERVER_ADDRESS.c_str(), CLIENT_ID.c_str(),
                           MQTTCLIENT_PERSISTENCE_NONE, NULL);
    if (rc != MQTTCLIENT_SUCCESS) {
        std::cerr << "Failed to create client, return code: " << rc << std::endl;
        return EXIT_FAILURE;
    }

    // Set up connection options
    // Clean session ensures that no previous session information is used or stored
    conn_opts.cleansession = 1; // Use 1 for true, 0 for false in C API

    // Attempt to connect to the MQTT broker
    std::cout << "Attempting to connect to the MQTT broker at " << SERVER_ADDRESS << "..." << std::endl;
    // Arguments: client handle, connection options
    rc = MQTTClient_connect(client, &conn_opts);
    if (rc != MQTTCLIENT_SUCCESS) {
        std::cerr << "Failed to connect, return code: " << rc << std::endl;
        // Provide more detailed error message for common connection failures
        if (rc == MQTTCLIENT_DISCONNECTED) {
            std::cerr << "  (Broker not running or incorrect address/port?)" << std::endl;
        }
        MQTTClient_destroy(&client); // Clean up client before exiting
        return EXIT_FAILURE;
    }
    std::cout << "Successfully connected to the broker." << std::endl;

    // Publish 100 random messages
    for (int i = 1; i <= 10; ++i) {
        // Generate a random message content
        std::string message_content_str = "Random message " + std::to_string(distrib(gen));
        std::string payload_str = "Message num. " + std::to_string(i) + ": " + message_content_str;

        // Set the payload for the MQTT message
        // The C API expects a char* and the length of the payload
        pubmsg.payload = (void*)payload_str.c_str();
        pubmsg.payloadlen = payload_str.length();
        pubmsg.qos = QOS;
        pubmsg.retained = 0; // 0 for false, 1 for true

        // Publish the message
        std::cout << "Publishing to topic '" << TOPIC << "': " << payload_str << std::endl;
        // Arguments: client handle, topic (C-string), message structure, delivery token pointer
        rc = MQTTClient_publishMessage(client, TOPIC.c_str(), &pubmsg, &token);
        if (rc != MQTTCLIENT_SUCCESS) {
            std::cerr << "Failed to publish message, return code: " << rc << std::endl;
            // Continue trying to send messages, but log the error
        } else {
            // Wait for the message to be delivered (optional, but good for QoS > 0)
            // This blocks until the message is acknowledged by the broker
            rc = MQTTClient_waitForCompletion(client, token, 1000L); // 1000ms timeout
            if (rc != MQTTCLIENT_SUCCESS) {
                std::cerr << "Message delivery failed or timed out, return code: " << rc << std::endl;
            }
        }

        // Wait for 1 second before sending the next message
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Disconnect from the broker
    std::cout << "\nAll messages sent. Disconnecting from the broker..." << std::endl;
    // Arguments: client handle, timeout in milliseconds
    rc = MQTTClient_disconnect(client, 10000); // 10 seconds timeout for graceful disconnect
    if (rc != MQTTCLIENT_SUCCESS) {
        std::cerr << "Failed to disconnect, return code: " << rc << std::endl;
        MQTTClient_destroy(&client);
        return EXIT_FAILURE;
    }
    std::cout << "Disconnected." << std::endl;

    // Destroy the client to free resources
    MQTTClient_destroy(&client);
    
    // std::cout << "exiting!\n";
    if (2<3) {
        return 0;
    }

    Utils::init_logs(params.log_level);
    std::srand(params.seed);

    auto metrics = MetricsTracker::getInstance();
    metrics->start();

    auto learner = setupLearner(params);

    // Setup agents
    std::vector<std::shared_ptr<Agent>> agents;
    std::vector<std::thread> agent_threads = setupAgents(params, agents, *learner);

    // Cleanup
    cleanup(params, *learner, agent_threads);

    return 0; 
}