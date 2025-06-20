#include <cmath>
#include <iostream>
#include <argparse/argparse.hpp>
#include "freeimpala/learner.h"
#include "freeimpala/agent.h"

#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>
#include <ctime>
#include <cmath>

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
};

// Setup argument parser with all parameters
argparse::ArgumentParser setupArgumentParser() {
    argparse::ArgumentParser program("freeimpala");
    program.add_description("Parallel consumer-producer system for game simulation");

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

    return program;
}

// Parse command line arguments and extract parameters
bool parseParameters(
    int argc,
    char** argv,
    ProgramParams& params,
    std::stringstream& ss
) {
    auto program = setupArgumentParser();
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
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

    return true;
}

// Validate parameters
bool validateParameters(const ProgramParams& params, std::stringstream& ss) {
    if (params.batch_size > params.buffer_capacity) {
        ss << "Error: Batch size must be less than buffer capacity" << std::endl;
        std::cerr << ss.str();
        return false;
    }
    
    if (params.game_steps > params.entry_size) {
        ss << "Error: Game steps must be less than or equal to entry size" << std::endl;
        std::cerr << ss.str();
        return false;
    }
    
    return true;
}

// Setup and start the learner
std::unique_ptr<Learner> setupLearner(const ProgramParams& params, std::stringstream& ss) {
    ss << "Creating learner..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();

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

    ss << "Starting learner..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();
    
    // Start learner
    learner->start();

    return learner;
}

// Setup and start agents
std::vector<std::thread> setupAgents(
    const ProgramParams& params, 
    std::vector<std::shared_ptr<Agent>>& agents,
    Learner& learner,
    std::stringstream& ss
) {
    ss << "Creating and starting " << params.num_agents << " agents..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();

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
    std::vector<std::thread>& agent_threads, 
    std::stringstream& ss
) {
    ss << "Waiting for agents to complete..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();

    for (auto& thread : agent_threads) {
        thread.join();
    }

    ss << "Stopping learner..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();
    learner.stop();

    auto metrics = MetricsTracker::getInstance();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    metrics->stop();
    
    metrics->printMetricsSummary();
    
    if (!params.metrics_file.empty()) {
        metrics->saveMetricsToCSV(params.metrics_file);
        ss << "Metrics saved to " << params.metrics_file << std::endl;
        std::cerr << ss.str();
        ss.str("");
        ss.clear();
    }

    ss << "Execution completed successfully!" << std::endl;
    std::cerr << ss.str();
}

int main(int argc, char** argv) {
    std::stringstream ss;
    ProgramParams params;

    // Initialize metrics and random seed
    auto metrics = MetricsTracker::getInstance();
    metrics->start();
    std::srand(std::time(nullptr));

    // Parse and validate parameters
    if (!parseParameters(argc, argv, params, ss)) {
        return 1;
    }
    
    if (!validateParameters(params, ss)) {
        return 1;
    }

    auto learner = setupLearner(params, ss);

    // Setup agents
    std::vector<std::shared_ptr<Agent>> agents;
    // Dereference the pointer with '*' to pass the Learner object by reference.
    std::vector<std::thread> agent_threads = setupAgents(params, agents, *learner, ss);

    // Cleanup
    // Dereference the pointer here as well.
    cleanup(params, *learner, agent_threads, ss);

    return 0; 
    // As main ends, 'learner' goes out of scope, and the unique_ptr's 
    // destructor is called, which safely deletes the Learner object.
}