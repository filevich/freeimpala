#include <cmath>
#include <iostream>
#include <argparse/argparse.hpp>
#include "freeimpala/learner.h"
#include "freeimpala/agent.h"

int main(int argc, char** argv) {
    // Parse command line arguments
    argparse::ArgumentParser program("parallel-game-system");
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
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    
    // Extract parameters
    size_t num_players = program.get<int>("--players");
    size_t total_iterations = program.get<int>("--iterations");
    size_t entry_size = program.get<int>("--entry-size");
    size_t buffer_capacity = program.get<int>("--buffer-capacity");
    size_t batch_size = program.get<int>("--batch-size");
    size_t learner_time = program.get<int>("--learner-time");
    size_t checkpoint_freq = program.get<int>("--checkpoint-freq");
    std::string checkpoint_location = program.get<std::string>("--checkpoint-location");
    std::string starting_model = program.get<std::string>("--starting-model");
    size_t num_agents = program.get<int>("--agents");
    size_t game_steps = program.get<int>("--game-steps");
    size_t agent_time = program.get<int>("--agent-time");
    std::string metrics_file = program.get<std::string>("--metrics-file");

    // For thread-safe output printing
    std::stringstream ss;
    
    // Initialize metrics tracker
    auto metrics = MetricsTracker::getInstance();
    metrics->start();
    
    // Validate parameters
    if (batch_size > buffer_capacity) {
        std::cerr << "Error: Batch size must be less than buffer capacity" << std::endl;
        return 1;
    }
    
    if (game_steps > entry_size) {
        std::cerr << "Error: Game steps must be less than or equal to entry size" << std::endl;
        return 1;
    }
    
    // Initialize random seed
    std::srand(std::time(nullptr));
    
    // Create and start the learner
    ss << "Creating learner..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();

    size_t learner_iterations = ceil((num_agents * total_iterations) / batch_size);
    Learner learner(
        num_players,
        buffer_capacity,
        entry_size,
        batch_size,
        learner_time,
        checkpoint_freq,
        checkpoint_location,
        starting_model,
        learner_iterations
    );
    
    // Get shared resources for agents
    auto shared_buffers = learner.getSharedBuffers();
    auto model_manager = learner.getModelManager();
    
    // Start the learner
    ss << "Starting learner..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();
    learner.start();
    
    // Create and start agents
    ss << "Creating and starting " << num_agents << " agents..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();

    std::vector<std::shared_ptr<Agent>> agents;
    std::vector<std::thread> agent_threads;
    
    for (size_t a = 0; a < num_agents; a++) {
        auto agent = std::make_shared<Agent>(
            a,
            num_players,
            entry_size,
            game_steps,
            agent_time,
            total_iterations,
            shared_buffers,
            model_manager
        );
        
        agents.push_back(agent);
        
        // Launch agent in its own thread
        agent_threads.emplace_back([agent] {
            agent->run();
        });
    }
    
    // Wait for all agent threads to finish
    ss << "Waiting for agents to complete..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();

    for (auto& thread : agent_threads) {
        thread.join();
    }
    
    // Stop the learner
    ss << "Stopping learner..." << std::endl;
    std::cerr << ss.str();
    ss.str("");
    ss.clear();
    learner.stop();
    
    // Stop metrics collection
    // Allow final model syncs
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    metrics->stop();
    
    // Print metrics summary
    metrics->printMetricsSummary();
    
    // Save metrics to file if specified
    if (!metrics_file.empty()) {
        metrics->saveMetricsToCSV(metrics_file);
        ss << "Metrics saved to " << metrics_file << std::endl;
        std::cerr << ss.str();
        ss.str("");
        ss.clear();
    }
    
    ss << "Execution completed successfully!" << std::endl;
    std::cout << ss.str();
    ss.str("");
    ss.clear();
    return 0;
}