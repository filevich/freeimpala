#include <mpi.h>
#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <ctime>
#include <cmath>
#include "freeimpala/learner_mpi.h"
#include "freeimpala/agent_mpi.h"

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
};

// Setup argument parser with all parameters
argparse::ArgumentParser setupArgumentParser() {
    argparse::ArgumentParser program("freeimpala-mpi");
    program.add_description("MPI-based parallel consumer-producer system for game simulation");

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

    return program;
}

// Parse command line arguments and extract parameters
bool parseParameters(
    int argc,
    char** argv,
    ProgramParams& params
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
    params.seed = program.get<unsigned int>("--seed");

    return true;
}

// Validate parameters
bool validateParameters(const ProgramParams& params, int world_size) {
    std::stringstream ss;

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
    
    if (world_size != params.num_agents + 1) {
        ss << "Error: MPI world size (" << world_size << ") must equal num_agents + 1 (learner)" << std::endl;
        ss << "Please run with: mpirun -n " << (params.num_agents + 1) << " ..." << std::endl;
        std::cerr << ss.str();
        return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    ProgramParams params;
    
    // Parse and validate parameters
    if (!parseParameters(argc, argv, params)) {
        MPI_Finalize();
        return 1;
    }
    
    if (!validateParameters(params, world_size)) {
        MPI_Finalize();
        return 1;
    }
    
    // Set random seed with rank offset to ensure different seeds per process
    std::srand(params.seed + world_rank);
    
    // Initialize metrics tracker for this process
    auto metrics = MetricsTracker::getInstance();
    metrics->start();
    
    if (world_rank == 0) {
        // Rank 0 is the learner
        std::stringstream ss;
        ss << "Starting Learner process (rank 0)" << std::endl;
        std::cerr << ss.str();
        
        // Calculate learner iterations
        size_t learner_iterations = std::ceil(
            static_cast<double>(params.num_agents * params.total_iterations) / params.batch_size
        );
        
        // Create and run learner
        LearnerMPI learner(
            params.num_players,
            params.buffer_capacity,
            params.entry_size,
            params.batch_size,
            params.learner_time,
            params.checkpoint_freq,
            params.checkpoint_location,
            params.starting_model,
            learner_iterations,
            params.num_agents,
            world_size
        );
        
        learner.run();
        
        // Print learner metrics
        metrics->stop();
        ss.str("");
        ss << "\n===== Learner Metrics =====\n";
        std::cerr << ss.str();
        metrics->printMetricsSummary();
        
        if (!params.metrics_file.empty()) {
            std::string learner_metrics_file = params.metrics_file + ".learner.csv";
            metrics->saveMetricsToCSV(learner_metrics_file);
            ss.str("");
            ss << "Learner metrics saved to " << learner_metrics_file << std::endl;
            std::cerr << ss.str();
        }
        
    } else {
        // Other ranks are agents
        size_t agent_id = world_rank - 1;  // Agent IDs start at 0
        
        std::stringstream ss;
        ss << "Starting Agent " << agent_id << " (rank " << world_rank << ")" << std::endl;
        std::cerr << ss.str();
        
        // Create and run agent
        AgentMPI agent(
            world_rank,
            agent_id,
            params.num_players,
            params.entry_size,
            params.game_steps,
            params.agent_time,
            params.total_iterations
        );
        
        agent.run();
        
        // Print agent metrics
        metrics->stop();
        ss.str("");
        ss << "\n===== Agent " << agent_id << " Metrics =====\n";
        std::cerr << ss.str();
        metrics->printMetricsSummary();
        
        if (!params.metrics_file.empty()) {
            std::string agent_metrics_file = params.metrics_file + ".agent" + 
                                           std::to_string(agent_id) + ".csv";
            metrics->saveMetricsToCSV(agent_metrics_file);
            ss.str("");
            ss << "Agent " << agent_id << " metrics saved to " << agent_metrics_file << std::endl;
            std::cerr << ss.str();
        }
    }
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
