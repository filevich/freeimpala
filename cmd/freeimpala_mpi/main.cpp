#include <mpi.h>
#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
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
    size_t game_steps;
    size_t agent_time;
    std::string metrics_file;
    unsigned int seed;
};

// Setup argument parser with all parameters
argparse::ArgumentParser setupArgumentParser() {
    argparse::ArgumentParser program("freeimpala_mpi");
    program.add_description("Distributed IMPALA implementation using MPI");

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
    program.add_argument("--game-steps")
        .help("Number of steps in each game simulation")
        .default_value(100)
        .scan<'i', int>();
    
    program.add_argument("--agent-time")
        .help("Simulated game play time for agents (in ms)")
        .default_value(200)
        .scan<'i', int>();
        
    program.add_argument("--metrics-file")
        .help("Base name for metrics files (will append process type and rank)")
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
    std::stringstream ss;
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
    params.game_steps = program.get<int>("--game-steps");
    params.agent_time = program.get<int>("--agent-time");
    params.metrics_file = program.get<std::string>("--metrics-file");
    params.seed = program.get<unsigned int>("--seed");

    return true;
}

// Validate parameters
bool validateParameters(const ProgramParams& params) {
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
    
    return true;
}

// Broadcast parameters to all processes
void broadcastParameters(ProgramParams& params, int world_rank) {
    // Broadcast numerical parameters
    MPI_Bcast(&params.num_players, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.total_iterations, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.entry_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.buffer_capacity, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.batch_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.learner_time, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.checkpoint_freq, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.game_steps, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.agent_time, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Broadcast strings
    auto broadcast_string = [world_rank](std::string& str) {
        size_t length = 0;
        if (world_rank == 0) length = str.size();
        MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        char* buffer = new char[length + 1];
        if (world_rank == 0) strcpy(buffer, str.c_str());
        MPI_Bcast(buffer, length + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (world_rank != 0) str = std::string(buffer);
        delete[] buffer;
    };
    
    broadcast_string(params.checkpoint_location);
    broadcast_string(params.starting_model);
    broadcast_string(params.metrics_file);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ProgramParams params;
    bool parse_success = true;
    
    if (world_rank == 0) {
        parse_success = parseParameters(argc, argv, params);
        if (parse_success) {
            parse_success = validateParameters(params);
        }
    }
    
    // Broadcast parse status
    MPI_Bcast(&parse_success, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!parse_success) {
        MPI_Finalize();
        return 1;
    }

    // Broadcast parameters to all processes
    broadcastParameters(params, world_rank);

    // Set random seed per process
    std::srand(params.seed + world_rank);

    // Initialize metrics
    auto metrics = MetricsTracker::getInstance();
    metrics->start();

    // Generate metrics filename based on process type
    std::string metrics_file = params.metrics_file;
    if (!metrics_file.empty()) {
        if (world_rank == 0) {
            metrics_file += "_learner.csv";
        } else {
            metrics_file += "_agent_" + std::to_string(world_rank - 1) + ".csv";
        }
    }

    if (world_rank == 0) {
        // Learner process
        LearnerMPI learner(
            params.num_players,
            params.buffer_capacity,
            params.entry_size,
            params.batch_size,
            params.learner_time,
            params.checkpoint_freq,
            params.checkpoint_location,
            params.starting_model,
            world_size - 1,  // num_agents = world_size - 1
            params.total_iterations
        );
        learner.run();
    } else {
        // Agent process
        AgentMPI agent(
            world_rank - 1,  // agent_id
            params.num_players,
            params.entry_size,
            params.game_steps,
            params.agent_time,
            params.total_iterations
        );
        agent.run();
    }

    // Finalize metrics
    metrics->printMetricsSummary();
    if (!metrics_file.empty()) {
        metrics->saveMetricsToCSV(metrics_file);
    }
    
    MPI_Finalize();
    return 0;
}