#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>
#include <ctime>
#include <cmath>
#include "freeimpala/learner.h"
#include "freeimpala/agent.h"
#include <mpi.h>
#include <atomic>
#include <cstdint>
#include <cstring>

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

void process_tag(
    const int tag,
    const std::shared_ptr<ModelManager>& models,
    const int src,
    std::atomic<int>& done_actors,
    int nbyt,
    const std::vector<std::shared_ptr<SharedBuffer>>& buffers
) {
    switch (tag)
    {
    case MessageTag::TAG_VERSION_REQ: {
            uint32_t p;
            MPI_Recv(&p, 1, MPI_UNSIGNED, src, MessageTag::TAG_VERSION_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            uint32_t v = models->getLatestVersion(p);
            if (MPI_Send(&v, 1, MPI_UNSIGNED, src, MessageTag::TAG_VERSION_RES, MPI_COMM_WORLD) != MPI_SUCCESS) {
                spdlog::error("Failed to send version response to rank {} for player {} ", src, p);
            }
            break;
        }

    case MessageTag::TAG_WEIGHTS_REQ: {
            uint32_t p;
            MPI_Recv(&p, 1, MPI_UNSIGNED, src, MessageTag::TAG_WEIGHTS_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            auto modelCopy = models->getModel(p)->createCopy();

            /* pack version + bytes */
            uint32_t v = modelCopy->getVersion();
            auto sz = modelCopy->getData().size();
            std::vector<uint8_t> out(sizeof(uint32_t) + sz);
            std::memcpy(out.data(), &v, sizeof(uint32_t));
            std::memcpy(out.data()+sizeof(uint32_t), modelCopy->getData().data(), sz);

            if (MPI_Send(out.data(), out.size(), MPI_BYTE, src, MessageTag::TAG_WEIGHTS_RES, MPI_COMM_WORLD) != MPI_SUCCESS) {
                spdlog::error("Failed to send weights response to rank {} for player {}", src, p);
            }
            break;
        }

    case MessageTag::TAG_TERMINATE: {
            MPI_Recv(nullptr, 0, MPI_CHAR, src, MessageTag::TAG_TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            done_actors.fetch_add(1);
            break;
        }

    default: {
            // it's a `TAG_TRAJECTORY`
            // we need to "extract" the `player_idx` from the tag as follows
            int player_idx = tag - MessageTag::TAG_TRAJECTORY_BASE;

            std::vector<char> data(nbyt);
            MPI_Recv(data.data(), nbyt, MPI_CHAR, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Write into the learner's local shared buffer – blocks if full
            buffers[player_idx]->write(data);
            break;
        }
    }
}

// rank-0 (learner) thread
void mpi_receiver(
    const ProgramParams& params,
    const std::vector<std::shared_ptr<SharedBuffer>>& buffers,
    std::atomic<int>& done_actors,
    int world_size,
    const std::shared_ptr<ModelManager>& models,
    std::size_t max_msg_bytes
) {
    while (done_actors.load() < world_size - 1) {
        MPI_Status st;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        const int tag  = st.MPI_TAG;
        const int src  = st.MPI_SOURCE;
        int nbyt = 0;
        MPI_Get_count(&st, MPI_BYTE, &nbyt);
        process_tag(
            tag,
            models,
            src,
            done_actors,
            nbyt,
            buffers
        );
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        spdlog::error("MPI library does not provide MPI_THREAD_MULTIPLE");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ProgramParams params;
    if (!parseParameters(argc, argv, params) || !validateParameters(params)) {
        MPI_Finalize(); return 1;
    }

    // override num_agents
    params.num_agents = world_size - 1;

    // learner process
    if (rank == 0) {
        spdlog::info("slim sync");
        auto metrics = MetricsTracker::getInstance();
        metrics->start();

        auto learner         = setupLearner(params);
        auto shared_buffers  = learner->getSharedBuffers();

        std::atomic<int> terminated{0};

        std::size_t max_traj_bytes = params.entry_size * ELEMENT_SIZE;
        const std::size_t MAX_MSG_BYTES = std::max<std::size_t>(
            max_traj_bytes,
            sizeof(uint64_t) + learner->getModelManager()->getModel(0)->getData().size()
        );

        mpi_receiver(
            std::cref(params),
            std::cref(shared_buffers),
            std::ref(terminated),
            world_size,
            learner->getModelManager(),
            MAX_MSG_BYTES
        );

        // learner's worker threads already started by setupLearner()
        // wait until all actors have finished

        learner->stop();
        metrics->stop();
        metrics->printMetricsSummary();
        MPI_Finalize();
        return 0;
    }

    // actor processes
    {
        // Each rank>0 owns ONE agent. Re-use existing class.
        // We pass dummy shared_buffers because the MPI send happens inside
        // Agent::transferThread (see next section).
        std::vector<std::shared_ptr<SharedBuffer>> dummy;
        auto dummy_model_mgr = std::make_shared<ModelManager>(params.num_players, 6*1024*1024, params.checkpoint_location);

        Agent agent(
                    rank - 1,
                    params.num_players,
                    params.entry_size,
                    params.game_steps,
                    params.agent_time,
                    params.total_iterations,
                    dummy, dummy_model_mgr
                );

        agent.run(); // same loop as before

        // Tell learner we are done
        if (MPI_Send(nullptr, 0, MPI_CHAR, 0, MessageTag::TAG_TERMINATE, MPI_COMM_WORLD) != MPI_SUCCESS) {
            spdlog::error("Failed to send TAG_TERMINATE message to learner from rank {}", rank);
        }
    }

    MPI_Finalize();
    return 0;
}