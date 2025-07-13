#include <argparse/argparse.hpp>
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

// Setup and start the learner
std::unique_ptr<Learner> setupLearner(const ProgramParams& params) {
    std::stringstream ss;
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

void process_tag(
    const int tag,
    std::vector<char> &buf,
    const std::shared_ptr<ModelManager>& models,
    const int src,
    std::atomic<int>& done_actors,
    int nbyt,
    const std::vector<std::shared_ptr<SharedBuffer>>& buffers
) {
    switch (tag)
    {
    case TAG_VERSION_REQ: {
        uint32_t player;
        std::memcpy(&player, buf.data(), sizeof(player));

        uint64_t ver = models->getLatestVersion(player);
        auto res = MPI_Send(
                        &ver,
                        1,
                        MPI_UNSIGNED_LONG_LONG,
                        src,
                        TAG_VERSION_RES,
                        MPI_COMM_WORLD
                    );
        if (res != MPI_SUCCESS)
            std::cerr << "MPI_Send(version_res) failed\n";
        break;
    }

    case TAG_WEIGHTS_REQ: {
        uint32_t player;
        std::memcpy(&player, buf.data(), sizeof(player));

        auto model_copy = models->getModel(player)->createCopy();
        uint64_t ver    = model_copy->getVersion();
        const auto& w   = model_copy->getData();

        std::vector<uint8_t> out(sizeof(uint64_t) + w.size());
        std::memcpy(out.data(), &ver, sizeof(uint64_t));
        std::memcpy(out.data() + sizeof(uint64_t), w.data(), w.size());
        auto res = MPI_Send(
                        out.data(),
                        out.size(),
                        MPI_BYTE,
                        src,
                        TAG_WEIGHTS_RES,
                        MPI_COMM_WORLD
                    );

        if (res != MPI_SUCCESS)
            std::cerr << "MPI_Send(weights_res) failed\n";
        break;
    }

    case TAG_TERMINATE:
        done_actors.fetch_add(1, std::memory_order_relaxed);
        break;

    default:
        if (tag >= TAG_TRAJECTORY_BASE) {
            int player = tag - TAG_TRAJECTORY_BASE;
            std::vector<char> traj(buf.begin(), buf.begin() + nbyt);
            buffers[player]->write(std::move(traj));
        } else {
            std::cerr << "Unexpected tag " << tag << '\n';
        }
        break;
    }
}

// rank-0 (learner) thread
void mpi_receiver_posted(
    const ProgramParams& params,
    const std::vector<std::shared_ptr<SharedBuffer>>& buffers,
    std::atomic<int>& done_actors,
    int world_size,
    const std::shared_ptr<ModelManager>& models,
    std::size_t max_traj_bytes
) {
    constexpr int NUM_SLOTS = 128;
    const std::size_t MAX_MSG_BYTES = std::max<std::size_t>(
        max_traj_bytes,
        sizeof(uint64_t) + models->getModel(0)->getData().size()
    );

    std::vector<std::vector<char>> bufs(NUM_SLOTS, std::vector<char>(MAX_MSG_BYTES));
    std::vector<MPI_Request> reqs(NUM_SLOTS);
    std::vector<MPI_Status>  stats(NUM_SLOTS); // optional bookkeeping

    /* initial Irecvs */
    for (int i = 0; i < NUM_SLOTS; ++i)
        MPI_Irecv(
            bufs[i].data(),
            bufs[i].size(),
            MPI_BYTE,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &reqs[i]
        );

    while (done_actors.load(std::memory_order_relaxed) < world_size - 1)
    {
        int idx;
        MPI_Status st; // <-- local status
        MPI_Waitany(NUM_SLOTS, reqs.data(), &idx, &st);
        if (idx == MPI_UNDEFINED) continue; // safety

        stats[idx] = st; // keep a copy (optional)

        const int tag  = st.MPI_TAG;
        const int src  = st.MPI_SOURCE;
        int       nbyt = 0;
        MPI_Get_count(&st, MPI_BYTE, &nbyt);
        auto&     buf  = bufs[idx];

        process_tag(
            tag,
            buf,
            models,
            src,
            done_actors,
            nbyt,
            buffers
        );

        // repost slot
        MPI_Irecv(
            buf.data(),
            buf.size(),
            MPI_BYTE,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            MPI_COMM_WORLD, &reqs[idx]
        );
    }

    for (auto& b : buffers)
        b->setDraining();
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI library does not provide MPI_THREAD_MULTIPLE\n";
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
        auto metrics = MetricsTracker::getInstance();
        metrics->start();

        auto learner         = setupLearner(params);
        auto shared_buffers  = learner->getSharedBuffers();

        std::atomic<int> terminated{0};

        std::size_t max_traj_bytes = params.entry_size * ELEMENT_SIZE;

        mpi_receiver_posted(
            std::cref(params),
            std::cref(shared_buffers),
            std::ref(terminated),
            world_size,
            learner->getModelManager(),
            max_traj_bytes
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
        // We pass dummy `shared_buffers` because the MPI send happens inside
        // `Agent::transferThread`
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
        if (MPI_Send(nullptr, 0, MPI_CHAR, 0, TAG_TERMINATE, MPI_COMM_WORLD) != MPI_SUCCESS) {
            std::stringstream ss;
            ss << "Error: Failed to send TAG_TERMINATE message to learner from rank " << rank << std::endl;
            std::cerr << ss.str();
        }
    }

    MPI_Finalize();
    return 0;
}