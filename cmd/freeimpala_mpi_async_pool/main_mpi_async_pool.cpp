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

// queue
#include <queue>
#include <mutex>
#include <condition_variable>

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
void setupArgumentParser(argparse::ArgumentParser& program) {
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
}

// Parse command line arguments and extract parameters
bool parseParameters(
    int argc,
    char** argv,
    ProgramParams& params
) {
    std::stringstream ss;
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

// Struct to hold a message for processing
struct _MPI_Message {
    int tag;
    int source;
    std::vector<char> buffer;
};

// A simple thread-safe queue
template<typename T>
class ConcurrentQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};

public:
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        lock.unlock();
        cv_.notify_one();
    }

    bool try_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stop_.load(); });
        if (queue_.empty()) {
            return false; // Stopped
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void stop() {
        stop_.store(true);
        cv_.notify_all();
    }
};

// The original logic, now in a standalone function
void process_message(
    _MPI_Message& msg,
    const std::shared_ptr<ModelManager>& models,
    std::atomic<int>& done_actors,
    const std::vector<std::shared_ptr<SharedBuffer>>& buffers
) {
    // Re-use the existing switch logic from process_tag
    switch (msg.tag)
    {
        case MessageTag::TAG_VERSION_REQ: {
            uint32_t player;
            std::memcpy(&player, msg.buffer.data(), sizeof(player));

            uint64_t ver = models->getLatestVersion(player);
            auto res = MPI_Send(&ver, 1, MPI_UNSIGNED_LONG_LONG, msg.source, MessageTag::TAG_VERSION_RES, MPI_COMM_WORLD);
            if (res != MPI_SUCCESS) spdlog::error("MPI_Send(version_res) failed");
            break;
        }

        case MessageTag::TAG_WEIGHTS_REQ: {
            uint32_t player;
            std::memcpy(&player, msg.buffer.data(), sizeof(player));

            auto model_copy = models->getModel(player)->createCopy();
            uint64_t ver    = model_copy->getVersion();
            const auto& w   = model_copy->getData();

            std::vector<uint8_t> out(sizeof(uint64_t) + w.size());
            std::memcpy(out.data(), &ver, sizeof(uint64_t));
            std::memcpy(out.data() + sizeof(uint64_t), w.data(), w.size());
            auto res = MPI_Send(out.data(), out.size(), MPI_BYTE, msg.source, MessageTag::TAG_WEIGHTS_RES, MPI_COMM_WORLD);
            if (res != MPI_SUCCESS) spdlog::error("MPI_Send(weights_res) failed");
            break;
        }

        // case MessageTag::TAG_TERMINATE: {
        //     spdlog::info("rcv MessageTag::TAG_TERMINATE");
        //     done_actors.fetch_add(1, std::memory_order_relaxed);
        //     break;
        // }

        default: {
            if (msg.tag >= MessageTag::TAG_TRAJECTORY_BASE) {
                int player = msg.tag - MessageTag::TAG_TRAJECTORY_BASE;
                // The buffer in msg already has the correct size
                buffers[player]->write(std::move(msg.buffer));
            } else {
                spdlog::error("Unexpected tag {}", msg.tag);
            }
            break;
        }
    }
}

// The function for our new processor threads
void processor_worker(
    ConcurrentQueue<_MPI_Message>& queue,
    const std::shared_ptr<ModelManager>& models,
    std::atomic<int>& done_actors,
    const std::vector<std::shared_ptr<SharedBuffer>>& buffers
) {
    _MPI_Message msg;
    while (queue.try_pop(msg)) {
        process_message(msg, models, done_actors, buffers);
    }
    spdlog::info("exiting processor_worker");
}

void mpi_receiver_posted(
    ConcurrentQueue<_MPI_Message>& work_queue,
    std::atomic<int>& done_actors,
    int world_size,
    std::size_t max_msg_bytes
) {
    constexpr int NUM_SLOTS = 128;
    std::vector<std::vector<char>> bufs(NUM_SLOTS, std::vector<char>(max_msg_bytes));
    std::vector<MPI_Request> reqs(NUM_SLOTS);

    for (int i = 0; i < NUM_SLOTS; ++i)
        MPI_Irecv(bufs[i].data(), bufs[i].size(), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[i]);

    while (done_actors.load(std::memory_order_relaxed) < world_size - 1) {
        int idx;
        MPI_Status st;
        MPI_Waitany(NUM_SLOTS, reqs.data(), &idx, &st);
        if (idx == MPI_UNDEFINED) continue;

        const int tag = st.MPI_TAG;

        // Handle MessageTag::TAG_TERMINATE as a special case right here
        if (tag == MessageTag::TAG_TERMINATE) {
            done_actors.fetch_add(1, std::memory_order_relaxed);
            // No need to push to the queue, just handle it and loop.
            // The while condition will handle the exit.
        } else {
            // For all other messages, delegate to the processor threads.
            int nbyt = 0;
            MPI_Get_count(&st, MPI_BYTE, &nbyt);

            _MPI_Message msg;
            msg.tag = tag;
            msg.source = st.MPI_SOURCE;
            msg.buffer.assign(bufs[idx].begin(), bufs[idx].begin() + nbyt);
            
            work_queue.push(std::move(msg));
        }

        // Always repost the receive for the used slot
        MPI_Irecv(bufs[idx].data(), bufs[idx].size(), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[idx]);
    }

    spdlog::info("about to exit mpi_receiver_posted");
    work_queue.stop();
    spdlog::info("exiting mpi_receiver_posted");
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

    // learner process for rank == 0
    if (rank == 0) {
        spdlog::info("boost SLIM 8t");
        auto metrics = MetricsTracker::getInstance();
        metrics->start();

        auto learner = setupLearner(params);
        auto shared_buffers = learner->getSharedBuffers();
        auto model_manager = learner->getModelManager();

        std::atomic<int> terminated{0};
        std::size_t max_traj_bytes = params.entry_size * ELEMENT_SIZE;
        // const std::size_t MAX_MSG_BYTES = /* same as before */;
        const std::size_t MAX_MSG_BYTES = std::max<std::size_t>(
            max_traj_bytes,
            sizeof(uint64_t) + model_manager->getModel(0)->getData().size()
        );

        // 1. Create the work queue
        ConcurrentQueue<_MPI_Message> work_queue;

        // 2. Create and launch processor threads
        const int num_processor_threads = 8; // A good starting point
        std::vector<std::thread> processor_threads;
        for (int i = 0; i < num_processor_threads; ++i) {
            processor_threads.emplace_back(
                processor_worker,
                std::ref(work_queue),
                std::ref(model_manager),
                std::ref(terminated),
                std::ref(shared_buffers)
            );
        }
        
        // 3. Launch the receiver thread (now the producer)
        mpi_receiver_posted(
            std::ref(work_queue),
            std::ref(terminated),
            world_size,
            MAX_MSG_BYTES // Pass max size instead of full objects
        );

        // Wait until all actors have terminated

        // 4. Wait for processor threads to finish
        for (auto& th : processor_threads) {
            th.join();
        }

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
            spdlog::error("Failed to send MessageTag::TAG_TERMINATE message to learner from rank {}", rank);
        }
    }

    MPI_Finalize();
    return 0;
}