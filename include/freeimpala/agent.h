#ifndef AGENT_H
#define AGENT_H

#include <future>
#include "freeimpala/data_structures.h"
#include "freeimpala/metrics_tracker.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

class Agent {
private:
    // Configuration
    size_t agent_id;
    size_t num_players;
    size_t entry_size;
    size_t game_steps;
    size_t game_time_ms;
    size_t total_iterations;
    
    // Buffers and models
    std::vector<std::shared_ptr<Buffer>> local_buffers;
    std::vector<std::shared_ptr<SharedBuffer>> shared_buffers;
    std::shared_ptr<ModelManager> model_manager;
    std::vector<uint64_t> current_model_versions;
    std::vector<std::shared_ptr<Model>> local_models;  // Local models for each player
    
    // Thread management
    std::vector<std::thread> worker_threads;
    std::atomic<bool> should_stop;
    
    // Simulate playing the game
    void simulateGame() {
        // Use scoped timer to measure simulation time
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createSimulationTimer();
        
        // Reset all local buffers
        for (auto& buffer : local_buffers) {
            buffer->reset();
        }
        
        // Sleep to simulate game play time
        std::this_thread::sleep_for(std::chrono::milliseconds(game_time_ms));
        
        // Generate random data for each step of the game
        for (size_t step = 0; step < game_steps; step++) {
            // Determine which player's turn it is
            size_t player_index = step % num_players;
            
            // In a real implementation, we would use the local model here:
            // if (local_models[player_index]) {
            //     // Use the model to generate a move/action
            //     // This would involve passing game state to the neural network
            // }
            
            // Generate random data for this player's turn
            auto& entry = local_buffers[player_index]->getEntry();
            
            // Calculate where to write in the buffer based on steps
            size_t offset = (step / num_players) * ELEMENT_SIZE;
            
            // Make sure we don't exceed buffer size
            if (offset + ELEMENT_SIZE <= entry.data.size()) {
                // Generate random bytes for this step
                for (size_t i = 0; i < ELEMENT_SIZE; i++) {
                    entry.data[offset + i] = static_cast<char>(rand() % 256);
                }
                
                // Mark entry as filled
                entry.filled = true;
            }
        }
    }
    
    // Transfer thread function (one per player)
    void transferThread(size_t player_index, std::promise<void>& promise) {
        auto metrics = MetricsTracker::getInstance();
        auto timer   = metrics->createTransferTimer();

        auto& entry = local_buffers[player_index]->getEntry();

        if (entry.filled) {
#ifdef USE_MPI
            // send buffer, to learner (rank 0)
            const int tag = TAG_TRAJECTORY_BASE + static_cast<int>(player_index);
            if (MPI_Send(entry.data.data(), entry.data.size(), MPI_CHAR, 0, tag, MPI_COMM_WORLD) != MPI_SUCCESS) {
                std::stringstream ss;
                ss << "Error: Agent " << agent_id << " failed to send trajectory data for player "
                   << player_index << " via MPI_Send (tag=" << tag << ")" << std::endl;
                std::cerr << ss.str();
            }
            // count it (to metrics)
            metrics->recordDataTransfer();
#else
            bool success = shared_buffers[player_index]->write(entry.data);
            if (success) {
                metrics->recordDataTransfer();
            } else {
                std::cerr << "Agent " << agent_id
                        << ": failed to write data for player "
                        << player_index << '\n';
            }
#endif
        }
        promise.set_value();
    }
    
    // Model update thread function (one per player)
    void modelUpdateThread(size_t player_index, std::promise<void>& promise) {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createSyncTimer();

#ifdef USE_MPI
        uint32_t p32 = static_cast<uint32_t>(player_index);
        if (MPI_Send(&p32, 1, MPI_UNSIGNED, 0, TAG_VERSION_REQ, MPI_COMM_WORLD) != MPI_SUCCESS) {
            std::stringstream ss;
            ss << "Error: Agent " << agent_id << " failed to send version request for player "
               << player_index << " via MPI_Send (tag=" << TAG_VERSION_REQ << ")" << std::endl;
            std::cerr << ss.str();
        }

        // Wait for the learner’s answer (blocking)
        uint32_t latest;
        MPI_Recv(&latest, 1, MPI_UNSIGNED, 0, TAG_VERSION_RES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (latest > current_model_versions[player_index]) {
            // Request the weights for that player
            if (MPI_Send(&p32, 1, MPI_UNSIGNED, 0, TAG_WEIGHTS_REQ, MPI_COMM_WORLD) != MPI_SUCCESS) {
                std::stringstream ss;
                ss << "Error: Agent " << agent_id << " failed to send weights request for player "
                << player_index << " via MPI_Send (tag=" << TAG_WEIGHTS_REQ << ")" << std::endl;
                std::cerr << ss.str();
            }

            // Get the data size
            const size_t data_size = local_models[player_index]->getData().size();
            const size_t total_size = sizeof(uint32_t) + data_size;

            // Create a buffer to hold version + data
            std::vector<uint8_t> buffer(total_size);

            // Receive the data
            MPI_Recv(buffer.data(), total_size, MPI_BYTE, 0, TAG_WEIGHTS_RES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Extract version (be careful about endianness if needed)
            uint32_t new_version;
            std::memcpy(&new_version, buffer.data(), sizeof(uint32_t));

            // Extract data and convert to std::vector<char>
            std::vector<char> new_data(buffer.begin() + sizeof(uint32_t), buffer.end());

            // Call update function
            local_models[player_index]->update(new_data, new_version);
            current_model_versions[player_index] = new_version;
            metrics->recordAgentModelSync();
        }
#else
        // Check if there's a new model available
        uint64_t current_version = current_model_versions[player_index];
        uint64_t latest_version = model_manager->getLatestVersion(player_index);
        
        if (latest_version > current_version) {
            std::stringstream ss;
            ss << "Agent " << agent_id << " updating model for player " << player_index 
                      << " from version " << current_version << " to " << latest_version << std::endl;
            std::cerr << ss.str();
            
            // Get the latest model from the manager
            auto shared_model = model_manager->getModel(player_index);
            if (shared_model) {
                // Create a deep copy for this agent
                local_models[player_index] = shared_model->createCopy();
                
                // In a real implementation with neural networks, we would:
                // 1. Load the weights from the shared model
                // 2. Update our local neural network instance
                // 3. Use the updated model for future gameplay
                
                // For the dummy version, we'll simulate using the model:
                std::vector<char> model_data = local_models[player_index]->getData();
                // (In a real implementation, we would do something with this data)
                
                // Update our record of the current version
                current_model_versions[player_index] = latest_version;
                
                // Record that a model update occurred
                metrics->recordAgentModelSync();
            }
        }
#endif        
        promise.set_value();
    }

public:
    Agent(
        size_t id,         // Agent ID
        size_t p,          // Number of players
        size_t S,          // Entry size
        size_t steps,      // Game steps
        size_t r,          // Game time (ms)
        size_t T,          // Total iterations
        const std::vector<std::shared_ptr<SharedBuffer>>& buffers,
        const std::shared_ptr<ModelManager>& models
    ) : 
        agent_id(id),
        num_players(p),
        entry_size(S),
        game_steps(steps),
        game_time_ms(r),
        total_iterations(T),
        shared_buffers(buffers),
        model_manager(models),
        current_model_versions(p, 0),
        local_models(p),  // Initialize local models vector
        should_stop(false)
    {
        // Create local buffers for each player
        for (size_t p = 0; p < num_players; p++) {
            local_buffers.push_back(std::make_shared<Buffer>(entry_size));
            
            // Create initial local models (deep copy from model manager)
            auto shared_model = model_manager->getModel(p);
            if (shared_model) {
                // Create a deep copy for this agent
                local_models[p] = shared_model->createCopy();
                current_model_versions[p] = local_models[p]->getVersion();
                
                std::stringstream ss;
                ss << "Agent " << agent_id << " initialized model for player " 
                        << p << " with version " << current_model_versions[p] << std::endl;
                std::cerr << ss.str();
            }
        }
    }
    
    ~Agent() {
        stop();
    }
    
    // Run the agent's main loop
    void run() {
        auto metrics = MetricsTracker::getInstance();
        size_t iteration = 0;
        
        while (!should_stop.load() && iteration < total_iterations) {
            // Record the start of an agent iteration
            metrics->startAgentIteration(agent_id);
            
            // Step 1: Simulate the game
            simulateGame();
            
            // Step 2: Transfer data to shared buffers
            std::vector<std::thread> transfer_threads;
            std::vector<std::promise<void>> transfer_promises(num_players);
            std::vector<std::future<void>> transfer_futures;
            
            for (size_t p = 0; p < num_players; p++) {
                transfer_futures.push_back(transfer_promises[p].get_future());
                transfer_threads.emplace_back(
                    [this, p, &promise = transfer_promises[p]] {
                        this->transferThread(p, promise);
                    }
                );
            }
            
            // Wait for all transfers to complete
            for (auto& future : transfer_futures) {
                future.wait();
            }
            
            // Join all transfer threads
            for (auto& thread : transfer_threads) {
                thread.join();
            }
            
            // Step 3: Update models if needed
            std::vector<std::thread> update_threads;
            std::vector<std::promise<void>> update_promises(num_players);
            std::vector<std::future<void>> update_futures;
            
            for (size_t p = 0; p < num_players; p++) {
                update_futures.push_back(update_promises[p].get_future());
                update_threads.emplace_back(
                    [this, p, &promise = update_promises[p]] {
                        this->modelUpdateThread(p, promise);
                    }
                );
            }
            
            // Wait for all updates to complete
            for (auto& future : update_futures) {
                future.wait();
            }
            
            // Join all update threads
            for (auto& thread : update_threads) {
                thread.join();
            }
            
            // Record the end of an agent iteration
            metrics->endAgentIteration(agent_id);
            
            // Increment iteration counter
            iteration++;
        }
    }
    
    // Stop the agent
    void stop() {
        should_stop.store(true);
    }
};

#endif // AGENT_H