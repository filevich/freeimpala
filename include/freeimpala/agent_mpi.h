#ifndef AGENT_MPI_H
#define AGENT_MPI_H

#include <mpi.h>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <random>
#include "data_structures_mpi.h"
#include "metrics_tracker.h"

class AgentMPI {
private:
    // Configuration
    size_t agent_id;
    size_t num_players;
    size_t entry_size;
    size_t game_steps;
    size_t game_time_ms;
    size_t total_iterations;
    
    // Local state
    std::vector<std::shared_ptr<Buffer>> local_buffers;
    std::vector<std::shared_ptr<Model>> local_models;
    std::vector<uint64_t> current_model_versions;
    std::atomic<bool> should_stop;
    
    // Simulate playing the game
    void simulateGame() {
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
    
    // Transfer data to learner
    void transferData() {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createTransferTimer();
        
        for (size_t p = 0; p < num_players; p++) {
            auto& local_buffer = local_buffers[p];
            
            // If local buffer is filled, transfer to learner
            if (local_buffer->getEntry().filled) {
                // Send player index
                MPI_Send(&p, 1, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD);
                
                // Send data
                MPI_Send(local_buffer->getEntry().data.data(), 
                         local_buffer->getEntry().data.size(), 
                         MPI_BYTE, 0, DATA_TAG, MPI_COMM_WORLD);
                
                metrics->recordDataTransfer();
            }
        }
    }
    
    // Check for model updates
    void checkModelUpdates() {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createSyncTimer();
        
        for (size_t p = 0; p < num_players; p++) {
            // Request latest version from learner
            MPI_Send(&p, 1, MPI_INT, 0, VERSION_REQUEST_TAG, MPI_COMM_WORLD);
            
            uint64_t latest_version;
            MPI_Recv(&latest_version, 1, MPI_UNSIGNED_LONG_LONG, 0, 
                     VERSION_REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (latest_version > current_model_versions[p]) {
                // Request new model
                MPI_Send(&p, 1, MPI_INT, 0, MODEL_REQUEST_TAG, MPI_COMM_WORLD);
                
                // Receive version and model data
                uint64_t new_version;
                MPI_Recv(&new_version, 1, MPI_UNSIGNED_LONG_LONG, 0, 
                         MODEL_REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                std::vector<char> model_data(6 * 1024 * 1024);
                MPI_Recv(model_data.data(), model_data.size(), MPI_BYTE, 0,
                         MODEL_REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Update local model
                local_models[p]->setData(model_data, new_version);
                current_model_versions[p] = new_version;
                
                metrics->recordAgentModelSync();
            }
        }
    }

public:
    AgentMPI(
        size_t id,         // Agent ID
        size_t p,          // Number of players
        size_t S,          // Entry size
        size_t steps,      // Game steps
        size_t r,          // Game time (ms)
        size_t T           // Total iterations
    ) : 
        agent_id(id),
        num_players(p),
        entry_size(S),
        game_steps(steps),
        game_time_ms(r),
        total_iterations(T),
        should_stop(false),
        current_model_versions(p, 0)
    {
        // Create local buffers for each player
        for (size_t i = 0; i < num_players; i++) {
            local_buffers.push_back(std::make_shared<Buffer>(entry_size * ELEMENT_SIZE));
        }
        
        // Create initial local models
        for (size_t i = 0; i < num_players; i++) {
            local_models.push_back(std::make_shared<Model>(6 * 1024 * 1024, ""));
        }
    }
    
    // Run the agent
    void run() {
        auto metrics = MetricsTracker::getInstance();
        size_t iteration = 0;
        bool running = true;
        
        while (running && iteration < total_iterations) {
            metrics->startAgentIteration(agent_id);

            if (iteration % 10 == 0) {
                std::cerr << "Agent " << agent_id << ": Completed iteration " 
                        << iteration << "/" << total_iterations << std::endl;
            }
            
            // Step 1: Simulate the game
            simulateGame();
            
            // Step 2: Transfer data to learner
            transferData();
            
            // Step 3: Check for model updates
            checkModelUpdates();
            
            metrics->endAgentIteration(agent_id);
            iteration++;
            
            // Check for termination signal with timeout
            int flag;
            MPI_Status status;
            MPI_Iprobe(0, TERMINATE_TAG, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                MPI_Recv(nullptr, 0, MPI_BYTE, 0, TERMINATE_TAG, MPI_COMM_WORLD, &status);
                running = false;
            }
        }
        
        // If we finished iterations but haven't received termination, wait for it
        if (running) {
            MPI_Recv(nullptr, 0, MPI_BYTE, 0, TERMINATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
};

#endif // AGENT_MPI_H