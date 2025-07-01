#ifndef AGENT_MPI_H
#define AGENT_MPI_H

#include "data_structures_mpi.h"
#include "metrics_tracker.h"
#include <sstream>
#include <thread>

class AgentMPI {
private:
    // Configuration
    int agent_rank;
    size_t agent_id;
    size_t num_players;
    size_t entry_size;
    size_t game_steps;
    size_t game_time_ms;
    size_t total_iterations;
    
    // Local buffers and models
    std::vector<std::shared_ptr<Buffer>> local_buffers;
    std::vector<std::shared_ptr<Model>> local_models;
    std::vector<uint64_t> current_model_versions;
    
    // State
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
    
    // Send data to learner via MPI
    void sendDataToLearner(size_t player_index) {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createTransferTimer();
        
        auto& entry = local_buffers[player_index]->getEntry();
        
        if (entry.filled) {
            int tag = TAG_DATA_TRANSFER + player_index;
            
            // Send to learner (rank 0)
            MPI_Send(entry.data.data(), entry.data.size(), MPI_CHAR, 0, tag, MPI_COMM_WORLD);
            
            metrics->recordDataTransfer();
            
            // Reset the buffer after sending
            entry.filled = false;
        }
    }
    
    // Check for model updates via MPI - only when signaled by learner
    bool checkForModelUpdateSignal() {
        int flag;
        MPI_Status status;
        
        // Check if learner is signaling a model update
        MPI_Iprobe(0, TAG_MODEL_UPDATE, MPI_COMM_WORLD, &flag, &status);
        
        if (flag) {
            // Receive the signal
            int player_index;
            MPI_Recv(&player_index, 1, MPI_INT, 0, TAG_MODEL_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Now participate in the broadcast for this specific player
            receiveModelUpdate(player_index);
            return true;
        }
        
        return false;
    }
    
    // Receive model update for a specific player
    void receiveModelUpdate(size_t player_index) {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createSyncTimer();
        
        // Participate in broadcast for version (learner is root)
        uint64_t new_version;
        MPI_Bcast(&new_version, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        
        // Participate in broadcast for model data
        std::vector<char> model_data(local_models[player_index]->getSize());
        MPI_Bcast(model_data.data(), model_data.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
        
        // Update if this is a new version
        if (new_version > current_model_versions[player_index]) {
            local_models[player_index]->update(model_data, new_version);
            current_model_versions[player_index] = new_version;
            
            std::stringstream ss;
            ss << "Agent " << agent_id << ": Updated model " << player_index 
               << " to version " << new_version << std::endl;
            std::cerr << ss.str();
            
            metrics->recordAgentModelSync();
        }
    }
    
    // Check for shutdown signal
    bool checkShutdown() {
        int flag;
        MPI_Status status;
        
        MPI_Iprobe(0, TAG_SHUTDOWN, MPI_COMM_WORLD, &flag, &status);
        
        if (flag) {
            int shutdown;
            MPI_Recv(&shutdown, 1, MPI_INT, 0, TAG_SHUTDOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return true;
        }
        
        return false;
    }

public:
    AgentMPI(
        int rank,          // MPI rank
        size_t id,         // Agent ID (0-based among agents)
        size_t p,          // Number of players
        size_t S,          // Entry size
        size_t steps,      // Game steps
        size_t r,          // Game time (ms)
        size_t T           // Total iterations
    ) : 
        agent_rank(rank),
        agent_id(id),
        num_players(p),
        entry_size(S),
        game_steps(steps),
        game_time_ms(r),
        total_iterations(T),
        should_stop(false)
    {
        // Create local buffers for each player
        for (size_t i = 0; i < num_players; i++) {
            local_buffers.push_back(std::make_shared<Buffer>(entry_size));
            
            // Create local models
            std::string dummy_path = "/tmp/agent_" + std::to_string(agent_id) + "_model_" + std::to_string(i);
            local_models.push_back(std::make_shared<Model>(6 * 1024 * 1024, dummy_path));
            current_model_versions.push_back(0);
        }
        
        // Receive initial models from learner
        for (size_t i = 0; i < num_players; i++) {
            // Wait for initial model signal
            int player_index;
            MPI_Recv(&player_index, 1, MPI_INT, 0, TAG_MODEL_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Receive the model
            receiveModelUpdate(player_index);
        }
        
        std::stringstream ss;
        ss << "Agent " << agent_id << " (rank " << agent_rank << ") initialized" << std::endl;
        std::cerr << ss.str();
    }
    
    // Run the agent's main loop
    void run() {
        auto metrics = MetricsTracker::getInstance();
        size_t iteration = 0;
        
        while (!should_stop && iteration < total_iterations) {
            // Check for shutdown signal
            if (checkShutdown()) {
                std::stringstream ss;
                ss << "Agent " << agent_id << ": Received shutdown signal" << std::endl;
                std::cerr << ss.str();
                break;
            }
            
            // Record the start of an agent iteration
            metrics->startAgentIteration(agent_id);
            
            // Step 1: Simulate the game
            simulateGame();
            
            // Step 2: Send data to learner for each player
            for (size_t p = 0; p < num_players; p++) {
                sendDataToLearner(p);
            }
            
            // Step 3: Check for model updates (non-blocking)
            while (checkForModelUpdateSignal()) {
                // Process any pending model updates
            }
            
            // Record the end of an agent iteration
            metrics->endAgentIteration(agent_id);
            
            // Increment iteration counter
            iteration++;
        }
        
        std::stringstream ss;
        ss << "Agent " << agent_id << ": Completed " << iteration << " iterations" << std::endl;
        std::cerr << ss.str();
    }
    
    void stop() {
        should_stop = true;
    }
};

#endif // AGENT_MPI_H