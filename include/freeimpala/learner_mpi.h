#ifndef LEARNER_MPI_H
#define LEARNER_MPI_H

#include "data_structures_mpi.h"
#include "metrics_tracker.h"
#include <sstream>
#include <thread>

class LearnerMPI {
private:
    // Configuration
    size_t num_players;
    size_t buffer_capacity;
    size_t entry_size;
    size_t batch_size;
    size_t train_time_ms;
    size_t checkpoint_frequency;
    std::string checkpoint_location;
    std::string starting_model;
    size_t total_iterations;
    size_t num_agents;
    int world_size;
    
    // Models and buffers
    std::vector<std::shared_ptr<Model>> models;
    std::vector<std::unique_ptr<MPIReceiveBuffer>> receive_buffers;
    std::vector<uint64_t> checkpoint_counters;
    
    // State
    std::atomic<bool> should_stop;
    std::vector<size_t> iteration_counts;

    // Training function (simulated with sleep)
    void trainModel(size_t player_index, const std::vector<std::vector<char>>& batch) {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createTrainingTimer();
        
        // Sleep to simulate training time
        std::this_thread::sleep_for(std::chrono::milliseconds(train_time_ms));
        
        // For the dummy implementation, just increment version and generate new random data
        models[player_index]->generateRandomData();
        
        // Record training metrics
        metrics->recordLearnerModelUpdate();
    }
    
    // Broadcast model update to all agents
    void broadcastModelUpdate(size_t player_index) {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createSyncTimer();
        
        // First, signal all agents that a model update is coming
        for (int rank = 1; rank < world_size; rank++) {
            int player_idx = static_cast<int>(player_index);
            MPI_Send(&player_idx, 1, MPI_INT, rank, TAG_MODEL_UPDATE, MPI_COMM_WORLD);
        }
        
        // Now do the broadcast
        uint64_t version = models[player_index]->getVersion();
        const auto& data = models[player_index]->getData();
        
        // Broadcast version first
        MPI_Bcast(&version, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        
        // Then broadcast model data
        MPI_Bcast(const_cast<char*>(data.data()), data.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
        
        std::stringstream ss;
        ss << "Learner: Broadcasted model " << player_index << " version " << version << std::endl;
        std::cerr << ss.str();
    }
    
    // Save model checkpoint
    void saveModelCheckpoint(size_t player_index, size_t iteration) {
        std::string timestamp = std::to_string(iteration);
        std::string versioned_filepath = checkpoint_location + "/model_" + 
                                       std::to_string(player_index) + "_" + timestamp + ".bin";
        std::string latest_filepath = checkpoint_location + "/model_" + 
                                    std::to_string(player_index) + "_latest.bin";
        
        // Create a model with the versioned filepath
        auto checkpoint_model = std::make_shared<Model>(models[player_index]->getSize(), versioned_filepath);
        checkpoint_model->update(models[player_index]->getData(), models[player_index]->getVersion());
        
        if (checkpoint_model->saveToDisk()) {
            std::stringstream ss;
            ss << "Learner: Saved checkpoint for player " << player_index 
               << " at iteration " << iteration << std::endl;
            std::cerr << ss.str();
            
            // Also save as latest
            auto latest_model = std::make_shared<Model>(models[player_index]->getSize(), latest_filepath);
            latest_model->update(models[player_index]->getData(), models[player_index]->getVersion());
            latest_model->saveToDisk();
        }
    }

public:
    LearnerMPI(
        size_t p,          // Number of players
        size_t B,          // Buffer capacity
        size_t S,          // Entry size
        size_t M,          // Batch size
        size_t r,          // Training time (ms)
        size_t c,          // Checkpoint frequency
        const std::string& l,  // Checkpoint location
        const std::string& m,  // Starting model
        size_t T,          // Total iterations
        size_t a,          // Number of agents
        int ws             // World size
    ) : 
        num_players(p),
        buffer_capacity(B),
        entry_size(S),
        batch_size(M),
        train_time_ms(r),
        checkpoint_frequency(c),
        checkpoint_location(l),
        starting_model(m),
        total_iterations(T),
        num_agents(a),
        world_size(ws),
        should_stop(false),
        iteration_counts(p, 0),
        checkpoint_counters(p, 0)
    {
        // Create models for each player
        for (size_t i = 0; i < num_players; i++) {
            std::string filepath = checkpoint_location + "/model_" + std::to_string(i) + "_latest.bin";
            models.push_back(std::make_shared<Model>(6 * 1024 * 1024, filepath)); // 6MB model
            
            // Try to load from disk if starting model specified
            if (!starting_model.empty()) {
                std::string load_path = starting_model + "/model_" + std::to_string(i) + "_latest.bin";
                auto temp_model = std::make_shared<Model>(6 * 1024 * 1024, load_path);
                if (temp_model->loadFromDisk()) {
                    models[i]->update(temp_model->getData(), temp_model->getVersion());
                    std::stringstream ss;
                    ss << "Learner: Loaded model " << i << " from " << load_path 
                       << ", version: " << models[i]->getVersion() << std::endl;
                    std::cerr << ss.str();
                }
            }
            
            // Create receive buffer for this player
            receive_buffers.push_back(std::make_unique<MPIReceiveBuffer>(entry_size, buffer_capacity));
        }
        
        // Send initial models to all agents
        for (size_t i = 0; i < num_players; i++) {
            broadcastModelUpdate(i);
        }
    }
    
    // Main learner loop
    void run() {
        auto metrics = MetricsTracker::getInstance();
        bool all_done = false;
        
        while (!should_stop && !all_done) {
            all_done = true;
            
            // Process each player
            for (size_t p = 0; p < num_players; p++) {
                if (iteration_counts[p] >= total_iterations) {
                    continue;
                }
                
                all_done = false;
                
                // Try to receive data from any agent for this player
                int tag = TAG_DATA_TRANSFER + p;
                bool received_any = false;
                
                // Check all agents
                for (size_t a = 0; a < num_agents; a++) {
                    int agent_rank = a + 1; // Agents start at rank 1
                    if (receive_buffers[p]->tryReceive(agent_rank, tag)) {
                        received_any = true;
                        metrics->recordDataTransfer();
                    }
                }
                
                // If we have enough data for a batch, train
                if (receive_buffers[p]->hasData(batch_size)) {
                    auto batch = receive_buffers[p]->getBatch(batch_size);
                    
                    // Train the model
                    trainModel(p, batch);
                    
                    // Broadcast updated model
                    broadcastModelUpdate(p);
                    
                    // Increment iteration count
                    iteration_counts[p]++;
                    
                    // Checkpoint if needed
                    if (checkpoint_frequency > 0 && iteration_counts[p] % checkpoint_frequency == 0) {
                        saveModelCheckpoint(p, iteration_counts[p]);
                    }
                }
            }
            
            // Small sleep to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Send shutdown signal to all agents
        std::stringstream ss;
        ss << "Learner: Sending shutdown signal to all agents" << std::endl;
        std::cerr << ss.str();
        
        for (int rank = 1; rank < world_size; rank++) {
            int shutdown = 1;
            MPI_Send(&shutdown, 1, MPI_INT, rank, TAG_SHUTDOWN, MPI_COMM_WORLD);
        }
        
        // Final checkpoint
        ss.str("");
        ss << "Learner: Performing final checkpoint" << std::endl;
        std::cerr << ss.str();
        
        for (size_t p = 0; p < num_players; p++) {
            saveModelCheckpoint(p, iteration_counts[p]);
        }
    }
    
    void stop() {
        should_stop = true;
    }
};

#endif // LEARNER_MPI_H