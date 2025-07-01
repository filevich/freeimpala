#ifndef LEARNER_MPI_H
#define LEARNER_MPI_H

#include <mpi.h>
#include <queue>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "data_structures_mpi.h"
#include "metrics_tracker.h"

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
    size_t num_agents;
    size_t total_iterations;
    
    // Data structures
    std::shared_ptr<ModelManager> model_manager;
    std::vector<std::queue<std::vector<char>>> player_buffers;
    std::vector<size_t> training_counts;
    std::atomic<bool> should_stop;
    std::mutex buffer_mutex;
    
    // Checkpoint threads
    std::vector<std::thread> checkpoint_threads;
    std::mutex checkpoint_mutex;
    
    // Process incoming messages
    void processMessages() {
        MPI_Status status;
        int flag;
        
        // Non-blocking check for messages
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        
        while (flag) {
            int source = status.MPI_SOURCE;
            int tag = status.MPI_TAG;
            
            if (tag == DATA_TAG) {
                // Receive player index
                int player_index;
                MPI_Recv(&player_index, 1, MPI_INT, source, DATA_TAG, MPI_COMM_WORLD, &status);
                
                // Receive data
                size_t data_size = entry_size * ELEMENT_SIZE;
                std::vector<char> data(data_size);
                MPI_Recv(data.data(), data_size, MPI_BYTE, source, DATA_TAG, MPI_COMM_WORLD, &status);
                
                // Add to buffer
                {
                    std::lock_guard<std::mutex> lock(buffer_mutex);
                    player_buffers[player_index].push(data);
                }
            }
            else if (tag == VERSION_REQUEST_TAG) {
                // Receive player index
                int player_index;
                MPI_Recv(&player_index, 1, MPI_INT, source, VERSION_REQUEST_TAG, MPI_COMM_WORLD, &status);
                
                // Send back version
                uint64_t version = model_manager->getLatestVersion(player_index);
                MPI_Send(&version, 1, MPI_UNSIGNED_LONG_LONG, source, VERSION_REQUEST_TAG, MPI_COMM_WORLD);
            }
            else if (tag == MODEL_REQUEST_TAG) {
                // Receive player index
                int player_index;
                MPI_Recv(&player_index, 1, MPI_INT, source, MODEL_REQUEST_TAG, MPI_COMM_WORLD, &status);
                
                // Get model data
                auto model = model_manager->getModel(player_index);
                std::vector<char> model_data = model->getData();
                uint64_t version = model->getVersion();
                
                // Send model data
                MPI_Send(&version, 1, MPI_UNSIGNED_LONG_LONG, source, MODEL_REQUEST_TAG, MPI_COMM_WORLD);
                MPI_Send(model_data.data(), model_data.size(), MPI_BYTE, source, MODEL_REQUEST_TAG, MPI_COMM_WORLD);
            }
            
            // Check for next message
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        }
    }
    
    // Train model for a player
    void trainModel(size_t player_index) {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createTrainingTimer();
        
        // Sleep to simulate training time
        std::this_thread::sleep_for(std::chrono::milliseconds(train_time_ms));
        
        // Create a new model with random data
        auto current_model = model_manager->getModel(player_index);
        auto new_model = current_model->createCopy();
        new_model->generateRandomData();
        
        // Update the model
        model_manager->updateModel(player_index, new_model);
        
        // Record training metrics
        metrics->recordLearnerModelUpdate();
    }
    
    // Checkpoint function for a specific player's model
    void checkpointModel(size_t player_index) {
        std::lock_guard<std::mutex> lock(checkpoint_mutex);
        
        // Clean up completed checkpoint threads
        for (auto it = checkpoint_threads.begin(); it != checkpoint_threads.end();) {
            if (it->joinable()) {
                it->join();
                it = checkpoint_threads.erase(it);
            } else {
                ++it;
            }
        }
        
        // Start a new checkpoint thread
        checkpoint_threads.emplace_back([this, player_index] {
            model_manager->saveModel(player_index, training_counts[player_index]);
        });
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
        size_t a,          // Number of agents
        size_t T           // Total iterations
    ) : 
        num_players(p),
        buffer_capacity(B),
        entry_size(S),
        batch_size(M),
        train_time_ms(r),
        checkpoint_frequency(c),
        checkpoint_location(l),
        starting_model(m),
        num_agents(a),
        total_iterations(T),
        should_stop(false),
        player_buffers(p),
        training_counts(p, 0)
    {
        // Create model manager
        model_manager = std::make_shared<ModelManager>(
            num_players, 
            6 * 1024 * 1024,  // 6MB for dummy model
            checkpoint_location
        );
        
        // Try to load models if specified
        if (!starting_model.empty()) {
            model_manager->loadModels(starting_model);
        }
    }
    
    ~LearnerMPI() {
        // Join any remaining checkpoint threads
        {
            std::lock_guard<std::mutex> lock(checkpoint_mutex);
            for (auto& thread : checkpoint_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
    }
    
    void run() {
        auto metrics = MetricsTracker::getInstance();
        metrics->start();
        
        // Calculate total batches needed
        size_t total_batches = (num_agents * total_iterations + batch_size - 1) / batch_size;
        
        std::cerr << "Learner starting. Total batches needed: " 
                << total_batches << std::endl;
        
        size_t processed_batches = 0;
        bool all_done = false;
        
        while (!should_stop) {
            // Process incoming messages
            processMessages();
            
            // Check if we have enough data to train
            bool trained = false;
            for (size_t p = 0; p < num_players; p++) {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                if (player_buffers[p].size() >= batch_size && training_counts[p] < total_batches) {
                    // Form batch
                    std::vector<std::vector<char>> batch;
                    for (size_t i = 0; i < batch_size; i++) {
                        batch.push_back(player_buffers[p].front());
                        player_buffers[p].pop();
                    }
                    
                    // Train model
                    trainModel(p);
                    training_counts[p]++;
                    processed_batches++;
                    trained = true;
                    
                    std::cerr << "Trained model for player " << p 
                            << ". Total trained: " << training_counts[p]
                            << "/" << total_batches << std::endl;
                    
                    // Checkpoint if needed
                    if (checkpoint_frequency > 0 && training_counts[p] % checkpoint_frequency == 0) {
                        checkpointModel(p);
                    }
                }
            }
            
            // Check if all training is complete
            all_done = true;
            for (size_t p = 0; p < num_players; p++) {
                if (training_counts[p] < total_batches) {
                    all_done = false;
                    break;
                }
            }
            
            if (all_done) {
                should_stop = true;
                break;
            }
            
            // Sleep if no training occurred to avoid busy waiting
            if (!trained) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        // After training is done:
        std::cerr << "Learner completed training. Sending termination signals to " 
                << num_agents << " agents." << std::endl;

        // Send termination multiple times to ensure delivery
        for (int retry = 0; retry < 3; retry++) {
            for (int i = 1; i <= num_agents; i++) {
                MPI_Send(nullptr, 0, MPI_BYTE, i, TERMINATE_TAG, MPI_COMM_WORLD);
                std::cerr << "Sent termination signal to agent " << i << " (attempt " 
                        << (retry + 1) << ")" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Save final model state
        model_manager->saveAllModels(processed_batches);
        std::cerr << "Learner saved final models." << std::endl;
    }
};

#endif // LEARNER_MPI_H