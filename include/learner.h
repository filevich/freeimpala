#ifndef LEARNER_H
#define LEARNER_H

#include "data_structures.h"
#include "metrics_tracker.h"

class Learner {
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
    
    // Buffers and models
    std::vector<std::shared_ptr<SharedBuffer>> shared_buffers;
    std::shared_ptr<ModelManager> model_manager;
    
    // Thread management
    std::vector<std::thread> worker_threads;
    std::vector<std::thread> checkpoint_threads;
    std::atomic<bool> should_stop;
    std::mutex learner_mutex;
    std::mutex checkpoint_mutex;

    // Training function (simulated with sleep)
    void trainModel(size_t player_index, const std::vector<std::vector<char>>& batch) {
        auto metrics = MetricsTracker::getInstance();
        auto timer = metrics->createTrainingTimer();
        
        // Sleep to simulate training time
        std::this_thread::sleep_for(std::chrono::milliseconds(train_time_ms));
        
        // For the dummy implementation, just create a new model with random data
        auto current_model = model_manager->getModel(player_index);
        auto new_model = current_model->createCopy();
        new_model->generateRandomData();
        
        // Update the model
        model_manager->updateModel(player_index, new_model);
        
        // Record training metrics
        metrics->recordLearnerModelUpdate();
    }
    
    // Checkpoint function for a specific player's model
    void checkpointModel(size_t player_index, uint64_t current_iteration) {
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
        
        // Start a new checkpoint thread only for this player's model
        checkpoint_threads.emplace_back([this, player_index, current_iteration] {
            model_manager->saveModel(player_index, current_iteration);
        });
    }
    
    // Worker thread function (one per player)
    void workerThread(size_t player_index) {
        size_t iteration_count = 0;
        
        while (!should_stop.load() && iteration_count < total_iterations) {
            // Wait for enough data in the buffer
            auto batch = shared_buffers[player_index]->readBatch(batch_size);
            
            // Train the model with the batch
            trainModel(player_index, batch);
            
            // Increment the iteration counter
            iteration_count++;
            
            // Checkpoint if needed - now each thread only saves its own model
            if (checkpoint_frequency > 0 && iteration_count % checkpoint_frequency == 0) {
                checkpointModel(player_index, iteration_count);
            }
        }
    }

public:
    Learner(
        size_t p,          // Number of players
        size_t B,          // Buffer capacity
        size_t S,          // Entry size
        size_t M,          // Batch size
        size_t r,          // Training time (ms)
        size_t c,          // Checkpoint frequency
        const std::string& l,  // Checkpoint location
        const std::string& m,  // Starting model
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
        total_iterations(T),
        should_stop(false)
    {
        // Create model manager
        model_manager = std::make_shared<ModelManager>(
            num_players, 
            6 * 1024 * 1024,  // 6MB for dummy model; same as DouZero
            checkpoint_location
        );
        
        // Try to load models if specified
        if (!starting_model.empty()) {
            model_manager->loadModels(starting_model);
        }
        
        // Create shared buffers for each player
        for (size_t p = 0; p < num_players; p++) {
            shared_buffers.push_back(
                std::make_shared<SharedBuffer>(entry_size, buffer_capacity)
            );
        }
    }
    
    ~Learner() {
        stop();
        
        // Join any remaining checkpoint threads
        {
            std::lock_guard<std::mutex> lock(checkpoint_mutex);
            for (auto& thread : checkpoint_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            checkpoint_threads.clear();
        }
    }
    
    // Start the learner
    void start() {
        // Launch worker threads (one per player)
        for (size_t p = 0; p < num_players; p++) {
            worker_threads.emplace_back([this, p] { this->workerThread(p); });
        }
    }
    
    // Stop the learner
    void stop() {
        should_stop.store(true);
        
        // Wait for all worker threads to finish
        for (auto& thread : worker_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        // Clear threads
        worker_threads.clear();
        
        // Save final model state if needed
        std::stringstream ss;
        ss << "Performing final model save before exit..." << std::endl;
        std::cerr << ss.str();
        uint64_t final_iteration = total_iterations; // Use total iterations as final checkpoint number
        model_manager->saveAllModels(final_iteration);
        
        // Wait for any in-progress checkpoint threads
        std::lock_guard<std::mutex> lock(checkpoint_mutex);
        for (auto& thread : checkpoint_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        checkpoint_threads.clear();
    }
    
    // Get shared buffers for agents to use
    std::vector<std::shared_ptr<SharedBuffer>> getSharedBuffers() {
        return shared_buffers;
    }
    
    // Get model manager for agents to use
    std::shared_ptr<ModelManager> getModelManager() {
        return model_manager;
    }
};

#endif // LEARNER_H