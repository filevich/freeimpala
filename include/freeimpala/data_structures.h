#ifndef DATA_H
#define DATA_H

#include <spdlog/spdlog.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <filesystem>
#include <queue>
#include <functional>

enum MessageTag : int {
    /* -------- actor -> learner -------- */
    TAG_TRAJECTORY_BASE = 100, // +player_idx, payload: [byte] buffer
    TAG_VERSION_REQ     = 200, // payload: uint32_t player_idx
    TAG_WEIGHTS_REQ     = 210, // payload: uint32_t player_idx

    /* -------- actor <- learner -------- */
    TAG_VERSION_RES     = 201, // payload: uint64_t latest_version
    TAG_WEIGHTS_RES     = 211, // payload: uint64_t latest_version  +  [byte] weights

    TAG_TERMINATE       = 999
};

// Size of each element in bytes
constexpr size_t ELEMENT_SIZE = 1024;

// Forward declarations
class Model;
class Buffer;
class SharedBuffer;

// Simple model class that represents our "neural network"
class Model {
private:
    std::vector<char> data;
    std::atomic<uint64_t> version;
    std::string filepath;
    mutable std::mutex model_mutex;

public:
    // Create a new model with random data
    Model(size_t size_bytes, const std::string& path) : 
        data(size_bytes, 0), 
        version(0),
        filepath(path) {
        
        // Initialize with random data
        generateRandomData();
    }

    // Load model from disk if it exists
    bool loadFromDisk() {
        if (!std::filesystem::exists(filepath)) {
            return false;
        }

        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            return false;
        }

        // First read version
        uint64_t stored_version;
        file.read(reinterpret_cast<char*>(&stored_version), sizeof(stored_version));
        
        // Then read data
        file.read(data.data(), data.size());
        
        if (file) {
            version.store(stored_version);
            return true;
        }
        return false;
    }

    // Save model to disk
    bool saveToDisk() {
        // Safety check for empty filepath
        if (filepath.empty()) {
            spdlog::error("Error: Cannot save model with empty filepath");
            return false;
        }
        
        std::filesystem::path directory = std::filesystem::path(filepath).parent_path();
        if (!std::filesystem::exists(directory)) {
            std::filesystem::create_directories(directory);
        }

        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            spdlog::info("Error: Could not open file for writing: {}", filepath);
            return false;
        }

        // First write version
        uint64_t current_version = version.load();
        file.write(reinterpret_cast<char*>(&current_version), sizeof(current_version));
        
        // Then write data
        file.write(data.data(), data.size());
        
        return static_cast<bool>(file);
    }
    
    // Get the model filepath
    std::string getFilePath() const {
        return filepath;
    }

    // Generate random data for the model
    void generateRandomData() {
        std::lock_guard<std::mutex> lock(model_mutex);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<char>(rand() % 256);
        }
        version++;
    }

    // Get the model version
    uint64_t getVersion() const {
        return version.load();
    }

    // Get a copy of the model data
    std::vector<char> getData() const {
        std::lock_guard<std::mutex> lock(model_mutex);
        return data;
    }

    // Update model with new data and optionally set version
    void update(const std::vector<char>& new_data, std::optional<uint64_t> new_version = std::nullopt) {
        std::lock_guard<std::mutex> lock(model_mutex);
        if (new_data.size() == data.size()) {
            data = new_data;
            version = new_version.has_value() ? *new_version : version + 1;
        }
    }

    // Create a deep copy of this model
    std::shared_ptr<Model> createCopy() const {
        std::lock_guard<std::mutex> lock(model_mutex);
        std::shared_ptr<Model> copy = std::make_shared<Model>(data.size(), filepath);
        copy->data = data;
        copy->version.store(version.load());
        return copy;
    }
};

// Represents a single entry in a buffer
struct BufferEntry {
    std::vector<char> data;
    bool filled;

    BufferEntry(size_t size) : data(size, 0), filled(false) {}
};

// Local buffer with a single entry (used by Agents)
class Buffer {
private:
    std::vector<BufferEntry> entries;

public:
    Buffer(size_t entry_size, size_t num_entries = 1) {
        for (size_t i = 0; i < num_entries; i++) {
            entries.emplace_back(entry_size * ELEMENT_SIZE);
        }
    }

    BufferEntry& getEntry(size_t index = 0) {
        return entries[index];
    }

    void reset() {
        for (auto& entry : entries) {
            entry.filled = false;
        }
    }
};

// Shared buffer with multiple entries (used between Agents and Learner)
class SharedBuffer {
private:
    std::vector<BufferEntry> entries;
    std::mutex buffer_mutex;
    std::condition_variable not_full;
    std::condition_variable not_empty;
    size_t write_index;
    size_t read_index;
    size_t count;
    size_t capacity;
    // fix:
    std::atomic<bool> draining_{false};

public:
    SharedBuffer(size_t entry_size, size_t buffer_capacity) 
        : entries(buffer_capacity, BufferEntry(entry_size * ELEMENT_SIZE)),
          write_index(0),
          read_index(0),
          count(0),
          capacity(buffer_capacity) {}

    void setDraining() {
        draining_ = true;
        not_empty.notify_all(); // Wake up waiting readers
        not_full.notify_all();  // wake any blocked writers
    }

    // Write data to the buffer (used by Agents) - blocking version
    bool write(const std::vector<char>& data) {
        std::unique_lock<std::mutex> lock(buffer_mutex);
        
        // Wait until buffer has space or should_stop is signaled
        not_full.wait(lock, [this] { return count < capacity; });
        
        // Copy data to the current write position
        if (data.size() <= entries[write_index].data.size()) {
            std::copy(data.begin(), data.end(), entries[write_index].data.begin());
            entries[write_index].filled = true;
            
            // Update write index and count
            write_index = (write_index + 1) % capacity;
            count++;
            
            // Notify readers
            lock.unlock();
            not_empty.notify_one();
            return true;
        }
        
        return false;
    }

    // Try to write data without blocking (returns false if buffer is full)
    bool try_write(const std::vector<char>& data) {
        std::unique_lock<std::mutex> lock(buffer_mutex, std::try_to_lock);
        
        if (!lock.owns_lock() || count >= capacity) {
            return false;
        }
        
        if (data.size() <= entries[write_index].data.size()) {
            std::copy(data.begin(), data.end(), entries[write_index].data.begin());
            entries[write_index].filled = true;
            
            write_index = (write_index + 1) % capacity;
            count++;
            
            lock.unlock();
            not_empty.notify_one();
            return true;
        }
        
        return false;
    }

    // Read multiple entries from the buffer (used by Learner)
    std::vector<std::vector<char>> readBatch(size_t batch_size) {
        std::unique_lock<std::mutex> lock(buffer_mutex);

        // Wait until: 
        //   a) full batch is available (i.e., we have enough entries) OR 
        //   b) (fix) we're draining and should exit
        not_empty.wait(lock, [this, batch_size] { 
            return count >= batch_size || draining_.load(); 
        });

        // drop if BOTH draining AND incomplete
        if (draining_.load() && count < batch_size) {
            return {};  // Return empty batch during drain
        }
        
        std::vector<std::vector<char>> batch;
        batch.reserve(batch_size);
        
        // Read up to batch_size entries
        for (size_t i = 0; i < batch_size; i++) {
            batch.push_back(entries[read_index].data);
            entries[read_index].filled = false;
            
            // Update read index and count
            read_index = (read_index + 1) % capacity;
            count--;
        }
        
        // Notify writers
        lock.unlock();
        not_full.notify_all();
        
        return batch;
    }

    // Get the number of filled entries
    size_t getFilledCount() {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        return count;
    }
};

// Model manager for handling model updates and synchronization
class ModelManager {
private:
    std::vector<std::shared_ptr<Model>> models;
    std::vector<std::mutex> model_mutexes;
    std::vector<std::condition_variable> model_updated;
    std::vector<std::atomic<uint64_t>> latest_versions;
    std::string model_directory;
    std::vector<uint64_t> checkpoint_counters;  // Added counters for checkpoint iterations

public:
    ModelManager(size_t num_players, size_t model_size, const std::string& directory) 
        : models(num_players),
          model_mutexes(num_players),
          model_updated(num_players),
          latest_versions(num_players),
          model_directory(directory),
          checkpoint_counters(num_players, 0) {  // Initialize checkpoint counters
        
        // Create or load models for each player
        for (size_t p = 0; p < num_players; p++) {
            std::string filepath = model_directory + "/model_" + std::to_string(p) + "_latest.bin";
            models[p] = std::make_shared<Model>(model_size, filepath);
            latest_versions[p].store(models[p]->getVersion());
        }
    }

    // Try to load models from disk
    void loadModels(const std::string& modelPath) {
        if (modelPath.empty()) return;
        
        for (size_t p = 0; p < models.size(); p++) {
            // First try to load a specific checkpoint if specified
            std::string filepath = modelPath + "/model_" + std::to_string(p) + "_latest.bin";
            
            // Check if directory exists but not the latest file - try to find most recent checkpoint
            if (std::filesystem::exists(modelPath) && !std::filesystem::exists(filepath)) {
                // Pattern to search for: model_p_*.bin
                std::string pattern = "model_" + std::to_string(p) + "_";
                uint64_t highest_checkpoint = 0;
                std::string highest_file = "";
                
                for (const auto& entry : std::filesystem::directory_iterator(modelPath)) {
                    std::string filename = entry.path().filename().string();
                    if (filename.find(pattern) == 0) {
                        // Extract iteration number
                        size_t start_pos = pattern.length();
                        size_t end_pos = filename.find(".bin");
                        if (end_pos != std::string::npos) {
                            std::string iter_str = filename.substr(start_pos, end_pos - start_pos);
                            try {
                                uint64_t iter_num = std::stoull(iter_str);
                                if (iter_num > highest_checkpoint) {
                                    highest_checkpoint = iter_num;
                                    highest_file = entry.path().string();
                                }
                            } catch (...) {
                                // Skip files with non-numeric iteration part
                            }
                        }
                    }
                }
                
                if (!highest_file.empty()) {
                    filepath = highest_file;
                    spdlog::info("Found highest checkpoint for player {}: {}", p, filepath);
                    checkpoint_counters[p] = highest_checkpoint + 1; // Start from next iteration
                }
            }
            
            models[p] = std::make_shared<Model>(models[p]->getData().size(), filepath);
            if (models[p]->loadFromDisk()) {
                latest_versions[p].store(models[p]->getVersion());
                spdlog::info("Loaded model {} from disk, version: {}", p, models[p]->getVersion());
            }
        }
    }

    // Save a specific model to disk with versioned filename
    void saveModel(size_t player_index, uint64_t current_iteration = 0) {
        if (player_index >= models.size() || !models[player_index]) {
            spdlog::error("Error: Invalid model index or null model: {}", player_index);
            return;
        }
        
        // Create a deep copy of the model
        std::shared_ptr<Model> model_copy = models[player_index]->createCopy();
        if (!model_copy) {
            spdlog::error("Error: Failed to create model copy for player {}", player_index);
            return;
        }
        
        // Create timestamped filename
        std::string timestamp = std::to_string(current_iteration > 0 ? current_iteration : checkpoint_counters[player_index]++);
        std::string versioned_filepath = model_directory + "/model_" + std::to_string(player_index) + "_" + timestamp + ".bin";
        
        // Also create a "latest" symlink/copy
        std::string latest_filepath = model_directory + "/model_" + std::to_string(player_index) + "_latest.bin";
        
        // Update model filepath to the versioned one
        model_copy = std::make_shared<Model>(model_copy->getData().size(), versioned_filepath);
        model_copy->update(model_copy->getData());  // Keep same data, new path
        
        // Save versioned file
        if (model_copy->saveToDisk()) {
            spdlog::info("Saved checkpoint for player {} at iteration {} to {}", player_index, timestamp, versioned_filepath);
            
            // Also save a copy as "latest"
            auto latest_model = std::make_shared<Model>(model_copy->getData().size(), latest_filepath);
            latest_model->update(model_copy->getData());
            latest_model->saveToDisk();
        } else {
            spdlog::error("Error: Failed to save checkpoint for player {}", player_index);
        }
    }
    
    // Save all models to disk (for final checkpoints or manual saving)
    void saveAllModels(uint64_t current_iteration = 0) {
        for (size_t p = 0; p < models.size(); p++) {
            saveModel(p, current_iteration);
        }
    }

    // Get a specific model
    std::shared_ptr<Model> getModel(size_t player_index) {
        if (player_index < models.size()) {
            return models[player_index];
        }
        return nullptr;
    }

    // Update a model and notify waiting threads
    void updateModel(size_t player_index, const std::shared_ptr<Model>& new_model) {
        if (player_index >= models.size()) return;
        
        {
            std::lock_guard<std::mutex> lock(model_mutexes[player_index]);
            models[player_index] = new_model;
            latest_versions[player_index].store(new_model->getVersion());
        }
        
        model_updated[player_index].notify_all();
    }

    // Wait for a model update (returns true if updated, false if timed out)
    bool waitForModelUpdate(size_t player_index, uint64_t current_version, int timeout_ms) {
        if (player_index >= models.size()) return false;
        
        std::unique_lock<std::mutex> lock(model_mutexes[player_index]);
        
        // Check if already updated
        if (latest_versions[player_index].load() > current_version) {
            return true;
        }
        
        // Wait for notification with timeout
        return model_updated[player_index].wait_for(
            lock, 
            std::chrono::milliseconds(timeout_ms),
            [this, player_index, current_version] { 
                return latest_versions[player_index].load() > current_version; 
            }
        );
    }

    // Get the latest version for a model
    uint64_t getLatestVersion(size_t player_index) {
        if (player_index < latest_versions.size()) {
            return latest_versions[player_index].load();
        }
        return 0;
    }
};

#endif // DATA_H