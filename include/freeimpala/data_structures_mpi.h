#ifndef DATA_STRUCTURES_MPI_H
#define DATA_STRUCTURES_MPI_H

#include <iostream>
#include <vector>
#include <mutex>
#include <atomic>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <queue>
#include <functional>
#include <sstream>

// MPI tags
#define DATA_TAG 1
#define VERSION_REQUEST_TAG 2
#define MODEL_REQUEST_TAG 3
#define TERMINATE_TAG 4

// Size of each element in bytes
constexpr size_t ELEMENT_SIZE = 1024;

class Model {
private:
    std::vector<char> data;
    std::atomic<uint64_t> version;
    std::string filepath;
    mutable std::mutex model_mutex;

public:
    Model(size_t size_bytes, const std::string& path) : 
        data(size_bytes, 0), 
        version(0),
        filepath(path) {
        generateRandomData();
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
    
    // Set model data directly (for MPI updates)
    void setData(const std::vector<char>& new_data, uint64_t new_version) {
        std::lock_guard<std::mutex> lock(model_mutex);
        if (new_data.size() == data.size()) {
            data = new_data;
            version.store(new_version);
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
    BufferEntry entry;

public:
    Buffer(size_t size) : entry(size) {}

    BufferEntry& getEntry() {
        return entry;
    }

    void reset() {
        entry.filled = false;
        std::fill(entry.data.begin(), entry.data.end(), 0);
    }
};

// Model manager for handling model updates and synchronization
class ModelManager {
private:
    std::vector<std::shared_ptr<Model>> models;
    std::vector<std::atomic<uint64_t>> latest_versions;
    std::string model_directory;

public:
    ModelManager(size_t num_players, size_t model_size, const std::string& directory) 
        : models(num_players),
          latest_versions(num_players) {
        
        // Create models for each player
        for (size_t p = 0; p < num_players; p++) {
            std::string filepath = directory + "/model_" + std::to_string(p) + "_latest.bin";
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
            
            // Check if file exists
            if (std::filesystem::exists(filepath)) {
                // In a real implementation, we would load the model from disk
                // For this dummy version, we'll just generate new random data
                models[p]->generateRandomData();
                latest_versions[p].store(models[p]->getVersion());
                
                std::stringstream ss;
                ss << "Loaded model " << p << " from disk, version: " << models[p]->getVersion() << std::endl;
                std::cerr << ss.str();
            }
        }
    }

    // Save a specific model to disk with versioned filename
    void saveModel(size_t player_index, uint64_t current_iteration = 0) {
        if (player_index >= models.size() || !models[player_index]) {
            std::cerr << "Error: Invalid model index or null model: " << player_index << std::endl;
            return;
        }
        
        // Create a deep copy of the model
        std::shared_ptr<Model> model_copy = models[player_index]->createCopy();
        if (!model_copy) {
            std::cerr << "Error: Failed to create model copy for player " << player_index << std::endl;
            return;
        }
        
        // Create timestamped filename
        std::string versioned_filepath = model_directory + "/model_" + std::to_string(player_index) + "_" + std::to_string(current_iteration) + ".bin";
        
        // In a real implementation, we would save the model to disk
        std::stringstream ss;
        ss << "Saved checkpoint for player " << player_index << " at iteration " << current_iteration 
                  << " to " << versioned_filepath << std::endl;
        std::cerr << ss.str();
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

    // Update a model
    void updateModel(size_t player_index, const std::shared_ptr<Model>& new_model) {
        if (player_index >= models.size()) return;
        
        models[player_index] = new_model;
        latest_versions[player_index].store(new_model->getVersion());
    }

    // Get the latest version for a model
    uint64_t getLatestVersion(size_t player_index) {
        if (player_index < latest_versions.size()) {
            return latest_versions[player_index].load();
        }
        return 0;
    }
};

#endif // DATA_STRUCTURES_MPI_H