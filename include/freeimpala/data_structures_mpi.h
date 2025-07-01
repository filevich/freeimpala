#ifndef DATA_MPI_H
#define DATA_MPI_H

#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <filesystem>
#include <mpi.h>

// Size of each element in bytes
constexpr size_t ELEMENT_SIZE = 1024;

// MPI message tags
enum MPITags {
    TAG_DATA_TRANSFER = 1000,      // Base tag for data transfers (add player_id)
    TAG_MODEL_VERSION = 2000,      // Base tag for model version queries (add player_id)
    TAG_MODEL_UPDATE = 3000,       // Signal that a model update is coming
    TAG_SHUTDOWN = 9999            // Shutdown signal
};

// Simple model class that represents our "neural network"
class Model {
private:
    std::vector<char> data;
    std::atomic<uint64_t> version;
    std::string filepath;

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
        if (filepath.empty()) {
            std::cerr << "Error: Cannot save model with empty filepath" << std::endl;
            return false;
        }
        
        std::filesystem::path directory = std::filesystem::path(filepath).parent_path();
        if (!std::filesystem::exists(directory)) {
            std::filesystem::create_directories(directory);
        }

        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return false;
        }

        // First write version
        uint64_t current_version = version.load();
        file.write(reinterpret_cast<char*>(&current_version), sizeof(current_version));
        
        // Then write data
        file.write(data.data(), data.size());
        
        return static_cast<bool>(file);
    }

    // Generate random data for the model
    void generateRandomData() {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<char>(rand() % 256);
        }
        version++;
    }

    // Get the model version
    uint64_t getVersion() const {
        return version.load();
    }

    // Get a reference to the model data
    std::vector<char>& getData() {
        return data;
    }

    // Get a const reference to the model data
    const std::vector<char>& getData() const {
        return data;
    }

    // Update model with new data and increment version
    void update(const std::vector<char>& new_data, uint64_t new_version) {
        if (new_data.size() == data.size()) {
            data = new_data;
            version.store(new_version);
        }
    }

    // Get model size
    size_t getSize() const {
        return data.size();
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

// MPI-based buffer for receiving data at the learner
class MPIReceiveBuffer {
private:
    std::vector<std::vector<char>> buffer;
    size_t entry_size;
    size_t capacity;
    size_t write_pos;
    size_t read_pos;
    size_t count;

public:
    MPIReceiveBuffer(size_t entry_sz, size_t cap) 
        : entry_size(entry_sz * ELEMENT_SIZE),
          capacity(cap),
          write_pos(0),
          read_pos(0),
          count(0) {
        buffer.resize(capacity);
        for (auto& entry : buffer) {
            entry.resize(entry_size);
        }
    }

    // Try to receive data via MPI (non-blocking check)
    bool tryReceive(int source_rank, int tag) {
        int flag;
        MPI_Status status;
        
        // Check if there's an incoming message
        MPI_Iprobe(source_rank, tag, MPI_COMM_WORLD, &flag, &status);
        
        if (flag && count < capacity) {
            // Receive the data
            MPI_Recv(buffer[write_pos].data(), entry_size, MPI_CHAR, 
                    status.MPI_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            write_pos = (write_pos + 1) % capacity;
            count++;
            return true;
        }
        
        return false;
    }

    // Get a batch of entries
    std::vector<std::vector<char>> getBatch(size_t batch_size) {
        std::vector<std::vector<char>> batch;
        
        size_t available = std::min(batch_size, count);
        for (size_t i = 0; i < available; i++) {
            batch.push_back(buffer[read_pos]);
            read_pos = (read_pos + 1) % capacity;
            count--;
        }
        
        return batch;
    }

    // Check if we have enough data for a batch
    bool hasData(size_t batch_size) const {
        return count >= batch_size;
    }

    size_t getCount() const {
        return count;
    }
};

#endif // DATA_MPI_H