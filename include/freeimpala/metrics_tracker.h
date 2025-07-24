#pragma once

#include <spdlog/spdlog.h>
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <iostream>
#include <fstream>
#include <atomic>
#include <memory>
#include <iomanip>

// Define a high-resolution clock type for timing
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::nanoseconds;

// Singleton class to track system-wide metrics
class MetricsTracker {
private:
    // Private constructor for singleton
    MetricsTracker() : 
        system_start_time(Clock::now()),
        total_iterations(0),
        total_learner_model_updates(0),
        total_agent_model_syncs(0),
        total_data_transfers(0),
        total_simulation_time(0),
        total_training_time(0),
        total_transfer_time(0),
        total_sync_time(0),
        is_running(true) {}
    
    // Static instance
    static std::shared_ptr<MetricsTracker> instance;
    static std::mutex instance_mutex;
    
    // Timing data
    TimePoint system_start_time;
    
    // Atomic counters for thread-safe updates
    std::atomic<uint64_t> total_iterations;
    std::atomic<uint64_t> total_learner_model_updates;
    std::atomic<uint64_t> total_agent_model_syncs;
    std::atomic<uint64_t> total_data_transfers;
    std::atomic<uint64_t> total_simulation_time;  // nanoseconds
    std::atomic<uint64_t> total_training_time;    // nanoseconds
    std::atomic<uint64_t> total_transfer_time;    // nanoseconds
    std::atomic<uint64_t> total_sync_time;        // nanoseconds
    
    // Per-agent timing data 
    std::map<size_t, std::vector<uint64_t>> agent_iteration_times;
    std::map<size_t, uint64_t> agent_total_time;
    mutable std::mutex agent_mutex;
    
    // Thread-specific timing points
    thread_local static TimePoint thread_start_time;
    
    // Flag to indicate if metrics are being collected
    std::atomic<bool> is_running;

public:
    // Delete copy and move constructors/assignments
    MetricsTracker(const MetricsTracker&) = delete;
    MetricsTracker& operator=(const MetricsTracker&) = delete;
    MetricsTracker(MetricsTracker&&) = delete;
    MetricsTracker& operator=(MetricsTracker&&) = delete;
    
    // Get singleton instance
    static std::shared_ptr<MetricsTracker> getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex);
        if (!instance) {
            instance = std::shared_ptr<MetricsTracker>(new MetricsTracker());
        }
        return instance;
    }
    
    // Start collecting metrics
    void start() {
        system_start_time = Clock::now();
        is_running = true;
    }
    
    // Stop collecting metrics
    void stop() {
        is_running = false;
    }
    
    // Start timing an agent iteration
    void startAgentIteration(size_t agent_id) {
        if (!is_running) return;
        thread_start_time = Clock::now();
    }
    
    // End timing an agent iteration
    void endAgentIteration(size_t agent_id) {
        if (!is_running) return;
        auto end_time = Clock::now();
        auto duration = std::chrono::duration_cast<Duration>(end_time - thread_start_time).count();
        
        std::lock_guard<std::mutex> lock(agent_mutex);
        agent_iteration_times[agent_id].push_back(duration);
        agent_total_time[agent_id] += duration;
        total_iterations++;
    }
    
    void recordLearnerModelUpdate() {
        if (!is_running) return;
        total_learner_model_updates++;
    }

    void recordAgentModelSync() {
        if (!is_running) return;
        total_agent_model_syncs++;
    }

    // Record a data transfer
    void recordDataTransfer() {
        if (!is_running) return;
        total_data_transfers++;
    }
    
    // Record timing for specific operations
    void recordSimulationTime(uint64_t nanoseconds) {
        if (!is_running) return;
        total_simulation_time += nanoseconds;
    }
    
    void recordTrainingTime(uint64_t nanoseconds) {
        if (!is_running) return;
        total_training_time += nanoseconds;
    }
    
    void recordTransferTime(uint64_t nanoseconds) {
        if (!is_running) return;
        total_transfer_time += nanoseconds;
    }
    
    void recordSyncTime(uint64_t nanoseconds) {
        if (!is_running) return;
        total_sync_time += nanoseconds;
    }
    
    // Timer class for measuring duration of scoped operations
    class ScopedTimer {
    private:
        TimePoint start_time;
        std::function<void(uint64_t)> callback;
        
    public:
        ScopedTimer(std::function<void(uint64_t)> cb) : start_time(Clock::now()), callback(cb) {}
        
        ~ScopedTimer() {
            auto end_time = Clock::now();
            auto duration = std::chrono::duration_cast<Duration>(end_time - start_time).count();
            callback(duration);
        }
    };
    
    // Create scoped timers for different operations
    ScopedTimer createSimulationTimer() {
        return ScopedTimer([this](uint64_t ns) { this->recordSimulationTime(ns); });
    }
    
    ScopedTimer createTrainingTimer() {
        return ScopedTimer([this](uint64_t ns) { this->recordTrainingTime(ns); });
    }
    
    ScopedTimer createTransferTimer() {
        return ScopedTimer([this](uint64_t ns) { this->recordTransferTime(ns); });
    }
    
    ScopedTimer createSyncTimer() {
        return ScopedTimer([this](uint64_t ns) { this->recordSyncTime(ns); });
    }
    
    // Get total execution time
    uint64_t getTotalExecutionTime() const {
        auto now = Clock::now();
        return std::chrono::duration_cast<Duration>(now - system_start_time).count();
    }
    
private:
    // Get average time per iteration for a specific agent - PRIVATE to prevent deadlocks
    double getAverageIterationTime(size_t agent_id) const {
        std::lock_guard<std::mutex> lock(agent_mutex);
        if (agent_iteration_times.find(agent_id) == agent_iteration_times.end() || 
            agent_iteration_times.at(agent_id).empty()) {
            return 0.0;
        }
        
        const auto& times = agent_iteration_times.at(agent_id);
        uint64_t total = 0;
        for (const auto& time : times) {
            total += time;
        }
        return static_cast<double>(total) / times.size();
    }
    
public:
    
    // Get iterations per second
    double getIterationsPerSecond() const {
        uint64_t total_time_ns = getTotalExecutionTime();
        if (total_time_ns == 0) return 0.0;
        
        return static_cast<double>(total_iterations) / (total_time_ns / 1e9);
    }
    
    // Get learner model updates per second
    double getLearnerUpdatesPerSecond() const {
        uint64_t total_time_ns = getTotalExecutionTime();
        if (total_time_ns == 0) return 0.0;
        return static_cast<double>(total_learner_model_updates) / (total_time_ns / 1e9);
    }

    // Get agent model syncs per second
    double getAgentSyncsPerSecond() const {
        uint64_t total_time_ns = getTotalExecutionTime();
        if (total_time_ns == 0) return 0.0;
        return static_cast<double>(total_agent_model_syncs) / (total_time_ns / 1e9);
    }
    
    // Get data transfers per second
    double getDataTransfersPerSecond() const {
        uint64_t total_time_ns = getTotalExecutionTime();
        if (total_time_ns == 0) return 0.0;
        
        return static_cast<double>(total_data_transfers) / (total_time_ns / 1e9);
    }
    
    // Get time distribution percentages
    std::map<std::string, double> getTimeDistribution() const {
        uint64_t total_time = total_simulation_time + total_training_time + 
                             total_transfer_time + total_sync_time;
        
        if (total_time == 0) {
            return {
                {"simulation", 0.0},
                {"training", 0.0},
                {"transfer", 0.0},
                {"sync", 0.0}
            };
        }
        
        return {
            {"simulation", 100.0 * static_cast<double>(total_simulation_time) / total_time},
            {"training", 100.0 * static_cast<double>(total_training_time) / total_time},
            {"transfer", 100.0 * static_cast<double>(total_transfer_time) / total_time},
            {"sync", 100.0 * static_cast<double>(total_sync_time) / total_time}
        };
    }
    
    // Get raw counter values for custom calculations
    uint64_t getTotalIterations() const { return total_iterations; }
    uint64_t getTotalDataTransfers() const { return total_data_transfers; }
    uint64_t getTotalSimulationTime() const { return total_simulation_time; }
    uint64_t getTotalTrainingTime() const { return total_training_time; }
    uint64_t getTotalTransferTime() const { return total_transfer_time; }
    uint64_t getTotalSyncTime() const { return total_sync_time; }
    
    // Save metrics to a CSV file
    void saveMetricsToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) {
            spdlog::error("Error: Could not open file for writing: {}", filename);
            return;
        }
        
        // Write header
        file << "Metric,Value\n";
        
        // Write total times
        file << "TotalExecutionTime_ns," << getTotalExecutionTime() << "\n";
        file << "TotalSimulationTime_ns," << total_simulation_time << "\n";
        file << "TotalTrainingTime_ns," << total_training_time << "\n";
        file << "TotalTransferTime_ns," << total_transfer_time << "\n";
        file << "TotalSyncTime_ns," << total_sync_time << "\n";
        
        // Write counters
        file << "TotalIterations," << total_iterations << "\n";
        file << "TotalLearnerModelUpdates," << total_learner_model_updates << "\n";
        file << "TotalAgentModelSyncs," << total_agent_model_syncs << "\n";
        file << "TotalDataTransfers," << total_data_transfers << "\n";
        
        // Write rates
        file << "IterationsPerSecond," << getIterationsPerSecond() << "\n";
        file << "LearnerUpdatesPerSecond," << getLearnerUpdatesPerSecond() << "\n";
        file << "AgentSyncsPerSecond," << getAgentSyncsPerSecond() << "\n";
        file << "DataTransfersPerSecond," << getDataTransfersPerSecond() << "\n";
        
        // Write time distribution
        auto distribution = getTimeDistribution();
        for (const auto& [key, value] : distribution) {
            file << "TimePercentage_" << key << "," << value << "\n";
        }
        
        // Write per-agent metrics
        {
            // Only acquire the mutex once for all agent data
            std::lock_guard<std::mutex> lock(agent_mutex);
            
            for (const auto& [agent_id, times] : agent_iteration_times) {
                if (times.empty()) continue;
                
                // Calculate statistics directly
                uint64_t total = 0;
                uint64_t min_time = std::numeric_limits<uint64_t>::max();
                uint64_t max_time = 0;
                
                for (const auto& time : times) {
                    total += time;
                    min_time = std::min(min_time, time);
                    max_time = std::max(max_time, time);
                }
                
                double avg_time = static_cast<double>(total) / times.size();
                
                file << "Agent_" << agent_id << "_TotalTime_ns," << agent_total_time.at(agent_id) << "\n";
                file << "Agent_" << agent_id << "_AvgIterationTime_ns," << avg_time << "\n";
                file << "Agent_" << agent_id << "_MinIterationTime_ns," << min_time << "\n";
                file << "Agent_" << agent_id << "_MaxIterationTime_ns," << max_time << "\n";
            }
        }
        
        file.close();
    }
    
    // Print a summary of metrics to stdout
    void printMetricsSummary() const {
        std::stringstream ss;
        ss << "\n===== Performance Metrics Summary =====\n";
        ss << "Total Execution Time: " << std::fixed << std::setprecision(3) 
                  << (getTotalExecutionTime() / 1e9) << " seconds\n";
        
        ss << "\n--- Throughput Metrics ---\n";
        ss << "Iterations Per Second: " << std::fixed << std::setprecision(2) 
                  << getIterationsPerSecond() << "\n";
        ss << "Learner Model Updates Per Second: " << std::fixed << std::setprecision(2) 
                << getLearnerUpdatesPerSecond() << "\n";
        ss << "Agent Model Syncs Per Second: " << std::fixed << std::setprecision(2) 
                << getAgentSyncsPerSecond() << "\n";
        ss << "Data Transfers Per Second: " << std::fixed << std::setprecision(2) 
                  << getDataTransfersPerSecond() << "\n";
        
        ss << "\n--- Time Distribution ---\n";
        auto distribution = getTimeDistribution();
        for (const auto& [key, value] : distribution) {
            ss << key << ": " << std::fixed << std::setprecision(1) << value << "%\n";
        }
        
        ss << "\n--- Total Counts ---\n";
        ss << "Total Iterations: " << total_iterations << "\n";
        ss << "Total Learner Model Updates: " << total_learner_model_updates << "\n";
        ss << "Total Agent Model Syncs: " << total_agent_model_syncs << "\n";
        ss << "Total Data Transfers: " << total_data_transfers << "\n";
        
        ss << "\n--- Per-Agent Metrics ---\n";
        {
            // Only acquire the mutex once for all agent data
            std::lock_guard<std::mutex> lock(agent_mutex);
            
            for (const auto& [agent_id, times] : agent_iteration_times) {
                if (times.empty()) continue;
                
                // Calculate average directly instead of calling getAverageIterationTime()
                uint64_t total = 0;
                for (const auto& time : times) {
                    total += time;
                }
                double avg_time = static_cast<double>(total) / times.size();
                
                ss << "Agent " << agent_id << " Avg Iteration Time: " 
                          << std::fixed << std::setprecision(3) << (avg_time / 1e6) << " ms\n";
            }
        }
        
        ss << "=====================================\n";
        std::cout << ss.str();
    }
};

// Initialize static members
std::shared_ptr<MetricsTracker> MetricsTracker::instance = nullptr;
std::mutex MetricsTracker::instance_mutex;
thread_local TimePoint MetricsTracker::thread_start_time;