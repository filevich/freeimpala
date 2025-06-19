#!/bin/bash

# Parallel Performance Benchmark Script
# This script runs a series of experiments with different configurations
# and collects performance metrics to measure parallel scaling

# Configuration parameters
PLAYERS=(2 4)                 # Number of players
BUFFER_SIZES=(5 10 20)        # Buffer capacity
BATCH_SIZES=(3 5 10)          # Batch size
ITERATIONS=200                # Total iterations
AGENTS=(1 2 4 8 16)              # Number of agents
OUTPUT_DIR="benchmark_results" # Directory to store results
CHECKPOINT_DIR="./checkpoints" # Checkpoint location
LEARNER_TIME=500              # Learner training time (ms)
AGENT_TIME=200                # Agent gameplay time (ms)
GAME_STEPS=100                # Game steps
CHECKPOINT_FREQ=20            # Checkpoint frequency

# Create output directory
mkdir -p $OUTPUT_DIR

# Create summary CSV file
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"
echo "Config,Agents,Players,BufferSize,BatchSize,ExecutionTime,SpeedUp,Efficiency,Overhead,IterationsPerSec,ModelsPerSec,DataTransfersPerSec" > $SUMMARY_FILE

# Run the baseline configuration first (with 1 agent)
echo "Running baseline configuration with 1 agent..."
# For the baseline configuration:
CMD="./build/freeimpala \
    --players 2 \
    --iterations $ITERATIONS \
    --entry-size $GAME_STEPS \
    --buffer-capacity 10 \
    --batch-size 5 \
    --learner-time $LEARNER_TIME \
    --checkpoint-freq $CHECKPOINT_FREQ \
    --checkpoint-location $CHECKPOINT_DIR \
    --starting-model \"\" \
    --agents 1 \
    --game-steps $GAME_STEPS \
    --agent-time $AGENT_TIME \
    --metrics-file \"$OUTPUT_DIR/baseline.csv\""
echo "Executing command: $CMD"
eval $CMD

# Extract baseline time
BASELINE_TIME=$(grep "TotalExecutionTime_ns" "$OUTPUT_DIR/baseline.csv" | cut -d',' -f2)
BASELINE_TIME_SEC=$(echo "scale=6; $BASELINE_TIME/1000000000" | bc)
echo "Baseline time (1 agent): $BASELINE_TIME_SEC seconds"

# Run experiments
echo "Running experiments..."

# 1. Scaling the number of agents
echo "Experiment 1: Scaling agents"
for AGENT_COUNT in ${AGENTS[@]:1}; do  # Skip the first agent (1) since we already ran it
    echo "  Running with $AGENT_COUNT agents..."
    CONFIG_NAME="agents_${AGENT_COUNT}"

    # For the agent scaling experiment:
    CMD="./build/freeimpala \
        --players 2 \
        --iterations $ITERATIONS \
        --entry-size $GAME_STEPS \
        --buffer-capacity 10 \
        --batch-size 5 \
        --learner-time $LEARNER_TIME \
        --checkpoint-freq $CHECKPOINT_FREQ \
        --checkpoint-location $CHECKPOINT_DIR \
        --starting-model \"\" \
        --agents $AGENT_COUNT \
        --game-steps $GAME_STEPS \
        --agent-time $AGENT_TIME \
        --metrics-file \"$OUTPUT_DIR/${CONFIG_NAME}.csv\" \
        --baseline-time $BASELINE_TIME_SEC"
    echo "Executing command: $CMD"
    eval $CMD
    
    # Extract metrics for summary
    EXEC_TIME=$(grep "TotalExecutionTime_ns" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    EXEC_TIME_SEC=$(echo "scale=6; $EXEC_TIME/1000000000" | bc)
    SPEEDUP=$(echo "scale=4; $BASELINE_TIME_SEC/$EXEC_TIME_SEC" | bc)
    EFFICIENCY=$(echo "scale=4; $SPEEDUP/$AGENT_COUNT" | bc)
    OVERHEAD=$(echo "scale=4; ($AGENT_COUNT*$EXEC_TIME_SEC)-$BASELINE_TIME_SEC" | bc)
    ITER_PER_SEC=$(grep "IterationsPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    MODELS_PER_SEC=$(grep "ModelUpdatesPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    TRANSFERS_PER_SEC=$(grep "DataTransfersPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    
    # Add to summary
    echo "AgentScaling,$AGENT_COUNT,2,10,5,$EXEC_TIME_SEC,$SPEEDUP,$EFFICIENCY,$OVERHEAD,$ITER_PER_SEC,$MODELS_PER_SEC,$TRANSFERS_PER_SEC" >> $SUMMARY_FILE
done

# 2. Scaling the buffer size with a fixed number of agents (8)
echo "Experiment 2: Scaling buffer size"
for BUFFER_SIZE in ${BUFFER_SIZES[@]}; do
    echo "  Running with buffer size $BUFFER_SIZE..."
    CONFIG_NAME="buffer_${BUFFER_SIZE}"

    # For the buffer scaling experiment:
    CMD="./build/freeimpala \
        --players 2 \
        --iterations $ITERATIONS \
        --entry-size $GAME_STEPS \
        --buffer-capacity $BUFFER_SIZE \
        --batch-size $(($BUFFER_SIZE / 2)) \
        --learner-time $LEARNER_TIME \
        --checkpoint-freq $CHECKPOINT_FREQ \
        --checkpoint-location $CHECKPOINT_DIR \
        --starting-model \"\" \
        --agents 8 \
        --game-steps $GAME_STEPS \
        --agent-time $AGENT_TIME \
        --metrics-file \"$OUTPUT_DIR/${CONFIG_NAME}.csv\" \
        --baseline-time $BASELINE_TIME_SEC"
    echo "Executing command: $CMD"
    eval $CMD
    
    # Extract metrics for summary
    EXEC_TIME=$(grep "TotalExecutionTime_ns" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    EXEC_TIME_SEC=$(echo "scale=6; $EXEC_TIME/1000000000" | bc)
    SPEEDUP=$(echo "scale=4; $BASELINE_TIME_SEC/$EXEC_TIME_SEC" | bc)
    EFFICIENCY=$(echo "scale=4; $SPEEDUP/8" | bc)
    OVERHEAD=$(echo "scale=4; (8*$EXEC_TIME_SEC)-$BASELINE_TIME_SEC" | bc)
    ITER_PER_SEC=$(grep "IterationsPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    MODELS_PER_SEC=$(grep "ModelUpdatesPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    TRANSFERS_PER_SEC=$(grep "DataTransfersPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    
    # Add to summary
    echo "BufferScaling,8,2,$BUFFER_SIZE,$(($BUFFER_SIZE / 2)),$EXEC_TIME_SEC,$SPEEDUP,$EFFICIENCY,$OVERHEAD,$ITER_PER_SEC,$MODELS_PER_SEC,$TRANSFERS_PER_SEC" >> $SUMMARY_FILE
done

# 3. Scaling the number of players
echo "Experiment 3: Scaling players"
for PLAYER_COUNT in ${PLAYERS[@]}; do
    echo "  Running with $PLAYER_COUNT players..."
    CONFIG_NAME="players_${PLAYER_COUNT}"
    
    # For the player scaling experiment:
    CMD="./build/freeimpala \
        --players $PLAYER_COUNT \
        --iterations $ITERATIONS \
        --entry-size $GAME_STEPS \
        --buffer-capacity 10 \
        --batch-size 5 \
        --learner-time $LEARNER_TIME \
        --checkpoint-freq $CHECKPOINT_FREQ \
        --checkpoint-location $CHECKPOINT_DIR \
        --starting-model \"\" \
        --agents 8 \
        --game-steps $GAME_STEPS \
        --agent-time $AGENT_TIME \
        --metrics-file \"$OUTPUT_DIR/${CONFIG_NAME}.csv\" \
        --baseline-time $BASELINE_TIME_SEC"
    echo "Executing command: $CMD"
    eval $CMD
    
    # Extract metrics for summary
    EXEC_TIME=$(grep "TotalExecutionTime_ns" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    EXEC_TIME_SEC=$(echo "scale=6; $EXEC_TIME/1000000000" | bc)
    SPEEDUP=$(echo "scale=4; $BASELINE_TIME_SEC/$EXEC_TIME_SEC" | bc)
    EFFICIENCY=$(echo "scale=4; $SPEEDUP/8" | bc)
    OVERHEAD=$(echo "scale=4; (8*$EXEC_TIME_SEC)-$BASELINE_TIME_SEC" | bc)
    ITER_PER_SEC=$(grep "IterationsPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    MODELS_PER_SEC=$(grep "ModelUpdatesPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    TRANSFERS_PER_SEC=$(grep "DataTransfersPerSecond" "$OUTPUT_DIR/${CONFIG_NAME}.csv" | cut -d',' -f2)
    
    # Add to summary
    echo "PlayerScaling,8,$PLAYER_COUNT,10,5,$EXEC_TIME_SEC,$SPEEDUP,$EFFICIENCY,$OVERHEAD,$ITER_PER_SEC,$MODELS_PER_SEC,$TRANSFERS_PER_SEC" >> $SUMMARY_FILE
done

# Generate summary and plots using Python
if command -v python3 > /dev/null; then
    echo "Generating summary and plots with Python..."
    python3 generate_plots.py "$OUTPUT_DIR" --metric TotalIterations
else
    echo "Python 3 not found. Skipping plot generation."
fi

echo "All experiments completed. Results saved in $OUTPUT_DIR"
echo "Summary data available in $SUMMARY_FILE"
