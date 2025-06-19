#!/bin/bash
#SBATCH --job-name=freeimpala
#SBATCH --output=freeimpala_%j.out
#SBATCH --error=freeimpala_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --partition=compute

# Load necessary modules (adjust as needed for your cluster)
module load cmake
module load gcc

# Set the number of agents to use (default: use half of available CPUs)
if [ -z "$NUM_AGENTS" ]; then
    NUM_AGENTS=$((SLURM_CPUS_PER_TASK / 2))
fi

# Set other parameters (or use defaults from the program)
PLAYERS=${PLAYERS:-4}
ITERATIONS=${ITERATIONS:-1000}
ENTRY_SIZE=${ENTRY_SIZE:-100}
BUFFER_CAPACITY=${BUFFER_CAPACITY:-20}
BATCH_SIZE=${BATCH_SIZE:-10}
LEARNER_TIME=${LEARNER_TIME:-500}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-50}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
STARTING_MODEL=${STARTING_MODEL:-""}
GAME_STEPS=${GAME_STEPS:-100}
AGENT_TIME=${AGENT_TIME:-200}

# Create build directory and compile
mkdir -p build
cd build
cmake ..
make -j$SLURM_CPUS_PER_TASK

# Run the program
./freeimpala \
    --players $PLAYERS \
    --iterations $ITERATIONS \
    --entry-size $ENTRY_SIZE \
    --buffer-capacity $BUFFER_CAPACITY \
    --batch-size $BATCH_SIZE \
    --learner-time $LEARNER_TIME \
    --checkpoint-freq $CHECKPOINT_FREQ \
    --checkpoint-location $CHECKPOINT_DIR \
    --starting-model "$STARTING_MODEL" \
    --agents $NUM_AGENTS \
    --game-steps $GAME_STEPS \
    --agent-time $AGENT_TIME

# Print completion message
echo "Job completed at $(date)"