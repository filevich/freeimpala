import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import platform
import os
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

def get_loss_function(loss_function_name):
    """Return the appropriate loss function based on name."""
    if loss_function_name.lower() == 'mse':
        return nn.MSELoss()
    elif loss_function_name.lower() == 'mae':
        return nn.L1Loss()
    elif loss_function_name.lower() == 'huber':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_function_name}")

def get_optimizer(optimizer_type, model_parameters, learning_rate):
    """Return the appropriate optimizer based on type."""
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def determine_device(args):
    """Determine which device to use based on args and available hardware."""
    if args.cpu:
        return torch.device("cpu")
    
    if args.gpu == 'auto':
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif args.gpu == 'cuda' and torch.cuda.is_available():
        return torch.device("cuda")
    elif args.gpu == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def generate_synthetic_data(batch_size, seq_length, device):
    """Generate synthetic data for the model."""
    # z is the sequence input for LSTM: [batch_size, seq_length, input_features]
    z = torch.randn(batch_size, seq_length, 162, device=device)
    
    # x is the additional features input: [batch_size, 484]
    x = torch.randn(batch_size, 484, device=device)
    
    # Create synthetic target values
    targets = torch.randn(batch_size, 1, device=device)
    
    return z, x, targets

def run_single_training_iteration(model, z, x, targets, criterion, optimizer, device):
    """Run a single training iteration and return the elapsed time and loss."""
    # Clear gradients
    optimizer.zero_grad()
    
    # Start timing
    start_time = time.time()
    
    # Forward pass
    output = model(z, x, return_value=True)
    loss = criterion(output['values'], targets)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # End timing - synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for all GPU operations to finish
    end_time = time.time()
    
    return end_time - start_time, loss.item()

def run_profiled_iteration(model, z, x, targets, criterion, optimizer, device):
    """Run a single training iteration with profiling enabled."""
    # Clear gradients
    optimizer.zero_grad()
    
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_training"):
            # Start timing
            start_time = time.time()
            
            # Forward pass
            output = model(z, x, return_value=True)
            loss = criterion(output['values'], targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # End timing - synchronize if using CUDA
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return end_time - start_time, loss.item()

def perform_warmup(model, batch_size, seq_length, criterion, optimizer, device, num_warmup=5):
    """Perform multiple warmup runs to ensure system is fully initialized."""
    print("\nPerforming warmup runs...")
    
    for i in range(num_warmup):
        # Generate fresh data for each warmup run
        z, x, targets = generate_synthetic_data(batch_size, seq_length, device)
        
        # Run a complete training iteration
        elapsed_time, loss = run_single_training_iteration(model, z, x, targets, criterion, optimizer, device)
        print(f"  Warmup {i+1}/{num_warmup}: {elapsed_time*1000:.2f} ms")
    
    print("Warmup complete. Starting benchmark...\n")

def get_model_stats(model):
    """Get statistics about the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_gpu_info():
    """Get information about available GPUs or accelerators."""
    # Check for CUDA
    if torch.cuda.is_available():
        info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info.append(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.2f} GB, "
                      f"Compute Capability: {props.major}.{props.minor}")
        return "\n".join(info)
    # Check for MPS (Metal Performance Shaders on Mac)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "Apple Metal (MPS) available"
    else:
        return "No GPU accelerators available"

def get_system_info():
    """Get system information."""
    # Detect MPS availability for Metal support on Mac
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CPU": platform.processor() or "Unknown",
        "CPU Cores": os.cpu_count(),
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cuDNN Version": torch.backends.cudnn.version() if torch.cuda.is_available() and torch.backends.cudnn.is_available() else "N/A",
        "MPS (Metal) Available": has_mps,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return info

def collect_environment_info():
    """Collect and print environment information."""
    system_info = get_system_info()
    gpu_info = get_gpu_info()
    
    print("\n=== BENCHMARK ENVIRONMENT ===")
    print(f"PyTorch Version: {system_info['PyTorch Version']}")
    print(f"Python Version: {system_info['Python Version']}")
    print(f"Platform: {system_info['Platform']}")
    print(f"CUDA: {system_info['CUDA Version']}")
    print(f"cuDNN: {system_info['cuDNN Version']}")
    if "MPS (Metal) Available" in system_info:
        print(f"MPS (Metal): {'Yes' if system_info['MPS (Metal) Available'] else 'No'}")
    print("\n=== GPU/ACCELERATOR INFORMATION ===")
    print(gpu_info)
    
    return system_info, gpu_info

def print_benchmark_config(args):
    """Print benchmark configuration."""
    print("\n=== BENCHMARK CONFIGURATION ===")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Loss Function: {args.loss_function}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Number of Runs: {args.runs}")
    print(f"Number of Warmup Runs: {args.warmup_runs}")
    print(f"Save Directory: {args.save_dir}")

def run_benchmark_iterations(model, args, criterion, optimizer, device):
    """Run all benchmark iterations and return the results."""
    train_times = []
    losses = []
    
    for run in range(args.runs):
        # Generate fresh data for each run
        z, x, targets = generate_synthetic_data(args.batch_size, args.seq_length, device)
        
        # Run iteration with or without profiling
        if args.profile and run == 0:
            print("\nProfiling first run:")
            elapsed_time, loss = run_profiled_iteration(model, z, x, targets, criterion, optimizer, device)
        else:
            elapsed_time, loss = run_single_training_iteration(model, z, x, targets, criterion, optimizer, device)
        
        train_times.append(elapsed_time)
        losses.append(loss)
        print(f"Run {run+1}/{args.runs}: Training time: {elapsed_time*1000:.2f} ms, Loss: {loss:.6f}")
    
    return train_times, losses

def compute_and_print_results(times, losses, args, total_params, trainable_params):
    """Compute and print benchmark results."""
    avg_time = sum(times) / len(times)
    throughput = args.batch_size / avg_time
    
    print("\n=== BENCHMARK RESULTS ===")
    for i, (time_val, loss) in enumerate(zip(times, losses)):
        print(f"Run {i+1}/{args.runs}: Training time: {time_val*1000:.2f} ms, Loss: {loss:.6f}")
    
    print(f"\nAverage training time over {args.runs} runs: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/second")
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return avg_time, throughput

def save_benchmark_results(results, args, filename="/tmp/benchmark_results.txt"):
    """Save benchmark results to a file."""
    with open(filename, 'w') as f:
        f.write("=== FarmerLstmModel Benchmark Results ===\n\n")
        
        # Write timestamp and system info
        f.write(f"Timestamp: {results['system_info']['Timestamp']}\n\n")
        
        f.write("System Information:\n")
        for key, value in results['system_info'].items():
            if key != "Timestamp":  # Already printed above
                f.write(f"  {key}: {value}\n")
        
        # GPU information
        f.write("\nGPU/Accelerator Information:\n")
        f.write(f"  {results['gpu_info']}\n")
        
        # Benchmark configuration
        f.write("\nBenchmark Configuration:\n")
        for key, value in vars(args).items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        
        # Model information
        f.write("\nModel Information:\n")
        f.write(f"  Total Parameters: {results['total_params']:,}\n")
        f.write(f"  Trainable Parameters: {results['trainable_params']:,}\n")
        
        # Performance results
        f.write("\nPerformance Results:\n")
        f.write(f"  Average Training Time: {results['avg_time']*1000:.2f} ms\n")
        f.write(f"  Throughput: {results['throughput']:.2f} samples/second\n")
        
        # Individual run information
        f.write("\nIndividual Run Times:\n")
        for i, (time_ms, loss) in enumerate(zip(results['times'], results['losses'])):
            f.write(f"  Run {i+1}: {time_ms*1000:.2f} ms, Loss: {loss:.6f}\n")
    
    print(f"\nResults saved to {filename}")

def maybe_save_results(args, results):
    """Save benchmark results to a file if save_results is enabled."""
    if not args.no_save:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/benchmark_results_{timestamp}.txt"
        save_benchmark_results(results, args, filename)

def save_model(model, save_dir):
    """Save the model to the specified directory."""
    try:
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f"farmer_lstm_model_{timestamp}.pth")
        # Save the model's state dictionary
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        return model_path
    except Exception as e:
        print(f"\nError saving model: {str(e)}")
        return None

def benchmark_training(args):
    """Main benchmark function."""
    # Collect environment info
    system_info, gpu_info = collect_environment_info()
    
    # Print benchmark configuration
    print_benchmark_config(args)
    
    # Determine which device to use
    device = determine_device(args)
    print(f"\nUsing device: {device}")
    
    # Create model and move to device
    model = FarmerLstmModel().to(device)
    
    # Set up optimizer and loss function
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    criterion = get_loss_function(args.loss_function)
    
    # Perform warmup
    perform_warmup(model, args.batch_size, args.seq_length, criterion, optimizer, device, args.warmup_runs)
    
    # Run benchmark iterations
    train_times, losses = run_benchmark_iterations(model, args, criterion, optimizer, device)
    
    # Get model statistics
    total_params, trainable_params = get_model_stats(model)
    
    # Compute and print results
    avg_time, throughput = compute_and_print_results(train_times, losses, args, total_params, trainable_params)
    
    # Save the trained model
    saved_model_path = save_model(model, args.save_dir)
    
    results = {
        'system_info': system_info,
        'gpu_info': gpu_info,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'avg_time': avg_time,
        'throughput': throughput,
        'times': train_times,
        'losses': losses,
        'saved_model_path': saved_model_path
    }
    maybe_save_results(args, results)
    return avg_time, throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark FarmerLstmModel training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=10, help='Sequence length for LSTM input')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'mae', 'huber'], 
                        help='Loss function to use (MSE, MAE, or Huber)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], 
                        help='Optimizer to use')
    parser.add_argument('--gpu', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], 
                        help='GPU backend to use (auto, cuda, mps, or cpu)')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU even if GPU is available')
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiling for detailed analysis')
    parser.add_argument('--runs', type=int, default=5, help='Number of training runs to average timing over')
    parser.add_argument('--warmup-runs', type=int, default=5, help='Number of warmup runs before benchmark')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    parser.add_argument('--save-dir', type=str, default='/tmp', help='Directory to save the trained model')
    
    args = parser.parse_args()
    benchmark_training(args)