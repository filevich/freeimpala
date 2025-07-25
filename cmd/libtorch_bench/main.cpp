#include <torch/torch.h>
#include <torch/serialize.h>
#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

using clock_type = std::chrono::high_resolution_clock;
using ms = std::chrono::duration<double, std::milli>;

// Model definition
struct FarmerLstmModel : torch::nn::Module {
    FarmerLstmModel() {
        lstm   = register_module("lstm"  , torch::nn::LSTM(torch::nn::LSTMOptions(162, 128).batch_first(true)));
        dense1 = register_module("dense1", torch::nn::Linear(484 + 128, 512));
        dense2 = register_module("dense2", torch::nn::Linear(512, 512));
        dense3 = register_module("dense3", torch::nn::Linear(512, 512));
        dense4 = register_module("dense4", torch::nn::Linear(512, 512));
        dense5 = register_module("dense5", torch::nn::Linear(512, 512));
        dense6 = register_module("dense6", torch::nn::Linear(512,   1));
    }

    torch::Tensor forward(const torch::Tensor& z, const torch::Tensor& x) {
        // z: [B, T, 162]  |  x: [B, 484]
        auto lstm_out_tuple = lstm->forward(z);
        auto last_timestep  = std::get<0>(lstm_out_tuple).select(1, z.size(1)-1); // [B, 128]

        auto cat = torch::cat({last_timestep, x}, /*dim=*/1); // [B, 612]
        cat = torch::relu(dense1(cat));
        cat = torch::relu(dense2(cat));
        cat = torch::relu(dense3(cat));
        cat = torch::relu(dense4(cat));
        cat = torch::relu(dense5(cat));
        return dense6(cat); // [B, 1]
    }

    torch::nn::LSTM   lstm{nullptr};
    torch::nn::Linear dense1{nullptr}, dense2{nullptr}, dense3{nullptr},
                      dense4{nullptr}, dense5{nullptr}, dense6{nullptr};
};

// Utility helpers
torch::Device pick_device(const std::string& gpu_flag, bool force_cpu) {
    if (force_cpu) return torch::kCPU;

    if (gpu_flag == "auto") {
        if (torch::cuda::is_available()) return torch::kCUDA;
#ifdef __APPLE__
        if (torch::mps::is_available()) return torch::kMPS;
#endif
        return torch::kCPU;
    }
    if (gpu_flag == "cuda" && torch::cuda::is_available()) return torch::kCUDA;
#ifdef __APPLE__
    if (gpu_flag == "mps"  && torch::mps::is_available()) return torch::kMPS;
#endif
    return torch::kCPU;
}

torch::TensorOptions opts(torch::Device device) {
    return torch::TensorOptions().dtype(torch::kFloat32).device(device);
}

struct Flags {
    int    batch;
    int    seq;
    int    runs;
    int    warmups;
    double lr;
    std::string loss;
    std::string opt;
    std::string gpu;
    bool   force_cpu;
    bool   no_save;
    std::string save_dir;
};

// Synthetic data
struct Synthetic {
    torch::Tensor z, x, target;
};

Synthetic make_batch(int B, int T, torch::Device dev) {
    Synthetic s;
    s.z      = torch::randn({B, T, 162}, opts(dev));
    s.x      = torch::randn({B,     484}, opts(dev));
    s.target = torch::randn({B,       1}, opts(dev));
    return s;
}

// Loss / optimizer factory
std::shared_ptr<torch::optim::Optimizer> make_optimizer(
    const std::string& name,
    torch::nn::Module& model,
    double lr
) {
    if (name == "adam") return std::make_shared<torch::optim::Adam>(model.parameters(), torch::optim::AdamOptions(lr));
    if (name == "sgd") return std::make_shared<torch::optim::SGD >(model.parameters(), torch::optim::SGDOptions (lr));
    if (name == "adamw") return std::make_shared<torch::optim::AdamW>(model.parameters(), torch::optim::AdamWOptions(lr));
    throw std::runtime_error("Unsupported optimizer: " + name);
}

torch::Tensor criterion(
    const std::string& name,
    const torch::Tensor& y_pred,
    const torch::Tensor& y_true
) {
    if (name == "mse") return torch::mse_loss(y_pred, y_true);
    if (name == "mae") return torch::l1_loss(y_pred, y_true);
    if (name == "huber") return torch::smooth_l1_loss(y_pred, y_true);
    throw std::runtime_error("Unsupported loss: " + name);
}

// One training iteration
double train_step(
    FarmerLstmModel& model,
    Synthetic& batch,
    const std::string& loss_name,
    torch::optim::Optimizer& opt,
    torch::Device device
) {
    opt.zero_grad();
    auto start = clock_type::now();

    auto y     = model.forward(batch.z, batch.x);
    auto loss  = criterion(loss_name, y, batch.target);
    loss.backward();
    opt.step();

    if (device.is_cuda()) torch::cuda::synchronize();
    auto end = clock_type::now();
    return ms(end - start).count(); // milliseconds
}

// Main
int main(int argc, char** argv) {
    argparse::ArgumentParser program("farmer_benchmark");

    program.add_argument("--batch-size")
       .default_value(32)
       .scan<'i', int>();

    program.add_argument("--seq-length")
        .default_value(10)
        .scan<'i', int>();

    program.add_argument("--runs")
        .default_value(5)
        .scan<'i', int>();

    program.add_argument("--warmup-runs")
        .default_value(5)
        .scan<'i', int>();

    program.add_argument("--learning-rate")
        .default_value(0.001)
        .scan<'g', double>();      // 'g' for generic/float

    program.add_argument("--loss-function")
           .default_value(std::string("mse"))
           .choices("mse","mae","huber");
    program.add_argument("--optimizer")
           .default_value(std::string("adam"))
           .choices("adam","sgd","adamw");

    program.add_argument("--gpu")
           .default_value(std::string("auto"))
           .choices("auto","cuda","mps","cpu");

    program.add_argument("--cpu").default_value(false).implicit_value(true);
    program.add_argument("--no-save").default_value(false).implicit_value(true);
    program.add_argument("--save-dir").default_value(std::string("/tmp"));

    try { program.parse_args(argc, argv); }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n\n" << program;
        return 1;
    }

    Flags f;
    f.batch     = program.get<int>("--batch-size");
    f.seq       = program.get<int>("--seq-length");
    f.runs      = program.get<int>("--runs");
    f.warmups   = program.get<int>("--warmup-runs");
    f.lr        = program.get<double>("--learning-rate");
    f.loss      = program.get<std::string>("--loss-function");
    f.opt       = program.get<std::string>("--optimizer");
    f.gpu       = program.get<std::string>("--gpu");
    f.force_cpu = program.get<bool>("--cpu");
    f.no_save   = program.get<bool>("--no-save");
    f.save_dir  = program.get<std::string>("--save-dir");

    // Device & model
    torch::Device device = pick_device(f.gpu, f.force_cpu);
    std::cout << "Using device: " << device << '\n';

    FarmerLstmModel model;
    model.to(device);
    auto opt = make_optimizer(f.opt, model, f.lr);

    // Warm‑up
    std::cout << "\nWarm-up (" << f.warmups << " runs)\n";
    for (int i=0;i<f.warmups;++i) {
        auto batch = make_batch(f.batch, f.seq, device);
        double t   = train_step(model, batch, f.loss, *opt, device);
        std::cout << "  " << i+1 << "/" << f.warmups << ": " << std::fixed << std::setprecision(2) << t << " ms\n";
    }

    // Benchmark
    std::vector<double> times_ms, losses;
    std::cout << "\nBenchmark (" << f.runs << " runs)\n";
    for (int run = 0; run < f.runs; ++run) {
        auto batch = make_batch(f.batch, f.seq, device);

        double t = train_step(model, batch, f.loss, *opt, device);
        times_ms.push_back(t);

        model.eval();
        auto l = criterion(f.loss, model.forward(batch.z,batch.x), batch.target).item<double>();
        model.train();
        losses.push_back(l);

        std::cout << "  Run " << run+1 << "/" << f.runs
                  << ": " << std::fixed << std::setprecision(2) << t << " ms, "
                  << "Loss: " << std::setprecision(6) << l << '\n';
    }

    // Results
    double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    double throughput = f.batch / (avg_ms/1000.0);

    std::cout << "\nAverage time: " << std::fixed << std::setprecision(2) << avg_ms << " ms\n";
    std::cout << "Throughput  : " << std::setprecision(2) << throughput << " samples/sec\n";

    size_t total_params     = 0;
    size_t trainable_params = 0;
    for (const auto& p : model.parameters()) {
        total_params += p.numel();
        if (p.requires_grad()) trainable_params += p.numel();
    }
    std::cout << "Parameters  : " << total_params << " (trainable " << trainable_params << ")\n";

    // Save model & results
    if (!f.no_save) {
        std::filesystem::create_directories(f.save_dir);
        auto timestamp = std::time(nullptr);
        std::string model_path = f.save_dir + "/farmer_lstm_" + std::to_string(timestamp) + ".pt";

        torch::serialize::OutputArchive archive;
        model.save(archive);          // dumps parameters and buffers
        archive.save_to(model_path);  // write to disk

        std::cout << "Model saved to " << model_path << '\n';
    }
    
    return 0;
}
