#pragma once
#include <vector>
#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include "Game.h"
#include "mcts.h"
#include "concurrentqueue.h"
#include "readerwriterqueue.h"
// #include "torch/torch.h"
#include "user_interface.h"
#include "fsm.h"
#include "generated/config.h"


#ifdef NEURAL_ENABLED

// ── Residual Block ────────────────────────────────────────────────────────────

struct NeuralConfig {
    int num_filters      = 128;
    int num_res_blocks   = 10;
    int policy_filters   = 4;    // filters in policy head before FC
    int value_hidden     = 128;  // FC size in value head
    bool use_se          = false; // squeeze-excitation blocks
};


inline void save_config(const NeuralConfig& cfg, const std::string& path) {
    std::ofstream f(path + ".cfg");
    if (!f) throw std::runtime_error("Could not open config file for writing: " + path + ".cfg");
    f << cfg.num_filters    << "\n"
      << cfg.num_res_blocks << "\n"
      << cfg.policy_filters << "\n"
      << cfg.value_hidden   << "\n";
}

inline NeuralConfig load_config(const std::string& path) {
    std::ifstream f(path + ".cfg");
    if (!f) throw std::runtime_error("Could not open config file for reading: " + path + ".cfg");
    NeuralConfig cfg;
    f >> cfg.num_filters
      >> cfg.num_res_blocks
      >> cfg.policy_filters
      >> cfg.value_hidden;
    return cfg;
}

struct ResBlockImpl : torch::nn::Module {
    torch::nn::Conv2d      conv1, conv2;
    torch::nn::BatchNorm2d bn1,   bn2;

    ResBlockImpl(int filters)
        : conv1(torch::nn::Conv2dOptions(filters, filters, 3).padding(1).bias(false))
        , conv2(torch::nn::Conv2dOptions(filters, filters, 3).padding(1).bias(false))
        , bn1(filters)
        , bn2(filters)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("bn1",   bn1);
        register_module("bn2",   bn2);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        auto skip = x;
        x = torch::relu(bn1(conv1(x)));
        x = bn2(conv2(x));
        return torch::relu(x + skip);
    }
};
TORCH_MODULE(ResBlock);

template <GameTraits T>
struct NeuralMCTSCoreImpl : torch::nn::Module {

    // Trunk
    torch::nn::Conv2d      input_conv;
    torch::nn::BatchNorm2d input_bn;
    torch::nn::ModuleList  res_blocks;

    // Policy head
    torch::nn::Conv2d      policy_conv;
    torch::nn::BatchNorm2d policy_bn;
    torch::nn::Linear      policy_fc;

    // Value head
    torch::nn::Conv2d      value_conv;
    torch::nn::BatchNorm2d value_bn;
    torch::nn::Linear      value_fc1;
    torch::nn::Linear      value_fc2;

    static constexpr int H = T::neural_input_height;
    static constexpr int W = T::neural_input_width;

    NeuralMCTSCoreImpl(NeuralConfig cfg = {})
        // Trunk
        : input_conv(torch::nn::Conv2dOptions(T::neural_feature_channels, cfg.num_filters, 3)
                         .padding(1).bias(false))
        , input_bn(cfg.num_filters)
        // Policy head
        , policy_conv(torch::nn::Conv2dOptions(cfg.num_filters, cfg.policy_filters, 1)
                          .bias(false))
        , policy_bn(cfg.policy_filters)
        , policy_fc(cfg.policy_filters * H * W, T::policy_size)
        // Value head
        , value_conv(torch::nn::Conv2dOptions(cfg.num_filters, 1, 1).bias(false))
        , value_bn(1)
        , value_fc1(H * W, cfg.value_hidden)
        , value_fc2(cfg.value_hidden, 1)
    {
        register_module("input_conv", input_conv);
        register_module("input_bn",   input_bn);

        for (int i = 0; i < cfg. ; ++i)
            res_blocks->push_back(ResBlock(cfg.num_filters));
        register_module("res_blocks", res_blocks);

        register_module("policy_conv", policy_conv);
        register_module("policy_bn",   policy_bn);
        register_module("policy_fc",   policy_fc);

        register_module("value_conv",  value_conv);
        register_module("value_bn",    value_bn);
        register_module("value_fc1",   value_fc1);
        register_module("value_fc2",   value_fc2);
    }

    // Returns {log_policy [B, policy_size], value [B, 1]}
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // ── Trunk ──────────────────────────────────────────────────────────────
        x = torch::relu(input_bn(input_conv(x)));

        for (auto& block : *res_blocks)
            x = block->as<ResBlock>()->forward(x);

        // ── Policy head ────────────────────────────────────────────────────────
        auto p = torch::relu(policy_bn(policy_conv(x)));
        p = p.flatten(1);
        p = torch::log_softmax(policy_fc(p), /*dim=*/1);

        // ── Value head ─────────────────────────────────────────────────────────
        auto v = torch::relu(value_bn(value_conv(x)));
        v = v.flatten(1);
        v = torch::relu(value_fc1(v));
        v = torch::tanh(value_fc2(v));

        return {p, v};
    }
};

template <GameTraits T>
TORCH_MODULE_IMPL(NeuralMCTSCore, NeuralMCTSCoreImpl<T>);

#endif


enum class DispatcherState {
    Running,    // Normal MCTS inference loop
    Training,   // Consuming game history, training NN
    Paused,     // External pause
    Dead        // Shutdown
};


//=============================================================
// Class: InferenceDispatcher
// * Single thread — batches inference requests and dispatches
//   to GPU, returns results to per-agent completed queues
//=============================================================
template <GameTraits T>
class InferenceDispatcher {
    // Reference to thread-safe queue used to collect state samples
    moodycamel::ConcurrentQueue<InferenceRequest<T>>& _request_queue;

    // Array of 'InferenceRequest' to de-load from queue locally
    std::array<InferenceRequest<T>, MAX_BATCH_SIZE> _batch;    

    // // Device in which NN math is performed
#ifdef NEURAL_ENABLED
    torch::Device _device;
    torch::Tensor _staging;
#endif
    ThreadContext* _ctx;

    MCTSTree<T>* _tree;

    std::barrier<BarrierCompletion>* _barriers;
    std::atomic<bool>* _training;
    bool* _shutdown;     // thread lifetime
    bool* _paused;       // pause/resume


    // Returns false if the queue was empty an nothing needed
    bool process_batch();

public:
    InferenceDispatcher(ThreadContext* ctx,
                        MCTSTree<T>* tree,
#ifdef NEURAL_ENABLED
                        torch::Device device,
                        std::optional<std::string> model_path,
                        NeuralConfig cfg = {},
#endif
                        std::barrier<BarrierCompletion>* barriers,
                        std::atomic<bool>* training,
                        bool* paused,
                        bool* shutdown,
                        moodycamel::ConcurrentQueue<InferenceRequest<T>>& request_queue
    );
        
    void run();
};

