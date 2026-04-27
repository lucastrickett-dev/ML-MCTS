#pragma once
#include <vector>
#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include "Game.h"
#include "mcts.h"
#include "concurrentqueue.h"
#include "readerwriterqueue.h"
#include "user_interface.h"
#include "fsm.h"
#include "config.h"


#ifdef NEURAL_ENABLED


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Config
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct NeuralConfig {
    int  num_filters    = 128;
    int  num_res_blocks = 10;
    int  policy_filters = 4;    // conv filters in policy head before FC
    int  value_hidden   = 128;  // hidden size in value head FC

    // Training hyper-parameters — persisted alongside weights so a loaded
    // checkpoint resumes with the same schedule it was saved with.
    float lr           = 0.01f;
    float lr_decay     = 0.95f; // multiplicative decay applied each game
    int   train_epochs = 5;     // passes over the game's data per training call
};

inline void save_config(const NeuralConfig& cfg, const std::string& path) {
    std::ofstream f(path + ".cfg");
    if (!f) throw std::runtime_error("Could not open config file for writing: " + path + ".cfg");
    f << cfg.num_filters    << "\n"
      << cfg.num_res_blocks << "\n"
      << cfg.policy_filters << "\n"
      << cfg.value_hidden   << "\n"
      << cfg.lr             << "\n"
      << cfg.lr_decay       << "\n"
      << cfg.train_epochs   << "\n";
}

inline NeuralConfig load_config(const std::string& path) {
    std::ifstream f(path + ".cfg");
    if (!f) throw std::runtime_error("Could not open config file for reading: " + path + ".cfg");
    NeuralConfig cfg;
    f >> cfg.num_filters
      >> cfg.num_res_blocks
      >> cfg.policy_filters
      >> cfg.value_hidden
      >> cfg.lr
      >> cfg.lr_decay
      >> cfg.train_epochs;
    return cfg;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Building block: Residual Block
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Neural Network
//
// Architecture (AlphaZero-style):
//   Input  [B, C, H, W]  — board feature planes
//   Trunk               — Conv + BN + ReLU, then N residual blocks
//   Policy head         — Conv + BN + ReLU + FC → policy_size logits
//   Value head          — Conv + BN + ReLU + FC + ReLU + FC + tanh → scalar
//
// Two outputs:
//
//   forward()  [INFERENCE]
//     Returns softmax probabilities over actions.
//     Used directly as PUCT priors — must be proper probabilities in [0,1].
//
//   forward_train()  [TRAINING]
//     Returns log_softmax over the same logits.
//     Used in the policy loss: -sum(target * log_pred).
//     This is mathematically identical to forward() + log(), but log_softmax
//     is numerically stable (avoids underflow when probabilities are tiny).
//
// The weights being trained are shared — both paths read the same logits.
// The network learns to make softmax(logits) match the MCTS visit distribution.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template <GameTraits T>
struct NeuralMCTSCoreImpl : torch::nn::Module {

    // ── Trunk ────────────────────────────────────────────────────────────────
    torch::nn::Conv2d      input_conv;
    torch::nn::BatchNorm2d input_bn;
    torch::nn::ModuleList  res_blocks;

    // ── Policy head ──────────────────────────────────────────────────────────
    torch::nn::Conv2d      policy_conv;
    torch::nn::BatchNorm2d policy_bn;
    torch::nn::Linear      policy_fc;      // → raw logits [B, policy_size]

    // ── Value head ───────────────────────────────────────────────────────────
    torch::nn::Conv2d      value_conv;
    torch::nn::BatchNorm2d value_bn;
    torch::nn::Linear      value_fc1;
    torch::nn::Linear      value_fc2;     // → tanh scalar [B, 1]  in [-1, 1]

    static constexpr int H = T::neural_input_height;
    static constexpr int W = T::neural_input_width;

    NeuralMCTSCoreImpl(NeuralConfig cfg = {})
        : input_conv(torch::nn::Conv2dOptions(T::neural_feature_channels, cfg.num_filters, 3)
                         .padding(1).bias(false))
        , input_bn(cfg.num_filters)
        , policy_conv(torch::nn::Conv2dOptions(cfg.num_filters, cfg.policy_filters, 1).bias(false))
        , policy_bn(cfg.policy_filters)
        , policy_fc(cfg.policy_filters * H * W, T::policy_size)
        , value_conv(torch::nn::Conv2dOptions(cfg.num_filters, 1, 1).bias(false))
        , value_bn(1)
        , value_fc1(H * W, cfg.value_hidden)
        , value_fc2(cfg.value_hidden, 1)
    {
        register_module("input_conv",  input_conv);
        register_module("input_bn",    input_bn);

        for (int i = 0; i < cfg.num_res_blocks; ++i)
            res_blocks->push_back(ResBlock(cfg.num_filters));
        register_module("res_blocks",  res_blocks);

        register_module("policy_conv", policy_conv);
        register_module("policy_bn",   policy_bn);
        register_module("policy_fc",   policy_fc);

        register_module("value_conv",  value_conv);
        register_module("value_bn",    value_bn);
        register_module("value_fc1",   value_fc1);
        register_module("value_fc2",   value_fc2);
    }

private:
    // ── Shared trunk — returns raw logits and tanh value ─────────────────────
    // Both forward paths below call this. Neither the policy logits nor the
    // value have any final activation applied here.
    std::pair<torch::Tensor, torch::Tensor> trunk(torch::Tensor x) {
        // Trunk
        x = torch::relu(input_bn(input_conv(x)));
        for (auto& block : *res_blocks)
            x = block->as<ResBlock>()->forward(x);

        // Policy head → logits (no activation yet)
        auto p = torch::relu(policy_bn(policy_conv(x)));
        p = p.flatten(1);
        p = policy_fc(p);                              // [B, policy_size] raw logits

        // Value head → tanh scalar
        auto v = torch::relu(value_bn(value_conv(x)));
        v = v.flatten(1);
        v = torch::relu(value_fc1(v));
        v = torch::tanh(value_fc2(v));                 // [B, 1]  in [-1, 1]

        return { p, v };
    }

public:
    // ── INFERENCE ────────────────────────────────────────────────────────────
    // Input:  board tensor [B, C, H, W]
    // Output: { policy [B, policy_size], value [B, 1] }
    //
    // policy — softmax probabilities, used directly as PUCT priors P(s,a).
    //          Each entry is in [0,1] and the row sums to 1.
    // value  — tanh scalar in [-1, 1], estimate of game outcome from the
    //          current player's perspective.
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto [logits, value] = trunk(x);
        auto policy = torch::softmax(logits, /*dim=*/1);   // [B, policy_size]
        return { policy, value };
    }

    // ── TRAINING ─────────────────────────────────────────────────────────────
    // Input:  board tensor [B, C, H, W]
    // Output: { log_policy [B, policy_size], value [B, 1] }
    //
    // log_policy — log_softmax of the same logits as forward().
    //              Used in: policy_loss = -sum(target_visits * log_policy, dim=1).mean()
    //              Numerically stable vs softmax() followed by log().
    // value      — same tanh scalar as forward(), used in: mse_loss(value, target_outcome)
    //
    // Target 'target_visits' is the normalised MCTS visit count distribution
    // stored in LabelledData::policy. The loss pushes softmax(logits) toward
    // that distribution, so after training forward() will predict what MCTS
    // would choose without needing to search.
    std::pair<torch::Tensor, torch::Tensor> forward_train(torch::Tensor x) {
        auto [logits, value] = trunk(x);
        auto log_policy = torch::log_softmax(logits, /*dim=*/1);   // [B, policy_size]
        return { log_policy, value };
    }
};

template <GameTraits T>
TORCH_MODULE_IMPL(NeuralMCTSCore, NeuralMCTSCoreImpl<T>);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Training
//
// Called once per game, after self-play completes.
//
// Data:
//   Each LabelledData sample contains:
//     state  — board feature planes at that move
//     policy — normalised MCTS visit counts at that move (sums to 1)
//     value  — game outcome from that player's perspective (+1 win, -1 loss)
//
// Loss (AlphaZero):
//   policy_loss = -sum(target_policy * log_softmax(logits), dim=1).mean()
//   value_loss  =  mse_loss(pred_value, target_value)
//   total       =  policy_loss + value_loss
//
// Notes:
//   - Multiple epochs: a single game gives few hundred positions at most;
//     multiple passes let the model converge on small data.
//   - Gradient clipping: prevents exploding gradients on small batches.
//   - LR decay: caller (InferenceDispatcher::run) multiplies cfg.lr by
//     cfg.lr_decay after each call and rebuilds the optimiser next game.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template <GameTraits T>
void train_epoch(NeuralMCTSCoreImpl<T>&              model,
                 torch::optim::SGD&                  opt,
                 const std::vector<LabelledData<T>>& data,
                 int                                 batch_size,
                 torch::Device                       device,
                 int                                 num_epochs)
{
    const int N = static_cast<int>(data.size());
    if (N == 0) return;

    constexpr int state_size = T::neural_feature_channels
                             * T::neural_input_height
                             * T::neural_input_width;

    // ── Build full dataset tensors on CPU once ────────────────────────────────
    auto states_t   = torch::empty({N,
                                    (int64_t)T::neural_feature_channels,
                                    (int64_t)T::neural_input_height,
                                    (int64_t)T::neural_input_width}, torch::kFloat32);
    auto policies_t = torch::empty({N, (int64_t)T::policy_size}, torch::kFloat32);
    auto values_t   = torch::empty({N, 1}, torch::kFloat32);

    float* sp = states_t.data_ptr<float>();
    float* pp = policies_t.data_ptr<float>();
    float* vp = values_t.data_ptr<float>();

    for (int i = 0; i < N; i++) {
        std::copy(data[i].state.begin(),  data[i].state.end(),  sp + i * state_size);
        std::copy(data[i].policy.begin(), data[i].policy.end(), pp + i * (int)T::policy_size);
        vp[i] = data[i].value;
    }

    // ── Epoch loop ────────────────────────────────────────────────────────────
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        int   steps      = 0;

        // ── Mini-batch loop ───────────────────────────────────────────────────
        for (int start = 0; start < N; start += batch_size) {
            int end = std::min(start + batch_size, N);

            auto idx_t = torch::from_blob(
                indices.data() + start,
                { end - start },
                torch::kInt32
            ).to(torch::kLong);

            auto s = states_t  .index_select(0, idx_t).to(device);  // [B, C, H, W]
            auto p = policies_t.index_select(0, idx_t).to(device);  // [B, policy_size]  target visit dist
            auto v = values_t  .index_select(0, idx_t).to(device);  // [B, 1]            target outcome

            opt.zero_grad();

            // ── Forward (training path) ───────────────────────────────────────
            // log_policy: log_softmax of logits      [B, policy_size]
            // pred_value: tanh scalar                [B, 1]
            auto [log_policy, pred_value] = model.forward_train(s);

            // ── Policy loss ───────────────────────────────────────────────────
            // Target p is the normalised MCTS visit distribution (sums to 1).
            // Loss = -sum(p * log_policy, dim=1).mean()
            // This pushes softmax(logits) to match p — equivalent to KL divergence
            // up to a constant, and identical to what AlphaZero uses.
            auto policy_loss = -(p * log_policy).sum(/*dim=*/1).mean();

            // ── Value loss ────────────────────────────────────────────────────
            // Target v is the game outcome {-1, 0, +1} from that player's view.
            // MSE against tanh output is the standard AlphaZero formulation.
            auto value_loss = torch::mse_loss(pred_value, v);

            // ── Combined loss + backward ──────────────────────────────────────
            auto loss = policy_loss + value_loss;
            loss.backward();

            // Gradient clipping — important for small batches
            torch::nn::utils::clip_grad_norm_(model.parameters(), /*max_norm=*/1.0);

            opt.step();

            epoch_loss += loss.item<float>();
            steps++;
        }

        std::cout << "[train] epoch " << (epoch + 1) << "/" << num_epochs
                  << "  loss=" << (steps > 0 ? epoch_loss / steps : 0.0f) << "\n";
    }
}

#endif  // NEURAL_ENABLED


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// InferenceDispatcher
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

enum class DispatcherState {
    Running,    // Normal MCTS inference loop
    Training,   // Consuming game history, training NN
    Paused,     // External pause
    Dead        // Shutdown
};

template <GameTraits T>
class InferenceDispatcher {
    moodycamel::ConcurrentQueue<InferenceRequest<T>>& _request_queue;
    std::array<InferenceRequest<T>, MAX_BATCH_SIZE>   _batch;

#ifdef NEURAL_ENABLED
    torch::Device                          _device;
    std::optional<std::string>             _model_path;
    NeuralConfig                           _cfg;
    torch::Tensor                          _staging;
    float*                                 _staging_ptr = nullptr;
    size_t                                 _stride      = 0;
    std::shared_ptr<NeuralMCTSCoreImpl<T>> model;
#endif

    ThreadContext*                    _ctx;
    MCTSTree<T>*                      _tree;
    std::barrier<BarrierCompletion>*  _barriers;
    std::atomic<bool>*                _training;
    bool*                             _shutdown;
    bool*                             _paused;

    bool process_batch();

public:
    InferenceDispatcher(ThreadContext* ctx,
                        MCTSTree<T>*  tree,
#ifdef NEURAL_ENABLED
                        torch::Device              device,
                        std::optional<std::string> model_path,
                        NeuralConfig               cfg,
#endif
                        std::barrier<BarrierCompletion>*                  barriers,
                        std::atomic<bool>*                                training,
                        bool*                                             paused,
                        bool*                                             shutdown,
                        moodycamel::ConcurrentQueue<InferenceRequest<T>>& request_queue);

    void run();
};