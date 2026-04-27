#include <iostream>
#include <thread>
#include <chrono>
#include <barrier>
#include <forward_list>
#include <random>
#include "neural.h"
#include "concurrentqueue.h"
#include "fsm.h"
#include "config.h"
#ifdef NEURAL_ENABLED
#include "torch.h"
#endif


template class InferenceDispatcher<Game>;

template <GameTraits T>
InferenceDispatcher<T>::InferenceDispatcher(ThreadContext* ctx,
                                            MCTSTree<T>*  tree,
#ifdef NEURAL_ENABLED
                                            torch::Device              device,
                                            std::optional<std::string> model_path,
                                            NeuralConfig               cfg,
#endif
                                            std::barrier<BarrierCompletion>* barriers,
                                            std::atomic<bool>*               training,
                                            bool*                            paused,
                                            bool*                            shutdown,
                                            moodycamel::ConcurrentQueue<InferenceRequest<T>>& request_queue)
    : _ctx(ctx)
    , _tree(tree)
#ifdef NEURAL_ENABLED
    , _device(device)
    , _model_path(model_path)
    , _cfg(cfg)
#endif
    , _barriers(barriers)
    , _training(training)
    , _paused(paused)
    , _shutdown(shutdown)
    , _request_queue(request_queue)
{
#ifdef NEURAL_ENABLED
    // Create pinned staging tensor for fast H2D transfer
    _staging = torch::empty(
        { (int64_t)MAX_BATCH_SIZE,
          (int64_t)T::neural_feature_channels,
          (int64_t)T::neural_input_height,
          (int64_t)T::neural_input_width
        },
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .pinned_memory(true)
    );
    _staging_ptr = _staging.data_ptr<float>();
    _stride = T::neural_feature_channels * T::neural_input_height * T::neural_input_width;

    // Load model onto device
    if (model_path.has_value()) {
        _cfg  = load_config(*model_path);
        model = std::make_shared<NeuralMCTSCoreImpl<T>>(_cfg);
        torch::load(model, *model_path + ".pt");
    } else {
        model = std::make_shared<NeuralMCTSCoreImpl<T>>(_cfg);
    }
    model->to(device);
    model->eval();

    // Verify model I/O shapes with dummy forward pass
    auto dummy = torch::zeros(
        { (int64_t)MAX_BATCH_SIZE,
          (int64_t)T::neural_feature_channels,
          (int64_t)T::neural_input_height,
          (int64_t)T::neural_input_width
        }
    ).to(device);

    auto [policy_head, value_head] = model->forward(dummy);

    TORCH_CHECK(value_head.sizes()  == torch::IntArrayRef({(int64_t)MAX_BATCH_SIZE, 1}),
                "Value head shape mismatch");
    TORCH_CHECK(policy_head.sizes() == torch::IntArrayRef({(int64_t)MAX_BATCH_SIZE, (int64_t)T::policy_size}),
                "Policy head shape mismatch");
#endif
}


template <GameTraits T>
bool InferenceDispatcher<T>::process_batch() {
    size_t count = _request_queue.try_dequeue_bulk(_batch.data(), MAX_BATCH_SIZE);
    if (count == 0) return false;

#ifdef NEURAL_ENABLED
    // Pack states into pinned staging tensor [B, C, H, W]
    for (size_t i = 0; i < count; i++) {
        T::get_data(_batch[i].state, _staging_ptr + i * _stride);
    }

    // H2D transfer and inference
    torch::NoGradGuard no_grad;
    auto input = _staging.slice(0, 0, count).to(_device, /*non_blocking=*/true);
    auto [policy_head, value_head] = model->forward(input);

    // D2H — pull results back to CPU for routing
    auto values   = value_head.cpu();
    auto policies = policy_head.cpu();

    auto value_acc  = values.accessor<float, 2>();    // [B, 1]
    auto policy_acc = policies.accessor<float, 2>();  // [B, policy_size]

    // Route results back to the requesting agent's SPSC queue
    for (size_t i = 0; i < count; i++) {
        InferenceJob<T> job {
            .state = _batch[i].state,
            .leaf  = _batch[i].leaf,
            .root  = _batch[i].root,
            .value = value_acc[i][0],
        };

        for (size_t a = 0; a < T::policy_size; a++) {
            job.policy[a] = policy_acc[i][a];
        }

        _batch[i].queue->try_enqueue(job);
    }
#else
    std::array<typename T::Action, T::max_actions> action_buf;

    for (size_t i = 0; i < count; i++) {
        InferenceJob<T> job {
            .state = _batch[i].state,
            .leaf  = _batch[i].leaf,
            .root  = _batch[i].root
        };

        job.value = T::rollout(_batch[i].state);

        size_t move_count = T::get_actions(_batch[i].state, action_buf);

        job.policy.fill(0.0f);
        for (size_t j = 0; j < move_count; j++) {
            job.policy[action_buf[j]] = 1.0f;
        }

        bool ok = _batch[i].queue->try_enqueue(job);
        assert(ok && "completed_queue full — raise MAX_PENDING_REQUESTS");
    }
#endif
    _ctx->localSims += count;
    return true;
}


template <GameTraits T>
void InferenceDispatcher<T>::run() {
    while (!(*_shutdown))
    {
        // ── RESET ─────────────────────────────────────────────────────────────
        _ctx->status = ThreadStatus::Reset;
        InferenceRequest<T> request;
        _training->store(false, std::memory_order_release);
        while (_request_queue.try_dequeue(request)) {}

        _ctx->localSims   = 0;
        _ctx->pendingJobs = 0;

        _tree->reset();
        _barriers[static_cast<int>(BarrierPoint::RESET)].arrive_and_wait();

        // ── ACTIVE ────────────────────────────────────────────────────────────
        _ctx->status = ThreadStatus::Inference;
        (void)_barriers[static_cast<int>(BarrierPoint::ACTIVE)].arrive();

        auto last_dispatch = std::chrono::steady_clock::now();

        while (!_training->load(std::memory_order_acquire))
        {
            size_t queued     = _request_queue.size_approx();
            _ctx->pendingJobs = queued;

            auto now     = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_dispatch);

            if (queued >= MAX_BATCH_SIZE || (queued > 0 && elapsed.count() >= INFERENCE_MAX_DELAY))
            {
                process_batch();
                last_dispatch = now;
            } else {
                std::this_thread::yield();
            }
        }

        while (_request_queue.try_dequeue(request)) {}

        // ── TRAIN ─────────────────────────────────────────────────────────────
#ifdef NEURAL_ENABLED
        _ctx->status = ThreadStatus::Training;

        auto training_data = _tree->get_game_history();

        if (!training_data.empty()) {
            model->train();

            // Use cfg.lr, which decays across games (see below)
            torch::optim::SGD opt(
                model->parameters(),
                torch::optim::SGDOptions(_cfg.lr).momentum(0.9).weight_decay(1e-4)
            );

            train_epoch<T>(*model, opt, training_data, /*batch_size=*/256, _device, _cfg.train_epochs);

            model->eval();

            // Decay LR for next game so the model doesn't overfit as it matures
            _cfg.lr *= _cfg.lr_decay;

            const std::string save_path = _model_path.value_or("neural/neural");
            torch::save(model, save_path + ".pt");
            save_config(_cfg, save_path);   // persists the decayed LR for next run
        }

        _ctx->status = ThreadStatus::Stopped;
#endif
        _training->store(false, std::memory_order_release);
        _barriers[static_cast<int>(BarrierPoint::TRAIN)].arrive_and_wait();
    }
}