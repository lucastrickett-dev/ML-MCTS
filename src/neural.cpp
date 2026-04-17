#include <iostream>
#include <thread>
#include <chrono>
#include <forward_list>
#include <random>
#include "neural.h"
#include "concurrentqueue.h"
#ifdef NEURAL_ENABLED
#include "torch.h"
#endif


template <GameTraits T>
InferenceDispatcher<T>::InferenceDispatcher(ThreadContext* ctx,
                                            size_t max_batch_size,
#ifdef NEURAL_ENABLED
                                            torch::Device device,
                                            const std::string& model_path,
#endif
                                            std::chrono::milliseconds idle_timeout,
                                            moodycamel::ConcurrentQueue<InferenceRequest<T>>& request_queue)
    : _ctx(ctx)
    , _max_batch_size(max_batch_size) 
    , _max_wait_time_ms(idle_timeout)
#ifdef NEURAL_ENABLED
    , _device(device)
#endif
    , _request_queue(request_queue)
    , _batch(max_batch_size)
{
#ifdef NEURAL_ENABLED
    // Create pinned staging tensor for fast H2D transfer
    _staging = torch::empty(
        { (int64_t)max_batch_size,
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
    model = std::make_shared<NeuralNet>();
    torch::load(model, model_path);
    model->to(device);
    model->eval();

    // Verify model I/O shapes with dummy forward pass
    auto dummy = torch::zeros(
        { (int64_t)max_batch_size,
          (int64_t)T::neural_feature_channels,
          (int64_t)T::neural_input_height,
          (int64_t)T::neural_input_width 
        }
    ).to(device);
    
    auto [value_head, policy_head] = model->forward(dummy);

    TORCH_CHECK(value_head.sizes()  == torch::IntArrayRef({(int64_t)max_batch_size, 1}),
                "Value head shape mismatch");
    TORCH_CHECK(policy_head.sizes() == torch::IntArrayRef({(int64_t)max_batch_size, (int64_t)T::policy_size}),
                "Policy head shape mismatch");
#endif
}


template <GameTraits T>
void InferenceDispatcher<T>::process_batch() {
    size_t count = _request_queue.try_dequeue_bulk(_batch.data(), _max_batch_size);
    if (count == 0) return;

#ifdef NEURAL_ENABLED
    // Pack states into pinned staging tensor [B, C, H, W]
    for (size_t i = 0; i < count; i++) {
        T::get_data(_batch[i].state, _staging_ptr + i * _stride);
    }

    // H2D transfer and inference
    torch::NoGradGuard no_grad;
    auto input = _staging.slice(0, 0, count).to(_device, /*non_blocking=*/true);
    auto [value_head, policy_head] = model->forward(input);

    // D2H — pull results back to CPU for routing
    auto values   = value_head.cpu();
    auto policies = policy_head.cpu();

    auto value_acc  = values.accessor<float, 2>();    // [B, 1]
    auto policy_acc = policies.accessor<float, 2>();  // [B, policy_size]

    // Route results back to the requesting agent's SPSC queue
    for (size_t i = 0; i < count; i++) {
        InferenceJob<T> job {
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
        InferenceJob<T> job { .leaf = _batch[i].leaf, .root = _batch[i].root };

        // Random rollout for value from leaf's perspective
        job.value = random_rollout(_batch[i].state);

        // Uniform policy over legal moves only
        size_t legal_count = T::get_actions(_batch[i].state, action_buf);  // was 'state', wrong var

        job.policy.fill(0.0f);
        for (size_t j = 0; j < legal_count; j++)                        // was 'i', shadows outer i
            job.policy[action_buf[j]] = 1.0f;                           // was 'action', undefined

        _batch[i].queue->try_enqueue(job);
    }
#endif
}


#ifndef NEURAL_ENABLED
template <GameTraits T>
float InferenceDispatcher<T>::random_rollout(typename T::State state) {
    std::array<typename T::Action, T::max_actions> action_buf;
    int depth = 0;
    while (!T::is_terminal(state) && depth++ < 200) {
        size_t count = T::get_actions(state, action_buf);
        if (count == 0) break;
        T::apply_action(state, action_buf[rand() % count]);
    }
    return T::get_winner(state);
}
#endif




// Function: Main logic loop for thread
// Entire purpose of thread is to take input requests and 
// output jobs for MCTS threads to complete
// * It should also do training since theres no reason for it not to I believe
template <GameTraits T>
void InferenceDispatcher<T>::run() {

}

template <GameTraits T>
bool InferenceDispatcher<T>::resume() {
    _paused.store(false);
}

template <GameTraits T>
bool InferenceDispatcher<T>::stop() {
    _paused.store(true);
}

template <GameTraits T>
bool InferenceDispatcher<T>::shutdown() {
    _alive.store(false);
}