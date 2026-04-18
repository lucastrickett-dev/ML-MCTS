#pragma once
#include <vector>
#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include "Game.h"
#include "mcts.h"
#include "concurrentqueue.h"
#include "readerwriterqueue.h"
#include "thread_interface.h"
// #include "torch/torch.h"
#include "user_interface.h"


#define MAX_BATCH_SIZE 64

#ifdef NEURAL_ENABLED
template <GameTraits T>
struct LabelledData {
    std::array<float, T::neural_feature_channels *
                      T::neural_input_height     *
                      T::neural_input_width>  state;

    std::array<float, T::policy_size>         policy;
    float                                     value = 0.0f;
};
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

    // Maximum time to wait before dispatching a non-empty batch
    std::chrono::milliseconds _max_wait_time_ms;

    // // Device in which NN math is performed
#ifdef NEURAL_ENABLED
    torch::Device _device;
    torch::Tensor _staging;
#endif

    ThreadContext* ctx;

    std::atomic<bool> _alive = true;     // thread lifetime
    std::atomic<bool> _paused = false;   // pause/resume
    std::barrier<>* _barriers;

    // Returns false if the queue was empty an nothing needed
    bool process_batch();

public:
    InferenceDispatcher(ThreadContext* ctx,
                        MCTSTree<T>* tree,
#ifdef NEURAL_ENABLED
                        torch::Device device,
                        const std::string& model_path,
#endif
                        std::barrier<>* barriers,
                        std::chrono::milliseconds max_wait_time_ms,
                        moodycamel::ConcurrentQueue<InferenceRequest<T>>& request_queue
    );
        
    void run();
};


