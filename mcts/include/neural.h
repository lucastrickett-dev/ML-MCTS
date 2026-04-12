// #pragma once
// #include <vector>
// #include <atomic>
// #include <cstdint>
// #include <shared_mutex>
// #include "Game.h"
// #include "concurrentqueue.h"
// #include "readerwriterqueue.h"
// #include "thread_interface.h"
// #include "torch/torch.h"


// template <GameTraits T>
// struct LabelledData {
//     std::array<float, T::neural_feature_channels *
//                       T::neural_input_height     *
//                       T::neural_input_width>  state;

//     std::array<float, T::policy_size>         policy;
//     float                                     value = 0.0f;
// };

// //=============================================================
// // Class: InferenceDispatcher
// // * Single thread — batches inference requests and dispatches
// //   to GPU, returns results to per-agent completed queues
// //=============================================================
// template <GameTraits T>
// class InferenceDispatcher {
//     // Reference to thread-safe queue used to collect state samples
//     moodycamel::ConcurrentQueue<InferenceRequest<T>>& _request_queue;

//     // Array of 'InferenceRequest' to de-load from queue locally
//     std::array<InferenceRequest<T>, MAX_BATCH_SIZE> _batch;

//     // atomic bool to signify process is active
//     std::atomic<bool> _running = false;

//     // Maximum batch size before forcing a dispatch
//     size_t _max_batch_size;

//     // Maximum time to wait before dispatching a non-empty batch
//     std::chrono::milliseconds _max_wait_time_ms;

//     // Device in which NN math is performed
//     torch::Device _device;
//     torch::Tensor _staging;



//     void run();

//     void process_batch();

// public:
//     InferenceDispatcher(size_t batch_size,
//                         torch::Device device,
//                         const std::string& model_path,
//                         std::chrono::milliseconds idle_timeout,
//                         moodycamel::ConcurrentQueue<InferenceRequest<T>>& request_queue);

//     bool start();
//     bool stop();
// };



