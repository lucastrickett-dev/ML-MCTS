// #include <iostream>
// #include <thread>
// #include <chrono>
// #include <forward_list>
// #include <random>
// #include "neural.h"


// template <GameTraits T>
// InferenceDispatcher<T>::InferenceDispatcher(size_t batch_size,
//                                             torch::Device device,
//                                             const std::string& model_path,
//                                             std::chrono::milliseconds idle_timeout,
//                                             moodycamel::ConcurrentQueue<InferenceRequest<T>>& request_queue)
//     : _max_batch_size(max_batch_size) 
//     , _max_wait_time_ms(max_wait_time_ms)
//     , _request_queue(request_queue)
// {
//     _staging = torch::empty(
//         { (int64_t)max_batch_size,
//           (int64_t)T::neural_feature_channels,
//           (int64_t)T::neural_input_height,
//           (int64_t)T::neural_input_width 
//         },
//         torch::TensorOptions()
//             .dtype(torch::kFloat32)
//             .pinned_memory(true)
//     );
// }


// // Process a batch of inference requests and route results back to agents
// template <GameTraits T>
// void InferenceDispatcher<T>::process_batch() {
//     // Collect requests from request queue in bulk
//     size_t count = _request_queue.try_dequeue_bulk(_batch.data(), _max_batch_size);
//     if (count == 0) return;

//     // process all samples into a 4D tensor of dimensions [B, C, H, W]
//     for (size_t i = 0; i < count; i++) {
//         T::get_data(_batch[i].state, _staging_ptr + i * _stride);
//     }

//     // Pass to GPU - Dont do yet

//     // Post jobs back to threads using thread pointers
//     for (size_t i = 0; i < count; i++) {
//         InferenceJob<T> job {
//             .state  = _batch[i].state,
//             .leaf   = _batch[i].leaf,
//             .root   = _batch[i].root,
//             .value  = 0.0f,                          // TODO: from model output
//             .policy = {}                             // TODO: from model output
//         };
//         _batch[i].queue->try_enqueue(job);
//     }
// }


// template <GameTraits T>
// void InferenceDispatcher<T>::run() {

// }


// template <GameTraits T>
// bool InferenceDispatcher<T>::start() {

// }

// template <GameTraits T>
// bool InferenceDispatcher<T>::stop() {

// }




// template <GameTraits T>
// struct AlphaZeroNetwork : torch::nn::Module {

//     //=========================================================
//     // Submodules
//     //=========================================================

// };