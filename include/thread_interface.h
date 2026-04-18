#pragma once
#include <vector>
#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include "Game.h"
#include "readerwriterqueue.h"


enum class BarrierPoint {
    RESET,
    ACTIVE,
    TRAIN,
    COUNT       // useful trick to get array size
};


template <GameTraits T> struct InferenceJob;
template <GameTraits T> struct InferenceRequest;

//=============================================================
// Struct: InferenceRequest
// * Sent to InferenceDispatcher — contains state for GPU, nodes
//   for tree ops, and where to send the result back
//=============================================================
template <GameTraits T>
struct InferenceRequest {
    typename T::State                               state;
    MCTSNode<T>*                                    leaf;
    MCTSNode<T>*                                    root;
    moodycamel::ReaderWriterQueue<InferenceJob<T>>* queue;
};

//=============================================================
// Struct: InferenceJob
// * Returned by InferenceDispatcher after GPU inference completes
// * Contains everything SimulationEvaluator needs
//=============================================================
template <GameTraits T>
struct InferenceJob {
    typename T::State                  state;
    MCTSNode<T>*                       leaf;
    MCTSNode<T>*                       root;
    float                              value;
    std::array<float, T::policy_size>  policy;
};