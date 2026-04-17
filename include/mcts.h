#pragma once
#include <vector>
#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include "Game.h"
#include "concurrentqueue.h"
#include "readerwriterqueue.h"
#include "thread_interface.h"
#include "user_interface.h"

#define VIRTUAL_LOSS_VALUE  1.0f
#define MCTS_PUCT_CONSTANT  1.25f
#define MAX_THREAD_INFERENCES 64

// Forward declarations
template <GameTraits T> struct MCTSNode;


template <GameTraits T>
struct MCTSAgentSetup {
    ThreadContext*                                    ctx;
    MCTSTree<T>*                                      tree;
    moodycamel::ConcurrentQueue<InferenceRequest<T>>* request_queue;
};


//=============================================================
// Struct: MCTSNode
// * Open-loop node — stores action that led here, tree pointers,
//   and MCTS statistics. Does NOT store game state.
//=============================================================
template <GameTraits T>
struct MCTSNode {
    //----- Game data -----
    typename T::Action action = {};

    //----- Tree structure -----
    MCTSNode*              parent_ptr  = nullptr;
    std::span<MCTSNode<T>> children    = {};
    size_t                 child_count = 0;

    //----- Expansion flags -----
    std::atomic<bool> is_terminal  = false;  // no legal moves / game over
    std::atomic<bool> is_expanding = false;  // expansion in progress (gate)
    std::atomic<bool> is_expanded  = false;  // children exist and are visible

    bool is_leaf() const {
        return !is_expanded.load(std::memory_order_acquire)
            || is_terminal.load(std::memory_order_relaxed);
    }

    //----- MCTS statistics -----
    float                  prior      = 0.0f;
    std::atomic<float>     value_sum  = 0.0f;
    std::atomic<uint32_t>  visit_count = 0;

    // PUCT score for selection
    float PUCT(uint32_t parent_visits) const {
        uint32_t n = visit_count.load(std::memory_order_relaxed);
        float    w = value_sum.load(std::memory_order_relaxed);

        float q = (n == 0) ? 0.0f : (w / static_cast<float>(n));
        float u = MCTS_PUCT_CONSTANT * prior * std::sqrt(static_cast<float>(parent_visits)) / (1.0f + n);
        return q + u;
    }

    // Atomically update visit count and value sum
    void update(float value, uint32_t visit) {
        visit_count.fetch_add(visit, std::memory_order_relaxed);

        // atomic float add via CAS loop
        float current = value_sum.load(std::memory_order_relaxed);
        while (!value_sum.compare_exchange_weak(current, current + value,
               std::memory_order_relaxed, std::memory_order_relaxed));
    }
};


template <GameTraits T>
struct MCTSTree {
    //=========================================================
    // Training Data
    //=========================================================
    // Store the history of the game for neural training
    std::vector<LabelledData<T>> game_history;
    std::atomic<bool> _game_over = false;

    //=========================================================
    // Tree Data
    //=========================================================
    // Contiguous node pool for cache efficiency
    std::unique_ptr<MCTSNode<T>[]> _nodes;
    
    // Atomic bump allocator — 0 reserved for root
    std::atomic<size_t> unallocated_index = 1;
    
    // Root protection — shared for readers, exclusive for advance_root
    mutable std::shared_mutex root_mutex;
    
    // Current root node and its corresponding game state
    MCTSNode<T>*      root_node  = nullptr;
    typename T::State root_state = {};
    
    // Constructor — initialise root node and game state
    MCTSTree(size_t max_nodes) 
        : _nodes(std::make_unique<MCTSNode<T>[]>(max_nodes))
    {
        reset();
    }
    
    // Function: reset
    // * Clear tree and reinitialise root to a fresh game state
    void reset() {
        unallocated_index.store(1, std::memory_order_relaxed);
        _nodes[0]  = MCTSNode<T>{};
        root_node  = &_nodes[0];
        game_history.clear();
        T::initialise(root_state);  // set root state to valid game start
    }
    
    // Allocate children for a node in one contiguous batch (thread safe)
    // * For a given parent, create empty children and connect both together
    std::span<MCTSNode<T>> create_children(MCTSNode<T>* parent, size_t count) {
        size_t start = unallocated_index.fetch_add(count, std::memory_order_release);
        
        MCTSNode<T>* ptr = &_nodes[start];
        for (size_t i = 0; i < count; i++) {
            ptr[i]            = MCTSNode<T>{};
            ptr[i].parent_ptr = parent;
        }

        parent->children    = std::span<MCTSNode<T>>(ptr, count);
        parent->child_count = count;

        return parent->children;
    }

    // Function: record_sim
    // MCTS Agents perform this when completing a new simulation, if we have 
    // met the required number of simulations per a move, we advance the root
    // * not perfect, simulation staged just before advance would increment counter
    //   correpsonding to next root, but simulation count only needs to be rough estimation
    template <GameTraits T>
    void MCTSTree<T>::record_sim() {
        uint64_t current = _localSims.fetch_add(1, std::memory_order_relaxed) + 1;
        if (current >= _targetSims) {
            _localSims.store(0, std::memory_order_relaxed);
            advance_root();
        }
    }

    // Function: advance_root
    // * Advance root to the most visited child (real move played)
    // * Returns false if game is over or no children exist
    bool advance_root() {
        // Check if game has finished
        if (!root_node || T::is_terminal(root_state) || root_node->child_count == 0 ) {
            _game_over = true;
            return false;
        };

        std::unique_lock lock(root_mutex);

        MCTSNode<T>* best_child  = nullptr;
        size_t       best_visits = 0;
        float        total_visits = 0.0f;

        // Single pass — find best child and sum visits for policy normalisation
        for (MCTSNode<T>& child : root_node->children) {
            size_t visits = child.visit_count.load(std::memory_order_relaxed);
            total_visits += visits;
            if (visits > best_visits) {
                best_visits = visits;
                best_child  = &child;
            }
        }

        if (!best_child) return false;

        // Capture labelled data before advancing
        LabelledData<T> data_sample;
        T::get_data(root_state, data_sample.state.data());

        // Second pass — normalise visit counts into policy
        for (MCTSNode<T>& child : root_node->children)
            data_sample.policy[child.action] = child.visit_count.load(std::memory_order_relaxed) / total_visits;

        game_history.push_back(data_sample);

        T::apply_action(root_state, best_child->action);
        root_node = best_child;

        return true;
    }

    // Function: get_root_snapshot
    // * Returns a copy of root pointer and root state under shared lock
    std::pair<MCTSNode<T>*, typename T::State> get_root_snapshot() const {
        std::shared_lock lock(root_mutex);
        return { root_node, root_state };
    }

    // Go through entire vector, add correct value (based of root_state)
    // and return
    std::vector<LabelledData<T>> get_game_history() {
        float outcome = T::get_winner(root_state);

        // Fill value retrospectively from terminal backwards
        for (int i = (int)game_history.size() - 1; i >= 0; i--) {
            game_history[i].value = outcome;
            outcome = -outcome;  // flip for alternating players
        }

        return game_history;
    }
};


//=============================================================
// Class: MCTSAgent
// * One agent per thread — runs selection and evaluation loop
//=============================================================
template <GameTraits T>
class MCTSAgent {
    // Shared tree
    MCTSTree<T>* _tree;

    // Shared request queue (many agents write, dispatcher reads)
    moodycamel::ConcurrentQueue<InferenceRequest<T>>* _request_queue;

    // Private result queue (dispatcher writes, this agent reads — SPSC)
    moodycamel::ReaderWriterQueue<InferenceJob<T>>* _completed_queue;

    // Jobs currently in flight awaiting GPU response
    size_t _pending_inference_count = 0;

    // Shared simulation counter across all agents
    size_t _local_simulation_count = 0;

    // Stop signal
    std::atomic<bool> _alive = true;     // thread lifetime
    std::atomic<bool> _paused = false;   // pause/resume

    std::atomic<bool> _stop_flag = false;

    // Output context of thread form which to update UI stuff
    ThreadContext* ctx;

    //----- Private methods -----
    void backpropagate(float value, int visit, MCTSNode<T>* start, MCTSNode<T>* end);
    void SimulationSelector();
    void SimulationEvaluator(InferenceJob<T> inference_job);
    
    public:
    MCTSAgent(MCTSAgentSetup<T> ctx);
    
    bool reset(); // Reset agent for new game/tree
    
    void run();
    
    bool resume(); // Resume agent (ie continue working after stopping)
    
    bool stop();  // Hard stop the running state

    bool shutdown();  // Hard stop the running state
};