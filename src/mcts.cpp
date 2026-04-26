#include <iostream>
#include <forward_list>
#include "mcts.h"
#include <barrier>
#include "fsm.h"
#include "generated/config.h"

template class MCTSAgent<Game>;

template <GameTraits T>
MCTSAgent<T>::MCTSAgent(MCTSAgentSetup<T> ctx)
    : _ctx(ctx.ctx)
    , _barriers(ctx.barriers)
    , _training(ctx.training)
    , _paused(ctx.paused)
    , _shutdown(ctx.shutdown)
    , _tree(ctx.tree)
    , _request_queue(ctx.request_queue)
    , _completed_queue(MAX_PENDING_REQUESTS)
{

}

// Function: backpropagate
// * From a given leaf node, iteratively undo virtual-loss features
//   move up to parent, and repeat until applied to root
template <GameTraits T>
void MCTSAgent<T>::backpropagate(float value, int visit, MCTSNode<T>* start, MCTSNode<T>* end)
{
    MCTSNode<T>* current = start;

    while (current != end && current->parent_ptr != nullptr)
    {
        // Undoing virtual loss inherient into the backpropogation
        current->update(value + VIRTUAL_LOSS_VALUE, visit);
        current = current->parent_ptr;
        value = -value;  // flip for negamax
    }
    // Apply to end node (root)
    current->update(value + VIRTUAL_LOSS_VALUE, visit);
}

// Function: SimulationSelector
// * First half of simulation (selection -> inference request)
template <GameTraits T>
void MCTSAgent<T>::SimulationSelector()
{
    // Snapshot root — thread safe copy of root ptr and state
    std::pair<MCTSNode<T>*, typename T::State> snapshot = _tree->get_root_snapshot();
    typename T::State local_state = snapshot.second;
    MCTSNode<T>*      local_root  = snapshot.first;

    MCTSNode<T>* current = local_root;

    // Iteratively travel tree until leaf is reached
    while (!current->is_leaf())
    {
        // Apply virtual loss to discourage other threads from selecting same path
        current->update(-VIRTUAL_LOSS_VALUE, 1);
                    
        uint32_t parent_visits = current->visit_count.load(std::memory_order_relaxed);

        // Look through all children to find highest PUCT
        MCTSNode<T>* best_child = nullptr;
        float        best_score = -std::numeric_limits<float>::infinity();

        for (MCTSNode<T>& child : current->children) {
            float score = child.PUCT(parent_visits);
            if (score > best_score) {
                best_score = score;
                best_child = &child;
            }
        }

        if (!best_child) return;

        current = best_child;
        T::apply_action(local_state, current->action);
    }

    // Apply virtual loss to leaf
    current->update(-VIRTUAL_LOSS_VALUE, 1);

    // Turn on expanding flag to add node lock
    // - Needed for multithreading so different threads dont work on same node
    bool expected = false;
    if (!current->is_expanding.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        // Another thread is already expanding — undo virtual loss and bail
        backpropagate(0, -1, current, local_root);
        return;
    } 

    // Check if terminal flag has already been set (faster than calling T::is_terminal)
    if (current->is_terminal.load(std::memory_order_acquire)) {
        // Turn off expanding flag to remove node lock
        current->is_expanding.store(false, std::memory_order_release);

        // Backpropogate result and record simulation (a valid simulation)
        backpropagate(T::get_winner(local_state), 0, current, local_root);
        _tree->register_simulation();
        _ctx->localSims++;
        return;
    }

    // Check if state is terminal (but placed after since this is the first terminal
    // check, and faster to do quick check on flag first)
    if (T::is_terminal(local_state)) {
        // Update node flags
        current->is_terminal.store(true, std::memory_order_relaxed);
        current->is_expanded.store(true, std::memory_order_relaxed);

        // Turn off expanding flag to remove node lock
        current->is_expanding.store(false, std::memory_order_release);

        // Considered a simulation, so update
        backpropagate(T::get_winner(local_state), 0, current, local_root);
        _tree->register_simulation();
        _ctx->localSims++;
        return;
    }

    // If we get here, we will have to actually evaluation the node
    // Build and submit inference request
    InferenceRequest<T> request {
        .state  = local_state,
        .leaf   = current,
        .root   = local_root,
        .queue  = &_completed_queue
    };

    // Try to add to request queue — if fails, undo virtual loss and discard
    if (_request_queue->enqueue(request)) {
        _pending_inference_count++;
        _ctx->pendingJobs = _pending_inference_count;
    } else {
        // Exit node, update is_expanding so others can use
        current->is_expanding.store(false, std::memory_order_release);
        backpropagate(0, -1, current, local_root);
    }
}


// Function: SimulationEvaluator
// * Second half of simulation (inference result -> expand -> backpropagate)
// * The InferenceDispacher passes queswt thgrough NN and puts it back into 
//   the local queue using the '.queue pointer' in the Request (line 123)
template <GameTraits T>
void MCTSAgent<T>::SimulationEvaluator(InferenceJob<T> inference_job)
{
    // Unpack job
    float                              value  = inference_job.value;
    MCTSNode<T>*                       leaf   = inference_job.leaf;
    MCTSNode<T>*                       root   = inference_job.root;
    typename T::State                  state  = inference_job.state;
    std::array<float, T::policy_size>  policy = inference_job.policy;

    // Retrieve legal actions from leaf state
    std::array<typename T::Action, T::max_actions> actions;
    size_t num_actions = T::get_actions(state, actions);

    // Allocate children in tree
    std::span<MCTSNode<T>> children = _tree->create_children(leaf, num_actions);

    // First pass — sum legal policy values for normalisation
    float normal_sum = 0.0f;
    for (size_t i = 0; i < num_actions; i++) {
        normal_sum += policy[actions[i]];
    }

    if (normal_sum <= 0.0f) normal_sum = 1.0f;

    // Assign actions and normalised priors to children
    for (size_t i = 0; i < num_actions; i++) {
        children[i].action = actions[i];
        children[i].prior  = policy[actions[i]] / normal_sum;
    }

    // Publish children before opening node to other threads
    leaf->is_expanded.store(true, std::memory_order_relaxed);
    leaf->is_expanding.store(false, std::memory_order_release);

    // Backpropagate real value up to root
    backpropagate(value, 0, leaf, root);

    // Update global simulation tracker and register simulation tracker in tree
    _tree->register_simulation();
    _ctx->localSims++;
    _pending_inference_count--;
    _ctx->pendingJobs = _pending_inference_count;
}

// Main while loop for an MCTS worker thread
// The main idea is that when MCTS has progressed to a point where
// the tree has finished (ie root is a terminal state), we stop simulating job
// * We allow ongoing jobs to continue as they likely reflect pre-existing simulations
//   from the previous root
// * Once the number of ongoing jobs has reached 0, we then halt the agent all together
//=============================================================
// Class: MCTSAgent
// * One agent per thread — runs selection and evaluation loop
//=============================================================

template <GameTraits T>
void MCTSAgent<T>::run() {
    while (!(*_shutdown)) {
        /*********************** RESET PHASE **********************/ 
        _ctx->status = ThreadStatus::Reset;
        _pending_inference_count = 0;
        _ctx->localSims   = 0;
        _ctx->pendingJobs = 0;

        InferenceJob<T> job;
        while (_completed_queue.try_dequeue(job)) {}
        _barriers[static_cast<int>(BarrierPoint::RESET)].arrive_and_wait(); // WAIT POINT <----------

        /*********************** ACTIVE PHASE **********************/ 
        // Exit Condition: Game Tree is over
        
        bool _running = true;
        while (_running) 
        {
            while (_completed_queue.try_dequeue(job)) {
                SimulationEvaluator(job);
            }

            if (_tree->_game_over.load(std::memory_order_relaxed)) {
                if (_pending_inference_count == 0) {
                    _running = false;
                } else {
                    _ctx->status = ThreadStatus::Stopping;
                }
            } else if (_pending_inference_count < MAX_PENDING_REQUESTS) {
                _ctx->status = ThreadStatus::Running;
                SimulationSelector();
            }
        }
        _ctx->status = ThreadStatus::Stopped;
        _barriers[static_cast<int>(BarrierPoint::ACTIVE)].arrive_and_wait(); // WAIT POINT <----------
        
        // Set a bool to allow the GPU to start training, t
        // * his will allow it to exit the while loop, boll reset after training it done
        _training->store(true, std::memory_order_release);

        /*********************** TRAIN PHASE **********************/ 
        // MCTS Doesnt really do anything during GPU training

        _barriers[static_cast<int>(BarrierPoint::TRAIN)].arrive_and_wait(); // WAIT POINT <----------
    }
}

