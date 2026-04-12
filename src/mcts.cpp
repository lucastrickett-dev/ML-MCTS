// #include <iostream>
// #include <forward_list>
// #include "mcts.h"


// template <GameTraits T>
// MCTSAgent<T>::MCTSAgent(MCTSTree<T>& tree,
//                         std::atomic<size_t>& simulation_count, 
//                         moodycamel::ConcurrentQueue<InferenceRequest<T>>& request_queue)
//     : _tree(&tree)
//     , _simulation_count(simulation_count)
//     , _request_queue(request_queue)
// {}


// // Function: backpropagate
// // * From a given leaf node, iteratively undo virtual-loss features
// //   move up to parent, and repeat until applied to root
// template <GameTraits T>
// void MCTSAgent<T>::backpropagate(float value, int visit, MCTSNode<T>* start, MCTSNode<T>* end)
// {
//     MCTSNode<T>* current = start;

//     while (current != end && current->parent_ptr != nullptr)
//     {
//         // Undoing virtual loss inherient into the backpropogation
//         current->update(value + VIRTUAL_LOSS_VALUE, visit);
//         current = current->parent_ptr;
//         value = -value;  // flip for negamax
//     }
//     // Apply to end node (root)
//     current->update(value, visit);
// }

// // Function: SimulationSelector
// // * First half of simulation (selection -> inference request)
// template <GameTraits T>
// void MCTSAgent<T>::SimulationSelector()
// {
//     // Snapshot root — thread safe copy of root ptr and state
//     std::pair<MCTSNode<T>*, typename T::State> snapshot = _tree->get_root_snapshot();
//     typename T::State local_state = snapshot.second;
//     MCTSNode<T>*      local_root  = snapshot.first;

//     MCTSNode<T>* current = local_root;

//     // Iteratively travel tree until leaf is reached
//     while (!current->is_leaf())
//     {
//         // Apply virtual loss to discourage other threads from selecting same path
//         current->update(-VIRTUAL_LOSS_VALUE, 1);
                    
//         uint32_t parent_visits = current->visit_count.load(std::memory_order_relaxed);

//         // Look through all children to find highest PUCT
//         MCTSNode<T>* best_child = nullptr;
//         float        best_score = -std::numeric_limits<float>::infinity();

//         for (MCTSNode<T>& child : current->children) {
//             float score = child.PUCT(parent_visits);
//             if (score > best_score) {
//                 best_score = score;
//                 best_child = &child;
//             }
//         }

//         if (!best_child) return;

//         current = best_child;
//         T::apply_action(local_state, current->action);
//     }

//     // Apply virtual loss to leaf
//     current->update(-VIRTUAL_LOSS_VALUE, 1);

//     // Ensure only one thread expands this leaf
//     bool expected = false;
//     if (!current->is_expanding.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
//         // Another thread is already expanding — undo virtual loss and bail
//         backpropagate(0, -1, current, local_root);
//         return;
//     } 

//     // Check if terminal flag has already been set (faster than calling T::is_terminal)
//     if (current->is_terminal.load(std::memory_order_acquire)) {
//         current->is_expanding.store(false, std::memory_order_release);
//         backpropagate(T::get_winner(local_state), 0, current, local_root);
//         _simulation_count.fetch_add(1, std::memory_order_relaxed);
//         return;
//     }

//     // Check if state is terminal
//     if (T::is_terminal(local_state)) {
//         current->is_terminal.store(true, std::memory_order_relaxed);
//         current->is_expanded.store(true, std::memory_order_relaxed);
//         current->is_expanding.store(false, std::memory_order_release);
//         backpropagate(T::get_winner(local_state), 0, current, local_root);
//         _simulation_count.fetch_add(1, std::memory_order_relaxed);
//         return;
//     }

//     // Build and submit inference request
//     InferenceRequest<T> request {
//         .state  = local_state,
//         .leaf   = current,
//         .root   = local_root,
//         .queue  = &_completed_queue
//     };

//     // Try to add to request queue — if fails, undo virtual loss and discard
//     if (_request_queue.enqueue(request)) {
//         _pending_inference_count++;
//     } else {
//         current->is_expanding.store(false, std::memory_order_release);
//         backpropagate(0, -1, current, local_root);
//     }
// }


// // Function: SimulationEvaluator
// // * Second half of simulation (inference result -> expand -> backpropagate)
// template <GameTraits T>
// void MCTSAgent<T>::SimulationEvaluator(InferenceJob<T> inference_job)
// {
//     // Unpack job
//     float                              value  = inference_job.value;
//     MCTSNode<T>*                       leaf   = inference_job.leaf;
//     MCTSNode<T>*                       root   = inference_job.root;
//     typename T::State                  state  = inference_job.leaf_state;
//     std::array<float, T::policy_size>  policy = inference_job.policy;

//     // Retrieve legal actions from leaf state
//     std::array<typename T::Action, T::max_actions> actions;
//     size_t num_actions = T::get_actions(state, actions);

//     // Allocate children in tree
//     std::span<MCTSNode<T>> children = _tree->create_children(leaf, num_actions);

//     // First pass — sum legal policy values for normalisation
//     float normal_sum = 0.0f;
//     for (size_t i = 0; i < num_actions; i++) {
//         normal_sum += policy[actions[i]];
//     }

//     if (normal_sum <= 0.0f) normal_sum = 1.0f;

//     // Assign actions and normalised priors to children
//     for (size_t i = 0; i < num_actions; i++) {
//         children[i].action = actions[i];
//         children[i].prior  = policy[actions[i]] / normal_sum;
//     }

//     // Publish children before opening node to other threads
//     leaf->is_expanded.store(true, std::memory_order_relaxed);
//     leaf->is_expanding.store(false, std::memory_order_release);

//     // Backpropagate real value up to root
//     backpropagate(value, 0, leaf, root);

//     // Update global simulation tracker
//     _simulation_count.fetch_add(1, std::memory_order_relaxed);
    
// }

// // Main while loop for an MCTS worker thread
// // The main idea is that when MCTS has progressed to a point where
// // the tree has finished (ie root is a terminal state), we stop simulating job
// // * We allow ongoing jobs to continue as they likely reflect pre-existing simulations
// //   from the previous root
// // * Once the number of ongoing jobs has reached 0, we then halt the agent all together
// template <GameTraits T>
// void MCTSAgent<T>::run()
// {
//     while (_running.load(std::memory_order_relaxed))
//     {
//         // Always drain completed jobs first
//         InferenceJob<T> job;
//         while (_completed_queue.try_dequeue(job)) {
//             _pending_inference_count--;
//             SimulationEvaluator(job);
//         }

//         if (_tree->_game_over.load(std::memory_order_relaxed)) {
//             // Game over — stop once all in-flight jobs are resolved
//             if (_pending_inference_count == 0)
//                 _running.store(false, std::memory_order_relaxed);
//         } else {
//             // Game active — run new selections if under limit
//             if (_pending_inference_count < _max_pending)
//                 SimulationSelector();
//         }
//     }
// }

// template <GameTraits T>
// bool MCTSAgent<T>::start() {
//     _running.store(true,  std::memory_order_relaxed);  // start

// }


// template <GameTraits T>
// bool MCTSAgent<T>::stop() {
//     _running.store(false,  std::memory_order_relaxed);  // stop

// }