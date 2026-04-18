#include "user_interface.h"
#include "Game.h"
#include <thread>
#include <barrier>
#include <vector>
#include "mcts.h"
#include "readerwriterqueue.h"
#include "concurrentqueue.h"
#include "neural.h"


// The main loop for the threads have the following phase gates:
// RESET PHASE GATE:
// * The run() loop will reset all conditions for the new run
//   and wait until all others have done the same
//
// RUNNING PHASE GATE:
// * Is the main loop each will do when running
//
// TRAINING PHASE GATE:
// * Wont do anything for MCTS, but will train the neural network
// * if the stop condition is active at the end, we exit the loop
std::barrier<> _barriers[static_cast<int>(BarrierPoint::COUNT)];



// SECTION TO EDIT
// Below edit Game to match your Game struct and include said header file
#include "SimpleGame.h"
using Game = SimpleGame;


#define SIMULATIONS_PER_MOVE 1000

#define PREALLOCATED_NODES 1000000

#define INFERENCE_BATCH_SIZE 64

#define MCTS_THREAD_WORKERS 6



int main () {
    // start process
    std::atomic<int> _thread_finished = 0;
    std::atomic<bool> shutdown = false;
    std::atomic<bool> paused = false;
    
    // Define the tree that MCTS works within
    MCTSTree<Game> tree(PREALLOCATED_NODES);  // shared, owned here
    
    // Setup communication queue between MCTS and InferenceThreads
    moodycamel::ConcurrentQueue<InferenceRequest<Game>> request_queue;

    // Thread context for keeping track of them
    GlobalContext globalContext;
    ThreadContext inferenceContext;
    std::vector<ThreadContext> mctsContext(MCTS_THREAD_WORKERS);
    
    // Create UI Terminal Interface
    UserInterface ui(MCTS_THREAD_WORKERS);
    ui.setup(globalContext, inferenceContext, mctsContext);
    ui.refresh(float(tree.unallocated_index.load(std::memory_order_relaxed)) / PREALLOCATED_NODES);

    // Shutdown stuff properly

}







// int main() {
//     FiniteStates state = FiniteStates::START;


//     // Assign thread ID's
//     inferenceContext.id = 0;
//     for (int i = 0; i < MCTS_THREAD_WORKERS; i++) {
//         mctsContext[i].id = i + 1;
//     }    
    
//     // Create the Inference Dispatcher thread
//     std::thread inferenceThread([&inferenceContext, &request_queue] {
//         InferenceDispatcher dispatcher (
//             &inferenceContext,
//             INFERENCE_BATCH_SIZE,
// #ifdef NEURAL_ENABLED
//             torch::Device device,
//             std::string("network/"),
// #endif
//             std::chrono::milliseconds(1),
//             request_queue
//         );
//         dispatcher.run();
//     });

//     // Create the MCTS Agent threads
//     std::vector<std::thread> threads;

//     for (int i = 0; i < MCTS_THREAD_WORKERS; i++) {
//         threads.emplace_back([&mctsContext, &tree, &request_queue, i] {
//             MCTSAgentSetup<Game> mctsSetup = {
//                 .ctx = &mctsContext[i],
//                 .tree = &tree,
//                 .request_queue = &request_queue
//             };    
//             MCTSAgent<Game> agent(mctsSetup);
//             agent.run();
//         });    
//     }    
    

//     for (auto& t : threads) t.join();
//     inferenceThread.join();
// }