#include "user_interface.h"
#include "Game.h"
#include <thread>
#include <barrier>
#include <vector>
#include "mcts.h"
#include "fsm.h"
#include "readerwriterqueue.h"
#include "concurrentqueue.h"
#include "neural.h"
#include "generated/config.h"
 
#define THREAD_COUNT (MCTS_THREAD_WORKERS + 1)


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



int main () {
    // start process
    std::atomic<bool> _training = false;

    bool pause             = true; // read inside completion, visible to all after barrier
    bool pause_snapshot    = true; // read inside completion, visible to all after barrier
    bool shutdown          = false; // read inside completion, visible to all after barrier
    bool shutdown_snapshot = false; // read inside completion, visible to all after barrier
    
    std::barrier<BarrierCompletion> barriers[] = {
        std::barrier<BarrierCompletion>(THREAD_COUNT, BarrierCompletion{pause_snapshot, pause, shutdown_snapshot, shutdown}),
        std::barrier<BarrierCompletion>(THREAD_COUNT, BarrierCompletion{pause_snapshot, pause, shutdown_snapshot, shutdown}),
        std::barrier<BarrierCompletion>(THREAD_COUNT, BarrierCompletion{pause_snapshot, pause, shutdown_snapshot, shutdown}),
    };
    
    // Thread context for keeping track of them
    GlobalContext globalContext;

    // Define the tree that MCTS works within
    MCTSTree<Game> tree(PREALLOCATED_NODES, &globalContext);  // shared, owned here
    
    // Setup communication queue between MCTS and InferenceThreads
    moodycamel::ConcurrentQueue<InferenceRequest<Game>> request_queue;
    
    // Spin up MCTS agent threads
    std::vector<ThreadContext> mctsContext(MCTS_THREAD_WORKERS);
    std::vector<std::thread> mcts_threads;
    mcts_threads.reserve(MCTS_THREAD_WORKERS);
    
    for (int i = 0; i < MCTS_THREAD_WORKERS; ++i) {
        mctsContext[i].id = i;
        MCTSAgentSetup<Game> setup {
            .ctx           = &mctsContext[i],
            .tree          = &tree,
            .barriers      = barriers,
            .request_queue = &request_queue,
            .training      = &_training,
            .paused        = &pause_snapshot,
            .shutdown      = &shutdown_snapshot,
        };
        mcts_threads.emplace_back([setup]() mutable {
            MCTSAgent<Game> agent(setup);
            agent.run();
        });
    }
    
    // Spin up inference thread
    ThreadContext inferenceContext;
    inferenceContext.id = MCTS_THREAD_WORKERS;
    std::thread inference_thread([&]() {
        InferenceDispatcher<Game> dispatcher(
            &inferenceContext,
            &tree,
#ifdef NEURAL_ENABLED
            torch::Device(torch::kCPU),
            std::nullopt,
            std::nullopt,
#endif
            barriers,
            &_training,
            &pause_snapshot,
            &shutdown_snapshot,
            request_queue
        );
        dispatcher.run();
    });
    
    // Create UI Terminal Interface
    UserInterface ui(MCTS_THREAD_WORKERS);
    ui.setup(globalContext, inferenceContext, mctsContext);
    
    std::thread ui_thread([&]() {
        while (true) {
            ui.refresh(float(tree.unallocated_index.load(std::memory_order_relaxed)) / PREALLOCATED_NODES);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // Join all threads before barriers go out of scope
    for (auto& t : mcts_threads) t.join();
    inference_thread.join();    
}




