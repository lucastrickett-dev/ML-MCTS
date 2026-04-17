#include "user_interface.h"
#include "Game.h"
#include <thread>
#include <vector>
#include "mcts.h"
#include "readerwriterqueue.h"
#include "concurrentqueue.h"
#include "neural.h"


// SECTION TO EDIT
// Below edit Game to match your Game struct and include said header file
#include "SimpleGame.h"
using Game = SimpleGame;


#define SIMULATIONS_PER_MOVE 1000

#define PREALLOCATED_NODES 1000000

#define INFERENCE_BATCH_SIZE 64

#define MCTS_THREAD_WORKERS 6


int main() {
    // Define the tree that MCTS works within
    MCTSTree<Game> tree(PREALLOCATED_NODES);  // shared, owned here
    
    // Setup communication queue between MCTS and InferenceThreads
    moodycamel::ConcurrentQueue<InferenceRequest<Game>> request_queue;

    // Thread context for keeping track of them
    GlobalContext globalContext;
    ThreadContext inferenceContext;
    std::vector<ThreadContext> mctsContext(MCTS_THREAD_WORKERS);

    // Assign thread ID's
    inferenceContext.id = 0;
    for (int i = 0; i < MCTS_THREAD_WORKERS; i++) {
        mctsContext[i].id = i + 1;
    }    
    
    // Create the Inference Dispatcher thread
    std::thread inferenceThread([&inferenceContext, &request_queue] {
        InferenceDispatcher dispatcher (
            &inferenceContext,
            INFERENCE_BATCH_SIZE,
#ifdef NEURAL_ENABLED
            torch::Device device,
            std::string("network/"),
#endif
            std::chrono::milliseconds(1),
            request_queue
        );
        dispatcher.run();
    });

    // Create the MCTS Agent threads
    std::vector<std::thread> threads;

    for (int i = 0; i < MCTS_THREAD_WORKERS; i++) {
        threads.emplace_back([&mctsContext, &tree, &request_queue, i] {
            MCTSAgentSetup<Game> mctsSetup = {
                .ctx = &mctsContext[i],
                .tree = &tree,
                .request_queue = &request_queue
            };    
            MCTSAgent<Game> agent(mctsSetup);
            agent.run();
        });    
    }    
    
    // Create UI Terminal Interface
    UserInterface ui(MCTS_THREAD_WORKERS);
    ui.setup(globalContext, inferenceContext, mctsContext);

    while (true) {
        // For loop and refresh UI at 10Hz
        ui.refresh(float(tree.unallocated_index.load(std::memory_order_relaxed)) / PREALLOCATED_NODES);
        std::this_thread::sleep_for(Ms(100));
    }

    for (auto& t : threads) t.join();
    inferenceThread.join();
}