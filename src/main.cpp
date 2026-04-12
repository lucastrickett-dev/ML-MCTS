#include <iostream>
#include <thread>
#include <chrono>
#include <forward_list>
#include <random>

using namespace std;


#define MCTS_SIMULATIONS_SIZE 100

// Function: MCTSWorkerThread
// Description: This function will be run by the MCTS worker thread. It will perform
// MCTS simulations and update the results to a shared buffer to be sent to the GPU
// for training the neural network.
void MCTSAgentThread()
{    
    // Main loop for MCTS simulations
    while (true) {
        
        // Perform MCTS simulations
        // Organise MCTS simulations and store results in a shared buffer
    }
}


// Function: GPUWorkerThread
// Description: This function will be run by the GPU worker thread. 
// It will receive the results from the MCTS worker thread, perform training 
// on the neural network, and update the neural network weights accordingly.
void InferenceDispatcherThread()
{
    
}



void ImGUIInterfaceSetup(){

}



int main()
{    
    ImGUIInterfaceSetup()


    // // one thread per MCTS worker
    // std::vector<std::thread> workers;

    // // one inference thread owning the GPU
    // std::thread inference_thread;

    // // one training thread
    // std::thread training_thread;    
 
 
    return 0;
}