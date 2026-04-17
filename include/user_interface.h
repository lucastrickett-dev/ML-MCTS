#pragma once
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::milliseconds;

enum class ThreadStatus   { Idle, Paused, Running, Stopping, Stopped, Waiting, Inference, Training };
enum class OverviewStatus { Idle, Active, Paused, Stopping, Training };

struct ThreadContext {
    int id = 0;
    std::atomic<ThreadStatus> status         { ThreadStatus::Idle };
    std::atomic<uint64_t>     localSims      { 0 };
    std::atomic<uint64_t>     pendingJobs    { 0 };
    std::atomic<uint64_t>     completedGames { 0 };
};


struct GlobalContext {
    std::atomic<uint64_t>       totalGames  { 0 };
    std::atomic<uint64_t>       localSims   { 0 };
    std::atomic<uint64_t>       root_index  { 0 };
    std::atomic<uint64_t>       time_sec    { 0 };
    std::atomic<OverviewStatus> status      { OverviewStatus::Active };
};

class UserInterface {
public:
    UserInterface(int mctsThreads);

    void setup(const GlobalContext&              global,
               const ThreadContext&              inference,
               const std::vector<ThreadContext>& mctsThreads);

    void refresh(float memory_usage);

private:
    void draw_row (int row, const std::string& s);

    std::string format_memory (float memory_usage);
    std::string format_global (const GlobalContext& g);
    std::string format_thread(const ThreadContext& t, bool is_inference);

    int                               _mctsCount;
    const GlobalContext*              _global      = nullptr;
    const ThreadContext*              _inference   = nullptr;
    const std::vector<ThreadContext>* _mctsThreads = nullptr;

    static constexpr int ROW_MEMORY      = 1;
    static constexpr int ROW_GLOBAL      = 9;
    static constexpr int ROW_INFERENCE   = 17;
    static constexpr int ROW_MCTS_START  = 19;
    static constexpr int ROW_MCTS_STEP   = 2;
};