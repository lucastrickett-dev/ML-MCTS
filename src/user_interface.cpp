#include "user_interface.h"
#include <iostream>
#include <sstream>
#include <iomanip>


static std::string overview_status_str(OverviewStatus s) {
    switch (s) {
        case OverviewStatus::Idle:      return "Idle";
        case OverviewStatus::Active:    return "Active";
        case OverviewStatus::Paused:    return "Paused";
        case OverviewStatus::Stopping:  return "Stopping";
        case OverviewStatus::Training:  return "Training";
        default:                        return "Unknown";
    }
}

static std::string thread_status_str(ThreadStatus s) {
    switch (s) {
        case ThreadStatus::Reset:     return "Reset";
        case ThreadStatus::Idle:      return "Idle";
        case ThreadStatus::Paused:    return "Paused";
        case ThreadStatus::Running:   return "Running";
        case ThreadStatus::Stopping:  return "Stopping";
        case ThreadStatus::Stopped:   return "Stopped";
        case ThreadStatus::Waiting:   return "Waiting";
        case ThreadStatus::Inference: return "Inference";
        case ThreadStatus::Training:  return "Training";
        default:                      return "Unknown";
    }
}

UserInterface::UserInterface(int mctsCount)
    : _mctsCount(mctsCount) {}

void UserInterface::setup(const GlobalContext&              global,
                          const ThreadContext&              inference,
                          const std::vector<ThreadContext>& mctsThreads)
{
    _global      = &global;
    _inference   = &inference;
    _mctsThreads = &mctsThreads;

    std::cout << "\033[2J\033[H" << std::flush;
    
    int r = 0;
    draw_row(r++, "");
    draw_row(r++, "");
    draw_row(r++, "############################################################################");
    draw_row(r++, "|    Memory    |                                                           |");
    draw_row(r++, "############################################################################");
    draw_row(r++, ""); 
    draw_row(r++, "############################################################################");  // 0
    draw_row(r++, "|                               MCTS Progress                              |");  // 1
    draw_row(r++, "============================================================================");  // 2
    draw_row(r++, "|    Status    |     Game     | Moves Played |  Simulation  |     Time     |");  // 9
    draw_row(r++, "============================================================================");  // 10
    draw_row(r++, "|              |              |              |              |  DD:HH:MM:SS |");  // 3  <- ROW_GLOBAL
    draw_row(r++, "############################################################################");  // 4
    draw_row(r++, "");                                                                              // 5
    draw_row(r++, "############################################################################");  // 6
    draw_row(r++, "|                             Thread Overview                              |");  // 7
    draw_row(r++, "============================================================================");  // 8
    draw_row(r++, "|    Thread    |    Status    |  Simulation  | Pending Sims |  Completed   |");  // header
    draw_row(r++, "============================================================================");  // 10
    draw_row(r++, "|              |              |              |              |              |");  // 11 <- ROW_INFERENCE
    draw_row(r++, "============================================================================");  // 12

    for (int i = 0; i < _mctsCount; i++) {
        draw_row(r++, "|              |              |              |              |              |");  // 13, 15, 17...
        draw_row(r++, "----------------------------------------------------------------------------"); // 14, 16, 18...
    }
    r--;
    // Overwrite r as i dont like the single ---- ending the table
    draw_row(r, "############################################################################");

    refresh(0.0f);

    int _total_rows = r + 3; // a few lines below the table
    std::cout << "\033[" << (_total_rows + 1) << ";1H" << std::flush;
}

void UserInterface::refresh(float memory_usage) {
    draw_row(ROW_MEMORY,    format_memory(memory_usage));
    draw_row(ROW_GLOBAL,    format_global(*_global));
    draw_row(ROW_INFERENCE, format_thread(*_inference, true));

    for (int i = 0; i < _mctsCount; i++) {
        draw_row(ROW_MCTS_START + i * ROW_MCTS_STEP, format_thread((*_mctsThreads)[i], false));
    }

    // Park cursor below the table after every refresh
    std::cout << "\033[" << (_total_rows + 2) << ";1H" << std::flush;

}

void UserInterface::draw_row(int row, const std::string& s) {
    std::cout << "\033[" << (row + 1) << ";1H"
              << "\033[2K"
              << s
              << std::flush;
}


std::string UserInterface::format_memory(float usage) {
    constexpr int BAR_TOTAL = 57;

    usage = std::clamp(usage, 0.0f, 1.0f);

    std::string pct_str = std::to_string(static_cast<int>(usage * 100)) + "%";
    int bar_width = BAR_TOTAL - static_cast<int>(pct_str.size()) - 1;

    int filled = static_cast<int>(bar_width * usage);
    int empty  = bar_width - filled;

    std::string bar_str;
    for (int i = 0; i < filled; i++) bar_str += "█";
    bar_str += std::string(empty, ' ');  // plain spaces instead
    bar_str += " " + pct_str;

    return "|    Memory    | " + bar_str + " |";
}


std::string UserInterface::format_global(const GlobalContext& g) {
    auto centre = [](const std::string& s, int width = 14) {
        int pad   = width - (int)s.size();
        int left  = pad / 2;
        int right = pad - left;
        return std::string(left, ' ') + s + std::string(right, ' ');
    };    
    
    uint64_t t  = g.time_sec.load();
    uint64_t dd = t / 86400;
    uint64_t hh = (t % 86400) / 3600;
    uint64_t mm = (t % 3600)  / 60;
    uint64_t ss = t % 60;

    char time_buf[32];
    std::snprintf(time_buf, sizeof(time_buf), "%02u:%02u:%02u:%02u",
        (unsigned)dd, (unsigned)hh, (unsigned)mm, (unsigned)ss);

    std::ostringstream out;
    out << "|" << centre(overview_status_str(g.status.load()))
        << "|" << centre(std::to_string(g.totalGames.load()))
        << "|" << centre(std::to_string(g.totalMoves.load()))
        << "|" << centre(std::to_string(g.totalSims.load()))
        << "|" << centre(time_buf)
        << "|";
    return out.str();
}


std::string UserInterface::format_thread(const ThreadContext& t, bool is_inference) {
    auto centre = [](const std::string& s, int width = 14) {
        int pad   = width - (int)s.size();
        int left  = pad / 2;
        int right = pad - left;
        return std::string(left, ' ') + s + std::string(right, ' ');
    };
    std::string name = is_inference
        ? "Inference"
        : "MCTS #" + std::to_string(t.id);

    std::ostringstream out;
    out << "|" << centre(name)
        << "|" << centre(thread_status_str(t.status.load()))
        << "|" << centre(std::to_string(t.localSims.load()))
        << "|" << centre(std::to_string(t.pendingJobs.load()))
        << "|" << centre(std::to_string(t.completedGames.load()))
        << "|";
    return out.str();
}
