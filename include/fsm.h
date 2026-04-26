#pragma once
#include <iostream>

enum class BarrierPoint {
    RESET,
    ACTIVE,
    TRAIN,
    COUNT       // useful trick to get array size
};

struct BarrierCompletion {
    bool& pause_snapshot;
    bool& paused;
    bool& shutdown_snapshot;
    bool& shutdown;

    void operator()() noexcept {
        pause_snapshot    = paused;
        shutdown_snapshot = shutdown;
    }
};