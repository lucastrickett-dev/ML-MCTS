#pragma once

#include "TicTacToe/TicTacToe.h"
using Game = TicTacToe;

constexpr int   MAX_BATCH_SIZE        = 64;
constexpr int   INFERENCE_MAX_DELAY   = 5;
constexpr float VIRTUAL_LOSS_VALUE    = 1.0;
constexpr float MCTS_PUCT_CONSTANT    = 1.2;
constexpr int   MCTS_THREAD_WORKERS   = 1;
constexpr int   PREALLOCATED_NODES    = 1000000;
constexpr int   SIMULATIONS_PER_MOVE  = 1000;
constexpr int   MAX_PENDING_REQUESTS  = 64;
