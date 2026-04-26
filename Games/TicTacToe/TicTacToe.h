#pragma once
#include "Game.h"
#include <array>
#include <cstdint>
#include <vector>

struct TicTacToe {
    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------
    struct State {
        // board[i] = 0 (empty), 1 (X / player 0), 2 (O / player 1)
        uint8_t board[9] = {};
        bool    x_turn   = true;   // true = player 0 (X), false = player 1 (O)
    };
    static_assert(std::is_trivially_copyable_v<State>);

    // ---------------------------------------------------------------
    // Action type
    // ---------------------------------------------------------------
    using Action = unsigned int;

    // ---------------------------------------------------------------
    // Compile-time constants
    // ---------------------------------------------------------------
    static constexpr std::size_t max_actions            = 9;
    static constexpr std::size_t policy_size            = 9;

    // 3x3 board, 2 channels: channel 0 = "my" pieces, channel 1 = "opponent" pieces
    static constexpr std::size_t neural_feature_channels = 2;
    static constexpr std::size_t neural_input_height     = 3;
    static constexpr std::size_t neural_input_width      = 3;

    // ---------------------------------------------------------------
    // Interface implementations
    // ---------------------------------------------------------------

    static void initialise(const State& s) {
        // cast away const for reset (matches concept signature)
        auto& ms = const_cast<State&>(s);
        ms = State{};
    }

    // Writes flattened [channel][row][col] floats:
    //   channel 0 = current player's pieces
    //   channel 1 = opponent's pieces
    static void get_data(const State& s, std::vector<float>& buf) {
        buf.assign(2 * 9, 0.0f);
        uint8_t my_mark  = s.x_turn ? 1 : 2;
        uint8_t opp_mark = s.x_turn ? 2 : 1;
        for (int i = 0; i < 9; ++i) {
            if (s.board[i] == my_mark)  buf[i]     = 1.0f;
            if (s.board[i] == opp_mark) buf[9 + i] = 1.0f;
        }
    }

    static bool is_terminal(const State& s) {
        return winner_mark(s) != 0 || board_full(s);
    }

    // Returns value from the perspective of the *current* player.
    // Call only when is_terminal() is true.
    static float get_winner(const State& s) {
        uint8_t w = winner_mark(s);
        if (w == 0) return 0.0f;                       // draw
        uint8_t my_mark = s.x_turn ? 1 : 2;
        // If the winning mark equals the current player's mark, the *previous*
        // player just won, so the current player has lost.
        return (w == my_mark) ? -1.0f : 1.0f;
    }

    static bool current_player(const State& s) {
        return s.x_turn;   // true = player 0
    }

    static size_t get_actions(
            const State& s,
            std::array<Action, max_actions>& action_buf) {
        size_t n = 0;
        for (unsigned i = 0; i < 9; ++i)
            if (s.board[i] == 0)
                action_buf[n++] = i;
        return n;
    }

    static void apply_action(State& s, Action a) {
        s.board[a] = s.x_turn ? 1 : 2;
        s.x_turn   = !s.x_turn;
    }

    // Simple random rollout: play random moves until terminal, then return result.
    static float rollout(const State& s) {
        State copy = s;
        std::array<Action, max_actions> buf;
        while (!is_terminal(copy)) {
            size_t n = get_actions(copy, buf);
            // deterministic "random": pick middle slot if available, else first
            apply_action(copy, buf[n / 2]);
        }
        return get_winner(copy);
    }

private:
    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------
    static constexpr int LINES[8][3] = {
        {0,1,2},{3,4,5},{6,7,8},   // rows
        {0,3,6},{1,4,7},{2,5,8},   // cols
        {0,4,8},{2,4,6}            // diagonals
    };

    static uint8_t winner_mark(const State& s) {
        for (auto& line : LINES) {
            uint8_t a = s.board[line[0]];
            if (a && a == s.board[line[1]] && a == s.board[line[2]])
                return a;
        }
        return 0;
    }

    static bool board_full(const State& s) {
        for (int i = 0; i < 9; ++i)
            if (s.board[i] == 0) return false;
        return true;
    }
};

static_assert(GameTraits<TicTacToe>, "TicTacToe does not satisfy GameTraits");