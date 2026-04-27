#pragma once
#include "Game.h"
#include <array>
#include <cstdint>
#include <vector>


#define ULTIMATEXO_SUB_INDEX(n, i) (9 * (n) + i)

struct UltimateXO {
    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------

    // 0   to 80  -> current player
    // 81  to 89  -> current player meta
    // 90  to 170 -> enemy player
    // 171 to 179 -> enemy player meta

    struct State {
        // board[i] = 0 (empty), 1 (X / player 0), 2 (O / player 1)
        uint8_t board[81] = {};
        uint8_t meta[9] = {};
        uint8_t index = 4;
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
    static constexpr std::size_t neural_feature_channels = 21;
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
    static void get_data(const State& s, float* buf) {
        std::fill(buf, buf + neural_input_width * neural_input_height * neural_feature_channels, 0.0f);
        uint8_t my_mark  = s.x_turn ? 1 : 2;
        uint8_t opp_mark = s.x_turn ? 2 : 1;

        for (int i = 0; i < 81; ++i) {
            if (s.board[i] == my_mark)  buf[i]      = 1.0f;
            if (s.board[i] == opp_mark) buf[81 + i] = 1.0f;
        }

        for (int i = 0; i < 9; ++i) {
            if (s.meta[i] == my_mark)  buf[162 + i] = 1.0f;
            if (s.meta[i] == opp_mark) buf[171 + i] = 1.0f;
        }

        buf[180 + s.index] = 1.0f;
    }

    static bool is_terminal(const State& s) {
        return winner_mark(s.meta) != 0 || board_full(s);
    }

    // Returns value from the perspective of the *current* player.
    // Call only when is_terminal() is true.
    static float get_winner(const State& s) {
        uint8_t w = winner_mark(s.meta);
        if (w == 0) return 0.0f;
        uint8_t my_mark = s.x_turn ? 1 : 2;
        return (w == my_mark) ? -1.0f : 1.0f;
    }

    static bool current_player(const State& s) {
        return s.x_turn;   // true = player 0
    }

    static size_t get_actions(
            const State& s,
            std::array<Action, max_actions>& action_buf) {
        size_t n = 0;
        for (unsigned i = 0; i < policy_size; ++i)
            if (s.board[ULTIMATEXO_SUB_INDEX(s.index, i)] == 0)
                action_buf[n++] = i;
        return n;
    }

    static void apply_action(State& s, Action a) {
        s.board[ULTIMATEXO_SUB_INDEX(s.index, a)] = s.x_turn ? 1 : 2;
        s.x_turn = !s.x_turn;
        
        if (s.meta[s.index] == 0) s.meta[s.index] = winner_mark(&s.board[ULTIMATEXO_SUB_INDEX(s.index, 0)]);  // check the local board
        s.index = (uint8_t)a;
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

    static uint8_t winner_mark(const uint8_t* cells) {
        for (auto& line : LINES) {
            uint8_t a = cells[line[0]];
            if (a && a == cells[line[1]] && a == cells[line[2]])
                return a;
        }
        return 0;
    }

    static bool board_full(const State& s) {
        for (int i = 0; i < 9; ++i)
            if (s.board[ULTIMATEXO_SUB_INDEX(s.index, i)] == 0) return false;
        return true;
    }
};

static_assert(GameTraits<UltimateXO>, "UltimateXO does not satisfy GameTraits");