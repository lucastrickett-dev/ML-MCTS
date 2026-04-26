#pragma once
#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include "Game.h"   // for GameTraits

struct SimpleGame {
    using State  = int;          // trivially copyable
    using Action = uint32_t;     // must be unsigned

    static constexpr size_t max_actions = 2;
    static constexpr size_t policy_size = 2;

    static constexpr size_t neural_feature_channels = 1;
    static constexpr size_t neural_input_height     = 1;
    static constexpr size_t neural_input_width      = 1;

    // --- Required interface ---

    // Non-const version (actual implementation)
    static void initialise(State& s) {
        s = 0;
    }

    // Const version to satisfy the concept
    static void initialise(const State& s) {
        auto& ns = const_cast<State&>(s);
        initialise(ns);
    }

    static void get_data(const State& s, std::vector<float>& buf) {
        buf.resize(neural_feature_channels *
                   neural_input_height *
                   neural_input_width);

        buf[0] = static_cast<float>(s);
    }

    static bool is_terminal(const State& s) {
        return s >= 5;
    }

    static float get_winner(const State& s) {
        return (s % 2 == 0) ? 1.0f : -1.0f;
    }

    static bool current_player(const State& s) {
        // false = player 0, true = player 1
        return (s % 2) != 0;
    }

    static size_t get_actions(const State&, std::array<Action, max_actions>& actions) {
        actions[0] = 0;
        actions[1] = 1;
        return 2;
    }

    static void apply_action(State& s, Action) {
        s += 1;
    }

    static float rollout(State s) {   // take by value — don't modify the original
        std::array<Action, max_actions> actions;

        while (!is_terminal(s)) {
            size_t count = get_actions(s, actions);
            Action a = actions[std::rand() % count];   // random action
            apply_action(s, a);
        }

        return get_winner(s);
    }
};

static_assert(GameTraits<SimpleGame>, "SimpleGame does not satisfy GameTraits");
