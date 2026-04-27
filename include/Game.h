#pragma once
#include <vector>
#include <concepts>



// Note:
// * Every action must be represented as an unsigned integer less than T::policy_size
//   and be unique within the set of all possible legal actiosns for any state.
//   (ie encode action on a board based on index of grid 0,1,2,...
//   - 0 - Action 0, 1 - Action 1, ...
// * There is a notable difference between policy_size and max_actions
//  - policy_size refers to the maximum possible number that youre action could be encoded as
//    (eg if you are designing a chess game and wish to denote a movement as 
//    'current-grid' * 64 + 'target-grid' where grids are 0-63, policy_size is 
//    63*64 + 63 = 4095 - not a possible move, but still) 
//    > this sets the size of the policy network.
//  - max_actions refers to the maximum amount of unique actions that could be performed at any given 
//    state during the game, (this sets the array in 'mcts.cpp' that 'get_actions' writes to)
//
// IMPORTANT:
// * If your code isnt consistent MCTS will not work, please ensure in your 'get_actions'
//   that when you are writing to action_buf[n] = k:
//    -  n stays between 0 - T::max_actions
//    -  k stays between 0 - T::policy_size
// * Make sure that T::player_size defines how many people are playing, and that
//    - 0 < current_player(s) < T::player_size

template <typename T>
concept GameTraits = requires(
    const typename T::State&                         s,
    typename T::State&                               ms,
    typename T::Action                               a,
    float*                                           buf,
    std::array<typename T::Action, T::max_actions>&  action_buf
) {
    typename T::State;
    requires std::is_trivially_copyable_v<typename T::State>;

    typename T::Action;
    requires std::is_unsigned_v<typename T::Action>;

    // Encodes the maximum actions possible at any given point
    // Since we use stack, we use this to determine how big to make the array
    { T::max_actions } -> std::convertible_to<std::size_t>;
    requires (T::max_actions > 0);

    // Defines the limit on how many actions the policy vector covers
    { T::policy_size } -> std::convertible_to<std::size_t>;
    requires (T::policy_size > 0);
    
    // Defines the dimensions of the 3D tensor that 'get_data' writes to 
    // for the neural network to use (e.g. dim(buf) - [height, width, channels]).
    // - for something like chess, 'height' and 'width' are 8 obviously. But
    //   the choice of feature_channels is up to you. Having 'enenmy' pieces 
    //   and 'your' pieces into seperate 8x8 layers can help the NN form more 
    //   specialised structure for each, the same would go for also sperating
    //   piece types (your pawns, their knights...) into different feature channels
    { T::neural_feature_channels } -> std::convertible_to<std::size_t>;
    { T::neural_input_height     } -> std::convertible_to<std::size_t>;
    { T::neural_input_width      } -> std::convertible_to<std::size_t>;

    // Should set state to its base-state (ie start of game)
    // This means setting turns, putting start pieces in like pawns, knights
    { T::initialise(s)              } -> std::same_as<void>;

    // Outputs the data as an array of float
    // Required: you need to output it form the perspecitve of the current player
    { T::get_data(s, buf)           } -> std::same_as<void>;

    // Outputs true when the game has ended
    { T::is_terminal(s)             } -> std::same_as<bool>;

    // Returns the value (-1.0 for loss, 1.0 for win, 0.0 for draw) as seen by the current player
    // This is usually always because this is only evaluated when 'is_terminal' is true
    // - meaning the current state has no possible moves for the current player
    //   usually because theyve been check mates, other player previously played winning move
    //   because  prev. player made a move resulting in stale-mate ...etc
    // - should ideally be 0 if game is on-going
    { T::get_winner(s)              } -> std::same_as<float>;

    // Returns true/flase for if its player0 or player 1 turn
    { T::current_player(s)          } -> std::same_as<bool>;

    // Fills in possible actions within 'action_buf' and returns 
    // how many actions were written as a 'size_t'
    { T::get_actions(s, action_buf) } -> std::same_as<size_t>;

    // Edits the data within State corresponding to doing a certain action 
    { T::apply_action(ms, a)        } -> std::same_as<void>;

    // If no neural network, need to have a rollout method
    { T::rollout(s)                 } -> std::same_as<float>;
};
