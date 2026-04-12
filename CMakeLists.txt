cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ML-MCTS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# wxWidgets (submodule)
# =========================
set(wxBUILD_SHARED OFF CACHE BOOL "" FORCE)
set(wxBUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/wxWidgets)


# =========================
# SOURCES
# =========================
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(ML-MCTS ${PROJECT_SOURCES})

# =========================
# LINK
# =========================
target_link_libraries(ML-MCTS PRIVATE wx::core wx::base)