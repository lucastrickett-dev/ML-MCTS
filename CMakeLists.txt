cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ML-MCTS CXX)

# ── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── ImGui + SDL + OpenGL ───────────────────────────────────────────────────
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ── libtorch ────────────────────────────────────────────────────────────────
set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── game selection ───────────────────────────────────────────────────────────
# Define GAME as a cache variable (user can override with -DGAME=...)
set(GAME "UltimateXOs" CACHE STRING "Game to build (folder name inside Games/)")

set(GAME_DIR "${CMAKE_SOURCE_DIR}/Games/${GAME}")

# Validate it exists
if(NOT EXISTS "${GAME_DIR}")
    file(GLOB AVAILABLE_GAMES RELATIVE "${CMAKE_SOURCE_DIR}/Games" "${CMAKE_SOURCE_DIR}/Games/*")

    message(FATAL_ERROR
        "Invalid GAME='${GAME}'\n"
        "Available games: ${AVAILABLE_GAMES}\n"
        "Use: cmake -B build -DGAME=<name>")
endif()

message(STATUS "Building with game: ${GAME}")


# ── gather sources ───────────────────────────────────────────────────────────
file(GLOB_RECURSE ALPHACORE_SOURCES CONFIGURE_DEPENDS
    "mcts/src/*.cpp"
)

file(GLOB_RECURSE GAME_SOURCES CONFIGURE_DEPENDS
    "${GAME_DIR}/src/*.cpp"
)

# ── executable ───────────────────────────────────────────────────────────────
add_executable(ML-MCTS
    src/main.cpp
    ${ALPHACORE_SOURCES}
    ${GAME_SOURCES}
    ${IMGUI_SOURCES}
)

# ── includes ─────────────────────────────────────────────────────────────────
target_include_directories(ML-MCTS PRIVATE
    mcts/include
    Games
    "${GAME_DIR}/include"
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# ── link ─────────────────────────────────────────────────────────────────────
target_link_libraries(ML-MCTS PRIVATE
    "${TORCH_LIBRARIES}"
    SDL2::SDL2
    OpenGL::GL
)

# Linux-specific fix (dl)
if(UNIX AND NOT APPLE)
    target_link_libraries(ML-MCTS PRIVATE dl)
endif()

# ── compile definitions ──────────────────────────────────────────────────────
target_compile_definitions(ML-MCTS PRIVATE
    GAME_NAME="${GAME}"
)

# ── RPATH (Linux / macOS) ────────────────────────────────────────────────────
if(APPLE)
    set_target_properties(ML-MCTS PROPERTIES
        BUILD_RPATH  "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "@loader_path/../lib"
    )
elseif(UNIX)
    set_target_properties(ML-MCTS PROPERTIES
        BUILD_RPATH  "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "$ORIGIN/../lib"
    )
endif()