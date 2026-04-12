cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ML-MCTS CXX)

# ==============================================================
# 🧠 C++ CONFIG
# ==============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================
# 📦 OUTPUT DIRS
# ==============================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

# ==============================================================
# 🚨 SDL3 SAFETY (prevents Wayland/X11 chaos)
# ==============================================================
set(SDL_VIDEO_WAYLAND OFF CACHE BOOL "" FORCE)
set(SDL_VIDEO_X11 ON CACHE BOOL "" FORCE)

# ==============================================================
# 🧩 SDL3
# ==============================================================
add_subdirectory(third_party/SDL EXCLUDE_FROM_ALL)

# ==============================================================
# 🖼️ IMGUI
# ==============================================================
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ==============================================================
# 📁 SOURCES
# ==============================================================
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# ==============================================================
# 🚀 EXECUTABLE
# ==============================================================
add_executable(ML-MCTS
    ${PROJECT_SOURCES}
    ${IMGUI_SOURCES}
)

# ==============================================================
# 📂 INCLUDE PATHS
# ==============================================================
target_include_directories(ML-MCTS PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# ==============================================================
# 🧩 DEPENDENCIES
# ==============================================================
find_package(OpenGL REQUIRED)

# ==============================================================
# 🔗 LINK LIBRARIES
# ==============================================================
target_link_libraries(ML-MCTS PRIVATE
    SDL3::SDL3
    OpenGL::GL
)

# ==============================================================
# 🐧 LINUX FIX
# ==============================================================
if(UNIX AND NOT APPLE)
    target_link_libraries(ML-MCTS PRIVATE dl)
endif()