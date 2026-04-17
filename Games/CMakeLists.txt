# Games/CMakeLists.txt

file(GLOB_RECURSE GAME_HEADERS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*/*.h"
)

add_library(Games ${GAME_HEADERS})
target_include_directories(Games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
