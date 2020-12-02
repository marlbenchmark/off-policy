cmake_minimum_required (VERSION 2.8.11)
project (hanabi_learning_environment)

set(CMAKE_C_FLAGS "-O2 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -Wall -Wextra -fPIC -Wno-sign-compare")

add_subdirectory (hanabi_lib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_library (pyhanabi SHARED pyhanabi.cc)
target_link_libraries (pyhanabi LINK_PUBLIC hanabi)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples)

add_executable (game_example examples/game_example.cc)
target_link_libraries (game_example LINK_PUBLIC hanabi)
