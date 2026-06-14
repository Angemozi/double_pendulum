# =============================================================================
# Double Pendulum RL -- CMake build
# -----------------------------------------------------------------------------
# Targets:
#   dp_core       (static lib)  physics + env + PPO + trainer  [NO external deps]
#   dp_train      (exe)         headless PPO training
#   dp_eval       (exe)         load checkpoint, evaluate, export replay CSV
#   dp_benchmark  (exe)         physics/integrator throughput + energy validation
#   dp_tests      (exe)         unit/validation tests  (ctest)
#   dp_viz        (exe)         SDL2 + Dear ImGui dashboard   [BUILD_RENDERER=ON]
#
# Options:
#   -DBUILD_RENDERER=ON   build the SDL2/ImGui visualizer (pulls ImGui via
#                         FetchContent; requires SDL2 via find_package).
#   -DBUILD_TESTS=ON      build dp_tests (default ON).
#   -DDP_WITH_LIBTORCH=ON reserved hook for a future LibTorch PPO backend.
#
# Example:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release -j
#   ctest --test-dir build -C Release
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(DoublePendulumRL LANGUAGES CXX VERSION 1.0.0)

# ---- Language / build defaults ----------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_RENDERER   "Build the legacy SDL2+ImGui visualizer" OFF)
option(BUILD_SIMULATOR  "Build the raylib real-time simulator"   OFF)
option(BUILD_TESTS      "Build the test suite"            ON)
option(DP_WITH_LIBTORCH "GPU LibTorch training backend"   OFF)

# ---- Warnings (cross-compiler) ----------------------------------------------
add_library(dp_warnings INTERFACE)
if(MSVC)
    target_compile_options(dp_warnings INTERFACE /W4 /permissive-)
else()
    target_compile_options(dp_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
endif()

# ---- Core library (dependency-free) -----------------------------------------
add_library(dp_core STATIC
    src/core/Config.cpp
    src/core/Trainer.cpp
    src/core/SimWorld.cpp
    src/core/Recorder.cpp
    src/physics/DoublePendulum.cpp
    src/rl/DoublePendulumEnv.cpp
    src/rl/NeuralNet.cpp
    src/rl/PPOAgent.cpp
)
target_include_directories(dp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dp_core PUBLIC dp_warnings)

find_package(Threads REQUIRED)
target_link_libraries(dp_core PUBLIC Threads::Threads)

if(DP_WITH_LIBTORCH)
    # GPU-capable backend. Provide -DCMAKE_PREFIX_PATH=/path/to/libtorch
    # (use the CUDA libtorch distribution for GPU acceleration).
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    add_executable(dp_train_torch src/app/train_torch_main.cpp)
    # dp_core is built with /W4 -Wconversion etc.; LibTorch headers are noisy, so
    # the Torch target links dp_core but does NOT inherit dp_warnings.
    target_link_libraries(dp_train_torch PRIVATE dp_core ${TORCH_LIBRARIES})
    target_compile_definitions(dp_train_torch PRIVATE DP_WITH_LIBTORCH=1)

    # On Windows, copy the LibTorch DLLs next to the executable.
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET dp_train_torch POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS} $<TARGET_FILE_DIR:dp_train_torch>)
    endif()
endif()

# ---- Executables ------------------------------------------------------------
add_executable(dp_train     src/app/train_main.cpp)
add_executable(dp_eval      src/app/eval_main.cpp)
add_executable(dp_benchmark src/app/benchmark_main.cpp)
foreach(exe dp_train dp_eval dp_benchmark)
    target_link_libraries(${exe} PRIVATE dp_core)
endforeach()

# ---- Tests ------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_executable(dp_tests tests/test_main.cpp)
    target_link_libraries(dp_tests PRIVATE dp_core)
    add_test(NAME dp_tests COMMAND dp_tests)
endif()

# ---- Real-time simulator (raylib) -------------------------------------------
if(BUILD_SIMULATOR)
    include(FetchContent)
    # raylib is a small, dependency-light C library -> fetched & built from source
    # so a fresh clone needs no system packages. Build only the library.
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        5.5)
    FetchContent_MakeAvailable(raylib)

    add_executable(dp_simulator src/app/simulator_main.cpp)
    target_link_libraries(dp_simulator PRIVATE dp_core raylib)
endif()

# ---- Optional legacy renderer (SDL2 + Dear ImGui) ---------------------------
if(BUILD_RENDERER)
    find_package(SDL2 REQUIRED)

    include(FetchContent)
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.5)
    FetchContent_MakeAvailable(imgui)

    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer2.cpp)
    target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
    target_link_libraries(imgui PUBLIC SDL2::SDL2)

    add_executable(dp_viz src/app/viz_main.cpp)
    target_compile_definitions(dp_viz PRIVATE DP_WITH_RENDERER=1)
    target_link_libraries(dp_viz PRIVATE dp_core imgui SDL2::SDL2)
    if(TARGET SDL2::SDL2main)
        target_link_libraries(dp_viz PRIVATE SDL2::SDL2main)
    endif()
endif()
