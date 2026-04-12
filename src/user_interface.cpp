#include "user_interface.h"
#include <cstdint>
#include "imgui.h"
#include "imgui_internal.h"

// =============================================================
// INIT / SHUTDOWN
// =============================================================

int UserInterface::CreateWindow()
{
    if (SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("Failed!");
        return EXIT_FAILURE;
    };
    return 0;
    
    SDL_CreateWindow("MCTS Multithreaded Progress",
                     320,
                     240,
                     SDL_WINDOW_BORDERLESS);
}

int UserInterface::CloseWindow()
{
    SDL_Quit();
}

// =============================================================
// FRAME BOUNDARY
// =============================================================


// =============================================================
// MAIN RENDER
// =============================================================

void UserInterface::Render(GlobalData& global,
                           const ThreadData& gpuThread,
                           const std::vector<ThreadData>& mctsThreads)
{
    ImGui::Begin("ML-MCTS Dashboard");

    RenderGlobalPanel(global);
    RenderGPUPanel(gpuThread);
    RenderMCTSThreads(mctsThreads);
    RenderBottomBar(global);

    ImGui::End();
}

// =============================================================
// GLOBAL PANEL
// =============================================================

void UserInterface::RenderGlobalPanel(GlobalData& global)
{
    ImGui::BeginChild("GlobalPanel", ImVec2(0, 80), true);

    ImGui::Text("GLOBAL STATE");
    ImGui::Separator();

    ImGui::Text("Threads: %d", global.totalThreads);
    ImGui::Text("Simulations/sec: %.2f", global.totalSimulationsPerSec);
    ImGui::Text("Running: %s", global.running ? "YES" : "NO");

    ImGui::EndChild();
}

// =============================================================
// GPU PANEL
// =============================================================

void UserInterface::RenderGPUPanel(const ThreadData& gpu)
{
    ImGui::BeginChild("GPUPanel", ImVec2(0, 100), true);

    ImGui::Text("GPU / MAIN THREAD");
    ImGui::Separator();

    ImGui::Text("Status: %s", gpu.status.c_str());
    ImGui::Text("Sims/sec: %.2f", gpu.simsPerSec);

    ImGui::ProgressBar(gpu.progress, ImVec2(-1, 0));

    ImGui::EndChild();
}

// =============================================================
// MCTS THREAD GRID
// =============================================================

void UserInterface::RenderMCTSThreads(const std::vector<ThreadData>& threads)
{
    ImGui::BeginChild("MCTSPanel", ImVec2(0, -80), true);

    ImGui::Text("MCTS THREADS");
    ImGui::Separator();

    const int columns = 3;
    ImGui::Columns(columns, nullptr, false);

    for (size_t i = 0; i < threads.size(); i++)
    {
        const auto& t = threads[i];

        ImGui::BeginChild(ImGui::GetID((void*)(intptr_t)i),
                          ImVec2(0, 120),
                          true);

        ImGui::Text("Thread %d", t.id);
        ImGui::Text("%s", t.status.c_str());
        ImGui::Text("SPS: %.2f", t.simsPerSec);

        ImGui::ProgressBar(t.progress, ImVec2(-1, 0));

        ImGui::EndChild();

        ImGui::NextColumn();
    }

    ImGui::Columns(1);

    ImGui::EndChild();
}

// =============================================================
// BOTTOM BAR
// =============================================================

void UserInterface::RenderBottomBar(GlobalData& global)
{
    ImGui::Separator();

    if (ImGui::Button(global.running ? "Pause" : "Resume"))
    {
        global.running = !global.running;
    }

    ImGui::SameLine();

    if (ImGui::Button("Stop"))
    {
        global.running = false;
        // later: signal worker threads to stop safely
    }
}