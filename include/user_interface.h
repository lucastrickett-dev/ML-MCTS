#pragma once
#include <vector>
#include <string>
#include <wx/wx.h>

class UserInterface
{
public:
    // =========================================================
    // DATA STRUCTS
    // =========================================================

    struct GlobalData
    {
        int totalThreads = 0;
        float totalSimulationsPerSec = 0.0f;
        bool running = false;
    };

    struct ThreadData
    {
        int id = 0;
        float progress = 0.0f;
        float simsPerSec = 0.0f;
        std::string status;
    };

    // =========================================================
    // LIFECYCLE
    // =========================================================

    int CreateWindow();

    int CloseWindow();

    // =========================================================
    // RENDER ENTRY
    // =========================================================

    void Render(GlobalData& global,
                const ThreadData& gpuThread,
                const std::vector<ThreadData>& mctsThreads);

private:
    // =========================================================
    // PANELS
    // =========================================================

    void RenderGlobalPanel(GlobalData& global);

    void RenderGPUPanel(const ThreadData& gpu);

    void (const std::vector<ThreadData>& threads);

    void RenderBottomBar(GlobalData& global);
};