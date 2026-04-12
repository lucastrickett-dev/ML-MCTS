#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_opengl3.h"

#include <vector>

#include "user_interface.h"

bool running = true;

// ─────────────────────────────────────────────
// YOUR UI SYSTEM
// ─────────────────────────────────────────────
UserInterface ui;
UserInterface::GlobalData global;
UserInterface::ThreadData gpu;
std::vector<UserInterface::ThreadData> mcts;

int main()
{
    // =========================================================
    // SDL3 INIT
    // =========================================================
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "ML-MCTS",
        1280, 720,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );

    if (!window)
    {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        return -1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1);

    // =========================================================
    // IMGUI INIT
    // =========================================================
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init("#version 130");

    ui.Init();

    // =========================================================
    // MAIN LOOP
    // =========================================================
    SDL_Event event;

    while (running)
    {
        // ─────────────────────────────
        // EVENTS
        // ─────────────────────────────
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);

            if (event.type == SDL_EVENT_QUIT)
                running = false;
        }

        // ─────────────────────────────
        // FRAME START
        // ─────────────────────────────
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ui.BeginFrame();

        // ─────────────────────────────
        // YOUR UI
        // ─────────────────────────────
        ui.Render(global, gpu, mcts);

        ui.EndFrame();

        ImGui::Render();

        // ─────────────────────────────
        // RENDER
        // ─────────────────────────────
        glViewport(0, 0, 1280, 720);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);
    }

    // =========================================================
    // CLEANUP
    // =========================================================
    ui.Shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}