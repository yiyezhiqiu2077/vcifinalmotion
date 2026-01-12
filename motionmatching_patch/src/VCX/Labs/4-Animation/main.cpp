#include "Assets/bundled.h"
#include "Labs/4-Animation/App.h"

int main() {
    using namespace VCX;
    return Engine::RunApp<Labs::Animation::App>(Engine::AppContextOptions {
        .Title      = "VCX Lab 4: Motion Matching",
        .WindowSize = { 1024, 768 },
        .FontSize   = 16,
        .IconFileNames = Assets::DefaultIcons,
        .FontFileNames = Assets::DefaultFonts,
    });
}
