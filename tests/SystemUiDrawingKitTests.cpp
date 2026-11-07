// Checks the draw list comparisons the element tests use, against drawings built by hand.
#include "SystemUiDrawingKit.h"

#include "imgui.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using SystemUiTest::ContainsDrawing;
using SystemUiTest::NearDrawing;
using SystemUiTest::NearGeometry;
using SystemUiTest::DRAWING_POSITION_SLACK;
using SystemUiTest::SameCommands;

constexpr float OUTPUT_WIDTH = 1920;
constexpr float OUTPUT_HEIGHT = 1080;
constexpr float HALF_PIXEL = 0.5f;
constexpr float OVER_SLACK = 1.5f;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const std::string &message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiDrawingKitTests: failed: %s\n", message.c_str());
    std::abort();
  }
}

// An ImGui context for draw lists, with anti-aliasing off so shapes have exact edges.
class ImGuiScope {
public:
  ImGuiScope() {
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.DisplaySize = {OUTPUT_WIDTH, OUTPUT_HEIGHT};
    io.DeltaTime = 1.0f / 60.0f;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.IniFilename = nullptr;
    ImGui::GetStyle().AntiAliasedFill = false;
    ImGui::GetStyle().AntiAliasedLines = false;
    ImGui::NewFrame();
  }
  ~ImGuiScope() {
    ImGui::EndFrame();
    ImGui::DestroyContext();
  }
  ImGuiScope(const ImGuiScope &) = delete;
  ImGuiScope &operator=(const ImGuiScope &) = delete;
};

// A draw list of its own that shares the context's draw list data, so two lists can be compared.
std::unique_ptr<ImDrawList> NewDrawList() {
  auto list = std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData());
  list->_ResetForNewFrame();
  list->PushClipRectFullScreen();
  return list;
}

constexpr ImU32 RED = IM_COL32(255, 0, 0, 255);
constexpr ImU32 GREEN = IM_COL32(0, 255, 0, 255);
constexpr ImU32 BLUE = IM_COL32(0, 0, 255, 255);

// One filled rectangle, which ImGui draws as four vertices and two triangles with anti-aliasing off.
struct Shape {
  float x;
  float y;
  ImU32 color;
};

// A list of 20 by 10 rectangles at `shapes`, drawn in order.
std::unique_ptr<ImDrawList> Drawn(const std::vector<Shape> &shapes) {
  constexpr float SHAPE_WIDTH = 20;
  constexpr float SHAPE_HEIGHT = 10;
  auto list = NewDrawList();
  for (const Shape &shape : shapes) {
    list->AddRectFilled({shape.x, shape.y}, {shape.x + SHAPE_WIDTH, shape.y + SHAPE_HEIGHT}, shape.color);
  }
  return list;
}

// The drawing with `first` inside a clip of `clip` and `second` outside it.
std::unique_ptr<ImDrawList> DrawnClipped(const Shape &first, const Shape &second, const ImVec4 &clip) {
  constexpr float SHAPE_WIDTH = 20;
  constexpr float SHAPE_HEIGHT = 10;
  auto list = NewDrawList();
  list->PushClipRect({clip.x, clip.y}, {clip.z, clip.w}, true);
  list->AddRectFilled({first.x, first.y}, {first.x + SHAPE_WIDTH, first.y + SHAPE_HEIGHT}, first.color);
  list->PopClipRect();
  list->AddRectFilled({second.x, second.y}, {second.x + SHAPE_WIDTH, second.y + SHAPE_HEIGHT}, second.color);
  return list;
}

// NearDrawing allows the position slack and nothing else.
void TestNearDrawing() {
  ImGuiScope imgui;
  const std::vector<Shape> base{{10, 10, RED}, {50, 10, GREEN}};
  const auto reference = Drawn(base);
  Check(NearDrawing(*reference, *Drawn(base)), "an identical drawing is near");
  Check(NearDrawing(*reference, *Drawn({{10 + HALF_PIXEL, 10, RED}, {50, 10 + HALF_PIXEL, GREEN}})), "a half pixel shift is near");
  Check(NearDrawing(*reference, *Drawn({{10 + DRAWING_POSITION_SLACK, 10, RED}, {50, 10, GREEN}})), "a shift of exactly the slack is near");
  Check(!NearDrawing(*reference, *Drawn({{10 + OVER_SLACK, 10, RED}, {50, 10, GREEN}})), "a shift in x past the slack is not");
  Check(!NearDrawing(*reference, *Drawn({{10, 10 + OVER_SLACK, RED}, {50, 10, GREEN}})), "nor a shift in y");
  Check(!NearDrawing(*reference, *Drawn({{10, 10, RED}, {50, 10, BLUE}})), "a colour change is not");
  Check(!NearDrawing(*reference, *Drawn({{10, 10, RED}, {50, 10, GREEN}, {90, 10, RED}})), "an extra shape is not");
  Check(!NearDrawing(*reference, *Drawn({{10, 10, RED}})), "nor a missing one");
  auto otherUv = Drawn(base);
  otherUv->VtxBuffer[0].uv.x += HALF_PIXEL;
  Check(!NearDrawing(*reference, *otherUv), "a UV change is not");
  auto otherIndex = Drawn(base);
  std::swap(otherIndex->IdxBuffer[0], otherIndex->IdxBuffer[1]);
  Check(!NearDrawing(*reference, *otherIndex), "nor a different triangulation");
  auto notANumber = Drawn(base);
  notANumber->VtxBuffer[0].pos.x = std::numeric_limits<float>::quiet_NaN();
  Check(!NearDrawing(*reference, *notANumber) && !NearDrawing(*notANumber, *notANumber), "a NaN position is not, even against itself");
}

// NearDrawing sees the clips and the way the commands are split.
void TestNearDrawingCommands() {
  ImGuiScope imgui;
  const Shape first{10, 10, RED};
  const Shape second{50, 10, GREEN};
  const ImVec4 clip{0, 0, 100, 100};
  const auto reference = DrawnClipped(first, second, clip);
  Check(NearDrawing(*reference, *DrawnClipped(first, second, clip)), "the same clip is near");
  Check(NearDrawing(*reference, *DrawnClipped(first, second, {0, 0, 100 + HALF_PIXEL, 100})), "a clip a half pixel wider is near");
  Check(!NearDrawing(*reference, *DrawnClipped(first, second, {0, 0, 100 + OVER_SLACK, 100})), "a clip past the slack is not");
  Check(!NearDrawing(*reference, *DrawnClipped(first, second, {0, 0, 100, 40})), "nor a clip of another height");
  Check(!NearDrawing(*reference, *Drawn({first, second})), "nor the same shapes with no clip between them");
  // NearGeometry leaves the commands out, so it takes the clip changes NearDrawing refuses.
  Check(NearGeometry(*reference, *DrawnClipped(first, second, {0, 0, 100, 40})), "NearGeometry accepts a clip of another height");
  Check(NearGeometry(*reference, *Drawn({first, second})), "and the same shapes with no clip between them");
  Check(!NearGeometry(*reference, *Drawn({first, {50, 10, BLUE}})), "but not a different colour");
  Check(!NearGeometry(*reference, *Drawn({first, {50 + OVER_SLACK, 10, GREEN}})), "nor a shift past the slack");
}

// SameCommands sees a clip that the geometry does not show.
void TestSameCommands() {
  ImGuiScope imgui;
  const auto plain = Drawn({{10, 10, RED}});
  auto clipped = NewDrawList();
  clipped->PushClipRect({0, 0}, {100, 100}, true);
  clipped->AddRectFilled({10, 10}, {30, 20}, RED);
  Check(SameCommands(*plain, *Drawn({{10, 10, RED}})), "the same commands match");
  Check(plain->VtxBuffer.Size == clipped->VtxBuffer.Size && std::memcmp(plain->VtxBuffer.Data, clipped->VtxBuffer.Data, sizeof(ImDrawVert) * plain->VtxBuffer.Size) == 0,
        "a clip leaves the vertices alone");
  Check(!SameCommands(*plain, *clipped), "so only the commands see it");
}

// ContainsDrawing finds a run of shapes inside a larger drawing, and nothing that isn't there in order.
void TestContainsDrawing() {
  ImGuiScope imgui;
  const Shape red{10, 10, RED};
  const Shape green{50, 10, GREEN};
  const Shape blue{90, 10, BLUE};
  const auto whole = Drawn({red, green, blue});
  Check(ContainsDrawing(*whole, *whole), "a drawing holds itself");
  Check(ContainsDrawing(*whole, *Drawn({green})), "and a shape in the middle");
  Check(ContainsDrawing(*whole, *Drawn({green, blue})), "and a run at the end");
  Check(ContainsDrawing(*whole, *Drawn({{50 + HALF_PIXEL, 10, GREEN}})), "and a shape within the slack");
  Check(!ContainsDrawing(*whole, *Drawn({{50 + OVER_SLACK, 10, GREEN}})), "but not one past it");
  Check(!ContainsDrawing(*whole, *Drawn({{50, 10 + OVER_SLACK, GREEN}})), "in either direction");
  Check(!ContainsDrawing(*whole, *Drawn({{50, 10, BLUE}})), "nor another colour");
  Check(!ContainsDrawing(*whole, *Drawn({red, blue})), "nor shapes that aren't next to each other");
  Check(!ContainsDrawing(*whole, *Drawn({green, red})), "nor shapes out of order");
  Check(!ContainsDrawing(*whole, *Drawn({})), "nor an empty drawing");
  Check(!ContainsDrawing(*Drawn({green}), *whole), "nor more than it has");
  auto otherUv = Drawn({green});
  otherUv->VtxBuffer[0].uv.x += HALF_PIXEL;
  Check(!ContainsDrawing(*whole, *otherUv), "nor another UV");
}

} // namespace

int main() {
  TestContainsDrawing();
  TestNearDrawing();
  TestNearDrawingCommands();
  TestSameCommands();
  std::puts("SystemUiDrawingKitTests: ok");
  return 0;
}
