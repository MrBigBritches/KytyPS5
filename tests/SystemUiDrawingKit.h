#ifndef EMULATOR_TESTS_SYSTEMUIDRAWINGKIT_H_
#define EMULATOR_TESTS_SYSTEMUIDRAWINGKIT_H_

// Comparisons of two draw lists, which the element tests use to hold a drawing to a reference.
// Header-only so any test target can include it.
#include "imgui.h"
#include "imgui_internal.h"

#include <cmath>

namespace SystemUiTest {

// How far a drawing's positions may drift from its reference, in output pixels. The element tree
// rounds lengths where a hand-placed reference may not, so a sub-pixel shift is expected. Anything
// the eye could see is not.
constexpr float DRAWING_POSITION_SLACK = 1.0f;

// True when two lists hold the same draw commands, each with the same clip rectangle, element
// count, offsets and texture. SameDrawing compares the vertices and indices only, so it can't see
// a clip, which changes the commands and leaves the geometry alone.
inline bool SameCommands(const ImDrawList &a, const ImDrawList &b) {
  if (a.CmdBuffer.Size != b.CmdBuffer.Size) {
    return false;
  }
  for (int i = 0; i < a.CmdBuffer.Size; ++i) {
    const ImDrawCmd &left = a.CmdBuffer[i];
    const ImDrawCmd &right = b.CmdBuffer[i];
    if (left.ClipRect.x != right.ClipRect.x || left.ClipRect.y != right.ClipRect.y ||
        left.ClipRect.z != right.ClipRect.z || left.ClipRect.w != right.ClipRect.w ||
        left.ElemCount != right.ElemCount || left.IdxOffset != right.IdxOffset || left.VtxOffset != right.VtxOffset ||
        left.TexRef._TexData != right.TexRef._TexData || left.TexRef._TexID != right.TexRef._TexID) {
      return false;
    }
  }
  return true;
}

// True when two lengths agree to within `slack`. Written so a NaN on either side fails, since every
// comparison with NaN is false and a plain "difference > slack" would let it through.
inline bool WithinSlack(float a, float b, float slack) { return std::fabs(a - b) <= slack; }

// True when two lists have the same draw commands in the same shape. The clip rectangles agree to
// within `slack` and the commands' element counts and offsets agree exactly. The textures are left
// out, since two ImGui contexts have textures of their own, so the UVs are what stands for them.
inline bool NearCommands(const ImDrawList &a, const ImDrawList &b, float slack) {
  if (a.CmdBuffer.Size != b.CmdBuffer.Size) {
    return false;
  }
  for (int i = 0; i < a.CmdBuffer.Size; ++i) {
    const ImDrawCmd &left = a.CmdBuffer[i];
    const ImDrawCmd &right = b.CmdBuffer[i];
    if (!WithinSlack(left.ClipRect.x, right.ClipRect.x, slack) || !WithinSlack(left.ClipRect.y, right.ClipRect.y, slack) ||
        !WithinSlack(left.ClipRect.z, right.ClipRect.z, slack) || !WithinSlack(left.ClipRect.w, right.ClipRect.w, slack) ||
        left.ElemCount != right.ElemCount || left.IdxOffset != right.IdxOffset || left.VtxOffset != right.VtxOffset) {
      return false;
    }
  }
  return true;
}

// True when two lists draw the same shapes in the same order, leaving out how they are clipped. The
// positions agree to within `slack` pixels, and the vertex and index counts, the indices, the UVs
// and the colours must match exactly. The commands are not looked at, so a clip that was added,
// moved or lost goes unseen and a test that cares must check it itself. Use it where two drawings
// are meant to clip differently. A NaN anywhere fails.
inline bool NearGeometry(const ImDrawList &a, const ImDrawList &b, float slack = DRAWING_POSITION_SLACK) {
  if (a.VtxBuffer.Size != b.VtxBuffer.Size || a.IdxBuffer.Size != b.IdxBuffer.Size) {
    return false;
  }
  for (int i = 0; i < a.VtxBuffer.Size; ++i) {
    const ImDrawVert &left = a.VtxBuffer[i];
    const ImDrawVert &right = b.VtxBuffer[i];
    if (!WithinSlack(left.pos.x, right.pos.x, slack) || !WithinSlack(left.pos.y, right.pos.y, slack) ||
        !WithinSlack(left.uv.x, right.uv.x, 0.0f) || !WithinSlack(left.uv.y, right.uv.y, 0.0f) || left.col != right.col) {
      return false;
    }
  }
  for (int i = 0; i < a.IdxBuffer.Size; ++i) {
    if (a.IdxBuffer[i] != b.IdxBuffer[i]) {
      return false;
    }
  }
  return true;
}

// True when two lists draw the same shapes with positions and clip rectangles that agree to within
// `slack` pixels. The vertex and index counts, the indices, the UVs, the colours, and the commands'
// element counts and offsets must match exactly, so a shape that moved a hair passes and one that
// changed colour, triangulation or the way the commands are split does not. A texture is not
// compared, so use SameCommands as well for lists from one context. A NaN anywhere fails.
inline bool NearDrawing(const ImDrawList &a, const ImDrawList &b, float slack = DRAWING_POSITION_SLACK) {
  return NearGeometry(a, b, slack) && NearCommands(a, b, slack);
}

// True when `inner`'s vertices lie in a row somewhere in `outer`, in the same order, so `outer` drew
// that drawing among others. Positions agree to within `slack` pixels, and the UVs and colours must
// match exactly. Indices and commands are left out, since they depend on what was drawn before. An
// empty `inner` is not held, as it would be held by anything. A NaN anywhere fails.
inline bool ContainsDrawing(const ImDrawList &outer, const ImDrawList &inner, float slack = DRAWING_POSITION_SLACK) {
  const int length = inner.VtxBuffer.Size;
  if (length == 0) {
    return false;
  }
  for (int start = 0; start + length <= outer.VtxBuffer.Size; ++start) {
    bool all_match = true;
    for (int i = 0; i < length && all_match; ++i) {
      const ImDrawVert &left = outer.VtxBuffer[start + i];
      const ImDrawVert &right = inner.VtxBuffer[i];
      all_match = WithinSlack(left.pos.x, right.pos.x, slack) && WithinSlack(left.pos.y, right.pos.y, slack) &&
                  WithinSlack(left.uv.x, right.uv.x, 0.0f) && WithinSlack(left.uv.y, right.uv.y, 0.0f) && left.col == right.col;
    }
    if (all_match) {
      return true;
    }
  }
  return false;
}

} // namespace SystemUiTest

#endif // EMULATOR_TESTS_SYSTEMUIDRAWINGKIT_H_
