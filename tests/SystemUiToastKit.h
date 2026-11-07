#ifndef EMULATOR_TESTS_SYSTEMUITOASTKIT_H_
#define EMULATOR_TESTS_SYSTEMUITOASTKIT_H_

// What the toast tests share, the set of toasts they run over and the check of a toast's clips.
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/imgui/convert.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace SystemUiTest {

// A title long enough to need its two lines and an ellipsis at the toast's width.
inline constexpr const char *LONG_TITLE =
    "Master Alchemist of the Envisioned Land, Seeker of Every Lost Memory and Keeper of the Ancient Recipes";
// A title that wraps to exactly two lines at the toast's width without being cut.
inline constexpr const char *WRAPPING_TITLE = "Slayer of the Seven Sleeping Giants of the Frozen Northern Wastes";

// Toasts with each optional part, a long title, a wrapped title with a cup and a glow. A
// toast that draws more than the plain one proves each part is drawn at all.
inline std::vector<Libs::Graphics::SystemUi::ToastProps> ToastVariants(ImTextureData *icon) {
  using namespace Libs::Graphics::SystemUi;
  ToastProps plain{.title = "Saved", .subtitle = "Slot 1"};
  ToastProps iconic = plain;
  iconic.icon = icon;
  iconic.icon_size = {64, 64};
  std::vector<ToastProps> variants{plain, iconic};
  ToastProps glowing = iconic;
  glowing.icon_glow = Rgb(0xE0B040);
  variants.push_back(glowing);
  ToastProps decorated = plain;
  decorated.title_cup = Rgb(0xC0C0C0);
  decorated.edge_bar = Rgb(0xFF0000);
  decorated.shine = 0.5f;
  variants.push_back(decorated);
  ToastProps longTitle = iconic;
  longTitle.title = LONG_TITLE;
  variants.push_back(longTitle);
  ToastProps wrapped = iconic;
  wrapped.title = WRAPPING_TITLE;
  wrapped.title_cup = Rgb(0xE0B040);
  wrapped.edge_bar = Rgb(0x3060FF);
  variants.push_back(wrapped);
  return variants;
}

// The colour channels without the alpha, for telling colours apart whatever their fade.
inline constexpr ImU32 RGB_MASK = 0x00ffffffu;
// How far two clip edges may differ and still be the same clip.
inline constexpr float CLIP_EPSILON = 0.001f;

// A clip rectangle as ImGui holds it, by its edges.
struct ClipEdges {
  float left;
  float top;
  float right;
  float bottom;
};

// The clip a draw command has, as edges.
inline ClipEdges EdgesOf(const ImVec4 &clip) { return {clip.x, clip.y, clip.z, clip.w}; }

// The clip a pushed rectangle `inner` makes inside `parent`, as ImGui intersects them.
inline ClipEdges Intersected(const ClipEdges &inner, const ClipEdges &parent) {
  return {std::max(inner.left, parent.left), std::max(inner.top, parent.top), std::min(inner.right, parent.right),
          std::min(inner.bottom, parent.bottom)};
}

// True when two clips agree on all four sides.
inline bool SameEdges(const ClipEdges &a, const ClipEdges &b) {
  return std::fabs(a.left - b.left) <= CLIP_EPSILON && std::fabs(a.top - b.top) <= CLIP_EPSILON &&
         std::fabs(a.right - b.right) <= CLIP_EPSILON && std::fabs(a.bottom - b.bottom) <= CLIP_EPSILON;
}

// What a drawn toast's clips are checked against.
struct ToastClipExpect {
  // The card as drawn, and its corner radius in pixels as the card draws it.
  Libs::Graphics::SystemUi::Rect card;
  float radius = 0.0f;
  // Whether the toast has an icon.
  bool has_icon = false;
  // The edge bar's colour, transparent for none.
  Libs::Graphics::SystemUi::Color edge_bar;
  // The glow behind the icon, transparent for none.
  Libs::Graphics::SystemUi::Color icon_glow;
  // Whether a shine is being drawn, which is drawn last and so ends the list.
  bool shine = false;
};

// The clip of the first command that draws a vertex of `rgb`, alpha ignored, or nothing.
inline std::optional<ClipEdges> FirstClipOfColour(const ImDrawList &list, ImU32 rgb) {
  for (const ImDrawCmd &command : list.CmdBuffer) {
    for (unsigned i = 0; i < command.ElemCount; i++) {
      const ImDrawVert &vertex = list.VtxBuffer[command.VtxOffset + list.IdxBuffer[command.IdxOffset + i]];
      if ((vertex.col & RGB_MASK) == (rgb & RGB_MASK)) {
        return EdgesOf(command.ClipRect);
      }
    }
  }
  return std::nullopt;
}

// Whether `vertex` samples a glyph or the icon, so it is text or image content and not a plain
// shape. A plain shape samples the draw list's white pixel, and a glyph or an image never does.
inline bool IsContent(const ImDrawList &list, const ImDrawVert &vertex) {
  const bool plain = vertex.uv.x == list._Data->TexUvWhitePixel.x && vertex.uv.y == list._Data->TexUvWhitePixel.y;
  return !plain;
}

// Looks over every draw command of a drawn toast and returns what is wrong with its clips, or
// nothing. The card clips its content, the glyphs and the icon, to the card inside its rounded
// corners, intersected with the clip it was drawn in. Everything else is drawn in that parent clip:
// the shadow, fill and border, the edge bar, and the glow, which spills past the card. Glyph and
// icon vertices must lie inside the clip. All four sides of every clip are compared.
inline const char *ToastClipFault(const ImDrawList &list, const ToastClipExpect &expect) {
  const auto &card = expect.card;
  const ImVec4 full = list._Data->ClipRectFullscreen;
  const ClipEdges parent = EdgesOf(full);
  const auto inside_rect = Libs::Graphics::SystemUi::InsideCorners(card, expect.radius);
  const ClipEdges inside = Intersected({inside_rect.x, inside_rect.y, inside_rect.Right(), inside_rect.Bottom()}, parent);
  bool any = false;
  for (const ImDrawCmd &command : list.CmdBuffer) {
    if (command.ElemCount == 0) {
      continue;
    }
    const ClipEdges clip = EdgesOf(command.ClipRect);
    if (!SameEdges(clip, parent) && !SameEdges(clip, inside)) {
      return "a command has a clip that is neither the parent's nor the card's inside";
    }
    for (unsigned i = 0; i < command.ElemCount; i++) {
      const ImDrawVert &vertex = list.VtxBuffer[command.VtxOffset + list.IdxBuffer[command.IdxOffset + i]];
      if (!IsContent(list, vertex)) {
        continue;
      }
      if (!SameEdges(clip, inside)) {
        return "text or the icon is drawn outside the card's content clip";
      }
      if (vertex.pos.x < inside.left - CLIP_EPSILON || vertex.pos.x > inside.right + CLIP_EPSILON ||
          vertex.pos.y < inside.top - CLIP_EPSILON || vertex.pos.y > inside.bottom + CLIP_EPSILON) {
        return "text or the icon reaches past the card's content clip";
      }
    }
    if (!any && !SameEdges(clip, parent)) {
      return "the shadow is not drawn in the parent clip";
    }
    any = true;
  }
  if (!any) {
    return "nothing was drawn";
  }
  const auto raised = FirstClipOfColour(list, Libs::Graphics::SystemUi::ToImU32(Libs::Graphics::SystemUi::DEFAULT_THEME.colors.surface.raised));
  const auto border = FirstClipOfColour(list, Libs::Graphics::SystemUi::ToImU32(Libs::Graphics::SystemUi::DEFAULT_THEME.colors.border.subtle));
  if (!raised || !SameEdges(*raised, parent) || !border || !SameEdges(*border, parent)) {
    return "the fill or border is not drawn in the parent clip";
  }
  if (expect.edge_bar.a > 0) {
    const auto bar = FirstClipOfColour(list, Libs::Graphics::SystemUi::ToImU32(expect.edge_bar));
    if (!bar || !SameEdges(*bar, parent)) {
      return "the edge bar is not drawn in the parent clip";
    }
  }
  if (expect.icon_glow.a > 0 && expect.has_icon) {
    const auto glow = FirstClipOfColour(list, Libs::Graphics::SystemUi::ToImU32(expect.icon_glow));
    if (!glow || !SameEdges(*glow, parent)) {
      return "the glow is not in the parent clip";
    }
  }
  if (expect.shine) {
    // The shine goes over the content and is clipped inside the card's corners with it, so the
    // last thing drawn is in the inner clip. Drawn in the parent clip it would spill over the corners.
    for (int i = list.CmdBuffer.Size - 1; i >= 0; --i) {
      if (list.CmdBuffer[i].ElemCount > 0) {
        return SameEdges(EdgesOf(list.CmdBuffer[i].ClipRect), inside) ? nullptr : "the shine is not drawn in the card's content clip";
      }
    }
  }
  return nullptr;
}

// The card's fill as drawn, found as the plain shapes in the raised surface colour, or nothing when
// the toast drew none, as at the start of its entrance.
inline std::optional<Libs::Graphics::SystemUi::Rect> DrawnCard(const ImDrawList &list) {
  const ImU32 raised = Libs::Graphics::SystemUi::ToImU32(Libs::Graphics::SystemUi::DEFAULT_THEME.colors.surface.raised);
  float left = 1e30f, top = 1e30f, right = -1e30f, bottom = -1e30f;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if ((vertex.col & RGB_MASK) == (raised & RGB_MASK) && !IsContent(list, vertex)) {
      left = std::min(left, vertex.pos.x);
      top = std::min(top, vertex.pos.y);
      right = std::max(right, vertex.pos.x);
      bottom = std::max(bottom, vertex.pos.y);
    }
  }
  if (left > right) {
    return std::nullopt;
  }
  return Libs::Graphics::SystemUi::Rect{left, top, right - left, bottom - top};
}

} // namespace SystemUiTest

#endif // EMULATOR_TESTS_SYSTEMUITOASTKIT_H_
