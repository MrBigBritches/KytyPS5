#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/layout/anchor.h"
#include "graphics/presentation/system/layout/flex.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::SystemUi::Align;
using Libs::Graphics::SystemUi::Anchor;
using Libs::Graphics::SystemUi::BelowEdge;
using Libs::Graphics::SystemUi::EdgeInsets;
using Libs::Graphics::SystemUi::FlexDirection;
using Libs::Graphics::SystemUi::FlexItem;
using Libs::Graphics::SystemUi::FlexMainLengths;
using Libs::Graphics::SystemUi::FlexScratch;
using Libs::Graphics::SystemUi::FlexStyle;
using Libs::Graphics::SystemUi::Justify;
using Libs::Graphics::SystemUi::Place;
using Libs::Graphics::SystemUi::Rect;
using Libs::Graphics::SystemUi::SolveFlex;
using Libs::Graphics::SystemUi::Vec2;

constexpr Rect ROW_BOX{0, 0, 300, 100};

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiLayoutTests: failed: %s\n", message);
    std::abort();
  }
}

// Three 50 by 20 items, the common fixture for the row cases.
std::vector<FlexItem> ThreeItems() { return {{.size = {50, 20}}, {.size = {50, 20}}, {.size = {50, 20}}}; }

// The x position of each rectangle.
std::vector<float> Lefts(const std::vector<Rect> &rects) {
  std::vector<float> lefts;
  for (const auto &rect : rects) {
    lefts.push_back(rect.x);
  }
  return lefts;
}

// The width of each rectangle.
std::vector<float> Widths(const std::vector<Rect> &rects) {
  std::vector<float> widths;
  for (const auto &rect : rects) {
    widths.push_back(rect.width);
  }
  return widths;
}

// Each justify mode places the same items differently. Start is the baseline, and every mode is
// compared with it, so a mode that falls back to Start fails.
void TestJustify() {
  const auto items = ThreeItems();
  const auto start = SolveFlex(ROW_BOX, {.justify = Justify::Start, .gap = 10}, items);
  Check(Lefts(start) == std::vector<float>{0, 60, 120}, "start packs items from the left");
  const auto center = SolveFlex(ROW_BOX, {.justify = Justify::Center, .gap = 10}, items);
  Check(Lefts(center) == std::vector<float>{65, 125, 185}, "center splits the 130 left over");
  const auto end = SolveFlex(ROW_BOX, {.justify = Justify::End, .gap = 10}, items);
  Check(Lefts(end) == std::vector<float>{130, 190, 250}, "end packs items to the right");
  const auto between = SolveFlex(ROW_BOX, {.justify = Justify::SpaceBetween, .gap = 10}, items);
  Check(Lefts(between) == std::vector<float>{0, 125, 250}, "space-between puts the rest between items");
  const auto single = SolveFlex(ROW_BOX, {.justify = Justify::SpaceBetween}, std::vector<FlexItem>{{.size = {50, 20}}});
  Check(Lefts(single) == std::vector<float>{0}, "a single item with space-between starts at the left");
}

// Growing items share the free space by their factors and no justify offset applies. The
// baseline is the same items without growth.
void TestGrow() {
  auto items = ThreeItems();
  Check(Widths(SolveFlex(ROW_BOX, {}, items)) == std::vector<float>{50, 50, 50}, "baseline without growth");
  items[0].grow = 1;
  items[1].grow = 3;
  const auto grown = SolveFlex(ROW_BOX, {.justify = Justify::End}, items);
  Check(Widths(grown) == std::vector<float>{88, 162, 50}, "150 free pixels split one to three");
  Check(Lefts(grown) == std::vector<float>{0, 88, 250}, "grown items fill from the start despite end");
}

// An item that reaches its maximum stops there, and the space it can't take goes to the other
// growing items. Clamping without sharing again would leave widths 60, 100 and 100.
void TestGrowCapped() {
  auto items = ThreeItems();
  for (auto &item : items) {
    item.grow = 1;
  }
  items[0].max_main = 60;
  const auto rects = SolveFlex(ROW_BOX, {}, items);
  Check(Widths(rects) == std::vector<float>{60, 120, 120}, "a capped item's share goes to the others");
}

// An item that collapses leaves when the others would be short of their minimums, and the others
// have the room. A 60 wide label beside a track that wants 56: a row 116 wide fits both exactly, one
// 115 wide drops the label and gives the track the lot. The same label that does not collapse is the
// baseline, which keeps its 60 and runs the row over. With no length known nothing collapses.
void TestCollapse() {
  const auto build = [](bool collapses) {
    return std::vector<FlexItem>{{.size = {0, 20}, .grow = 1, .min_main = 56}, {.size = {60, 20}, .collapses = collapses}};
  };
  const auto fits = SolveFlex({0, 0, 116, 100}, {}, build(true));
  Check(Widths(fits) == std::vector<float>{56, 60}, "a label that leaves the track its minimum stays");
  const auto short_row = SolveFlex({0, 0, 115, 100}, {}, build(true));
  Check(Widths(short_row) == std::vector<float>{115, 0}, "a label that does not collapses and the track takes the row");
  const auto baseline = SolveFlex({0, 0, 115, 100}, {}, build(false));
  Check(Widths(baseline) == std::vector<float>{56, 60}, "without collapsing the label stays and the row runs over");
  std::vector<float> lengths(2);
  const auto items = build(true);
  Check(Libs::Graphics::SystemUi::FlexMainLengths(std::numeric_limits<float>::infinity(), FlexDirection::Row, 0, items, lengths) &&
            lengths[1] == 60,
        "with no length known nothing collapses");
}

// A collapsed item gives back the gap beside it, wherever it sits, and a gap is only between two
// visible items. Three items 50 wide with a gap of 10 in a row 150 wide, one of them collapsing, leave
// the other two one gap apart. The collapsing one first, in the middle and last each gives the same
// 110 of items and gap, and its zero width sits where the next visible item starts. The baseline that
// does not collapse keeps both gaps.
void TestCollapseGivesBackItsGap() {
  const auto build = [](size_t collapsing, bool collapses) {
    std::vector<FlexItem> items{{.size = {50, 20}, .min_main = 50}, {.size = {50, 20}, .min_main = 50}, {.size = {50, 20}, .min_main = 50}};
    items[collapsing].collapses = collapses;
    items[collapsing].min_main = 0;
    items[collapsing].size = {60, 20};
    return items;
  };
  const Rect row{0, 0, 150, 100};
  const auto first = SolveFlex(row, {.gap = 10}, build(0, true));
  Check(Widths(first) == std::vector<float>{0, 50, 50} && Lefts(first) == std::vector<float>{0, 0, 60}, "a collapsing item first leaves no gap before the others");
  const auto middle = SolveFlex(row, {.gap = 10}, build(1, true));
  Check(Widths(middle) == std::vector<float>{50, 0, 50} && Lefts(middle) == std::vector<float>{0, 60, 60}, "in the middle it leaves one gap between them");
  const auto last = SolveFlex(row, {.gap = 10}, build(2, true));
  Check(Widths(last) == std::vector<float>{50, 50, 0} && Lefts(last) == std::vector<float>{0, 60, 120}, "last it leaves no gap after them");
  const auto kept = SolveFlex(row, {.gap = 10}, build(1, false));
  Check(Lefts(kept) == std::vector<float>{0, 60, 130}, "while without collapsing both gaps stay");
}

// A collapsing item first beside a growing item gives the grower the whole row. In a row 100 wide with
// a gap of 10 the track starts at 0 and ends at 100, not at 10 and 110, and the lengths the measure
// finds are the lengths the layout uses.
void TestCollapseFirstBesideAGrower() {
  const std::vector<FlexItem> items{{.size = {60, 20}, .collapses = true}, {.size = {0, 20}, .grow = 1, .min_main = 56}};
  const auto rects = SolveFlex({0, 0, 100, 100}, {.gap = 10}, items);
  Check(Widths(rects) == std::vector<float>{0, 100} && Lefts(rects) == std::vector<float>{0, 0}, "the grower takes the whole row from its start");
  std::vector<float> lengths(2);
  std::vector<bool> collapsed;
  Check(Libs::Graphics::SystemUi::FlexMainLengths(100, FlexDirection::Row, 10, items, lengths, &collapsed) && lengths == std::vector<float>{0, 100} &&
            collapsed.size() == 2 && collapsed[0] && !collapsed[1],
        "the measure finds the same lengths and says which item collapsed");
}

// A collapsed item takes no share of the space justify spreads. Two items 50 wide with a collapsing
// one between them in a row 150 wide, space-between, put the outer two at the ends of the row and not
// at the thirds that counting the collapsed one would give.
void TestCollapseTakesNoJustifyShare() {
  std::vector<FlexItem> items{{.size = {50, 20}, .min_main = 50}, {.size = {60, 20}, .collapses = true}, {.size = {50, 20}, .min_main = 50}};
  const auto rects = SolveFlex({0, 0, 120, 100}, {.justify = Justify::SpaceBetween}, items);
  Check(Widths(rects) == std::vector<float>{50, 0, 50} && rects[0].x == 0 && rects[2].x == 70, "the visible items are spread across the row");
}

// An item never measures less than its minimum or more than its maximum, even without growth.
void TestMinimumAndMaximum() {
  std::vector<FlexItem> items = {{.size = {20, 20}, .min_main = 40}, {.size = {90, 20}, .max_main = 70}};
  Check(Widths(SolveFlex(ROW_BOX, {}, items)) == std::vector<float>{40, 70}, "base sizes are limited");
}

// Padding shrinks the box the items are laid out in, on both axes.
void TestPadding() {
  const auto rects = SolveFlex(ROW_BOX, {.align = Align::Stretch, .padding = EdgeInsets::All(10)}, ThreeItems());
  Check(rects[0] == Rect{10, 10, 50, 80}, "padding moves the first item and shortens the cross axis");
}

// Each cross-axis alignment, with Start as the baseline, and a per-item override.
void TestAlign() {
  const std::vector<FlexItem> item = {{.size = {50, 20}}};
  Check(SolveFlex(ROW_BOX, {.align = Align::Start}, item)[0] == Rect{0, 0, 50, 20}, "start aligns to the top");
  Check(SolveFlex(ROW_BOX, {.align = Align::Center}, item)[0] == Rect{0, 40, 50, 20}, "center aligns to the middle");
  Check(SolveFlex(ROW_BOX, {.align = Align::End}, item)[0] == Rect{0, 80, 50, 20}, "end aligns to the bottom");
  Check(SolveFlex(ROW_BOX, {.align = Align::Stretch}, item)[0] == Rect{0, 0, 50, 100}, "stretch fills the height");
  const std::vector<FlexItem> overridden = {{.size = {50, 20}, .align_self = Align::End}};
  Check(SolveFlex(ROW_BOX, {.align = Align::Start}, overridden)[0].y == 80, "an item can override the alignment");
}

// A column lays items out top to bottom, and the cross axis is horizontal.
void TestColumn() {
  const Rect box{0, 0, 100, 300};
  const std::vector<FlexItem> items = {{.size = {20, 50}}, {.size = {20, 50}}};
  const auto rects = SolveFlex(box, {.direction = FlexDirection::Column, .align = Align::Center, .gap = 10}, items);
  Check(rects[0] == Rect{40, 0, 20, 50}, "the first column item is at the top, centred across");
  Check(rects[1] == Rect{40, 60, 20, 50}, "the second column item follows the gap downward");
}

// Items that don't fit keep their sizes and run past the end, even when justified to the centre
// or end, so their start never moves before the container.
void TestOverflow() {
  const std::vector<FlexItem> items = {{.size = {200, 20}}, {.size = {200, 20}}};
  for (const auto justify : {Justify::Start, Justify::Center, Justify::End}) {
    const auto rects = SolveFlex(ROW_BOX, {.justify = justify, .gap = 10}, items);
    Check(Lefts(rects) == std::vector<float>{0, 210}, "overflowing items start at the container's start");
  }
}

// Edges snap to whole pixels, and neighbours share them. Three equal growers in 100 pixels
// would be 33.33 wide and start at fractional positions without snapping.
void TestSnapping() {
  std::vector<FlexItem> items(3, FlexItem{.grow = 1});
  const auto rects = SolveFlex({0.4f, 0, 100, 10}, {}, items);
  Check(Lefts(rects) == std::vector<float>{0, 34, 67}, "item edges are whole pixels");
  Check(Widths(rects) == std::vector<float>{34, 33, 33}, "widths are whole pixels");
  Check(rects[0].Right() == rects[1].x && rects[1].Right() == rects[2].x, "neighbours share an edge");
}

// Six equal growers in a box starting at x = 1. The true edge between the third and fourth
// sits at 48.5, and summing it in two different orders gives 48.49999 and 48.5, which round
// apart and leave a hairline gap. One running edge makes them the same float.
void TestNoHairlineGaps() {
  std::vector<FlexItem> items(6, FlexItem{.grow = 1});
  const auto rects = SolveFlex({1, 0, 95, 10}, {}, items);
  for (size_t i = 0; i + 1 < rects.size(); i++) {
    Check(rects[i].Right() == rects[i + 1].x, "each item starts exactly where the previous one ends");
  }
  Check(rects.front().x == 1 && rects.back().Right() == 96, "the items fill the box edge to edge");
}

// A span keeps its length wherever it sits. Rounding halves away from zero would make the span
// from -10.5 to 9.5 one pixel longer than the same span from 0.5 to 20.5.
void TestSnapIsTranslationInvariant() {
  using Libs::Graphics::SystemUi::SnapPixel;
  Check(SnapPixel(9.5f) - SnapPixel(-10.5f) == SnapPixel(20.5f) - SnapPixel(0.5f), "a span keeps its length below 0");
  std::vector<FlexItem> items(2, FlexItem{.size = {20, 10}});
  const auto above = SolveFlex({0.5f, 0, 100, 10}, {}, items);
  const auto below = SolveFlex({-10.5f, 0, 100, 10}, {}, items);
  Check(Widths(above) == Widths(below), "flex widths don't change when the box moves below 0");
}

// The freeze loop's edge cases. A minimum above the maximum wins. A grower already at its
// maximum stays there and its share goes to the other grower. When every grower is capped the
// space left over is placed by justify, which the end offset shows.
void TestGrowEdgeCases() {
  const std::vector<FlexItem> inverted = {{.size = {20, 10}, .grow = 1, .min_main = 40, .max_main = 30}};
  Check(Widths(SolveFlex(ROW_BOX, {}, inverted)) == std::vector<float>{40}, "a minimum above the maximum wins");
  const std::vector<FlexItem> at_max = {{.size = {60, 10}, .grow = 1, .max_main = 60},
                                        {.size = {50, 10}, .grow = 1}};
  Check(Widths(SolveFlex(ROW_BOX, {}, at_max)) == std::vector<float>{60, 240}, "a grower at its maximum stays there");
  const std::vector<FlexItem> all_capped = {{.size = {50, 10}, .grow = 1, .max_main = 80},
                                            {.size = {50, 10}, .grow = 1, .max_main = 80}};
  const auto capped = SolveFlex(ROW_BOX, {.justify = Justify::End}, all_capped);
  Check(Widths(capped) == std::vector<float>{80, 80}, "every grower stops at its maximum");
  Check(Lefts(capped) == std::vector<float>{140, 220}, "the space every grower refused is justified");
}

// Values that are not finite are made safe instead of spreading NaN to the other items. Each
// case is compared with the same layout given the safe value directly.
void TestNonFiniteItems() {
  constexpr float NOT_A_NUMBER = std::numeric_limits<float>::quiet_NaN();
  constexpr float POSITIVE_INFINITY = std::numeric_limits<float>::infinity();
  const std::vector<FlexItem> nan_size = {{.size = {NOT_A_NUMBER, 10}}, {.size = {50, 10}}};
  Check(Lefts(SolveFlex(ROW_BOX, {}, nan_size)) == std::vector<float>{0, 0}, "a NaN width counts as zero");
  const std::vector<FlexItem> infinite_grow = {{.size = {50, 10}, .grow = POSITIVE_INFINITY}, {.size = {50, 10}}};
  Check(Widths(SolveFlex(ROW_BOX, {}, infinite_grow)) == std::vector<float>{50, 50}, "an infinite grow counts as zero");
  const std::vector<FlexItem> nan_max = {{.size = {50, 10}, .grow = 1, .min_main = 20, .max_main = NOT_A_NUMBER}};
  Check(Widths(SolveFlex(ROW_BOX, {}, nan_max)) == std::vector<float>{300}, "a NaN maximum means no maximum");
}

// An item taller than a row starts at the top whatever the alignment, as an overflowing row
// starts at the left. Centring it would move it above the container.
void TestCrossOverflow() {
  const std::vector<FlexItem> tall = {{.size = {50, 140}}};
  for (const auto align : {Align::Start, Align::Center, Align::End}) {
    Check(SolveFlex(ROW_BOX, {.align = align}, tall)[0].y == 0, "an overflowing item starts at the cross start");
  }
}

// Place snaps the far edges too, so a fractional measured size gives whole-pixel edges. The
// area starts at 0.3, so the right edge at 200.7 snaps to 201 and the width is 201, where
// rounding the size alone would give 200.
void TestPlaceSnapsFarEdges() {
  const auto placed = Place({0.3f, 0, 1000, 500}, {200.4f, 100.6f}, Anchor::TopLeft);
  Check(placed == Rect{0, 0, 201, 101}, "a fractional size snaps to whole-pixel edges");
}

// A NaN edge leaves the area whole rather than producing a NaN rectangle.
void TestBelowEdgeNaN() {
  const Rect area{0, 100, 400, 300};
  Check(BelowEdge(area, std::numeric_limits<float>::quiet_NaN()) == area, "a NaN edge leaves the area whole");
}

// The span overload agrees with the vector version across justify, grow, align and padding, and
// its results are also checked against fixed numbers, so both can't drift together unnoticed. A
// short output is refused without being written, and a long one keeps its extra rectangles.
void TestSpanOverload() {
  auto grown = ThreeItems();
  grown[0].grow = 1;
  grown[1].grow = 3;
  const std::vector<std::pair<FlexStyle, std::vector<FlexItem>>> cases = {
      {{.justify = Justify::Center, .gap = 10}, ThreeItems()},
      {{.justify = Justify::SpaceBetween}, ThreeItems()},
      {{.align = Align::End, .gap = 4, .padding = EdgeInsets::All(10)}, grown},
      {{.direction = FlexDirection::Column, .align = Align::Center, .gap = 10}, ThreeItems()},
  };
  for (const auto &[style, items] : cases) {
    std::vector<Rect> out(items.size());
    Check(SolveFlex(ROW_BOX, style, items, out), "an output of the right size is accepted");
    Check(out == SolveFlex(ROW_BOX, style, items), "the span overload matches the vector version");
  }
  std::vector<Rect> centred(3);
  Check(SolveFlex(ROW_BOX, cases[0].first, cases[0].second, centred), "centre case accepted");
  Check(Lefts(centred) == std::vector<float>{65, 125, 185}, "the span overload applies the justify offset");
  std::vector<Rect> padded(3);
  Check(SolveFlex(ROW_BOX, cases[2].first, grown, padded), "grow case accepted");
  Check(padded[0] == SolveFlex(ROW_BOX, cases[2].first, grown)[0] && padded[0].x == 10, "and the padding");

  const Rect untouched{-1, -1, -1, -1};
  std::vector<Rect> shortOut(2, untouched);
  Check(!SolveFlex(ROW_BOX, {}, ThreeItems(), shortOut), "an output that is too short is refused");
  Check(shortOut[0] == untouched && shortOut[1] == untouched, "and left untouched");
  std::vector<Rect> longOut(4, untouched);
  Check(SolveFlex(ROW_BOX, {}, ThreeItems(), longOut), "a longer output is accepted");
  Check(longOut[3] == untouched && longOut[2] != untouched, "and only the first rectangles are written");
}

// Items with a shrink factor give up length when they overflow, in proportion to their size and
// down to their minimums. Without a factor the same items overflow, which is the baseline and the
// behaviour the other tests rely on.
void TestShrink() {
  std::vector<FlexItem> items = {{.size = {150, 20}}, {.size = {150, 20}}};
  const Rect box{0, 0, 200, 40};
  Check(Widths(SolveFlex(box, {}, items)) == std::vector<float>{150, 150}, "without a factor items overflow");
  items[0].shrink = 1;
  items[1].shrink = 1;
  Check(Widths(SolveFlex(box, {}, items)) == std::vector<float>{100, 100}, "equal items shrink evenly");
  items[1].min_main = 120;
  Check(Widths(SolveFlex(box, {}, items)) == std::vector<float>{80, 120}, "a minimum stops one and the rest give more");
  std::vector<FlexItem> uneven = {{.size = {300, 20}, .shrink = 1}, {.size = {100, 20}, .shrink = 1}};
  Check(Widths(SolveFlex(box, {}, uneven)) == std::vector<float>{150, 50}, "shrinking follows each item's size");
  std::vector<FlexItem> stiff = {{.size = {150, 20}, .shrink = 1}, {.size = {150, 20}}};
  Check(Widths(SolveFlex(box, {}, stiff)) == std::vector<float>{50, 150}, "an item with no factor keeps its size");
}

// The main lengths the solver would give, asked for without placing anything, are the ones
// SolveFlex uses. A growing and a shrinking case are compared, and an unbounded space leaves the
// preferred lengths alone, where a limit would grow the grower.
void TestFlexMainLengths() {
  using Libs::Graphics::SystemUi::FlexMainLengths;
  std::vector<FlexItem> grow = {{.size = {100, 20}, .grow = 1}, {.size = {0, 20}, .grow = 1}};
  std::vector<float> lengths(2);
  Check(FlexMainLengths(200, FlexDirection::Row, 0, grow, lengths), "grow lengths are found");
  Check(lengths == Widths(SolveFlex({0, 0, 200, 20}, {}, grow)), "and match the solver");
  Check(lengths == std::vector<float>{150, 50}, "at the stated split");
  std::vector<FlexItem> shrink = {{.size = {150, 20}, .shrink = 1}, {.size = {150, 20}, .shrink = 1}};
  Check(FlexMainLengths(210, FlexDirection::Row, 10, shrink, lengths), "shrink lengths are found");
  Check(lengths == std::vector<float>{100, 100}, "with the gap taken out of the space");
  Check(FlexMainLengths(std::numeric_limits<float>::infinity(), FlexDirection::Row, 0, grow, lengths), "unbounded");
  Check(lengths == std::vector<float>{100, 0}, "unbounded space leaves preferred lengths");
  std::vector<float> tooShort(1);
  Check(!FlexMainLengths(200, FlexDirection::Row, 0, grow, tooShort), "a short output is refused");
}

// No items give no rectangles.
void TestEmpty() { Check(SolveFlex(ROW_BOX, {}, std::vector<FlexItem>{}).empty(), "no items give no rectangles"); }

// Every anchor lands at its own position, so no two are swapped.
void TestPlace() {
  const Rect area{100, 50, 1000, 500};
  const Vec2 size{200, 100};
  Check(Place(area, size, Anchor::TopLeft) == Rect{100, 50, 200, 100}, "top left");
  Check(Place(area, size, Anchor::Top) == Rect{500, 50, 200, 100}, "top");
  Check(Place(area, size, Anchor::TopRight) == Rect{900, 50, 200, 100}, "top right");
  Check(Place(area, size, Anchor::Left) == Rect{100, 250, 200, 100}, "left");
  Check(Place(area, size, Anchor::Center) == Rect{500, 250, 200, 100}, "center");
  Check(Place(area, size, Anchor::Right) == Rect{900, 250, 200, 100}, "right");
  Check(Place(area, size, Anchor::BottomLeft) == Rect{100, 450, 200, 100}, "bottom left");
  Check(Place(area, size, Anchor::Bottom) == Rect{500, 450, 200, 100}, "bottom");
  Check(Place(area, size, Anchor::BottomRight) == Rect{900, 450, 200, 100}, "bottom right");
  Check(Place({100, 50, 1001, 501}, size, Anchor::Center) == Rect{501, 251, 200, 100}, "a centred corner is snapped");
}

// The space below another panel's edge, for stacking. The edge is limited to the area.
void TestBelowEdge() {
  const Rect area{0, 100, 400, 300};
  Check(BelowEdge(area, 180) == Rect{0, 180, 400, 220}, "the area below an edge inside it");
  Check(BelowEdge(area, 20) == area, "an edge above the area leaves it whole");
  Check(BelowEdge(area, 900) == Rect{0, 400, 400, 0}, "an edge below the area leaves it empty");
}

// Rectangle helpers. Insets never make a size negative, and containment counts the top and
// left edges only.
static_assert(Rect{10, 20, 30, 40}.Inset(EdgeInsets::All(5)) == Rect{15, 25, 20, 30}, "insets shrink each side");
static_assert(Rect{10, 20, 30, 40}.Inset(EdgeInsets::Symmetric(20, 0)) == Rect{30, 20, 0, 40},
              "an inset wider than the rectangle leaves it empty");
static_assert(Rect{0, 0, 10, 10}.Contains({0, 0}) && !Rect{0, 0, 10, 10}.Contains({10, 5}),
              "the left edge is inside and the right edge is not");
static_assert(Rect{0, 0, 10, 20}.Center() == Vec2{5, 10}, "the centre is half the size in");

} // namespace

namespace {

// Scratch used for a solve keeps nothing of it. Each case is solved in turn with one scratch and one
// set of flags, big ones before small ones and a collapse before none, and has to give what a fresh
// solve of the same case gives, which is the baseline, as the fresh form shares no storage with it.
void TestReusedScratchMatchesFresh() {
  struct Case {
    Rect container;
    FlexStyle style;
    std::vector<FlexItem> items;
  };
  FlexItem grower;
  grower.size = {50, 20};
  grower.grow = 1.0f;
  FlexItem capped = grower;
  capped.max_main = 60.0f;
  FlexItem shrinker;
  shrinker.size = {200, 20};
  shrinker.shrink = 1.0f;
  shrinker.min_main = 40.0f;
  FlexItem fixed;
  fixed.size = {30, 20};
  fixed.min_main = 30.0f;
  FlexItem collapsing;
  collapsing.size = {80, 20};
  collapsing.collapses = true;
  const std::vector<Case> cases{
      {{0, 0, 400, 50}, {.gap = 8.0f}, {grower, capped, grower, fixed, grower}},
      {{0, 0, 100, 50}, {.gap = 4.0f}, {fixed, collapsing}},
      {{0, 0, 100, 50}, {.gap = 4.0f}, {fixed}},
      {{0, 0, 150, 50}, {}, {shrinker, shrinker, fixed}},
      {{0, 0, 300, 50}, {.justify = Justify::SpaceBetween}, {fixed, fixed}},
      {{0, 0, 200, 300}, {.direction = FlexDirection::Column, .align = Align::Stretch}, {grower, fixed}},
  };
  FlexScratch scratch;
  std::vector<uint8_t> flags(8, 1);
  for (const Case &test : cases) {
    std::vector<Rect> reused(test.items.size());
    std::fill(flags.begin(), flags.end(), uint8_t{1});
    Check(SolveFlex(test.container, test.style, test.items, reused, flags, scratch), "the solve fits its storage");
    std::vector<bool> fresh_collapsed;
    std::vector<Rect> fresh(test.items.size());
    Check(SolveFlex(test.container, test.style, test.items, fresh, &fresh_collapsed), "and so does a fresh one");
    std::vector<float> reused_lengths(test.items.size());
    std::vector<float> fresh_lengths(test.items.size());
    std::vector<uint8_t> length_flags(test.items.size(), 1);
    const bool row = test.style.direction == FlexDirection::Row;
    const float inner = row ? test.container.width : test.container.height;
    Check(FlexMainLengths(inner, test.style.direction, test.style.gap, test.items, reused_lengths, length_flags, scratch), "the lengths fit too");
    Check(FlexMainLengths(inner, test.style.direction, test.style.gap, test.items, fresh_lengths), "fresh as well");
    for (size_t i = 0; i < test.items.size(); ++i) {
      Check(reused[i].x == fresh[i].x && reused[i].y == fresh[i].y && reused[i].width == fresh[i].width && reused[i].height == fresh[i].height,
            "a reused scratch places each item as a fresh solve does");
      const bool was_collapsed = !fresh_collapsed.empty() && fresh_collapsed[i];
      Check((flags[i] == 1) == was_collapsed, "and reports the same collapses, whatever the flags held before");
      Check((length_flags[i] == 1) == was_collapsed, "as the lengths do");
      Check(reused_lengths[i] == fresh_lengths[i], "and gives the same lengths");
    }
  }
  std::vector<Rect> collapse_check(2);
  std::vector<bool> collapsed;
  Check(SolveFlex({0, 0, 100, 50}, {.gap = 4.0f}, cases[1].items, collapse_check, &collapsed) && collapsed.size() == 2 && collapsed[1],
        "one of the cases really collapses, so the flags are tested both ways");
}

// The forms that report through a vector replace what it held. It says which items collapsed, one flag
// each, or it is left empty when none did, so whatever was in it doesn't read as a collapse.
void TestCollapseReportReplacesItsVector() {
  const std::vector<FlexItem> collapsing{{.size = {60, 20}, .collapses = true}, {.size = {0, 20}, .min_main = 56}};
  const std::vector<FlexItem> roomy{{.size = {10, 20}, .collapses = true}, {.size = {10, 20}}};
  std::vector<Rect> rects(2);
  std::vector<float> lengths(2);
  std::vector<bool> report(5, true);
  Check(SolveFlex({0, 0, 400, 50}, {}, roomy, rects, &report) && report.empty(), "no collapse leaves the solve's report empty");
  report.assign(5, true);
  Check(FlexMainLengths(400, FlexDirection::Row, 0, roomy, lengths, &report) && report.empty(), "and the lengths' report");
  report.assign(5, true);
  Check(SolveFlex({0, 0, 100, 50}, {}, collapsing, rects, &report) && report.size() == 2 && report[0] && !report[1], "a collapse gives one flag per item");
  report.assign(5, true);
  Check(FlexMainLengths(100, FlexDirection::Row, 0, collapsing, lengths, &report) && report.size() == 2 && report[0] && !report[1], "for the lengths too");
}

// Storage that is too short is refused and left as it was, whichever part is short.
void TestScratchFormsRefuseShortStorage() {
  FlexItem item;
  item.size = {10, 10};
  const std::vector<FlexItem> items{item, item};
  FlexScratch scratch;
  std::vector<Rect> rects(2, Rect{1, 2, 3, 4});
  std::vector<Rect> short_rects(1, Rect{1, 2, 3, 4});
  std::vector<uint8_t> flags(2, 7);
  std::vector<uint8_t> short_flags(1, 7);
  std::vector<float> lengths(2, 9.0f);
  std::vector<float> short_lengths(1, 9.0f);
  Check(!SolveFlex({0, 0, 100, 50}, {}, items, short_rects, flags, scratch) && short_rects[0].x == 1 && flags[0] == 7, "short rectangles are refused");
  Check(!SolveFlex({0, 0, 100, 50}, {}, items, rects, short_flags, scratch) && rects[0].x == 1 && short_flags[0] == 7, "short flags are refused");
  Check(!FlexMainLengths(100, FlexDirection::Row, 0, items, short_lengths, flags, scratch) && short_lengths[0] == 9.0f && flags[0] == 7, "short lengths are refused");
  Check(!FlexMainLengths(100, FlexDirection::Row, 0, items, lengths, short_flags, scratch) && lengths[0] == 9.0f && short_flags[0] == 7, "so are short flags for lengths");
  Check(SolveFlex({0, 0, 100, 50}, {}, items, rects, flags, scratch) && rects[0].x == 0 && flags[0] == 0, "and storage that fits is used");
}

} // namespace

int main() {
  TestReusedScratchMatchesFresh();
  TestScratchFormsRefuseShortStorage();
  TestCollapseReportReplacesItsVector();
  TestJustify();
  TestGrow();
  TestGrowCapped();
  TestCollapse();
  TestCollapseGivesBackItsGap();
  TestCollapseFirstBesideAGrower();
  TestCollapseTakesNoJustifyShare();
  TestMinimumAndMaximum();
  TestPadding();
  TestAlign();
  TestColumn();
  TestOverflow();
  TestSnapping();
  TestNoHairlineGaps();
  TestSnapIsTranslationInvariant();
  TestGrowEdgeCases();
  TestNonFiniteItems();
  TestCrossOverflow();
  TestPlaceSnapsFarEdges();
  TestBelowEdgeNaN();
  TestSpanOverload();
  TestShrink();
  TestFlexMainLengths();
  TestEmpty();
  TestPlace();
  TestBelowEdge();
  std::puts("System UI layout tests passed");
  return 0;
}
