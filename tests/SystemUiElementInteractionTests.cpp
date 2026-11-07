#include "SystemUiElementTestHooks.h"
#include "SystemUiElementTestKit.h"
#include "SystemUiDrawingKit.h"
#include "graphics/presentation/system/components/box.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/list.h"
#include "graphics/presentation/system/components/toggle.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/frameInternals.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/interactContext.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics::SystemUi;
using SystemUiTest::Check;
using SystemUiTest::FrameRig;
using SystemUiTest::NavHarness;
using SystemUiTest::NearDrawing;

// Runs `body` with a begun frame inside an ImGui window, where ids have a seed to chain from.
template <class Body> void InWindow(Body &&body) {
  NavHarness nav;
  nav.Frame([&](InteractContext &) {
    FrameRig rig;
    rig.Begin();
    body(rig.frame);
  });
}

// A childless box.
Element Leaf(UiFrame &frame) { return Box(frame, {}, {}); }

// A box with `kids`, whose keys the caller has set.
Element Parent(UiFrame &frame, std::span<const Element> kids, ElementError *reason = nullptr) {
  return Box(frame, {}, {}, kids, reason);
}

// Every child's id is the ImGui id of its index seeded with its parent's id, and the root's is the
// id of index 0 seeded with the seed it was given. Siblings differ, so the index is part of it.
void TestChildIdIsIndexSeededByParent() {
  InWindow([](UiFrame &frame) {
    const std::array<Element, 3> kids{Leaf(frame), Leaf(frame), Leaf(frame)};
    const Element root = Parent(frame, kids);
    const ItemId seed = CurrentIdSeed();
    Check(IdOf(frame, root) == 0 && IdOf(frame, kids[0]) == 0, "ids are 0 until assigned");
    Check(!AssignIds(frame, root, seed).has_value(), "the ids assign");
    Check(IdOf(frame, root) == ImGui::GetIDWithSeed(0, seed), "the root is index 0 under the seed");
    for (int i = 0; i < 3; ++i) {
      Check(IdOf(frame, kids[i]) == ImGui::GetIDWithSeed(i, IdOf(frame, root)), "a child is its index under its parent");
    }
    Check(IdOf(frame, kids[0]) != IdOf(frame, kids[1]) && IdOf(frame, kids[1]) != IdOf(frame, kids[2]), "siblings differ");
  });
}

// A root keyed "ok" gets exactly the id IdFor gives "ok" in this window. The unkeyed root is the
// baseline, which does not.
void TestRootKeyMatchesIdFor() {
  InWindow([](UiFrame &frame) {
    const Element keyed = Keyed(frame, Leaf(frame), "ok");
    const Element unkeyed = Leaf(frame);
    Check(!AssignIds(frame, keyed, CurrentIdSeed()).has_value(), "the ids assign");
    Check(!AssignIds(frame, unkeyed, CurrentIdSeed()).has_value(), "the ids assign");
    Check(IdOf(frame, keyed) == IdFor("ok"), "a keyed root has the id IdFor gives");
    Check(IdOf(frame, unkeyed) != IdFor("ok"), "an unkeyed root does not");
    const Element numbered = Keyed(frame, Leaf(frame), 7u);
    Check(!AssignIds(frame, numbered, CurrentIdSeed()).has_value(), "the ids assign");
    Check(IdOf(frame, numbered) == ImGui::GetID(7), "a number key matches PushID's integer id");
  });
}

// Builds a parent of leaves keyed in the order `keys` names them, or of unkeyed leaves when `keyed`
// is false, assigns ids, and returns each child's id in order.
std::vector<ItemId> ChildIds(UiFrame &frame, const std::vector<std::string> &keys, bool keyed) {
  std::vector<Element> kids;
  for (const std::string &key : keys) {
    kids.push_back(keyed ? Keyed(frame, Leaf(frame), key) : Leaf(frame));
  }
  const Element root = Parent(frame, kids);
  Check(!AssignIds(frame, root, CurrentIdSeed()).has_value(), "the ids assign");
  std::vector<ItemId> ids;
  for (const Element kid : kids) {
    ids.push_back(IdOf(frame, kid));
  }
  return ids;
}

// Reordering the data moves a keyed child's id with it and leaves an index id at its position.
// The unkeyed children are the baseline, which behave the other way round.
void TestReorderingFollowsData() {
  NavHarness nav;
  FrameRig rig;
  nav.Frame([&](InteractContext &) {
    const std::vector<std::string> before{"a", "b", "c"};
    const std::vector<std::string> after{"c", "a", "b"};
    rig.Begin();
    const auto keyedBefore = ChildIds(rig.frame, before, true);
    rig.Begin();
    const auto keyedAfter = ChildIds(rig.frame, after, true);
    Check(keyedBefore[0] == keyedAfter[1] && keyedBefore[1] == keyedAfter[2] && keyedBefore[2] == keyedAfter[0], "keyed ids follow the data");
    Check(keyedBefore[0] != keyedAfter[0], "and not the position");
    rig.Begin();
    const auto indexBefore = ChildIds(rig.frame, before, false);
    rig.Begin();
    const auto indexAfter = ChildIds(rig.frame, after, false);
    Check(indexBefore == indexAfter, "index ids follow the position");
    Check(indexBefore[0] != keyedBefore[0], "and differ from keyed ones");
  });
}

// Siblings that repeat a key are refused with DuplicateKey, for text and for numbers, and the
// refusal frees the children for another parent. A text key and a number key never match, and
// distinct keys are the baseline, accepted.
void TestDuplicateKeysRefused() {
  InWindow([](UiFrame &frame) {
    ElementError reason = ElementError::None;
    const std::array<Element, 2> distinct{Keyed(frame, Leaf(frame), "x"), Keyed(frame, Leaf(frame), "y")};
    Check(!Parent(frame, distinct, &reason).IsNone(), "distinct keys are accepted");

    const std::array<Element, 3> repeated{Keyed(frame, Leaf(frame), "x"), Leaf(frame), Keyed(frame, Leaf(frame), "x")};
    Check(Parent(frame, repeated, &reason).IsNone() && reason == ElementError::DuplicateKey, "a repeated text key is refused");
    const std::array<Element, 2> sameNumber{Keyed(frame, Leaf(frame), 4u), Keyed(frame, Leaf(frame), 4u)};
    reason = ElementError::None;
    Check(Parent(frame, sameNumber, &reason).IsNone() && reason == ElementError::DuplicateKey, "and a repeated number key");
    const std::array<Element, 2> mixed{Keyed(frame, Leaf(frame), "4"), Keyed(frame, Leaf(frame), 4u)};
    Check(!Parent(frame, mixed).IsNone(), "a text key and a number key do not match");

    const std::array<Element, 2> freed{repeated[0], repeated[1]};
    Check(!Parent(frame, freed).IsNone(), "a refused parent leaves its children free");
  });
}

// Keyed refuses an empty key, an element that already has a parent and a none handle.
void TestKeyedRefusals() {
  InWindow([](UiFrame &frame) {
    ElementError reason = ElementError::None;
    const Element leaf = Leaf(frame);
    Check(Keyed(frame, leaf, "", &reason).IsNone() && reason == ElementError::InvalidKey, "an empty key is refused");
    const std::array<Element, 1> kids{leaf};
    (void)Parent(frame, kids);
    Check(Keyed(frame, leaf, "late", &reason).IsNone() && reason == ElementError::AlreadyParented, "a parented element is refused");
    reason = ElementError::InvalidKey;
    Check(Keyed(frame, Element{}, "k", &reason).IsNone() && reason == ElementError::None, "a none handle is refused");
  });
}

// Three keyed levels get the id the same names nested in PushID give, a child at an index the id
// of PushID of that number, and a component called under PushOverrideID(IdOf(node)) makes the ids
// it always did.
void TestKeysMatchPushIdNesting() {
  InWindow([](UiFrame &frame) {
    const Element grand = Keyed(frame, Leaf(frame), "c");
    const std::array<Element, 1> grandKids{grand};
    const Element child = Keyed(frame, Parent(frame, grandKids), "b");
    const std::array<Element, 1> childKids{child};
    const Element root = Keyed(frame, Parent(frame, childKids), "a");
    Check(!AssignIds(frame, root, CurrentIdSeed()).has_value(), "the ids assign");
    ImGui::PushID("a");
    ImGui::PushID("b");
    ImGui::PushID("c");
    Check(IdOf(frame, grand) == CurrentIdSeed(), "three keyed levels match three PushIDs");
    ImGui::PopID();
    ImGui::PopID();
    ImGui::PopID();

    const std::array<Element, 1> plainKids{Leaf(frame)};
    const Element plain = Keyed(frame, Parent(frame, plainKids), "p");
    Check(!AssignIds(frame, plain, CurrentIdSeed()).has_value(), "the ids assign");
    ImGui::PushID("p");
    ImGui::PushID(0);
    Check(IdOf(frame, plainKids[0]) == CurrentIdSeed(), "an index matches PushID of the number");
    ImGui::PopID();
    ImGui::PopID();

    ImGui::PushOverrideID(IdOf(frame, grand));
    Check(IdFor("button") == IdWithSeed("button", IdOf(frame, grand)), "a component under the node's id chains from it");
    ImGui::PopID();
  });
}

// A number key equal to a sibling's index gives the same id, which the raw keys cannot show since
// the sibling has none. AssignIds refuses it and leaves every id 0. The baseline is the same pair
// with a key that matches no index, which is accepted.
void TestNumberKeyCollidesWithIndex() {
  InWindow([](UiFrame &frame) {
    const std::array<Element, 2> clash{Keyed(frame, Leaf(frame), 1u), Leaf(frame)};
    const Element clashRoot = Parent(frame, clash);
    Check(!clashRoot.IsNone(), "the raw keys do not clash, so Box accepts the pair");
    const auto failure = AssignIds(frame, clashRoot, CurrentIdSeed());
    Check(failure == ElementError::IdCollision, "but their ids do, so AssignIds refuses");
    Check(IdOf(frame, clashRoot) == 0 && IdOf(frame, clash[0]) == 0 && IdOf(frame, clash[1]) == 0, "and leaves no ids behind");

    const std::array<Element, 2> fine{Keyed(frame, Leaf(frame), 5u), Leaf(frame)};
    const Element fineRoot = Keyed(frame, Parent(frame, fine), "fine");
    Check(!AssignIds(frame, fineRoot, CurrentIdSeed()).has_value(), "a number key matching no index is accepted");
    Check(IdOf(frame, fine[0]) != IdOf(frame, fine[1]), "with different ids");
  });
}

// ImGui ignores what comes before ### in an id, so "a###x" and "b###x" are different strings with
// the same id. AssignIds refuses them as siblings. "a##x" and "b##x" hash differently and are the
// baseline.
void TestHashAlikeKeysRefused() {
  InWindow([](UiFrame &frame) {
    const std::array<Element, 2> alike{Keyed(frame, Leaf(frame), "a###x"), Keyed(frame, Leaf(frame), "b###x")};
    const Element alikeRoot = Parent(frame, alike);
    Check(!alikeRoot.IsNone(), "the strings differ, so Box accepts them");
    Check(AssignIds(frame, alikeRoot, CurrentIdSeed()) == ElementError::IdCollision, "their ids match, so AssignIds refuses");

    const std::array<Element, 2> apart{Keyed(frame, Leaf(frame), "a##x"), Keyed(frame, Leaf(frame), "b##x")};
    const Element apartRoot = Keyed(frame, Parent(frame, apart), "apart");
    Check(!AssignIds(frame, apartRoot, CurrentIdSeed()).has_value(), "a ## suffix keeps the ids apart");
  });
}

// A clash deeper in the tree is found too, and clears the ids above it.
void TestNestedCollisionClearsTree() {
  InWindow([](UiFrame &frame) {
    const std::array<Element, 2> alike{Keyed(frame, Leaf(frame), "a###x"), Keyed(frame, Leaf(frame), "b###x")};
    const std::array<Element, 1> middle{Parent(frame, alike)};
    const Element root = Parent(frame, middle);
    Check(AssignIds(frame, root, CurrentIdSeed()) == ElementError::IdCollision, "a clash two levels down is refused");
    Check(IdOf(frame, root) == 0 && IdOf(frame, middle[0]) == 0, "and the levels above hold no ids");
  });
}

// Two unkeyed roots in one window share every id, so the second is refused. Keying it is the
// baseline, accepted, and the next frame starts clean.
void TestSecondRootNeedsKey() {
  NavHarness nav;
  FrameRig rig;
  nav.Frame([&](InteractContext &) {
    rig.Begin();
    const Element first = Leaf(rig.frame);
    const Element second = Leaf(rig.frame);
    const Element keyed = Keyed(rig.frame, Leaf(rig.frame), "second");
    Check(!AssignIds(rig.frame, first, CurrentIdSeed()).has_value(), "the first root is accepted");
    Check(AssignIds(rig.frame, second, CurrentIdSeed()) == ElementError::IdCollision, "a second unkeyed root is refused");
    Check(IdOf(rig.frame, second) == 0, "and gets no id");
    Check(!AssignIds(rig.frame, keyed, CurrentIdSeed()).has_value(), "a keyed second root is accepted");
    Check(IdOf(rig.frame, keyed) != IdOf(rig.frame, first), "with its own id");

    rig.Begin();
    Check(!AssignIds(rig.frame, Leaf(rig.frame), CurrentIdSeed()).has_value(), "a new frame forgets the roots");
  });
}

// A refused Keyed copies nothing. The text interned after the refusals lands right after the text
// interned before them.
void TestRefusedKeyInternsNothing() {
  NavHarness nav;
  FrameRig rig;
  nav.Frame([&](InteractContext &) {
    rig.Begin();
    const Element stale = Leaf(rig.frame);
    const Element parented = Leaf(rig.frame);
    const std::array<Element, 1> kids{parented};
    (void)Parent(rig.frame, kids);
    rig.Begin();
    const std::string_view before = rig.frame.Intern("zz");
    Check(Keyed(rig.frame, stale, "a stale handle's key").IsNone(), "a stale handle is refused");
    Check(Keyed(rig.frame, Element{}, "a none handle's key").IsNone(), "and a none handle");
    Check(Keyed(rig.frame, Leaf(rig.frame), "").IsNone(), "and an empty key");
    const Element adopted = Leaf(rig.frame);
    const std::array<Element, 1> adoptedKids{adopted};
    (void)Parent(rig.frame, adoptedKids);
    Check(Keyed(rig.frame, adopted, "a parented element's key").IsNone(), "and a parented element");
    const std::string_view after = rig.frame.Intern("yy");
    Check(after.data() == before.data() + before.size(), "none of them copied its key");
  });
}

// The same root given ids twice is accepted and keeps its ids, so a tree can be assigned and then
// rendered in one frame. A node inside a tree is refused as a root, since it would get other ids
// there than it has in its tree. A different node with the root's id stays refused, shown by the
// second unkeyed root test.
void TestSameRootAssignedTwice() {
  InWindow([](UiFrame &frame) {
    const std::array<Element, 2> kids{Keyed(frame, Leaf(frame), "x"), Leaf(frame)};
    const Element root = Keyed(frame, Parent(frame, kids), "tree");
    Check(!AssignIds(frame, root, CurrentIdSeed()).has_value(), "the first assignment is accepted");
    const ItemId rootId = IdOf(frame, root);
    const ItemId firstId = IdOf(frame, kids[0]);
    const ItemId secondId = IdOf(frame, kids[1]);
    Check(rootId != 0 && firstId != 0 && secondId != 0, "and gives ids");
    Check(!AssignIds(frame, root, CurrentIdSeed()).has_value(), "assigning the same root again is accepted");
    Check(IdOf(frame, root) == rootId && IdOf(frame, kids[0]) == firstId && IdOf(frame, kids[1]) == secondId, "with the same ids");
    Check(AssignIds(frame, kids[0], CurrentIdSeed()) == ElementError::AlreadyParented, "a node inside a tree is refused as a root");
    Check(IdOf(frame, kids[0]) == firstId, "and keeps its id");
  });
}

// ---- Render with interaction, against the older controls ------------------------------------

constexpr float TOLERANCE = 0.001f;
// How far the element button's vertices may sit from the real button's. The two reach the same
// positions by different sums, the flex solver's centring against the button's own, and the
// largest difference seen is 6e-5 of a pixel, so a thousandth of a pixel catches a real shift.
constexpr float GEOMETRY_SLACK = 0.001f;

// True when two values agree to within rounding noise.
bool Near(float actual, float expected) { return std::fabs(actual - expected) <= TOLERANCE; }

// A control for a dual run: where it is, what it asks of the player, and what it reported last.
struct Control {
  std::string name;
  Rect rect;
  bool disabled = false;
  DpadClaim claim = DpadClaim::None;
  // The fixed pose of a specimen, for the old control as InteractContext::pose and for the
  // element as Specimen.
  std::optional<ItemVisuals> pose;
  // A pose on the element only, which moves what it draws and must not move what it hits.
  std::optional<ElementPose> elementPose;
  Interaction last{};
  HorizontalPresses dpad{};
  int activations = 0;
};

// How the element side lays its controls out. Roots makes each its own tree at its own rectangle,
// Column stacks them in one tree, and Grid puts them two to a row, so the tree order is the order
// the old controls were drawn in.
enum class Shape { Roots, Column, Grid };

// Draws the same controls twice, once with the older Interact calls and once as a tree rendered by
// Render, each in a harness of its own, and holds the two to the same trace. After every frame it
// compares every control's state, visuals, activations and D-pad presses, exactly.
class DualRun {
public:
  DualRun(std::vector<Control> controls, Shape shape, bool followScroll = false)
      : old_(controls), ele_(std::move(controls)), shape_(shape), followScroll_(followScroll) {}

  void Frame() {
    oldNav_.Frame([&](InteractContext &context) { DrawOld(context); });
    eleNav_.Frame([&](InteractContext &context) { DrawElements(context); });
    Compare();
  }
  void Frames(int count) {
    for (int i = 0; i < count; ++i) {
      Frame();
    }
  }
  // Presses and releases `key` over two frames and runs one more, as NavHarness::Press does.
  void Press(ImGuiKey key) {
    Key(key, true);
    Frame();
    Key(key, false);
    Frames(2);
  }
  void Key(ImGuiKey key, bool down) {
    oldNav_.Key(key, down);
    eleNav_.Key(key, down);
  }
  void MouseButton(bool down) {
    oldNav_.MouseButton(down);
    eleNav_.MouseButton(down);
  }
  void MoveMouse(float x, float y) {
    oldNav_.MoveMouse(x, y, [&](InteractContext &context) { DrawOld(context); });
    eleNav_.MoveMouse(x, y, [&](InteractContext &context) { DrawElements(context); });
    Compare();
  }
  void Disable(size_t index, bool disabled) { old_[index].disabled = ele_[index].disabled = disabled; }

  // The control that has focus on the old side, or "none".
  std::string Focused() const {
    for (const Control &control : old_) {
      if (control.last.state.focused) {
        return control.name;
      }
    }
    return "none";
  }
  const Control &Old(size_t index) const { return old_[index]; }
  const Control &Ele(size_t index) const { return ele_[index]; }
  // How many frames of controls were compared, so a test can show it compared something.
  int compared() const { return compared_; }

private:
  void DrawOld(InteractContext &context) {
    const float shift = followScroll_ ? ImGui::GetWindowPos().y - ImGui::GetScrollY() : 0.0f;
    for (Control &control : old_) {
      Rect rect = control.rect;
      rect.y += shift;
      context.pose = control.pose ? &*control.pose : nullptr;
      const ItemId id = ImGui::GetID(control.name.c_str());
      control.last = Interact(context, id, rect, control.disabled);
      if (control.claim != DpadClaim::None) {
        control.dpad = ClaimHorizontal(id, control.claim == DpadClaim::Repeat);
      }
      control.activations += control.last.state.activated ? 1 : 0;
    }
    context.pose = nullptr;
  }

  // One control as a keyed, interactive box that is exactly its rectangle in size.
  Element Leaf(UiFrame &frame, const Control &control) {
    const ItemLayout item{.width = Dp(control.rect.width), .height = Dp(control.rect.height)};
    const Element leaf = Box(frame, {.item = item}, {});
    Check(!Keyed(frame, leaf, control.name).IsNone(), "the control is keyed");
    InteractProps props;
    props.disabled = control.disabled;
    props.dpad = control.claim;
    Check(!MakeInteractive(frame, leaf, props).IsNone(), "the control is interactive");
    if (control.pose) {
      Check(!Specimen(frame, leaf, *control.pose).IsNone(), "the specimen is accepted");
    }
    if (control.elementPose) {
      Check(!WithPose(frame, leaf, *control.elementPose).IsNone(), "the pose is accepted");
    }
    return leaf;
  }

  // Lays out and renders one tree at `rect`, and checks Render drew it.
  Element RenderTree(UiFrame &frame, Element root, const Rect &rect, ImDrawList &list) {
    (void)Measure(frame, root, {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()});
    Check(Arrange(frame, root, rect), "the tree arranges");
    const RenderResult result = Render(frame, root, list);
    Check(!result.error.has_value(), "the tree renders");
    return result.activated;
  }

  void DrawElements(InteractContext &context) {
    frame_.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const float shift = followScroll_ ? ImGui::GetWindowPos().y - ImGui::GetScrollY() : 0.0f;
    std::vector<Element> leaves;
    for (const Control &control : ele_) {
      leaves.push_back(Leaf(frame_, control));
    }
    if (shape_ == Shape::Roots) {
      for (size_t i = 0; i < leaves.size(); ++i) {
        Rect rect = ele_[i].rect;
        rect.y += shift;
        (void)RenderTree(frame_, leaves[i], rect, context.list);
        Check(InputOf(frame_, leaves[i])->id == ImGui::GetID(ele_[i].name.c_str()), "a keyed root has the id of its name");
      }
    } else {
      Element root = shape_ == Shape::Column ? Column(leaves) : GridOf(leaves);
      Rect rect = ele_[0].rect;
      rect.y += shift;
      rect.height = ele_.back().rect.Bottom() - ele_[0].rect.y;
      rect.width = GridWidth();
      (void)RenderTree(frame_, root, rect, context.list);
      for (size_t i = 0; i < leaves.size(); ++i) {
        Rect expected = ele_[i].rect;
        expected.y += shift;
        const Rect placed = Arranged(frame_, leaves[i]);
        Check(std::fabs(placed.x - expected.x) < 0.5f && std::fabs(placed.y - expected.y) < 0.5f, "the layout puts the control where the old one was");
      }
    }
    for (size_t i = 0; i < leaves.size(); ++i) {
      const auto input = InputOf(frame_, leaves[i]);
      Check(input.has_value(), "an interactive node has an input after Render");
      ele_[i].last = input->interaction;
      ele_[i].dpad = input->dpad;
      ele_[i].activations += input->interaction.state.activated ? 1 : 0;
    }
  }

  // The controls stacked down one column, with the gap the old rectangles leave.
  Element Column(const std::vector<Element> &leaves) {
    const float gap = ele_.size() > 1 ? ele_[1].rect.y - ele_[0].rect.Bottom() : 0.0f;
    Layout layout{.direction = FlexDirection::Column};
    layout.gap = Dp(gap);
    const Element root = Box(frame_, layout, {}, leaves);
    Check(!Keyed(frame_, root, "tree").IsNone(), "the tree is keyed");
    return root;
  }

  // The controls two to a row, with the gaps the old rectangles leave.
  Element GridOf(const std::vector<Element> &leaves) {
    std::vector<Element> rows;
    for (size_t i = 0; i < leaves.size(); i += 2) {
      Layout row{.direction = FlexDirection::Row};
      row.gap = Dp(ele_[i + 1].rect.x - ele_[i].rect.Right());
      const std::array<Element, 2> pair{leaves[i], leaves[i + 1]};
      rows.push_back(Box(frame_, row, {}, pair));
    }
    Layout layout{.direction = FlexDirection::Column};
    layout.gap = Dp(ele_[2].rect.y - ele_[0].rect.Bottom());
    const Element root = Box(frame_, layout, {}, rows);
    Check(!Keyed(frame_, root, "tree").IsNone(), "the grid is keyed");
    return root;
  }

  float GridWidth() const { return shape_ == Shape::Grid ? ele_[1].rect.Right() - ele_[0].rect.x : ele_[0].rect.width; }

  void Compare() {
    for (size_t i = 0; i < old_.size(); ++i) {
      const Control &a = old_[i];
      const Control &b = ele_[i];
      const ItemState &x = a.last.state;
      const ItemState &y = b.last.state;
      Check(x.hovered == y.hovered && x.focused == y.focused && x.held == y.held && x.activated == y.activated && x.disabled == y.disabled,
            "the element reports the state the old control does");
      Check(a.last.visuals.hover == b.last.visuals.hover && a.last.visuals.focus == b.last.visuals.focus && a.last.visuals.press == b.last.visuals.press,
            "and shows the same highlights");
      Check(a.activations == b.activations, "and activates as often");
      Check(a.dpad.left == b.dpad.left && a.dpad.right == b.dpad.right, "and sees the same D-pad presses");
      ++compared_;
    }
  }

  NavHarness oldNav_;
  NavHarness eleNav_;
  std::vector<Control> old_;
  std::vector<Control> ele_;
  Shape shape_;
  bool followScroll_;
  UiFrame frame_;
  int compared_ = 0;
};

// Three controls stacked down the screen.
std::vector<Control> ColumnControls() {
  return {{.name = "a", .rect = {100, 100, 200, 50}}, {.name = "b", .rect = {100, 200, 200, 50}}, {.name = "c", .rect = {100, 300, 200, 50}}};
}

// A window that appears focuses its first control, and the D-pad then moves focus by position.
// The element tree must register its controls in tree order for the first focus to land on "a".
void TestElementDpadMovesFocus() {
  for (const Shape shape : {Shape::Roots, Shape::Column}) {
    DualRun run(ColumnControls(), shape);
    run.Frames(2);
    Check(run.Focused() == "a" && run.Ele(0).last.state.focused, "the first control has focus when the window appears");
    run.Press(ImGuiKey_GamepadDpadDown);
    Check(run.Focused() == "b" && run.Ele(1).last.state.focused, "down moves to the control below");
    run.Press(ImGuiKey_GamepadDpadUp);
    Check(run.Focused() == "a" && run.Ele(0).last.state.focused, "up moves back");
    Check(run.compared() > 20, "frames were compared");
  }
}

// Moving right from the top left of a grid picks the control beside it.
void TestElementGridNavigation() {
  DualRun run({{.name = "top-left", .rect = {100, 100, 200, 50}},
               {.name = "top-right", .rect = {400, 100, 200, 50}},
               {.name = "bottom-left", .rect = {100, 200, 200, 50}},
               {.name = "bottom-right", .rect = {400, 200, 200, 50}}},
              Shape::Grid);
  run.Frames(2);
  Check(run.Focused() == "top-left" && run.Ele(0).last.state.focused, "the grid starts at the top left");
  run.Press(ImGuiKey_GamepadDpadRight);
  Check(run.Focused() == "top-right" && run.Ele(1).last.state.focused, "right moves to the control beside it");
}

// A disabled control is skipped: with the middle one disabled, down goes from the first to the
// third.
void TestElementDisabledIsSkipped() {
  auto controls = ColumnControls();
  controls[1].disabled = true;
  DualRun run(controls, Shape::Column);
  run.Frames(2);
  Check(run.Focused() == "a", "the first control starts with focus");
  run.Press(ImGuiKey_GamepadDpadDown);
  Check(run.Focused() == "c" && run.Ele(2).last.state.focused, "down skips the disabled control");
}

// Confirm activates the focused control once per press however long it is held, and a disabled
// control under the mouse never activates when clicked.
void TestElementActivation() {
  auto controls = ColumnControls();
  controls[2].disabled = true;
  DualRun run(controls, Shape::Column);
  run.Frames(2);
  run.Key(ImGuiKey_GamepadFaceDown, true);
  run.Frame();
  Check(run.Ele(0).activations == 1, "confirm activates the focused control on the press");
  run.Frames(30);
  run.Key(ImGuiKey_GamepadFaceDown, false);
  run.Frames(3);
  Check(run.Ele(0).activations == 1 && run.Ele(1).activations == 0, "once however long it is held, and only that control");
  run.MoveMouse(150, 320);
  run.MouseButton(true);
  run.Frame();
  run.MouseButton(false);
  run.Frames(2);
  Check(run.Ele(2).activations == 0, "a disabled control never activates");
  run.MoveMouse(150, 220);
  run.MouseButton(true);
  run.Frame();
  run.MouseButton(false);
  run.Frames(2);
  Check(run.Ele(1).activations == 1, "and an enabled one does, which is the baseline");
}

// A specimen shows its pose whatever the input, navigation skips it, and a click on it does
// nothing. The unposed column is the baseline, where down reaches the middle control.
void TestElementPose() {
  const ItemVisuals pose{0.25f, 1.0f, 0.5f};
  for (const bool posed : {false, true}) {
    auto controls = ColumnControls();
    if (posed) {
      controls[1].pose = pose;
    }
    DualRun run(controls, Shape::Roots);
    run.Frames(2);
    const ItemVisuals &shown = run.Ele(1).last.visuals;
    Check(posed == (Near(shown.hover, 0.25f) && Near(shown.focus, 1.0f) && Near(shown.press, 0.5f)),
          posed ? "a specimen shows its pose" : "the baseline shows no highlights");
    run.Press(ImGuiKey_GamepadDpadDown);
    Check(run.Focused() == (posed ? "c" : "b"), posed ? "navigation skips a specimen" : "the baseline reaches it");
    run.MoveMouse(150, 220);
    run.MouseButton(true);
    run.Frame();
    run.MouseButton(false);
    run.Frames(2);
    Check((run.Ele(1).activations == 0) == posed, posed ? "a click on a specimen does nothing" : "the baseline activates");
  }
}

// Holding confirm on the focused control shows its press, which falls away after release. The
// control below is the baseline that stays at rest.
void TestElementPressHighlight() {
  DualRun run(ColumnControls(), Shape::Column);
  run.Frames(2);
  run.Key(ImGuiKey_GamepadFaceDown, true);
  run.Frames(10);
  Check(run.Ele(0).last.state.held && Near(run.Ele(0).last.visuals.press, 1), "a held control shows its press");
  Check(Near(run.Ele(1).last.visuals.press, 0), "a control that is not held does not");
  run.Key(ImGuiKey_GamepadFaceDown, false);
  run.Frames(10);
  Check(Near(run.Ele(0).last.visuals.press, 0), "the press falls away after release");
}

// Focus reaches a control below the window's edge and scrolls the window to it, with the tree
// laid out again at the scrolled position each frame.
void TestElementFocusScrollsTheWindow() {
  std::vector<Control> controls;
  for (int index = 0; index < 20; ++index) {
    controls.push_back({.name = "row" + std::to_string(index), .rect = {100, 100.0f + 150.0f * static_cast<float>(index), 200, 50}});
  }
  DualRun run(controls, Shape::Column, true);
  run.Frames(2);
  for (int press = 0; press < 19; ++press) {
    run.Press(ImGuiKey_GamepadDpadDown);
  }
  run.Frames(10);
  Check(run.Ele(19).last.state.focused, "focus reaches the last control");
}

// A control that loses its enabled state while it has focus keeps the focus but shows no press.
// The enabled control is the baseline that does show the press.
void TestElementDisabledKeepsFocusButNoPress() {
  for (const bool disable : {false, true}) {
    DualRun run(ColumnControls(), Shape::Column);
    run.Frames(2);
    run.Disable(0, disable);
    run.Key(ImGuiKey_GamepadFaceDown, true);
    run.Frames(10);
    Check(run.Ele(0).last.state.focused, "the control keeps focus");
    Check(Near(run.Ele(0).last.visuals.press, disable ? 0.0f : 1.0f), disable ? "a disabled control shows no press" : "the baseline shows the press");
    Check(run.Ele(0).activations == (disable ? 0 : 1), "and activates only when enabled");
  }
}

// A control that claims the D-pad keeps left and right from navigation while it has focus and
// reports the presses. The unclaiming control beside it is the baseline, where right moves focus.
void TestElementClaimsHorizontal() {
  for (const bool claim : {false, true}) {
    std::vector<Control> controls{{.name = "a", .rect = {100, 100, 200, 50}, .claim = claim ? DpadClaim::Once : DpadClaim::None},
                                  {.name = "b", .rect = {400, 100, 200, 50}}};
    DualRun run(controls, Shape::Roots);
    run.Frames(2);
    run.Key(ImGuiKey_GamepadDpadRight, true);
    bool pressed = false;
    for (int i = 0; i < 3; ++i) {
      run.Frame();
      pressed = pressed || run.Ele(0).dpad.right;
    }
    run.Key(ImGuiKey_GamepadDpadRight, false);
    run.Frames(3);
    Check(claim ? run.Focused() == "a" : run.Focused() == "b", claim ? "a claiming control keeps focus" : "the baseline moves focus right");
    Check(pressed == claim, claim ? "and sees the press" : "while the baseline sees none");
  }
}

// An element that is posed moves what it draws, and not where it is hit. A click inside the layout
// rectangle but outside the posed one activates it, as it does the old control with no pose.
void TestElementHitsTheLayoutRect() {
  std::vector<Control> controls{{.name = "a", .rect = {100, 100, 200, 50}}};
  controls[0].elementPose = ElementPose{.offset = {0, 300}, .scale = 0.5f};
  DualRun run(controls, Shape::Roots);
  run.Frames(2);
  run.MoveMouse(110, 110);
  run.MouseButton(true);
  run.Frame();
  run.MouseButton(false);
  run.Frames(2);
  Check(run.Old(0).activations == 1 && run.Ele(0).activations == 1, "a click on the layout rectangle activates the posed element");
}

// Interactive nodes do not nest. A box holding one cannot be made interactive, nor can a child of
// one, in either order of building. Two siblings are the baseline, accepted.
void TestNestedInteractiveRefused() {
  InWindow([](UiFrame &frame) {
    ElementError reason = ElementError::None;
    const std::array<Element, 2> pair{Leaf(frame), Leaf(frame)};
    Check(!MakeInteractive(frame, pair[0], {}).IsNone() && !MakeInteractive(frame, pair[1], {}).IsNone(), "siblings are both accepted");
    const Element parent = Parent(frame, pair);
    Check(MakeInteractive(frame, parent, {}, &reason).IsNone() && reason == ElementError::NestedInteractive, "a box holding interactive nodes is refused");

    const Element inner = Leaf(frame);
    const std::array<Element, 1> kids{inner};
    const Element outer = Parent(frame, kids);
    Check(!MakeInteractive(frame, outer, {}).IsNone(), "the outer box is made interactive first");
    reason = ElementError::None;
    Check(MakeInteractive(frame, inner, {}, &reason).IsNone() && reason == ElementError::NestedInteractive, "then a child of it is refused");
  });
}

// A tree whose ids collide is not rendered and registers nothing, and says why.
void TestRenderReportsIdError() {
  NavHarness nav;
  FrameRig rig;
  nav.Frame([&](InteractContext &context) {
    rig.Begin();
    const Element first = Leaf(rig.frame);
    const Element second = Leaf(rig.frame);
    Check(!MakeInteractive(rig.frame, first, {}).IsNone() && !MakeInteractive(rig.frame, second, {}).IsNone(), "both are interactive");
    const auto inf = std::numeric_limits<float>::infinity();
    for (const Element root : {first, second}) {
      (void)Measure(rig.frame, root, {inf, inf});
      Check(Arrange(rig.frame, root, {0, 0, 50, 50}), "arranges");
    }
    Check(!Render(rig.frame, first, context.list).error.has_value(), "the first root renders");
    const RenderResult refused = Render(rig.frame, second, context.list);
    Check(refused.error == ElementError::IdCollision && refused.activated.IsNone(), "the second unkeyed root is refused");
    Check(!InputOf(rig.frame, second).has_value(), "and has no input");
  });
}


// Builds a clipping card 100 dp wide holding a node 200 dp wide, so the right half of the node lies
// outside the card, renders it and returns what the node reported. With `clip` off the card
// does not clip, which is the baseline.
struct ClipProbe {
  bool hovered = false;
  int activations = 0;
};

ClipProbe ClickInClippedCard(bool clip, float mouseX) {
  NavHarness nav;
  UiFrame frame;
  ClipProbe probe;
  const auto draw = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element node = Keyed(frame, Box(frame, {.item = {.width = Dp(200.0f), .height = Dp(60.0f)}}, {}), "node");
    Check(!MakeInteractive(frame, node, {}).IsNone(), "the node is interactive");
    Style cardStyle;
    cardStyle.overflow = clip ? Overflow::Clip : Overflow::Visible;
    const std::array<Element, 1> kids{node};
    const Element card = Keyed(frame, Box(frame, {.item = {.width = Dp(100.0f), .height = Dp(60.0f)}}, cardStyle, kids), "card");
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(frame, card, {inf, inf});
    Check(Arrange(frame, card, {100, 100, 100, 60}), "the card arranges");
    Check(!Render(frame, card, context.list).error.has_value(), "the card renders");
    const auto input = InputOf(frame, node);
    probe.hovered = input->interaction.state.hovered;
    probe.activations += input->interaction.state.activated ? 1 : 0;
  };
  nav.Frames(2, draw);
  nav.MoveMouse(mouseX, 130, draw);
  nav.Frame(draw);
  const bool hovered = probe.hovered;
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(2, draw);
  probe.hovered = hovered;
  return probe;
}

// A node half outside a clipping card can only be reached on its visible half. A click on the cut
// off half does nothing and does not hover it. Without the clip both halves react, which is the
// baseline that shows the click lands on the node.
void TestClipLimitsHitTesting() {
  constexpr float VISIBLE_X = 150;
  constexpr float CUT_OFF_X = 250;
  for (const bool clip : {false, true}) {
    const ClipProbe visible = ClickInClippedCard(clip, VISIBLE_X);
    Check(visible.hovered && visible.activations == 1, "the visible half reacts");
    const ClipProbe cutOff = ClickInClippedCard(clip, CUT_OFF_X);
    Check(cutOff.hovered == !clip && cutOff.activations == (clip ? 0 : 1), clip ? "the cut off half does not react" : "the baseline reacts there too");
  }
}

// A second Render of the same root in a frame is refused and registers nothing more. AssignIds
// first is fine, and a new frame starts clean.
void TestRenderTwiceRefused() {
  NavHarness nav;
  FrameRig rig;
  nav.Frame([&](InteractContext &context) {
    for (int frameNumber = 0; frameNumber < 2; ++frameNumber) {
      rig.Begin();
      const Element root = Keyed(rig.frame, Leaf(rig.frame), "root");
      Check(!MakeInteractive(rig.frame, root, {}).IsNone(), "the root is interactive");
      const auto inf = std::numeric_limits<float>::infinity();
      (void)Measure(rig.frame, root, {inf, inf});
      Check(Arrange(rig.frame, root, {0, 0, 50, 50}), "arranges");
      Check(!AssignIds(rig.frame, root, CurrentIdSeed()).has_value(), "ids can be assigned before Render");
      Check(!Render(rig.frame, root, context.list).error.has_value(), "so Render still works");
      const RenderResult again = Render(rig.frame, root, context.list);
      Check(again.error == ElementError::AlreadyRendered && again.activated.IsNone(), "a second Render of the root is refused");
    }
  });
}

// The vertices the Button element drew compared with those of a hand-built element button, a box
// with state styles, a centred label and MakeInteractive, under the same input. Colours and uvs
// match exactly, which means the same highlights, and positions to a pixel.
Style ElementButtonStyle(const Theme &theme, bool accent) {
  Style style{accent ? theme.colors.accent.fill : theme.colors.surface.overlay, accent ? Color{} : theme.colors.border.standard,
              Dp(theme.border.hair), Dp(ButtonDefaults(theme).radius), {}};
  const auto &colors = theme.colors;
  if (accent) {
    style.states.hover.fill = colors.accent.fill_hover;
    style.states.focus.fill = colors.accent.fill_hover;
    style.states.pressed.fill = colors.accent.fill_pressed;
  } else {
    style.states.hover.overlay = colors.state.hover;
    style.states.focus.overlay = colors.state.focus;
    style.states.pressed.overlay = colors.state.pressed;
  }
  style.states.focus.focus_ring = true;
  style.states.pressed.scale = theme.interaction.press_scale;
  return style;
}

// A button as an element: the label centred in a box of the button's size.
Element ElementButton(UiFrame &frame, std::string_view name, const Rect &rect, bool accent, bool disabled) {
  const Theme &theme = frame.ActiveTheme();
  const TextProps label{"Save", theme.type.label, accent ? theme.colors.text.on_accent : theme.colors.text.primary, TextAlign::Center};
  const std::array<Element, 1> kids{Text(frame, label)};
  Layout layout{.direction = FlexDirection::Column, .justify = Justify::Center, .align = Align::Stretch};
  layout.padding = DpInsets::Symmetric(Dp(ButtonDefaults(theme).padding_x), Dp(0.0f));
  layout.item = {.width = Dp(rect.width), .height = Dp(rect.height)};
  Style style = ElementButtonStyle(theme, accent);
  const Element button = Box(frame, layout, style, kids);
  Check(!Keyed(frame, button, name).IsNone(), "the button is keyed");
  InteractProps props;
  props.disabled = disabled;
  Check(!MakeInteractive(frame, button, props).IsNone(), "the button is interactive");
  return button;
}

// What this proves now is that the Button component draws and activates as a hand-built tree of
// primitives does: a box with state styles, a centred label and MakeInteractive. The two run side by
// side through confirm, release, hover and click, for the secondary and the accent look, and
// disabled, and a frame where they differ fails. It says nothing of the button's own correctness,
// which the direct assertions cover: SystemUiButtonTests (TestFocusedPrimary and the others),
// TestElementActivation and TestElementDisabledKeepsFocusButNoPress above.
void TestElementButtonMatchesRealButton() {
  const Rect rect{400, 300, 240, 56};
  for (const bool accent : {false, true}) {
    for (const bool disabled : {false, true}) {
      NavHarness realNav, eleNav;
      UiFrame realFrame, frame;
      int compared = 0;
      bool realActivated = false, eleActivated = false;
      std::vector<ImDrawVert> realVertices, eleVertices;
      const auto drawReal = [&](InteractContext &context) {
        realFrame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
        const Element button = Button(realFrame, "button", {.label = "Save", .variant = accent ? ButtonVariant::Primary : ButtonVariant::Secondary, .disabled = disabled});
        const auto inf = std::numeric_limits<float>::infinity();
        (void)Measure(realFrame, button, {inf, inf});
        Check(Arrange(realFrame, button, rect), "the real button arranges");
        const int begin = context.list.VtxBuffer.Size;
        const RenderResult result = Render(realFrame, button, context.list);
        Check(!result.error.has_value(), "the real button renders");
        realActivated = !result.activated.IsNone();
        realVertices.assign(context.list.VtxBuffer.begin() + begin, context.list.VtxBuffer.end());
      };
      const auto drawEle = [&](InteractContext &context) {
        frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
        const Element button = ElementButton(frame, "button", rect, accent, disabled);
        const auto inf = std::numeric_limits<float>::infinity();
        (void)Measure(frame, button, {inf, inf});
        Check(Arrange(frame, button, rect), "the button arranges");
        const int begin = context.list.VtxBuffer.Size;
        const RenderResult result = Render(frame, button, context.list);
        Check(!result.error.has_value(), "the button renders");
        eleActivated = !result.activated.IsNone();
        eleVertices.assign(context.list.VtxBuffer.begin() + begin, context.list.VtxBuffer.end());
      };
      const auto compare = [&]() {
        Check(realActivated == eleActivated, "both activate on the same frame");
        Check(realVertices.size() == eleVertices.size() && !realVertices.empty(), "both draw the same number of vertices");
        for (size_t v = 0; v < realVertices.size(); ++v) {
          Check(realVertices[v].col == eleVertices[v].col && realVertices[v].uv.x == eleVertices[v].uv.x && realVertices[v].uv.y == eleVertices[v].uv.y,
                "the colours and uvs are the same, so the highlights are");
          Check(std::fabs(realVertices[v].pos.x - eleVertices[v].pos.x) <= GEOMETRY_SLACK && std::fabs(realVertices[v].pos.y - eleVertices[v].pos.y) <= GEOMETRY_SLACK,
                "and the geometry agrees to float noise");
        }
        ++compared;
      };
      const auto frames = [&](int count) {
        for (int i = 0; i < count; ++i) {
          realNav.Frame(drawReal);
          eleNav.Frame(drawEle);
          compare();
        }
      };
      const auto key = [&](ImGuiKey code, bool down) {
        realNav.Key(code, down);
        eleNav.Key(code, down);
      };
      const auto mouse = [&](bool down) {
        realNav.MouseButton(down);
        eleNav.MouseButton(down);
      };
      frames(2);
      key(ImGuiKey_GamepadFaceDown, true);
      frames(10);
      key(ImGuiKey_GamepadFaceDown, false);
      frames(10);
      realNav.MoveMouse(520, 330, drawReal);
      eleNav.MoveMouse(520, 330, drawEle);
      compare();
      frames(10);
      mouse(true);
      frames(6);
      mouse(false);
      frames(6);
      Check(compared > 40, "frames were compared");
    }
  }
}

// ---- Callbacks, bound values and inherited input ---------------------------------------------

// The value a toggle-like tree flips and the knob reads. The caller owns it, as with `bool& on`.
// `seen` is what the knob's paint read, which shows whether the flip came before the paint.
struct ToggleProbe {
  bool on = false;
  bool seen = false;
};

// The knob of a toggle drawn as a Custom node. It reads the caller's bool when it paints, animates
// between the ends of the track with the same tween the Toggle element uses, and
// draws the same circle. Only pointers and an id are captured, so it is trivially copyable.
struct KnobPaint {
  ToggleProbe *probe;
  ItemId id;

  void operator()(const CustomPaint &paint) const {
    probe->seen = probe->on;
    const Theme &theme = paint.frame.ActiveTheme();
    const Track slide{theme.interaction.knob_ms, Ease::Emphasis};
    const float knob = paint.frame.Store().Animate(id, probe->on ? 1.0f : 0.0f, {slide, slide}, paint.frame.Now());
    const float radius = paint.painted.height / 2.0f;
    const float knobRadius = std::min(paint.frame.Scale().Px(ToggleDefaults(theme).knob), paint.painted.height) / 2.0f;
    const float travel = std::max(0.0f, paint.painted.width - 2.0f * radius);
    const ImVec2 centre{paint.painted.x + radius + travel * knob, paint.painted.y + radius};
    constexpr int KNOB_SEGMENTS = 32;
    paint.list.AddCircleFilled(centre, knobRadius, ToImU32(theme.colors.control.knob.MulAlpha(paint.opacity)), KNOB_SEGMENTS);
  }
};

struct ActivateFlip {
  ToggleProbe *probe;
  void operator()(Element) const { probe->on = !probe->on; }
};

// A toggle as a tree. The track is an interactive box that flips `probe->on` when activated, and
// the knob is a Custom node filling it that reads the bool.
Element ElementToggle(UiFrame &frame, ToggleProbe *probe, const Rect &rect) {
  const Theme &theme = frame.ActiveTheme();
  const ItemLayout size{.width = Dp(rect.width), .height = Dp(rect.height)};
  const std::array<Element, 1> kids{Custom(frame, size, KnobPaint{probe, ImGui::GetID("toggle")})};
  Layout layout{};
  layout.item = size;
  Style style{theme.colors.control.track_off, {}, Dp(0.0f), Dp(rect.height / 2.0f), {}};
  const Element track = Box(frame, layout, style, kids);
  Check(!Keyed(frame, track, "toggle").IsNone(), "the track is keyed");
  InteractProps props;
  props.dpad = DpadClaim::Once;
  Check(!MakeInteractive(frame, track, props).IsNone(), "the track is interactive");
  Check(!OnActivate(frame, track, ActivateFlip{probe}).IsNone(), "the callback is accepted");
  return track;
}

constexpr int KNOB_VERTICES = 32;

// The mean x of the first `limit` vertices of `list` from `begin` that have `colour`, which for a
// circle of that many segments is its centre. The knob comes before the focus ring, which has the
// knob's colour, so the first ones are the knob's.
float MeanX(const ImDrawList &list, int begin, ImU32 colour, int limit, int &count) {
  float sum = 0.0f;
  count = 0;
  for (int v = begin; v < list.VtxBuffer.Size && count < limit; ++v) {
    if (list.VtxBuffer[v].col == colour) {
      sum += list.VtxBuffer[v].pos.x;
      ++count;
    }
  }
  return count == 0 ? 0.0f : sum / static_cast<float>(count);
}

// What this proves now is that the Toggle component behaves as a hand-built tree of primitives does
// under the same confirm and click input: an interactive box whose OnActivate callback flips a
// bool, and a Custom knob that reads it. The bool must agree on every frame, and so must the bool
// the knob painted with, which is stale on the press frame when the callback runs after the paint.
// The knob x agrees to a thousandth of a pixel, since the Custom uses the same tween and the same
// circle. The direct assertions on the toggle are in SystemUiToggleTests (TestConfirmFlips,
// TestLeftAndRight, TestKnobSlides and TestActivationOnThePressFrame).
void TestCallbackToggleMatchesRealToggle() {
  NavHarness oldNav, eleNav;
  UiFrame realFrame, frame;
  bool oldOn = false;
  ToggleProbe probe;
  float oldX = 0.0f, eleX = 0.0f;
  int oldCount = 0, eleCount = 0, compared = 0;
  float minX = 1e9f, maxX = -1e9f;
  Rect rect{};
  const auto drawOld = [&](InteractContext &context) {
    realFrame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element real = Toggle(realFrame, "toggle", &oldOn);
    const auto unlimited = std::numeric_limits<float>::infinity();
    const Vec2 size = Measure(realFrame, real, {unlimited, unlimited});
    rect = {400, 300, size.x, size.y};
    Check(Arrange(realFrame, real, rect), "the real toggle arranges");
    const int begin = context.list.VtxBuffer.Size;
    Check(!Render(realFrame, real, context.list).error.has_value(), "the real toggle renders");
    oldX = MeanX(context.list, begin, ToImU32(context.ui.theme.colors.control.knob), KNOB_VERTICES, oldCount);
  };
  const auto drawEle = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element toggle = ElementToggle(frame, &probe, rect);
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(frame, toggle, {inf, inf});
    Check(Arrange(frame, toggle, rect), "the toggle arranges");
    const int begin = context.list.VtxBuffer.Size;
    Check(!Render(frame, toggle, context.list).error.has_value(), "the toggle renders");
    eleX = MeanX(context.list, begin, ToImU32(context.ui.theme.colors.control.knob), KNOB_VERTICES, eleCount);
  };
  const auto compare = [&]() {
    Check(oldOn == probe.on, "the bool is the same on every frame, the press frame included");
    Check(probe.seen == oldOn, "and the knob painted with the flipped bool on the press frame");
    Check(oldCount == KNOB_VERTICES && eleCount == KNOB_VERTICES, "the knob circle is drawn in the knob colour, on both sides");
    Check(Near(oldX, eleX), "and the knob is at the same x");
    minX = std::min(minX, eleX);
    maxX = std::max(maxX, eleX);
    ++compared;
  };
  const auto frames = [&](int count) {
    for (int i = 0; i < count; ++i) {
      oldNav.Frame(drawOld);
      eleNav.Frame(drawEle);
      compare();
    }
  };
  const auto key = [&](bool down) {
    oldNav.Key(ImGuiKey_GamepadFaceDown, down);
    eleNav.Key(ImGuiKey_GamepadFaceDown, down);
  };
  const auto mouse = [&](bool down) {
    oldNav.MouseButton(down);
    eleNav.MouseButton(down);
  };
  frames(3);
  key(true);
  frames(1);
  Check(oldOn && probe.on, "confirm turns both on on the press frame");
  frames(9);
  key(false);
  frames(30);
  const float restOn = eleX;
  oldNav.MoveMouse(430, 318, drawOld);
  eleNav.MoveMouse(430, 318, drawEle);
  compare();
  mouse(true);
  frames(5);
  Check(oldOn && probe.on, "holding the button down changes nothing yet");
  mouse(false);
  frames(1);
  Check(!oldOn && !probe.on, "a click turns both off on the frame it is released");
  frames(40);
  // The knob's centre travels the track less its two round ends, and the test wants most of it.
  Check(restOn - eleX > (rect.width - rect.height) / 2.0f, "the knob travelled across the track and back");
  Check(compared > 80 && minX < maxX, "frames were compared and the knob moved");
}

// An interactive node that claims the D-pad, with a plain one to its right. The first has focus and
// takes the held right, so focus stays on it. When it is disabled it gives the key back and the
// held right moves focus on to the second, as it repeats. The same node kept enabled is the baseline,
// which still owns the key, so focus never leaves it.
void TestDisabledNodeGivesBackTheDpad() {
  for (const bool disable : {true, false}) {
    NavHarness nav;
    UiFrame frame;
    bool disabled = false;
    bool second_focused = false;
    bool first_focused = false;
    ItemId first_id = 0;
    const auto draw = [&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const float unlimited = std::numeric_limits<float>::infinity();
      const Rect rects[] = {{100, 100, 200, 50}, {400, 100, 200, 50}};
      Element nodes[2];
      for (int i = 0; i < 2; ++i) {
        nodes[i] = Box(frame, {.item = {.width = Dp(rects[i].width), .height = Dp(rects[i].height)}}, {});
        Check(!Keyed(frame, nodes[i], i == 0 ? "first" : "second").IsNone(), "the node is keyed");
        InteractProps props;
        props.disabled = i == 0 && disabled;
        props.dpad = i == 0 ? DpadClaim::Once : DpadClaim::None;
        Check(!MakeInteractive(frame, nodes[i], props).IsNone(), "the node is interactive");
        (void)Measure(frame, nodes[i], {unlimited, unlimited});
        Check(Arrange(frame, nodes[i], rects[i]), "the node arranges");
        Check(!Render(frame, nodes[i], context.list).error.has_value(), "the node renders");
      }
      first_id = IdOf(frame, nodes[0]);
      first_focused = InputOf(frame, nodes[0])->interaction.state.focused;
      second_focused = InputOf(frame, nodes[1])->interaction.state.focused;
    };
    nav.Frames(2, draw);
    Check(first_focused, "the claiming node starts with focus");
    nav.Key(ImGuiKey_GamepadDpadRight, true);
    nav.Frames(2, draw);
    Check(first_focused && !second_focused, "holding right, it keeps focus since it owns the key");
    disabled = disable;
    nav.Frames(40, draw);
    // The key is left with no owner when the node is disabled, and still the node's when it is not.
    // The last frame's owner is read as the frames end, before the key is let go.
    const ImGuiID owner = ImGui::GetKeyOwner(ImGuiKey_GamepadDpadRight);
    Check(disable ? owner == ImGuiKeyOwner_NoOwner : owner == first_id,
          disable ? "disabled, nobody owns the key" : "enabled, the node still owns the key");
    nav.Key(ImGuiKey_GamepadDpadRight, false);
    nav.Frame(draw);
    Check(second_focused == disable && first_focused != disable,
          disable ? "disabled, it gives the key back and the held right moves focus on" : "enabled, it still owns the key and keeps focus");
  }
}

// How long the entrance of the animation in TestCollapsedAnimationReplaysItsEntrance takes, in ms.
constexpr float ENTRANCE_MS = 100.0f;

// A collapsed interactive child is not there to navigate to. Two interactive boxes in a row, the
// second collapsing, with focus on the first: D-pad right moves focus to the second when the row is
// wide, and never reaches it when the row is short enough to collapse it. The collapsed node
// registers nothing, so it has no input to report either.
void TestCollapsedInteractiveChildIsNotNavigable() {
  for (const bool wide : {true, false}) {
    NavHarness nav;
    UiFrame frame;
    bool first_focused = false;
    bool second_focused = false;
    bool second_reported = false;
    const float room = wide ? 400.0f : 100.0f;
    const auto draw = [&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      std::array<Element, 2> kids{Box(frame, {.item = {.width = Dp(50), .height = Dp(40)}}, {}),
                                  Box(frame, {.item = {.width = Dp(60), .height = Dp(40), .collapses = true}}, {})};
      const char *keys[] = {"first", "second"};
      for (int i = 0; i < 2; ++i) {
        Check(!Keyed(frame, kids[i], keys[i]).IsNone() && !MakeInteractive(frame, kids[i], InteractProps{}).IsNone(), "the child is interactive");
      }
      const Element row = Box(frame, {}, {}, kids);
      const float unlimited = std::numeric_limits<float>::infinity();
      (void)Measure(frame, row, {room, unlimited});
      Check(Arrange(frame, row, {100, 100, room, 40}), "the row arranges");
      Check(!Render(frame, row, context.list).error.has_value(), "the row renders");
      first_focused = InputOf(frame, kids[0])->interaction.state.focused;
      const auto second = InputOf(frame, kids[1]);
      second_reported = second.has_value();
      second_focused = second && second->interaction.state.focused;
    };
    nav.Frames(2, draw);
    Check(first_focused && !second_focused, "the first child starts with focus");
    nav.Press(ImGuiKey_GamepadDpadRight, draw);
    if (wide) {
      Check(second_focused && !first_focused, "in a wide row right moves focus to the second child");
    } else {
      Check(!second_reported && !second_focused && first_focused, "in a short row the collapsed child is never reached");
    }
  }
}

// A keyed animation on a node that collapses is not played while it is collapsed, so it is forgotten
// and starts its entrance again when the node comes back, as a node missing for a frame does. The
// node is shown, collapsed for a frame by a row too short, and shown again in a wide row.
void TestCollapsedAnimationReplaysItsEntrance() {
  NavHarness nav;
  std::optional<Phase> phase;
  const auto at = [](double ms) { return UiClock::time_point{} + std::chrono::duration_cast<UiClock::duration>(std::chrono::duration<double, std::milli>(ms)); };
  const auto frame_at = [&](double ms, float room) {
    nav.Frame([&](InteractContext &context) {
      UiFrame frame;
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, at(ms));
      Animate animate;
      animate.timeline.enter = {{ENTRANCE_MS, Ease::Standard}, {ENTRANCE_MS, Ease::Standard}, {0, 0}, 1.0f};
      Element moving = Keyed(frame, Box(frame, {.item = {.width = Dp(60), .height = Dp(20), .collapses = true}}, {}), "moving");
      Check(!WithAnimation(frame, moving, animate).IsNone(), "the animation is accepted");
      const std::array<Element, 2> kids{Box(frame, {.item = {.width = Dp(50), .height = Dp(20)}}, {}), moving};
      const Element row = Keyed(frame, Box(frame, {}, {}, kids), "row");
      const float unlimited = std::numeric_limits<float>::infinity();
      (void)Measure(frame, row, {room, unlimited});
      Check(Arrange(frame, row, {0, 0, room, 20}), "the row arranges");
      Check(!Render(frame, row, context.list).error.has_value(), "the row renders");
      phase = PhaseOf(frame, moving);
    });
  };
  frame_at(0, 400);
  frame_at(ENTRANCE_MS * 4, 400);
  Check(phase.has_value() && *phase == Phase::Shown, "the animation has finished while the node is wide");
  frame_at(ENTRANCE_MS * 5, 100);
  Check(!phase.has_value(), "collapsed, the node is not played");
  frame_at(ENTRANCE_MS * 6, 400);
  Check(phase.has_value() && *phase == Phase::Entering, "back again, it enters again");
}

// An element button with an activation callback that counts, the real Button, and the count in
// RenderResult and InputOf, through confirm and click. Every frame the four agree, so a callback
// that is skipped fails here while the return values still pass.
struct CountActivations {
  int *count;
  void operator()(Element) const { ++*count; }
};

struct CountInputs {
  int *calls;
  int *activated;
  void operator()(const NodeInput &input) const {
    ++*calls;
    *activated += input.interaction.state.activated ? 1 : 0;
  }
};

// A box of 200 by 50 at 100, 100 that is interactive with `disabled` and the shoulder keys, takes
// the D-pad through ClaimDpad, and counts activations. Returns the element, and sets `why` to what
// ClaimDpad said.
Element ClaimedBox(UiFrame &frame, bool disabled, int *activations, ElementError &why) {
  Layout layout{};
  layout.item = {.width = Dp(200), .height = Dp(50)};
  const Element box = Keyed(frame, Box(frame, layout, {}), "claimed");
  InteractProps props;
  props.disabled = disabled;
  props.keys = KeyOwner{.shoulders = true};
  Check(!MakeInteractive(frame, box, props).IsNone(), "the box is interactive");
  Check(!OnActivate(frame, box, CountActivations{activations}).IsNone(), "and counts activations");
  Check(!ClaimDpad(frame, box, DpadClaim::Once, &why).IsNone(), "and then claims the D-pad");
  return box;
}

// What a ClaimedBox did over a click on it, and whether it still owned its keys.
struct ClaimedOutcome {
  int activations = 0;
  bool ownsKeys = false;
};

// Clicks a ClaimedBox, with `disabled` as given, and reports what it did.
ClaimedOutcome ClickClaimedBox(bool disabled) {
  NavHarness nav;
  FrameRig rig;
  ClaimedOutcome outcome;
  const auto draw = [&](InteractContext &context) {
    rig.Begin();
    ElementError why = ElementError::None;
    const Element box = ClaimedBox(rig.frame, disabled, &outcome.activations, why);
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(rig.frame, box, {inf, inf});
    (void)Arrange(rig.frame, box, {100, 100, 200, 50});
    (void)Render(rig.frame, box, context.list);
    outcome.ownsKeys = KeyPressesOf(rig.frame, box).has_value();
  };
  nav.Frames(2, draw);
  nav.MoveMouse(150, 120, draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(2, draw);
  return outcome;
}

// Whether an interactive child filling a clipping box of 200 by 100 at 100, 100 with a radius of 40
// is hovered with the mouse at `x`, `y`, where the box is made a list when `as_list`.
bool ChildHoveredAt(bool as_list, float x, float y) {
  NavHarness nav;
  FrameRig rig;
  bool hovered = false;
  const auto draw = [&](InteractContext &context) {
    rig.Begin();
    Layout childLayout{};
    childLayout.item = {.width = Dp(200), .height = Dp(100)};
    const Element child = Keyed(rig.frame, Box(rig.frame, childLayout, {}), "child");
    Check(!MakeInteractive(rig.frame, child, {}).IsNone(), "the child is interactive");
    const std::array<Element, 1> kids{child};
    Layout layout{};
    layout.item = {.width = Dp(200), .height = Dp(100)};
    Style style{};
    style.radius = Dp(40);
    style.overflow = Overflow::Clip;
    const Element box = Keyed(rig.frame, Box(rig.frame, layout, style, kids), "box");
    if (as_list) {
      Check(!AsList(rig.frame, box).IsNone(), "the box is a list");
    }
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(rig.frame, box, {inf, inf});
    (void)Arrange(rig.frame, box, {100, 100, 200, 100});
    (void)Render(rig.frame, box, context.list);
    const auto input = InputOf(rig.frame, child);
    hovered = input && input->interaction.state.hovered;
  };
  nav.Frames(2, draw);
  nav.MoveMouse(x, y, draw);
  nav.Frames(2, draw);
  return hovered;
}

// A clipping box cuts hit tests at its rounded corners, so a child can't be hovered in the part of
// a corner the box doesn't draw. A list clips to its exact rectangle instead, so a row partly in
// view near a rounded corner can still be hovered there. The middle of each is the baseline, where
// the child is hovered whichever the box is.
void TestListHitClipIgnoresRoundedCorners() {
  Check(ChildHoveredAt(false, 200, 150) && ChildHoveredAt(true, 200, 150), "the middle of either box hovers the child");
  Check(!ChildHoveredAt(false, 101, 101), "a plain clipping box leaves its rounded corner unhoverable");
  Check(ChildHoveredAt(true, 101, 101), "a list's clip is its exact rectangle, corners included");
}

// A clipping box's hit clip ends with the box, so a control drawn after it, outside it, can still be
// hovered. The child inside is the baseline, which shows the pointer's place tells the two apart.
// TestClipLimitsHitTesting covers that the clip limits what's inside.
void TestHitClipEndsWithItsBox() {
  NavHarness nav;
  FrameRig rig;
  bool after_hovered = false;
  bool inside_hovered = false;
  const auto interactive = [&](const char *key) {
    Layout layout{};
    layout.item = {.width = Dp(200), .height = Dp(100)};
    const Element element = Keyed(rig.frame, Box(rig.frame, layout, {}), key);
    Check(!MakeInteractive(rig.frame, element, {}).IsNone(), "the control is interactive");
    return element;
  };
  const auto draw = [&](InteractContext &context) {
    rig.Begin();
    const Element inside = interactive("inside");
    const std::array<Element, 1> kids{inside};
    Layout layout{};
    layout.item = {.width = Dp(200), .height = Dp(100)};
    Style style{};
    style.overflow = Overflow::Clip;
    const Element box = Keyed(rig.frame, Box(rig.frame, layout, style, kids), "box");
    const Element after = interactive("after");
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(rig.frame, box, {inf, inf});
    (void)Arrange(rig.frame, box, {100, 100, 200, 100});
    (void)Render(rig.frame, box, context.list);
    (void)Measure(rig.frame, after, {inf, inf});
    (void)Arrange(rig.frame, after, {500, 100, 200, 100});
    (void)Render(rig.frame, after, context.list);
    const auto inside_input = InputOf(rig.frame, inside);
    const auto after_input = InputOf(rig.frame, after);
    inside_hovered = inside_input && inside_input->interaction.state.hovered;
    after_hovered = after_input && after_input->interaction.state.hovered;
  };
  nav.Frames(2, draw);
  nav.MoveMouse(200, 150, draw);
  nav.Frames(2, draw);
  Check(inside_hovered && !after_hovered, "inside the box, its own child is hovered");
  nav.MoveMouse(600, 150, draw);
  nav.Frames(2, draw);
  Check(after_hovered && !inside_hovered, "and a control after the box, outside it, can be hovered too");
}

// ClaimDpad changes only the D-pad claim. A disabled box that claims it stays disabled, so a click
// on it activates nothing, and the keys it owned before the claim are still owned. The same box
// enabled is the baseline, which does activate on the click, so the click really lands there.
void TestClaimDpadKeepsTheRestOfTheRecord() {
  const ClaimedOutcome disabled = ClickClaimedBox(true);
  const ClaimedOutcome enabled = ClickClaimedBox(false);
  Check(enabled.activations > 0, "the enabled box activates on the click, which is the baseline");
  Check(disabled.activations == 0, "a disabled box that claims the D-pad stays disabled");
  Check(disabled.ownsKeys && enabled.ownsKeys, "and the keys owned before the claim are kept");
}

// A second MakeInteractive on a node is refused with AlreadyInteractive and changes nothing, and
// ClaimDpad refuses a node that is not interactive, a stale handle and a none handle.
void TestSecondMakeInteractiveRefused() {
  InWindow([](UiFrame &frame) {
    ElementError reason = ElementError::None;
    const Element box = Leaf(frame);
    Check(!MakeInteractive(frame, box, {.disabled = true}, &reason).IsNone() && reason == ElementError::None, "the first call is accepted");
    Check(MakeInteractive(frame, box, {}, &reason).IsNone() && reason == ElementError::AlreadyInteractive, "a second call is refused");
    reason = ElementError::None;
    const Element plain = Leaf(frame);
    Check(ClaimDpad(frame, plain, DpadClaim::Once, &reason).IsNone() && reason == ElementError::NotInteractive, "a node that is not interactive can't claim the D-pad");
    // A none handle refers to nothing, which is what ElementError::None says.
    reason = ElementError::AlreadyRendered;
    Check(ClaimDpad(frame, Element{}, DpadClaim::Once, &reason).IsNone() && reason == ElementError::None, "a none handle is refused");
    reason = ElementError::None;
    Check(!ClaimDpad(frame, box, DpadClaim::Repeat, &reason).IsNone() && reason == ElementError::None, "an interactive node takes the claim, and again");
  });
}

// What this proves now is that the activation callback, the input callback, Render's result and
// InputOf all report the activations the Button component reports, so the callbacks of a hand-built
// tree of primitives count what the component's own Render result does. The direct framework
// assertions are TestCallbackFlagIsSetInsideAndClearedAfter, TestCallbacksRefusedOnNonInteractive and
// TestElementActivation, and SystemUiButtonTests for the button itself.
void TestCallbackCountsMatchActivations() {
  const Rect rect{400, 300, 240, 56};
  NavHarness oldNav, eleNav;
  UiFrame realFrame, frame;
  int oldCount = 0, callbackCount = 0, resultCount = 0, inputCount = 0, inputCalls = 0, inputActivated = 0, frameCount = 0;
  // The real Button element, which reports an activation only through Render's result.
  const auto drawOld = [&](InteractContext &context) {
    realFrame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element real = Button(realFrame, "button", {.label = "Save"});
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(realFrame, real, {inf, inf});
    Check(Arrange(realFrame, real, rect), "the real button arranges");
    oldCount += Render(realFrame, real, context.list).activated.IsNone() ? 0 : 1;
  };
  const auto drawEle = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element button = ElementButton(frame, "button", rect, false, false);
    Check(!OnActivate(frame, button, CountActivations{&callbackCount}).IsNone(), "the activation callback is accepted");
    Check(!OnInput(frame, button, CountInputs{&inputCalls, &inputActivated}).IsNone(), "the input callback is accepted");
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(frame, button, {inf, inf});
    Check(Arrange(frame, button, rect), "the button arranges");
    const RenderResult result = Render(frame, button, context.list);
    Check(!result.error.has_value(), "the button renders");
    resultCount += result.activated.IsNone() ? 0 : 1;
    inputCount += InputOf(frame, button)->interaction.state.activated ? 1 : 0;
    ++frameCount;
  };
  const auto frames = [&](int count) {
    for (int i = 0; i < count; ++i) {
      oldNav.Frame(drawOld);
      eleNav.Frame(drawEle);
      Check(callbackCount == oldCount && resultCount == oldCount && inputCount == oldCount && inputActivated == oldCount,
            "the callback count, the Render result, InputOf and OnInput all equal the real Button element's");
    }
  };
  frames(2);
  oldNav.Key(ImGuiKey_GamepadFaceDown, true);
  eleNav.Key(ImGuiKey_GamepadFaceDown, true);
  frames(10);
  oldNav.Key(ImGuiKey_GamepadFaceDown, false);
  eleNav.Key(ImGuiKey_GamepadFaceDown, false);
  frames(10);
  oldNav.MoveMouse(520, 330, drawOld);
  eleNav.MoveMouse(520, 330, drawEle);
  oldNav.MouseButton(true);
  eleNav.MouseButton(true);
  frames(4);
  oldNav.MouseButton(false);
  eleNav.MouseButton(false);
  frames(6);
  Check(oldCount == 2 && callbackCount == 2, "confirm and click each activated once, however long confirm was held");
  Check(inputCalls == frameCount, "OnInput ran on every frame the node interacted");
}

// With callbacks the frame's reserved bytes stop growing once warm. The callables are padded so the
// storage they take shows against the same tree without callbacks, and growth every frame, from
// storage that is never cleared, would show as a change.
struct PaddedCallback {
  int *count;
  std::array<char, 512> padding;
  void operator()(Element) const { ++*count; }
  void operator()(const NodeInput &) const { ++*count; }
};

size_t SteadyReserved(bool callbacks) {
  NavHarness nav;
  UiFrame frame;
  int count = 0;
  size_t warm = 0;
  for (int i = 0; i < 40; ++i) {
    nav.Frame([&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Rect rect{400, 300, 240, 56};
      const Element button = ElementButton(frame, "button", rect, false, false);
      if (callbacks) {
        Check(!OnActivate(frame, button, PaddedCallback{&count, {}}).IsNone(), "the callback is accepted");
        Check(!OnInput(frame, button, PaddedCallback{&count, {}}).IsNone(), "and so is the second");
      }
      const auto inf = std::numeric_limits<float>::infinity();
      (void)Measure(frame, button, {inf, inf});
      Check(Arrange(frame, button, rect), "the button arranges");
      Check(!Render(frame, button, context.list).error.has_value(), "the button renders");
    });
    if (i == 5) {
      warm = frame.ReservedBytes();
    }
    if (i > 5) {
      Check(frame.ReservedBytes() == warm, "reserved bytes are steady across frames");
    }
  }
  return warm;
}

void TestCallbacksKeepReservedBytesSteady() {
  const size_t with = SteadyReserved(true);
  const size_t without = SteadyReserved(false);
  constexpr size_t PADDED_BYTES = 2 * sizeof(PaddedCallback);
  Check(with >= without + PADDED_BYTES, "the callbacks' bytes are in the frame's storage, so steadiness is about them");
}

// A callback is refused on a node that is not interactive, whatever its kind, and on a stale handle
// with the reason that handle gets elsewhere. Refused, it is never called. The interactive node is
// the baseline that is accepted and called.
void TestCallbacksRefusedOnNonInteractive() {
  NavHarness nav;
  UiFrame frame;
  int count = 0;
  Element stale;
  for (int i = 0; i < 3; ++i) {
    nav.Frame([&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      if (i == 1) {
        ElementError reason = ElementError::None;
        Check(OnActivate(frame, stale, CountActivations{&count}, &reason).IsNone() && reason == ElementError::Stale, "a stale handle is refused as stale");
      }
      const ItemLayout size{.width = Dp(100.0f), .height = Dp(40.0f)};
      const Element plain = Box(frame, {.item = size}, {});
      const Element text = Text(frame, TextProps{"Label", frame.ActiveTheme().type.label, frame.ActiveTheme().colors.text.primary});
      ElementError reason = ElementError::None;
      Check(OnActivate(frame, plain, CountActivations{&count}, &reason).IsNone() && reason == ElementError::NotInteractive, "a plain box is refused");
      reason = ElementError::None;
      int calls = 0, activated = 0;
      Check(OnInput(frame, plain, CountInputs{&calls, &activated}, &reason).IsNone() && reason == ElementError::NotInteractive, "so is its input callback");
      reason = ElementError::None;
      Check(OnActivate(frame, text, CountActivations{&count}, &reason).IsNone() && reason == ElementError::NotInteractive, "and a text node");
      Check(OnActivate(frame, Element{}, CountActivations{&count}).IsNone(), "and a none handle");
      Check(!MakeInteractive(frame, plain, {}).IsNone(), "once made interactive");
      Check(!OnActivate(frame, plain, CountActivations{&count}).IsNone(), "the same node accepts it");
      stale = plain;
      Check(frame.ReservedBytes() > 0, "the frame is live");
    });
  }
  Check(count == 0, "nothing was activated, so the callback of the accepted node was not called either");
  // Confirm on a tree with a refused callback and an accepted one shows only the accepted is called.
  int refusedCount = 0, acceptedCount = 0;
  // A fresh window, since ImGui gives first focus when a window appears with something to focus.
  NavHarness pressNav;
  for (int i = 0; i < 4; ++i) {
    if (i == 2) {
      pressNav.Key(ImGuiKey_GamepadFaceDown, true);
    }
    pressNav.Frame([&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const ItemLayout size{.width = Dp(100.0f), .height = Dp(40.0f)};
      const Element refused = Box(frame, {.item = size}, {});
      const Element accepted = Box(frame, {.item = size}, {});
      Check(OnActivate(frame, refused, CountActivations{&refusedCount}).IsNone(), "the plain box is refused");
      Check(!MakeInteractive(frame, accepted, {}).IsNone(), "the other is interactive");
      Check(!OnActivate(frame, accepted, CountActivations{&acceptedCount}).IsNone(), "and accepts it");
      const std::array<Element, 2> pair{refused, accepted};
      const Element root = Box(frame, {.direction = FlexDirection::Column}, {}, pair);
      const auto inf = std::numeric_limits<float>::infinity();
      (void)Measure(frame, root, {inf, inf});
      Check(Arrange(frame, root, {100, 100, 100, 80}), "the pair arranges");
      Check(!Render(frame, root, context.list).error.has_value(), "the pair renders");
    });
  }
  pressNav.Key(ImGuiKey_GamepadFaceDown, false);
  Check(acceptedCount == 1 && refusedCount == 0, "confirm calls the accepted callback once and the refused one never");
}

// The look of the label in a button, by the colours the label box draws.
struct LabelLook {
  int base = 0;
  int hover = 0;
  bool hovered = false;
};

// A button box holding a label box whose own hover style changes its fill. The mouse is over the
// button when `hovering`. `explicitRest` gives the label an input of its own by hand, which must
// win over the button's. The counts are of the vertices in the label's resting and hover fills.
LabelLook RunLabel(bool hovering, bool explicitRest) {
  NavHarness nav;
  UiFrame frame;
  LabelLook look;
  const Rect rect{400, 300, 240, 56};
  for (int i = 0; i < 30; ++i) {
    if (i == 3 && hovering) {
      nav.MoveMouse(520, 330, [](InteractContext &) {});
    }
    nav.Frame([&](InteractContext &context) {
      const Theme &theme = context.ui.theme;
      frame.Begin(theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Color restFill = theme.colors.surface.overlay;
      const Color hoverFill = theme.colors.accent.fill;
      Style labelStyle{restFill, {}, Dp(0.0f), Dp(0.0f), {}};
      labelStyle.states.hover.fill = hoverFill;
      Layout labelLayout{};
      labelLayout.item = {.width = Dp(80.0f), .height = Dp(20.0f)};
      const Element label = Box(frame, labelLayout, labelStyle);
      if (explicitRest) {
        Check(SetInput(frame, label, {}), "the label's own input is accepted");
      }
      const std::array<Element, 1> kids{label};
      Layout layout{.direction = FlexDirection::Column, .justify = Justify::Center, .align = Align::Center};
      layout.item = {.width = Dp(rect.width), .height = Dp(rect.height)};
      const Element button = Box(frame, layout, Style{theme.colors.border.standard, {}, Dp(0.0f), Dp(0.0f), {}}, kids);
      Check(!Keyed(frame, button, "button").IsNone() && !MakeInteractive(frame, button, {}).IsNone(), "the button is interactive");
      const auto inf = std::numeric_limits<float>::infinity();
      (void)Measure(frame, button, {inf, inf});
      Check(Arrange(frame, button, rect), "the button arranges");
      const int begin = context.list.VtxBuffer.Size;
      Check(!Render(frame, button, context.list).error.has_value(), "the button renders");
      const ImU32 rest = ToImU32(restFill);
      const ImU32 hover = ToImU32(hoverFill);
      look = {};
      for (int v = begin; v < context.list.VtxBuffer.Size; ++v) {
        look.base += context.list.VtxBuffer[v].col == rest ? 1 : 0;
        look.hover += context.list.VtxBuffer[v].col == hover ? 1 : 0;
      }
      look.hovered = InputOf(frame, button)->interaction.visuals.hover >= 1.0f;
    });
  }
  return look;
}

// A state style on a label inside an interactive box follows the box's input, as CSS :hover on a
// parent does. Without hover the label rests, which is the baseline, and a label with an input of
// its own is left alone by the box's.
void TestDescendantStyleFollowsInteractiveAncestor() {
  const LabelLook resting = RunLabel(false, false);
  Check(!resting.hovered && resting.base > 0 && resting.hover == 0, "without hover the label rests");
  const LabelLook hovered = RunLabel(true, false);
  Check(hovered.hovered, "the button is fully hovered");
  Check(hovered.hover > 0 && hovered.base == 0, "and its label, which is not interactive, takes its hover fill");
  const LabelLook explicitRest = RunLabel(true, true);
  Check(explicitRest.hovered && explicitRest.hover == 0 && explicitRest.base > 0, "a label with an input of its own keeps it though the button is hovered");
}

// A disabled button fades everything inside it once. The button's box and the box inside it are
// compared with DrawBox at the theme's disabled opacity, as the real Button draws its parts, and
// the enabled button at full opacity is the baseline. A box that inherited the disabled input and
// faded again would draw at the square of it.
void DrawnTwice(bool disabled, const Rect &rect, const Rect &inner, std::vector<ImDrawVert> &element, std::vector<ImDrawVert> &expected) {
  NavHarness nav;
  UiFrame frame;
  for (int i = 0; i < 3; ++i) {
    nav.Frame([&](InteractContext &context) {
      const Theme &theme = context.ui.theme;
      frame.Begin(theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Color outerFill = theme.colors.border.standard;
      const Color innerFill = theme.colors.accent.fill;
      Layout innerLayout{};
      innerLayout.item = {.width = Dp(inner.width), .height = Dp(inner.height)};
      const Element label = Box(frame, innerLayout, Style{innerFill, {}, Dp(0.0f), Dp(0.0f), {}});
      const std::array<Element, 1> kids{label};
      Layout layout{.direction = FlexDirection::Column, .justify = Justify::Center, .align = Align::Center};
      layout.item = {.width = Dp(rect.width), .height = Dp(rect.height)};
      const Element button = Box(frame, layout, Style{outerFill, {}, Dp(0.0f), Dp(0.0f), {}}, kids);
      InteractProps props;
      props.disabled = disabled;
      Check(!MakeInteractive(frame, button, props).IsNone(), "the button is interactive");
      const auto inf = std::numeric_limits<float>::infinity();
      (void)Measure(frame, button, {inf, inf});
      Check(Arrange(frame, button, rect), "the button arranges");
      int begin = context.list.VtxBuffer.Size;
      Check(!Render(frame, button, context.list).error.has_value(), "the button renders");
      element.assign(context.list.VtxBuffer.begin() + begin, context.list.VtxBuffer.end());
      const float opacity = disabled ? theme.opacity.disabled : 1.0f;
      begin = context.list.VtxBuffer.Size;
      DrawBox(context.list, context.ui, rect, BoxStyle{outerFill, 0.0f, Color{}, 0.0f, {}}, opacity);
      DrawBox(context.list, context.ui, Arranged(frame, label), BoxStyle{innerFill, 0.0f, Color{}, 0.0f, {}}, opacity);
      expected.assign(context.list.VtxBuffer.begin() + begin, context.list.VtxBuffer.end());
    });
  }
}

void TestDisabledFadesInheritingBoxesOnce() {
  const Rect rect{400, 300, 240, 56};
  const Rect inner{0, 0, 80, 20};
  for (const bool disabled : {false, true}) {
    std::vector<ImDrawVert> element, expected;
    DrawnTwice(disabled, rect, inner, element, expected);
    Check(!element.empty() && element.size() == expected.size(), "the tree draws both boxes");
    for (size_t v = 0; v < element.size(); ++v) {
      Check(element[v].col == expected[v].col, "both boxes draw at the real button's opacity, faded once");
      Check(std::fabs(element[v].pos.x - expected[v].pos.x) <= GEOMETRY_SLACK && std::fabs(element[v].pos.y - expected[v].pos.y) <= GEOMETRY_SLACK, "in the same place");
    }
  }
}

// ---- C3 follow-ups ---------------------------------------------------------------------------

// What a callback saw of the frame's callback flag.
struct FlagProbe {
  const UiFrame *frame;
  bool *seen;
  void operator()(Element) const { *seen = FrameTestHooks::InCallback(*frame); }
  void operator()(const NodeInput &) const {}
};

struct InputFlagProbe {
  const UiFrame *frame;
  bool *seen;
  void operator()(const NodeInput &) const { *seen = FrameTestHooks::InCallback(*frame); }
};

// The flag that makes the primitives refuse to build elements is set while a callback runs and is
// clear again once Render returns, which runs in Release too. The frames before the press are the
// baseline, where no callback runs and the flag is clear.
void TestCallbackFlagIsSetInsideAndClearedAfter() {
  NavHarness nav;
  UiFrame frame;
  const Rect rect{400, 300, 240, 56};
  bool activateSaw = false, inputSaw = false;
  int inputFrames = 0;
  bool clearAfterEveryRender = true;
  const auto draw = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element button = ElementButton(frame, "button", rect, false, false);
    Check(!OnActivate(frame, button, FlagProbe{&frame, &activateSaw}).IsNone(), "the activation callback is accepted");
    Check(!OnInput(frame, button, InputFlagProbe{&frame, &inputSaw}).IsNone(), "the input callback is accepted");
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(frame, button, {inf, inf});
    Check(Arrange(frame, button, rect), "the button arranges");
    Check(!FrameTestHooks::InCallback(frame), "the flag is clear before Render");
    Check(!Render(frame, button, context.list).error.has_value(), "the button renders");
    clearAfterEveryRender = clearAfterEveryRender && !FrameTestHooks::InCallback(frame);
    ++inputFrames;
  };
  nav.Frames(2, draw);
  Check(!activateSaw && inputSaw, "OnInput saw the flag set on a frame with no activation, OnActivate has not run");
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(2, draw);
  Check(activateSaw, "OnActivate saw the flag set");
  Check(clearAfterEveryRender && inputFrames == 6, "and it was clear after every Render");
}

// SetInput takes the node's own input, so a caller can't hand it one marked as inherited to skip the
// disabled fade. The same box set up through DrawBox at the disabled opacity is what it must match.
void TestSetInputClearsInherited() {
  NavHarness nav;
  UiFrame frame;
  const Rect rect{400, 300, 240, 56};
  for (const bool disabled : {false, true}) {
    std::vector<ImDrawVert> element, expected;
    nav.Frame([&](InteractContext &context) {
      const Theme &theme = context.ui.theme;
      frame.Begin(theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Color fill = theme.colors.accent.fill;
      Layout layout{};
      layout.item = {.width = Dp(rect.width), .height = Dp(rect.height)};
      const Element box = Box(frame, layout, Style{fill, {}, Dp(0.0f), Dp(0.0f), {}});
      ItemInput input;
      input.disabled = disabled;
      input.inherited = true;
      Check(SetInput(frame, box, input), "the input is accepted");
      const auto inf = std::numeric_limits<float>::infinity();
      (void)Measure(frame, box, {inf, inf});
      Check(Arrange(frame, box, rect), "the box arranges");
      int begin = context.list.VtxBuffer.Size;
      Paint(frame, box, context.list);
      element.assign(context.list.VtxBuffer.begin() + begin, context.list.VtxBuffer.end());
      begin = context.list.VtxBuffer.Size;
      DrawBox(context.list, context.ui, rect, BoxStyle{fill, 0.0f, Color{}, 0.0f, {}}, disabled ? theme.opacity.disabled : 1.0f);
      expected.assign(context.list.VtxBuffer.begin() + begin, context.list.VtxBuffer.end());
    });
    Check(!element.empty() && element.size() == expected.size(), "the box draws");
    for (size_t v = 0; v < element.size(); ++v) {
      Check(element[v].col == expected[v].col, "a disabled box fades once however its input was marked");
    }
  }
}

// ---- Lists, against the List component ---------------------------------------------------------------
// The tests below hold a list built from primitives and AsList to the List component, through
// ListDual. They show the component matches the primitives. Behaviour of the list itself is asserted
// directly in each test's own checks and in SystemUiListTests.

constexpr Rect LIST_ABOVE{100, 40, 300, 56};
constexpr Rect LIST_AREA{100, 140, 300, 400};
constexpr Rect LIST_BELOW{100, 720, 300, 56};
constexpr float LIST_ROW = 56;
// The focus ring's width, three pixels and a two pixel keyline, which rows sit in from the edges.
constexpr float LIST_INSET = 5;

// What a list scene draws.
struct ListScene {
  int rows = 5;
  int selected = -1;
  int disabled = -1;
  bool draw_list = true;
  // Draws a control below the list, first, so it takes the first focus.
  bool below = false;
  // The corner radius of the element list's box, which the old list has no equivalent of.
  float radius = 0;
};

// What one side saw of a list scene on its last frame, and since the start.
struct ListSeen {
  bool above_focused = false;
  bool below_focused = false;
  int focused_row = -1;
  // Which control had focus once the list had ended, after its end may have sent focus back. A row
  // is its index, the control above is -2 and the one below is -3, and -1 is none.
  int focus_after = -1;
  int activated = -1;
  int activations = 0;
  float scroll = 0;
  bool bar = false;
  // The vertical extent of the vertices in the bar's colour.
  float bar_min = 0;
  float bar_max = 0;
  bool focus_above = false;
};

// True when some vertex any window drew in the last frame of the current ImGui context has `color`,
// and then the vertical extent of those vertices.
bool ListDrew(Color color, float &min_y, float &max_y) {
  bool found = false;
  for (ImGuiWindow *window : GImGui->Windows) {
    for (const ImDrawVert &vertex : window->DrawList->VtxBuffer) {
      if (vertex.col == ToImU32(color)) {
        min_y = found ? std::min(min_y, vertex.pos.y) : vertex.pos.y;
        max_y = found ? std::max(max_y, vertex.pos.y) : vertex.pos.y;
        found = true;
      }
    }
  }
  return found;
}

// Which control has focus now, as ListSeen::focus_after says. The old list names its rows with
// `names`, and the element list with the ids of `ids`.
int ListFocusAfter(const std::vector<std::string> &names, const std::vector<ItemId> *ids) {
  const ItemId focus = ImGui::GetFocusID();
  for (size_t index = 0; index < names.size(); ++index) {
    const ItemId row = ids != nullptr ? (*ids)[index] : ImGui::GetID(names[index].c_str());
    if (focus == row) {
      return static_cast<int>(index);
    }
  }
  if (focus == ImGui::GetID("above")) {
    return -2;
  }
  return focus == ImGui::GetID("below") ? -3 : -1;
}

// A row of the hand-built list, styled as the List component's rows are, so a selected row draws
// the accent bar and nothing else in the scene draws the accent line.
Element ListElementRow(UiFrame &frame, const Theme &theme, int index, const ListScene &scene, std::string_view key) {
  Style style{{}, {}, Dp(0.0f), Dp(ListDefaults(theme).row_radius), {}};
  style.states.selected.selection = true;
  style.states.hover.overlay = theme.colors.state.hover;
  style.states.focus.overlay = theme.colors.state.focus;
  style.states.focus.focus_ring = true;
  Layout layout{};
  layout.item = {.height = Dp(LIST_ROW)};
  const Element row = Box(frame, layout, style);
  Check(!Keyed(frame, row, key).IsNone(), "the row is keyed");
  InteractProps props;
  props.disabled = index == scene.disabled;
  props.selected = index == scene.selected;
  Check(!MakeInteractive(frame, row, props).IsNone(), "the row is interactive");
  return row;
}

// The list box around `rows`, laid out in LIST_AREA and made a list.
Element ListElementBox(UiFrame &frame, const std::vector<Element> &rows, float radius = 0.0f) {
  Layout layout{.direction = FlexDirection::Column, .align = Align::Stretch};
  layout.padding = DpInsets::All(Dp(LIST_INSET));
  layout.item = {.width = Dp(LIST_AREA.width), .height = Dp(LIST_AREA.height)};
  Style style{};
  style.overflow = Overflow::Clip;
  style.radius = Dp(radius);
  const Element list = Box(frame, layout, style, rows);
  Check(!Keyed(frame, list, "list").IsNone(), "the list is keyed");
  ElementError reason = ElementError::None;
  Check(!AsList(frame, list, &reason).IsNone(), "the box is a list");
  return list;
}

// A list of the scene's rows, built by the List component of ListRows when `component`, and by hand
// from boxes made a list with AsList when not.
Element BuildList(InteractContext &context, UiFrame &frame, const ListScene &scene, bool component, std::vector<Element> &rows) {
  for (int index = 0; index < scene.rows; index++) {
    const std::string key = "row " + std::to_string(index);
    rows.push_back(component ? ListRow(frame, key, {"row", {}, index == scene.selected, index == scene.disabled})
                             : ListElementRow(frame, context.ui.theme, index, scene, key));
  }
  if (!component) {
    return ListElementBox(frame, rows, scene.radius);
  }
  return List(frame, "list", rows, {.width = Dp(LIST_AREA.width), .height = Dp(LIST_AREA.height)});
}

// The list, hand-built or the List component.
void DrawElementList(InteractContext &context, UiFrame &frame, const ListScene &scene, ListSeen &seen, bool component = false) {
  frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
  if (scene.below) {
    seen.below_focused = Interact(context, ImGui::GetID("below"), LIST_BELOW, false).state.focused;
  }
  seen.above_focused = Interact(context, ImGui::GetID("above"), LIST_ABOVE, false).state.focused;
  if (seen.focus_above) {
    ImGui::SetFocusID(ImGui::GetID("above"), ImGui::GetCurrentWindow());
    seen.focus_above = false;
  }
  seen.focused_row = -1;
  if (!scene.draw_list) {
    return;
  }
  // The list's memory, since the root's key makes its id the name's.
  seen.scroll = context.store.List(ImGui::GetID("list")).scroll;
  std::vector<Element> rows;
  const Element list = BuildList(context, frame, scene, component, rows);
  const auto inf = std::numeric_limits<float>::infinity();
  (void)Measure(frame, list, {inf, inf});
  Check(Arrange(frame, list, LIST_AREA), "the list arranges");
  Check(!Render(frame, list, context.list).error.has_value(), "the list renders");
  std::vector<std::string> names;
  std::vector<ItemId> ids;
  for (int index = 0; index < scene.rows; index++) {
    names.push_back("row");
    ids.push_back(IdOf(frame, rows[index]));
  }
  seen.focus_after = ListFocusAfter(names, &ids);
  for (int index = 0; index < scene.rows; index++) {
    if (InputOf(frame, rows[index])->interaction.state.activated) {
      seen.activated = index;
      ++seen.activations;
    }
    // Read from what Interact saw, since Render's end of list may already have sent focus back.
    if (InputOf(frame, rows[index])->interaction.state.focused) {
      seen.focused_row = index;
    }
  }
}

// The List component and a list built by hand from boxes and AsList, in a harness each, driven by
// the same input. After every frame the two must agree on which control has focus, what activated,
// how far the list scrolled and whether the selected row's bar was drawn. So the tests that use it
// prove that the component matches a hand-built tree of primitives, and they also assert the
// behaviour they are about directly on the hand-built side. The component's own claims, remembered
// row, scrolling, skipped disabled rows and activation, are asserted in SystemUiListTests
// (TestRememberedRow, TestScrolls, TestDisabledRowSkipped, TestActivation).
class ListDual {
public:
  ListScene scene;

  void Frame() {
    oldNav_.Frame([&](InteractContext &context) { DrawElementList(context, realFrame_, scene, old_, true); });
    old_.bar = ListDrew(DEFAULT_THEME.colors.accent.line, old_.bar_min, old_.bar_max);
    eleNav_.Frame([&](InteractContext &context) { DrawElementList(context, frame_, scene, ele_); });
    ele_.bar = ListDrew(DEFAULT_THEME.colors.accent.line, ele_.bar_min, ele_.bar_max);
    Compare();
  }
  void Frames(int count) {
    for (int i = 0; i < count; ++i) {
      Frame();
    }
  }
  void Key(ImGuiKey key, bool down) {
    oldNav_.Key(key, down);
    eleNav_.Key(key, down);
  }
  // Presses and releases `key` over two frames and runs one more, as NavHarness::Press does.
  void Press(ImGuiKey key) {
    Key(key, true);
    Frame();
    Key(key, false);
    Frames(2);
  }
  void MouseButton(bool down) {
    oldNav_.MouseButton(down);
    eleNav_.MouseButton(down);
  }
  void MoveMouse(float x, float y) {
    oldNav_.MoveMouse(x, y, [&](InteractContext &context) { DrawElementList(context, realFrame_, scene, old_, true); });
    old_.bar = ListDrew(DEFAULT_THEME.colors.accent.line, old_.bar_min, old_.bar_max);
    eleNav_.MoveMouse(x, y, [&](InteractContext &context) { DrawElementList(context, frame_, scene, ele_); });
    ele_.bar = ListDrew(DEFAULT_THEME.colors.accent.line, ele_.bar_min, ele_.bar_max);
    Compare();
  }
  // Clicks at a point, with a frame for the hover to settle before the press.
  void Click(float x, float y) {
    MoveMouse(x, y);
    Frame();
    MouseButton(true);
    Frame();
    MouseButton(false);
    Frames(3);
  }
  // Moves focus to the control above on the next frame, as a screen change would.
  void FocusAbove() { old_.focus_above = ele_.focus_above = true; }
  const ListSeen &Ele() const { return ele_; }
  int compared() const { return compared_; }

private:
  void Compare() {
    Check(old_.above_focused == ele_.above_focused && old_.below_focused == ele_.below_focused, "the controls around the list have focus as they do beside the old list");
    Check(old_.focused_row == ele_.focused_row, "the same row has focus");
    Check(old_.focus_after == ele_.focus_after, "and so has the same control once the list has ended");
    Check(old_.activated == ele_.activated && old_.activations == ele_.activations, "and rows activate as often");
    Check(Near(old_.scroll, ele_.scroll), "and the list has scrolled as far");
    Check(old_.bar == ele_.bar, "and draws the selected row's bar alike");
    ++compared_;
  }

  NavHarness oldNav_;
  NavHarness eleNav_;
  UiFrame realFrame_;
  UiFrame frame_;
  ListSeen old_;
  ListSeen ele_;
  int compared_ = 0;
};

void TestListRememberedRow() {
  ListDual run;
  run.Frames(2);
  Check(run.Ele().above_focused, "the control above starts with focus");
  run.Press(ImGuiKey_GamepadDpadDown);
  Check(run.Ele().focused_row == 0, "down enters the list at its first row");
  run.Press(ImGuiKey_GamepadDpadDown);
  run.Press(ImGuiKey_GamepadDpadDown);
  Check(run.Ele().focused_row == 2, "down moves along the rows");
  run.FocusAbove();
  run.Frames(2);
  Check(run.Ele().above_focused, "focus leaves the list");
  run.Press(ImGuiKey_GamepadDpadDown);
  Check(run.Ele().focused_row == 2, "coming back returns to the row it left");
}

void TestListSelectedWithoutFocus() {
  ListDual run;
  run.scene.selected = 3;
  run.Frames(3);
  Check(run.Ele().above_focused && run.Ele().bar, "a selected row shows its bar without focus");
  ListDual plain;
  plain.Frames(3);
  Check(!plain.Ele().bar, "a list with nothing selected has no bar");
}

void TestListDisabledRowSkipped() {
  ListDual run;
  run.scene.disabled = 1;
  run.Frames(2);
  run.Press(ImGuiKey_GamepadDpadDown);
  run.Press(ImGuiKey_GamepadDpadDown);
  Check(run.Ele().focused_row == 2, "down skips the disabled row");
}

void TestListActivation() {
  ListDual run;
  run.Frames(2);
  run.Press(ImGuiKey_GamepadDpadDown);
  run.Press(ImGuiKey_GamepadDpadDown);
  run.Press(ImGuiKey_GamepadFaceDown);
  Check(run.Ele().activated == 1 && run.Ele().activations == 1, "confirm activates the focused row once");
}

void TestListScrolls() {
  ListDual run;
  run.scene.rows = 20;
  run.Frames(2);
  for (int press = 0; press < 16; press++) {
    run.Press(ImGuiKey_GamepadDpadDown);
  }
  run.Frames(10);
  const float top = LIST_AREA.y + LIST_INSET + 15 * LIST_ROW - run.Ele().scroll;
  Check(run.Ele().focused_row == 15 && run.Ele().scroll > 0 && top >= LIST_AREA.y && top + LIST_ROW <= LIST_AREA.Bottom(), "focus deep in a long list scrolls it into view");
}

void TestListHiddenRowsIgnoreTheMouse() {
  ListDual run;
  run.scene.rows = 20;
  run.Frames(2);
  run.Click(200, 670);
  Check(run.Ele().activated == -1 && run.Ele().focused_row == -1 && run.Ele().scroll == 0.0f, "a row outside the list's area ignores the mouse");
  run.Click(200, 180);
  Check(run.Ele().activated == 0, "a row inside it takes the click");
}

// Leaves the list from row 2, by moving focus to the control above.
void LeaveListFromRowTwo(ListDual &run) {
  run.Frames(2);
  run.Press(ImGuiKey_GamepadDpadDown);
  run.Press(ImGuiKey_GamepadDpadDown);
  run.Press(ImGuiKey_GamepadDpadDown);
  Check(run.Ele().focused_row == 2, "focus reaches row 2");
  run.FocusAbove();
  run.Frames(2);
  Check(run.Ele().above_focused, "focus leaves the list");
}

void TestListClickIsNotRedirected() {
  ListDual run;
  LeaveListFromRowTwo(run);
  run.Click(200, LIST_AREA.y + LIST_INSET + 4 * LIST_ROW + 20);
  Check(run.Ele().focused_row == 4 && run.Ele().activated == 4, "a click onto row 4 keeps focus on row 4");
}

void TestListDisabledRowNotRestored() {
  ListDual run;
  LeaveListFromRowTwo(run);
  run.scene.disabled = 2;
  run.Press(ImGuiKey_GamepadDpadDown);
  Check(run.Ele().focused_row == 0, "a disabled remembered row isn't restored");
}

void TestListScreenChange() {
  ListDual run;
  run.Frames(2);
  run.Press(ImGuiKey_GamepadDpadDown);
  run.Press(ImGuiKey_GamepadDpadDown);
  run.Press(ImGuiKey_GamepadDpadDown);
  run.scene.draw_list = false;
  run.FocusAbove();
  run.Frames(5);
  run.scene.draw_list = true;
  run.Frames(2);
  Check(run.Ele().above_focused, "focus is outside the list when it comes back");
  run.Press(ImGuiKey_GamepadDpadDown);
  Check(run.Ele().focused_row == 2, "the D-pad returns to the row it left");
}

// Up from a control below a long list lands on row 6, the last row in view, and not on rows 9 or
// 10, which are nearer but hidden below the area.
void TestListControlBelowTheList() {
  ListDual run;
  run.scene.rows = 20;
  run.scene.below = true;
  run.Frames(2);
  Check(run.Ele().below_focused, "the control below starts with focus");
  run.Key(ImGuiKey_GamepadDpadUp, true);
  run.Frame();
  run.Key(ImGuiKey_GamepadDpadUp, false);
  run.Frame();
  Check(run.Ele().focused_row == 6 && run.Ele().scroll == 0.0f, "up lands on the last row wholly in view");
}

void TestListHiddenRowsNotDrawn() {
  const auto bar_shown = [](int selected) {
    ListDual run;
    run.scene.rows = 20;
    run.scene.selected = selected;
    run.Frames(2);
    return run.Ele().bar;
  };
  Check(bar_shown(3), "a selected row in view shows its bar");
  Check(!bar_shown(15), "one scrolled out of view isn't drawn");
}

void TestListShrinkingScrollsBack() {
  ListDual run;
  run.scene.rows = 20;
  run.Frames(2);
  for (int press = 0; press < 16; press++) {
    run.Press(ImGuiKey_GamepadDpadDown);
  }
  run.Frames(5);
  Check(run.Ele().scroll > 0, "the long list has scrolled");
  run.scene.rows = 3;
  run.Frames(2);
  Check(run.Ele().scroll == 0.0f, "shrunk to three rows, it scrolls back to show them at once");
  Check(run.compared() > 50, "frames were compared");
}

// A scrolled list draws its rows higher by the scroll. Row 15 is selected and focus is moved down
// to it, which scrolls the list, and its fill must then lie inside the list's area. Unscrolled it
// would be drawn at 985, below the area.
void TestScrolledChildrenMove() {
  ListDual run;
  run.scene.rows = 20;
  run.scene.selected = 15;
  run.Frames(2);
  for (int press = 0; press < 16; press++) {
    run.Press(ImGuiKey_GamepadDpadDown);
  }
  run.Frames(10);
  Check(run.Ele().scroll > 0 && run.Ele().focused_row == 15, "the list has scrolled to row 15");
  Check(run.Ele().bar && run.Ele().bar_min >= LIST_AREA.y && run.Ele().bar_max <= LIST_AREA.Bottom(), "and draws it inside its area");
}

// Rows keyed by the data they show come back to the same data after a row is inserted above the
// remembered one. The index-keyed baseline returns to the old position, which is other data.
std::string RowAfterInsertAbove(bool keyedByData) {
  NavHarness nav;
  UiFrame frame;
  std::vector<std::string> data{"alpha", "beta", "gamma", "delta", "epsilon"};
  bool focusAbove = false;
  std::string focused;
  const auto draw = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    (void)Interact(context, ImGui::GetID("above"), LIST_ABOVE, false);
    if (focusAbove) {
      ImGui::SetFocusID(ImGui::GetID("above"), ImGui::GetCurrentWindow());
      focusAbove = false;
    }
    std::vector<Element> rows;
    const ListScene scene;
    for (size_t index = 0; index < data.size(); ++index) {
      const Element row = ListElementRow(frame, context.ui.theme, static_cast<int>(index), scene, "unused");
      const Element keyed = keyedByData ? Keyed(frame, row, std::string_view(data[index])) : Keyed(frame, row, static_cast<uint32_t>(index));
      Check(!keyed.IsNone(), "the row is rekeyed");
      rows.push_back(row);
    }
    const Element list = ListElementBox(frame, rows);
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(frame, list, {inf, inf});
    Check(Arrange(frame, list, LIST_AREA), "the list arranges");
    Check(!Render(frame, list, context.list).error.has_value(), "the list renders");
    focused.clear();
    for (size_t index = 0; index < data.size(); ++index) {
      if (InputOf(frame, rows[index])->interaction.state.focused) {
        focused = data[index];
      }
    }
  };
  nav.Frames(2, draw);
  for (int press = 0; press < 3; ++press) {
    nav.Key(ImGuiKey_GamepadDpadDown, true);
    nav.Frame(draw);
    nav.Key(ImGuiKey_GamepadDpadDown, false);
    nav.Frames(2, draw);
  }
  Check(focused == "gamma", "focus reaches the third row");
  focusAbove = true;
  nav.Frames(2, draw);
  data.insert(data.begin(), "aardvark");
  nav.Key(ImGuiKey_GamepadDpadDown, true);
  nav.Frame(draw);
  nav.Key(ImGuiKey_GamepadDpadDown, false);
  nav.Frames(2, draw);
  return focused;
}

void TestDataKeysSurviveInsertion() {
  Check(RowAfterInsertAbove(true) == "gamma", "rows keyed by data return to the same data");
  Check(RowAfterInsertAbove(false) == "beta", "rows keyed by index return to the old position, which is other data");
}

// A list refuses a node that is not a clipping box, and a clipping box is accepted.
void TestAsListRefusals() {
  InWindow([](UiFrame &frame) {
    ElementError reason = ElementError::None;
    const Element plain = Box(frame, {}, {});
    Check(AsList(frame, plain, &reason).IsNone() && reason == ElementError::NotClippingBox, "a box that doesn't clip is refused");
    const Element text = Text(frame, TextProps{"x", frame.ActiveTheme().type.label, frame.ActiveTheme().colors.text.primary});
    reason = ElementError::None;
    Check(AsList(frame, text, &reason).IsNone() && reason == ElementError::NotClippingBox, "so is text");
    Style clip{};
    clip.overflow = Overflow::Clip;
    Check(!AsList(frame, Box(frame, {}, clip)).IsNone(), "a clipping box is accepted");
    Check(AsList(frame, Element{}).IsNone(), "and a none handle is refused");
  });
}

// What this proves now is that a hand-built list with a rounded corner takes a press on a row
// partly in view near the corner, as the List component does, since a list clips hit testing to its
// exact rectangle. Row 7 shows its top 3 px at the list's bottom, and the point is inside the row
// and the rectangle but outside a radius 24 corner cut. A plain clipping box would clip it away, so
// the test also asserts directly that the press focused row 7. The framework claim is also covered
// by TestListHitClipIgnoresRoundedCorners.
void TestListCornerHoverMatchesTheListComponent() {
  ListDual run;
  run.scene.rows = 20;
  run.scene.radius = 24;
  run.Frames(2);
  run.Click(106, 539);
  // The press focuses the row. The list then scrolls it into view, so the release lands elsewhere
  // and nothing activates, which is the same on both sides.
  Check(run.Ele().focused_row == 7, "a row partly in view near a rounded corner takes the press, as with the List component");
}

// The rows of a list nested in a list, with the elements the tests ask about.
struct NestedTree {
  Element outer, inner, rowA, rowB, rowC;
};

// A list of 300 high holding a row, a list of 200 high with three rows, and a last row. All
// have explicit heights, so nothing shrinks, and both lists scroll.
NestedTree BuildNested(UiFrame &frame, const Theme &theme) {
  const ListScene scene;
  NestedTree tree;
  const auto tall = [&](float height, std::string_view key) {
    Layout layout{};
    layout.item = {.height = Dp(height)};
    const Element row = Box(frame, layout, Style{});
    Check(!Keyed(frame, row, key).IsNone(), "the row is keyed");
    Check(!MakeInteractive(frame, row, {}).IsNone(), "the row is interactive");
    return row;
  };
  (void)theme;
  (void)scene;
  tree.rowA = tall(200, "a");
  tree.rowB = tall(100, "b0");
  const std::vector<Element> inner_rows{tree.rowB, tall(100, "b1"), tall(100, "b2")};
  Layout layout{.direction = FlexDirection::Column, .align = Align::Stretch};
  layout.padding = DpInsets::All(Dp(LIST_INSET));
  Layout inner_layout = layout;
  inner_layout.item = {.height = Dp(200.0f)};
  Style clip{};
  clip.overflow = Overflow::Clip;
  tree.inner = Box(frame, inner_layout, clip, inner_rows);
  Check(!Keyed(frame, tree.inner, "inner").IsNone() && !AsList(frame, tree.inner).IsNone(), "the inner list is made");
  tree.rowC = tall(300, "c");
  const std::vector<Element> outer_rows{tree.rowA, tree.inner, tree.rowC};
  Layout outer_layout = layout;
  outer_layout.item = {.width = Dp(300.0f), .height = Dp(300.0f)};
  tree.outer = Box(frame, outer_layout, clip, outer_rows);
  Check(!Keyed(frame, tree.outer, "outer").IsNone() && !AsList(frame, tree.outer).IsNone(), "the outer list is made");
  return tree;
}

// Painted gives a row's arranged place less the scroll of its list, and less the sum of the scrolls
// of the lists around it when they nest. The lists themselves and a list's own box are placed by
// the lists above them only, and the unscrolled outer list is the baseline.
void TestPaintedFollowsListScroll() {
  NavHarness nav;
  UiFrame frame;
  constexpr float OUTER_SCROLL = 50.0f;
  constexpr float INNER_SCROLL = 30.0f;
  int rendered = 0;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const NestedTree tree = BuildNested(frame, context.ui.theme);
    const auto inf = std::numeric_limits<float>::infinity();
    (void)Measure(frame, tree.outer, {inf, inf});
    Check(Arrange(frame, tree.outer, {100, 100, 300, 300}), "the lists arrange");
    Check(!AssignIds(frame, tree.outer, CurrentIdSeed()).has_value(), "the lists have ids");
    context.store.List(IdOf(frame, tree.outer)).scroll = OUTER_SCROLL;
    context.store.List(IdOf(frame, tree.inner)).scroll = INNER_SCROLL;
    Check(!Render(frame, tree.outer, context.list).error.has_value(), "the lists render");
    const auto lifted = [&](Element element, float scroll) {
      const Rect laid = Arranged(frame, element);
      const Rect drawn = Painted(frame, element);
      Check(Near(drawn.y, laid.y - scroll) && Near(drawn.x, laid.x) && Near(drawn.height, laid.height), "the element is drawn lifted by the scrolls above it");
    };
    lifted(tree.outer, 0.0f);
    lifted(tree.rowA, OUTER_SCROLL);
    lifted(tree.inner, OUTER_SCROLL);
    lifted(tree.rowB, OUTER_SCROLL + INNER_SCROLL);
    lifted(tree.rowC, OUTER_SCROLL);
    ++rendered;
  });
  Check(rendered == 1, "the frame ran");
}

// A list nested in a list keeps its rows in the frame's pool, one record for each depth, so the
// reserved bytes are steady once warm.
void TestNestedListReservedBytesSteady() {
  NavHarness nav;
  UiFrame frame;
  size_t warm = 0;
  ItemId outerId = 0, innerId = 0, rowAId = 0;
  for (int i = 0; i < 40; ++i) {
    nav.Frame([&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const NestedTree tree = BuildNested(frame, context.ui.theme);
      const auto inf = std::numeric_limits<float>::infinity();
      (void)Measure(frame, tree.outer, {inf, inf});
      Check(Arrange(frame, tree.outer, {100, 100, 300, 300}), "the lists arrange");
      Check(!Render(frame, tree.outer, context.list).error.has_value(), "the lists render");
      Check(InputOf(frame, tree.rowB).has_value(), "a nested row is interactive");
      outerId = IdOf(frame, tree.outer);
      innerId = IdOf(frame, tree.inner);
      rowAId = IdOf(frame, tree.rowA);
    });
    if (i == 5) {
      warm = frame.ReservedBytes();
    }
    if (i > 5) {
      Check(frame.ReservedBytes() == warm, "a nested list's reserved bytes are steady across frames");
    }
  }
  // Each list records its own rows. The first row is the outer list's own and has first focus, so
  // the outer list remembers it and the inner one, which has none of the focus, does not. A list
  // that shared its record with the one inside it would lose that row.
  Check(nav.store.List(outerId).inside && nav.store.List(outerId).row == rowAId, "the outer list remembers its own row");
  Check(!nav.store.List(innerId).inside, "and the inner list, with no focus, does not");
}

// A list in steady state allocates nothing for its rows.
void TestListReservedBytesSteady() {
  NavHarness nav;
  UiFrame frame;
  ListScene scene;
  scene.rows = 20;
  ListSeen seen;
  size_t warm = 0;
  for (int i = 0; i < 40; ++i) {
    nav.Frame([&](InteractContext &context) { DrawElementList(context, frame, scene, seen); });
    if (i == 5) {
      warm = frame.ReservedBytes();
    }
    if (i > 5) {
      Check(frame.ReservedBytes() == warm, "a list's reserved bytes are steady across frames");
    }
  }
}

// ---- Keyed animation ---------------------------------------------------------------------------

// The time `ms` milliseconds after the clock's epoch, which is how the harness counts too.
UiClock::time_point At(double ms) {
  return UiClock::time_point{} + std::chrono::duration_cast<UiClock::duration>(std::chrono::duration<double, std::milli>(ms));
}

// What a probe node saw when it was painted.
struct Seen {
  Rect painted;
  float opacity = -1.0f;
};

struct SeenPaint {
  Seen *out;
  void operator()(const CustomPaint &paint) const { *out = {paint.painted, paint.opacity}; }
};

constexpr float ROW_WIDTH = 80.0f;
constexpr float ROW_HEIGHT = 40.0f;
constexpr float ENTER_MS = 200.0f;
constexpr float EXIT_MS = 100.0f;
constexpr float SLIDE_DP = 40.0f;
constexpr float HIDDEN_SCALE = 0.5f;
constexpr Rect STAGE{100, 100, 400, 300};
// The width of the harness's output, which Show lays out for.
constexpr float HARNESS_WIDTH = 1920.0f;
constexpr float HARNESS_HEIGHT = 1080.0f;

// A card that slides down and grows in over 200 ms, rests until dismissed, then slides further and
// shrinks away over 100 ms. Linear so each value can be read off the clock.
Timeline CardTimeline() {
  Timeline timeline;
  timeline.enter = {{ENTER_MS, Ease::Linear}, {ENTER_MS, Ease::Linear}, {0.0f, -SLIDE_DP}, HIDDEN_SCALE};
  timeline.exit = {{EXIT_MS, Ease::Linear}, {EXIT_MS, Ease::Linear}, {0.0f, SLIDE_DP}, HIDDEN_SCALE};
  return timeline;
}

// One probe row of a frame. An animation, an explicit pose and a hand-made timeline pose are each
// optional, and the key names the row among its siblings.
struct RowPlan {
  std::string key;
  std::optional<Animate> animate;
  std::optional<ElementPose> explicit_pose;
  Origin explicit_origin = CENTER;
  std::optional<Pose> manual;
};

// What one probe row showed.
struct RowSeen {
  Seen seen;
  Rect arranged;
  std::optional<Phase> phase;
};

// Renders `plans` as the rows of one column in one frame that begins at `ms`, and returns what each
// probe saw. Rows are keyed by their text unless `keyed` is false, which leaves them to their index.
// Each call is a whole frame of the harness, so the store forgets any row that is left out of it.
std::vector<RowSeen> Show(NavHarness &nav, double ms, const std::vector<RowPlan> &plans, bool keyed = true) {
  std::vector<RowSeen> rows(plans.size());
  nav.Frame([&](InteractContext &context) {
    UiFrame frame;
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, At(ms));
    const ItemLayout size{.width = Dp(ROW_WIDTH), .height = Dp(ROW_HEIGHT)};
    std::vector<Element> probes;
    for (size_t index = 0; index < plans.size(); ++index) {
      const RowPlan &plan = plans[index];
      Element probe = Custom(frame, size, SeenPaint{&rows[index].seen});
      if (keyed) {
        probe = Keyed(frame, probe, plan.key);
      }
      if (plan.explicit_pose.has_value()) {
        Check(!WithPose(frame, probe, *plan.explicit_pose, plan.explicit_origin).IsNone(), "the pose is accepted");
      }
      if (plan.animate.has_value()) {
        Check(!WithAnimation(frame, probe, *plan.animate).IsNone(), "the animation is accepted");
      }
      if (plan.manual.has_value()) {
        Check(!WithTimelinePose(frame, probe, *plan.manual).IsNone(), "the timeline pose is accepted");
      }
      probes.push_back(probe);
    }
    Layout layout{};
    layout.direction = FlexDirection::Column;
    const Element root = Keyed(frame, Box(frame, layout, {}, probes), "stage");
    const float inf = std::numeric_limits<float>::infinity();
    (void)Measure(frame, root, {inf, inf});
    Check(Arrange(frame, root, STAGE), "the stage arranges");
    Check(!Render(frame, root, context.list).error.has_value(), "the stage renders");
    for (size_t index = 0; index < plans.size(); ++index) {
      rows[index].arranged = Arranged(frame, probes[index]);
      rows[index].phase = PhaseOf(frame, probes[index]);
    }
  });
  return rows;
}

// Where `row` is drawn less where it is laid out, which is the same for two rows with the same pose
// whatever their place in the column.
Rect Shift(const RowSeen &row) {
  return {row.seen.painted.x - row.arranged.x, row.seen.painted.y - row.arranged.y, row.seen.painted.width - row.arranged.width,
          row.seen.painted.height - row.arranged.height};
}

bool SameShift(const RowSeen &first, const RowSeen &second) {
  const Rect a = Shift(first), b = Shift(second);
  return Near(a.x, b.x) && Near(a.y, b.y) && Near(a.width, b.width) && Near(a.height, b.height);
}

// The pose the timeline gives a card that started at `start` and was dismissed at `dismissed`.
Pose Expect(const Timeline &timeline, double start, std::optional<double> dismissed, double now) {
  std::optional<UiClock::time_point> at;
  if (dismissed.has_value()) {
    at = At(*dismissed);
  }
  return Evaluate(timeline, At(start), at, At(now));
}

// The animated card draws as a card posed by hand with the pose Evaluate gives, through the
// entrance, at rest and through the exit. The hand-posed row is the baseline, and the checks that
// the card really moved, faded and went through every phase show the baseline is not simply still.
void TestAnimationMatchesManualPose() {
  NavHarness nav;
  const Timeline timeline = CardTimeline();
  constexpr double START = 1000, DISMISS = 1800;
  bool phases[5] = {};
  bool moved = false, faded = false;
  for (const double ms : {1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1500.0, 1800.0, 1830.0, 1860.0, 1900.0}) {
    const bool dismissed = ms >= DISMISS;
    const Pose want = Expect(timeline, START, dismissed ? std::optional<double>(DISMISS) : std::nullopt, ms);
    const auto rows = Show(nav, ms, {{.key = "card", .animate = Animate{timeline, CENTER, dismissed}}, {.key = "manual", .manual = want}});
    Check(SameShift(rows[0], rows[1]), "the animated card is drawn where the hand-posed one is");
    Check(Near(rows[0].seen.opacity, rows[1].seen.opacity) && Near(rows[0].seen.opacity, want.alpha), "and as opaque");
    Check(rows[0].phase.has_value() && *rows[0].phase == want.phase, "PhaseOf reports the phase Evaluate gives");
    phases[static_cast<int>(want.phase)] = true;
    moved = moved || std::fabs(Shift(rows[0]).y) > 1.0f;
    faded = faded || rows[0].seen.opacity < 0.9f;
  }
  Check(phases[static_cast<int>(Phase::Entering)] && phases[static_cast<int>(Phase::Shown)] && phases[static_cast<int>(Phase::Exiting)] &&
            phases[static_cast<int>(Phase::Gone)],
        "the run went through every phase");
  Check(moved && faded, "and the card moved and faded");
}

// Runs a card "a" alone and then joined by "b" a moment later, then reorders them, and returns how
// opaque each is after the swap. Every row has the same timeline, so only its start tells them apart.
std::pair<float, float> ReorderedOpacities(bool keyed) {
  NavHarness nav;
  const Animate animate{CardTimeline(), CENTER, false};
  (void)Show(nav, 1000, {{.key = "a", .animate = animate}}, keyed);
  (void)Show(nav, 1100, {{.key = "a", .animate = animate}, {.key = "b", .animate = animate}}, keyed);
  const auto rows = Show(nav, 1150, {{.key = "b", .animate = animate}, {.key = "a", .animate = animate}}, keyed);
  return {rows[1].seen.opacity, rows[0].seen.opacity};
}

// Reordering keyed rows keeps each row's start, so "a", which began first, is further in than "b"
// wherever each now sits. With index keys the starts stay with the positions, so they swap. That
// is the baseline, and it proves the two rows differ at all.
void TestReorderKeepsStartWhenKeyed() {
  const Timeline timeline = CardTimeline();
  const float aStart = Expect(timeline, 1000, std::nullopt, 1150).alpha;
  const float bStart = Expect(timeline, 1100, std::nullopt, 1150).alpha;
  Check(aStart - bStart > 0.25f, "the two starts give different opacities");
  const auto [keyedA, keyedB] = ReorderedOpacities(true);
  Check(Near(keyedA, aStart) && Near(keyedB, bStart), "keyed rows keep their own start through a reorder");
  const auto [indexA, indexB] = ReorderedOpacities(false);
  Check(Near(indexA, bStart) && Near(indexB, aStart), "index keys hand the starts to the positions, which is the baseline");
}

// Holding dismissed plays the exit once. The exit runs from the first dismissed frame to its end,
// and the card is then gone, instead of restarting each frame. A dismissal at each frame, which
// would hold the card at the start of its exit, gives a different opacity at the same time.
void TestDismissalIsRecordedOnce() {
  NavHarness nav;
  const Timeline timeline = CardTimeline();
  constexpr double DISMISS = 1500;
  float midExit = -1.0f;
  std::optional<Phase> last;
  for (double ms = 1000; ms <= 1700; ms += 25) {
    const bool dismissed = ms >= DISMISS;
    const auto rows = Show(nav, ms, {{.key = "card", .animate = Animate{timeline, CENTER, dismissed}}});
    const Pose want = Expect(timeline, 1000, dismissed ? std::optional<double>(DISMISS) : std::nullopt, ms);
    Check(Near(rows[0].seen.opacity, want.alpha), "the card follows a single dismissal");
    if (ms == 1550) {
      midExit = rows[0].seen.opacity;
    }
    last = rows[0].phase;
  }
  const float restarted = Expect(timeline, 1000, 1550, 1550).alpha;
  Check(std::fabs(midExit - restarted) > 0.25f, "half way through the exit is not where a restarted exit would be");
  Check(last.has_value() && *last == Phase::Gone, "and the exit finishes");
}

// A card left out of one frame is forgotten and replays its entrance when it returns. The card
// that stayed in every frame is the baseline and keeps its first start.
void TestMissingFrameReplaysEntrance() {
  NavHarness nav;
  const Timeline timeline = CardTimeline();
  const Animate animate{timeline, CENTER, false};
  const std::vector<RowPlan> both = {{.key = "kept", .animate = animate}, {.key = "gap", .animate = animate}};
  (void)Show(nav, 1000, both);
  (void)Show(nav, 1100, both);
  (void)Show(nav, 1150, {{.key = "kept", .animate = animate}});
  const auto back = Show(nav, 1200, both);
  Check(back[1].phase.has_value() && *back[1].phase == Phase::Entering, "the returning card enters again");
  Check(Near(back[1].seen.opacity, 0.0f), "from nothing, as at its start");
  const auto later = Show(nav, 1250, both);
  Check(Near(later[1].seen.opacity, Expect(timeline, 1200, std::nullopt, 1250).alpha), "its start is the frame it came back");
  Check(Near(later[0].seen.opacity, Expect(timeline, 1000, std::nullopt, 1250).alpha), "while the card that stayed kept its start");
  Check(later[0].seen.opacity > later[1].seen.opacity + 0.25f, "and the two differ");
}

// Maps `point` by a scale `scale` about `origin` of `rect`, then by `offset`.
Vec2 About(Vec2 point, const Rect &rect, Origin origin, float scale, Vec2 offset) {
  const Vec2 centre{rect.x + origin.x * rect.width, rect.y + origin.y * rect.height};
  return {centre.x + scale * (point.x - centre.x) + offset.x, centre.y + scale * (point.y - centre.y) + offset.y};
}

// The animation pose applies inside the explicit one: it maps the card first and the explicit pose
// then maps that. The expected rectangle is composed by hand in that order. The same two maps in
// the other order give a rectangle that is elsewhere, which shows the order is what is tested.
void TestAnimationComposesInsideExplicitPose() {
  NavHarness nav;
  const Timeline timeline = CardTimeline();
  const ScaleContext scale(DEFAULT_THEME.scale, HARNESS_WIDTH, HARNESS_HEIGHT);
  constexpr double START = 1000, NOW = 1100;
  const ElementPose outer{0.5f, {30.0f, 12.0f}, 1.5f};
  constexpr Origin OUTER_ORIGIN{0.0f, 0.0f};
  (void)Show(nav, START, {{.key = "card", .animate = Animate{timeline, CENTER, false}, .explicit_pose = outer, .explicit_origin = OUTER_ORIGIN}});
  const auto rows = Show(nav, NOW, {{.key = "card", .animate = Animate{timeline, CENTER, false}, .explicit_pose = outer, .explicit_origin = OUTER_ORIGIN}});
  const Pose inner = Expect(timeline, START, std::nullopt, NOW);
  const Rect box = rows[0].arranged;
  const Vec2 innerOffset{scale.Px(inner.offset.x), scale.Px(inner.offset.y)};
  const Vec2 outerOffset{scale.Px(outer.offset.x), scale.Px(outer.offset.y)};
  const auto compose = [&](Vec2 point, bool animationFirst) {
    const auto animate = [&](Vec2 p) { return About(p, box, CENTER, inner.scale, innerOffset); };
    const auto pose = [&](Vec2 p) { return About(p, box, OUTER_ORIGIN, outer.scale, outerOffset); };
    return animationFirst ? pose(animate(point)) : animate(pose(point));
  };
  const auto rectOf = [&](bool animationFirst) {
    const Vec2 topLeft = compose({box.x, box.y}, animationFirst);
    const Vec2 bottomRight = compose({box.Right(), box.Bottom()}, animationFirst);
    return Rect{topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y};
  };
  const auto near = [](const Rect &a, const Rect &b) {
    return Near(a.x, b.x) && Near(a.y, b.y) && Near(a.width, b.width) && Near(a.height, b.height);
  };
  Check(near(rows[0].seen.painted, rectOf(true)), "the card is drawn where the animation then the explicit pose put it");
  Check(!near(rectOf(true), rectOf(false)), "and the other order would put it elsewhere");
  Check(Near(rows[0].seen.opacity, outer.alpha * inner.alpha), "the two alphas multiply");
}

// Animating is true while a timeline moves, which is the entrance, and false once the card has
// settled, since this one holds until dismissed. It is true again for the exit and false when the
// card is gone. A row with no animation is the baseline, which leaves the store at rest.
void TestAnimatingFollowsTimeline() {
  NavHarness plain;
  (void)Show(plain, 1000, {{.key = "card"}});
  Check(!plain.store.Animating(At(1000)), "a row with no animation leaves the store at rest");
  NavHarness nav;
  const Timeline timeline = CardTimeline();
  const auto frame = [&](double ms, bool dismissed) {
    (void)Show(nav, ms, {{.key = "card", .animate = Animate{timeline, CENTER, dismissed}}});
    return nav.store.Animating(At(ms));
  };
  Check(frame(1000, false), "the store is animating as the card starts");
  Check(frame(1100, false), "and part way through the entrance");
  Check(!frame(1300, false), "but not once it has settled");
  Check(frame(1400, true), "and again when it is dismissed");
  Check(frame(1450, true), "through the exit");
  Check(!frame(1600, true), "and not once it is gone");
}

// WithAnimation refuses a none handle, an origin that is not finite and a timeline whose hidden
// scale could flip or lose the card, and PhaseOf has nothing before Render or for a plain node.
void TestAnimationRefusalsAndPhase() {
  InWindow([](UiFrame &frame) {
    const Element card = Leaf(frame);
    const Animate good{CardTimeline(), CENTER, false};
    Check(!PhaseOf(frame, card).has_value(), "a plain node has no phase");
    ElementError why = ElementError::None;
    Check(WithAnimation(frame, Element{}, good, &why).IsNone(), "a none handle is refused");
    Animate flat = good;
    flat.timeline.enter.scale = 0.0f;
    Check(WithAnimation(frame, card, flat, &why).IsNone() && why == ElementError::InvalidPose, "a zero scale is refused");
    Animate lost = good;
    lost.origin.x = std::numeric_limits<float>::quiet_NaN();
    why = ElementError::None;
    Check(WithAnimation(frame, card, lost, &why).IsNone() && why == ElementError::InvalidPose, "and so is a NaN origin");
    Check(!WithAnimation(frame, card, good).IsNone(), "a good animation is accepted, which is the baseline");
    Check(!PhaseOf(frame, card).has_value(), "and has no phase before Render");
  });
}

// A row dismissed until it is gone and then undismissed replays its entrance from that frame,
// matching a hand-posed row at elapsed 0, 1 and 2 frames. The row that stays dismissed is the
// baseline, which stays gone, and shows the replay is the undismissing and not time passing.
void TestUndismissReplaysEntrance() {
  NavHarness nav;
  const Timeline timeline = CardTimeline();
  const auto rows = [&](double ms, bool back, const Pose *want) {
    std::vector<RowPlan> plans = {{.key = "back", .animate = Animate{timeline, CENTER, !back}},
                                  {.key = "stay", .animate = Animate{timeline, CENTER, true}}};
    if (want != nullptr) {
      plans.push_back({.key = "manual", .manual = *want});
    }
    return Show(nav, ms, plans);
  };
  (void)Show(nav, 1000, {{.key = "back", .animate = Animate{timeline, CENTER, false}}, {.key = "stay", .animate = Animate{timeline, CENTER, false}}});
  (void)rows(1100, false, nullptr);
  const auto gone = rows(1300, false, nullptr);
  Check(gone[0].phase == Phase::Gone && gone[1].phase == Phase::Gone, "both rows are gone after the exit");
  constexpr double BACK = 1400;
  constexpr double FRAME_MS = 1000.0 / 60.0;
  for (const double ms : {BACK, BACK + FRAME_MS, BACK + 2 * FRAME_MS}) {
    const Pose want = Expect(timeline, BACK, std::nullopt, ms);
    const auto shown = rows(ms, true, &want);
    Check(shown[0].phase == Phase::Entering, "the undismissed row enters again");
    Check(SameShift(shown[0], shown[2]) && Near(shown[0].seen.opacity, shown[2].seen.opacity), "and is drawn as a hand-posed entrance from the undismissing");
    Check(shown[1].phase == Phase::Gone && Near(shown[1].seen.opacity, 0.0f), "while the row that stays dismissed stays gone");
  }
}

// ---- Key ownership ----------------------------------------------------------------------------

// Which of R1 and back each reader saw over a run, and by which it was read.
struct KeysSeen {
  bool consumer_r1 = false;
  bool consumer_back = false;
  bool bar_r1 = false;
  bool bar_back = false;
  bool modal_r1 = false;
  bool modal_back = false;
  bool bar_has_record = false;
};

// A root that holds one interactive child, as a tab bar or a dialog holds its tabs or buttons, and
// owns `keys` when given. The child shows that owning keys does not need the container to be
// interactive. The root is laid out at `rect` and rendered.
Element KeyRoot(UiFrame &frame, InteractContext &context, std::string_view key, const std::optional<KeyOwner> &keys, const Rect &rect) {
  const ItemLayout size{.width = Dp(60), .height = Dp(30)};
  Layout layout{};
  layout.item = size;
  const Element child = Keyed(frame, Box(frame, layout, {}), "child");
  Check(!MakeInteractive(frame, child, {}).IsNone(), "the child is interactive");
  const std::array<Element, 1> kids{child};
  const Element root = Keyed(frame, Box(frame, {}, {}, kids), key);
  if (keys.has_value()) {
    Check(!OwnKeys(frame, root, *keys).IsNone(), "the root owns its keys");
  }
  const float inf = std::numeric_limits<float>::infinity();
  (void)Measure(frame, root, {inf, inf});
  Check(Arrange(frame, root, rect), "the root arranges");
  Check(!Render(frame, root, context.list).error.has_value(), "the root renders");
  return root;
}

// Runs a harness with a tab bar of elements that owns the shoulders at Control priority when
// `bar_owns`, and over it a dialog of elements that owns the shoulders and back at `modal_priority`
// when `with_modal`. A reader that sees keys as the gallery does, owned by no one, reads first in
// each frame. R1 and back are pressed in turn.
KeysSeen RunKeyOwners(bool bar_owns, bool with_modal, ClaimPriority modal_priority = ClaimPriority::Modal) {
  NavHarness nav;
  KeysSeen seen;
  const auto draw = [&](InteractContext &context) {
    seen.consumer_r1 = seen.consumer_r1 || ShoulderPresses().right;
    seen.consumer_back = seen.consumer_back || BackKeyPressed(NO_OWNER);
    UiFrame frame;
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    std::optional<KeyOwner> bar_keys;
    if (bar_owns) {
      bar_keys = KeyOwner{.shoulders = true};
    }
    const Element bar = KeyRoot(frame, context, "bar", bar_keys, {100, 100, 200, 100});
    seen.bar_has_record = KeyPressesOf(frame, bar).has_value();
    if (const auto found = KeyPressesOf(frame, bar)) {
      seen.bar_r1 = seen.bar_r1 || found->shoulders.right;
      seen.bar_back = seen.bar_back || found->back;
    }
    if (with_modal) {
      const Element modal = KeyRoot(frame, context, "dialog", KeyOwner{.shoulders = true, .back = true, .priority = modal_priority}, {400, 100, 200, 100});
      const auto found = KeyPressesOf(frame, modal);
      Check(found.has_value(), "the dialog reports its presses");
      seen.modal_r1 = seen.modal_r1 || found->shoulders.right;
      seen.modal_back = seen.modal_back || found->back;
    }
  };
  nav.Frames(3, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  nav.Press(ImGuiKey_GamepadFaceRight, draw);
  return seen;
}

// A bar of elements that owns the shoulders reads R1 as its own and keeps it from a reader that
// sees keys as the gallery does. The same bar without OwnKeys is the baseline, where that reader
// sees R1 and the bar has no record of it.
void TestElementBarOwnsShoulders() {
  const KeysSeen owned = RunKeyOwners(true, false);
  Check(owned.bar_r1 && !owned.consumer_r1, "an owning bar reads R1 and the gallery's reader does not");
  Check(!owned.bar_back && owned.consumer_back, "back, which it did not ask for, is left to the reader");
  const KeysSeen plain = RunKeyOwners(false, false);
  Check(plain.consumer_r1 && !plain.bar_has_record, "without OwnKeys the reader sees R1, which is the baseline");
}

// A dialog of elements owns the shoulders and back at Modal priority, over a bar that owns the
// shoulders. R1 and back reach only the dialog. The bar alone is the baseline, where it reads R1.
// A dialog at Control priority is the mutation's shape, which a bar that claimed first would
// keep the shoulders from, so the bar would read R1 under it.
void TestElementModalOverBar() {
  const KeysSeen over = RunKeyOwners(true, true);
  Check(over.modal_r1 && over.modal_back, "the dialog reads R1 and back");
  Check(!over.bar_r1 && !over.consumer_r1 && !over.consumer_back, "and neither the bar nor the gallery's reader does");
  const KeysSeen bar_alone = RunKeyOwners(true, false);
  Check(bar_alone.bar_r1, "the bar reads R1 when no dialog is open");
  const KeysSeen weak = RunKeyOwners(true, true, ClaimPriority::Control);
  Check(weak.bar_r1 && !weak.modal_r1, "a dialog that only asks as a control loses the shoulders to the bar");
}

// OwnKeys refuses a none handle, and a node that owns no keys reports no presses.
void TestOwnKeysRefusals() {
  InWindow([](UiFrame &frame) {
    ElementError why = ElementError::None;
    Check(OwnKeys(frame, Element{}, {.shoulders = true}, &why).IsNone(), "a none handle is refused");
    const Element node = Leaf(frame);
    Check(!KeyPressesOf(frame, node).has_value(), "a node that owns nothing reports nothing");
    Check(!OwnKeys(frame, node, {.back = true}).IsNone(), "a live node is accepted");
    Check(!KeyPressesOf(frame, node).has_value(), "and reports nothing before Render");
  });
}

// An element bar that appears in the frame a press was consumed does not act on that press, though
// nobody owned the key when it claimed. The same bar on a later press is the baseline, and acts.
void TestElementBarAppearingInConsumedFrameStandsDown() {
  NavHarness nav;
  bool bar_visible = false;
  bool consumed = false;
  bool bar_acted = false;
  const auto draw = [&](InteractContext &context) {
    if (ShoulderPresses().right) {
      consumed = true;
      ConsumeKey(KeyGroup::Shoulders, ImGui::GetID("shell"));
    }
    if (!bar_visible) {
      return;
    }
    UiFrame frame;
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element bar = KeyRoot(frame, context, "bar", KeyOwner{.shoulders = true}, {100, 100, 200, 100});
    const auto found = KeyPressesOf(frame, bar);
    Check(found.has_value(), "the bar reports its presses");
    bar_acted = bar_acted || found->shoulders.right;
  };
  nav.Frames(3, draw);
  bar_visible = true;
  nav.Key(ImGuiKey_GamepadR1, true);
  nav.Frame(draw);
  Check(consumed && !bar_acted, "a bar that appears in the consumed frame does not act on the press");
  nav.Key(ImGuiKey_GamepadR1, false);
  nav.Frames(4, draw);
  consumed = false;
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(bar_acted && !consumed, "and on a later press it does, with the reader standing down");
}

// An element that owns back and reports a press consumes it, so a legacy BackKeyPressed() read
// later in the same frame, which accepts any owner, sees nothing. A claim alone would not stop it.
// With no element owning back, the baseline, the same legacy read sees the press.
void TestElementBackIsConsumedForLegacyReaders() {
  for (const bool owned : {true, false}) {
    NavHarness nav;
    bool element_saw = false;
    bool legacy_saw = false;
    bool legacy_shoulder = false;
    const auto draw = [&](InteractContext &context) {
      UiFrame frame;
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      std::optional<KeyOwner> keys;
      if (owned) {
        keys = KeyOwner{.shoulders = true, .back = true};
      }
      const Element root = KeyRoot(frame, context, "dialog", keys, {100, 100, 200, 100});
      if (const auto found = KeyPressesOf(frame, root)) {
        element_saw = element_saw || found->back;
      }
      legacy_saw = legacy_saw || BackKeyPressed();
      legacy_shoulder = legacy_shoulder || ShoulderPresses(ANY_OWNER).right;
    };
    nav.Frames(3, draw);
    nav.Press(ImGuiKey_GamepadFaceRight, draw);
    nav.Press(ImGuiKey_GamepadR1, draw);
    Check(element_saw == owned, "the element reads back only when it owns it");
    Check(legacy_saw == !owned, owned ? "a legacy read after the element sees no press" : "and sees it when no element owns back");
    Check(legacy_shoulder == !owned, "the same holds for the shoulders");
  }
}

// ---- Render's afterInteract hook --------------------------------------------------------------

// A draw list of its own that shares the context's draw list data, so two renders can be compared.
std::unique_ptr<ImDrawList> OwnList() {
  auto list = std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData());
  list->_ResetForNewFrame();
  list->PushClipRectFullScreen();
  return list;
}

// Measures and arranges `root` at `rect`.
void Place(UiFrame &frame, Element root, const Rect &rect) {
  const auto inf = std::numeric_limits<float>::infinity();
  (void)Measure(frame, root, {inf, inf});
  Check(Arrange(frame, root, rect), "the tree arranges");
}

constexpr Rect HOOK_ROOT{100, 100, 300, 120};
constexpr float HOOK_CONTROL_WIDTH = 100;
constexpr float HOOK_CONTROL_HEIGHT = 50;

// Reads two nodes' input from inside the hook.
struct InputSpy {
  Element first;
  Element second;
  std::optional<NodeInput> *seenFirst;
  std::optional<NodeInput> *seenSecond;
  void operator()(UiFrame &frame, Element) const {
    *seenFirst = InputOf(frame, first);
    *seenSecond = InputOf(frame, second);
  }
};

// Two keyed interactive boxes side by side in a plain root.
struct TwoControls {
  Element root;
  Element first;
  Element second;
};

TwoControls BuildTwoControls(UiFrame &frame) {
  const ItemLayout size{.width = Dp(HOOK_CONTROL_WIDTH), .height = Dp(HOOK_CONTROL_HEIGHT)};
  TwoControls tree;
  tree.first = Box(frame, {.item = size}, {});
  tree.second = Box(frame, {.item = size}, {});
  Check(!Keyed(frame, tree.first, "first").IsNone() && !Keyed(frame, tree.second, "second").IsNone(), "the controls are keyed");
  Check(!MakeInteractive(frame, tree.first, {}).IsNone() && !MakeInteractive(frame, tree.second, {}).IsNone(), "and interactive");
  const std::array<Element, 2> kids{tree.first, tree.second};
  tree.root = Box(frame, {.direction = FlexDirection::Row}, {}, kids);
  return tree;
}

// The hook runs after the whole walk, so it sees the input of every node, the later sibling
// included, and the activation of the frame it was pressed in. Before Render the input is not
// there, which is the baseline for "after the walk".
void TestHookSeesEveryNodesInputAfterTheWalk() {
  NavHarness nav;
  UiFrame frame;
  std::optional<NodeInput> seenFirst, seenSecond;
  bool beforeRenderEmpty = true, activationSeen = false, idsMatch = true;
  const auto draw = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const TwoControls tree = BuildTwoControls(frame);
    Place(frame, tree.root, HOOK_ROOT);
    beforeRenderEmpty = beforeRenderEmpty && !InputOf(frame, tree.first).has_value() && !InputOf(frame, tree.second).has_value();
    seenFirst.reset();
    seenSecond.reset();
    const auto result = Render(frame, tree.root, context.list, {}, InputSpy{tree.first, tree.second, &seenFirst, &seenSecond});
    Check(!result.error.has_value(), "the tree renders");
    const auto afterFirst = InputOf(frame, tree.first);
    const auto afterSecond = InputOf(frame, tree.second);
    Check(seenFirst.has_value() && seenSecond.has_value() && afterFirst && afterSecond, "the hook saw both nodes' input");
    idsMatch = idsMatch && seenFirst->id == afterFirst->id && seenSecond->id == afterSecond->id && seenFirst->id != seenSecond->id;
    activationSeen = activationSeen || (seenFirst->interaction.state.activated && result.activated == tree.first);
  };
  nav.Frames(3, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(2, draw);
  Check(beforeRenderEmpty, "before Render there is no input to see");
  Check(idsMatch, "the hook saw the same two nodes' ids that InputOf reports after");
  Check(activationSeen, "and the hook saw the first control activated on the frame Render reported it");
}

// Sets a colour on a text node.
struct Recolour {
  Element text;
  Color color;
  void operator()(UiFrame &frame, Element) const { Check(SetTextColor(frame, text, color), "the hook colours the text"); }
};

// A label in a plain root, with the colour it starts with.
struct Label {
  Element root;
  Element text;
  TextProps props;
};

Label BuildLabel(UiFrame &frame, Color color) {
  Label label;
  label.props = TextProps{"Label", frame.ActiveTheme().type.label, color};
  label.text = Text(frame, label.props);
  const std::array<Element, 1> kids{label.text};
  label.root = Box(frame, {.item = {.width = Dp(HOOK_ROOT.width), .height = Dp(HOOK_ROOT.height)}}, {}, kids);
  return label;
}

// What DrawText draws for the label's text node in `colour`, placed where Paint places it.
std::unique_ptr<ImDrawList> DrawnLabel(const UiFrame &frame, const UiContext &ui, const Label &label, Color color) {
  TextProps props = label.props;
  props.color = color;
  const TextLayout layout = LayoutText(ui, props, std::numeric_limits<float>::infinity());
  const Rect arranged = Arranged(frame, label.text);
  const float spare = std::max(0.0f, arranged.height - layout.size.y);
  auto list = OwnList();
  DrawText(*list, layout, props, {arranged.x, arranged.y + spare / 2.0f, arranged.width, layout.size.y}, 1.0f);
  return list;
}

// A colour set in the hook is in the same frame's paint, exactly what DrawText draws with that
// colour. With no hook the label keeps its own colour, and that is what DrawText draws with it, so
// the two differ only in the hook's doing. The original and the hook's colour differ in their
// vertices, which shows the comparison can tell them apart.
void TestHookColourShowsInThatFramesPaint() {
  NavHarness nav;
  UiFrame frame;
  const Color original = Rgb(0xE0E0E0);
  const Color hooked = Rgb(0xFF3366);
  for (const bool withHook : {false, true}) {
    nav.Frame([&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Label label = BuildLabel(frame, original);
      Place(frame, label.root, HOOK_ROOT);
      auto list = OwnList();
      const auto result = withHook ? Render(frame, label.root, *list, {}, Recolour{label.text, hooked}) : Render(frame, label.root, *list);
      Check(!result.error.has_value(), "the label renders");
      const auto expectedOriginal = DrawnLabel(frame, context.ui, label, original);
      const auto expectedHooked = DrawnLabel(frame, context.ui, label, hooked);
      Check(list->VtxBuffer.Size > 0 && !NearDrawing(*expectedOriginal, *expectedHooked, 0.0f), "the two colours draw different vertices");
      Check(NearDrawing(*list, withHook ? *expectedHooked : *expectedOriginal, 0.0f),
            withHook ? "the hook's colour is in that frame's paint" : "with no hook the label keeps its colour");
      Check(!NearDrawing(*list, withHook ? *expectedOriginal : *expectedHooked, 0.0f), "and not the other colour");
    });
  }
}

// Sets a hover on a node from the hook.
struct HoverIt {
  Element target;
  void operator()(UiFrame &frame, Element) const { Check(SetInput(frame, target, ItemInput{.visuals = {.hover = 1.0f}}), "the hook sets the input"); }
};

// A root holding a box whose hover state changes its fill.
struct Styled {
  Element root;
  Element box;
};

Styled BuildStyled(UiFrame &frame) {
  Style style;
  style.fill = Rgb(0x202020);
  style.states.hover.fill = Rgb(0x2060A0);
  Styled tree;
  tree.box = Box(frame, {.item = {.width = Dp(HOOK_CONTROL_WIDTH), .height = Dp(HOOK_CONTROL_HEIGHT)}}, style);
  const std::array<Element, 1> kids{tree.box};
  tree.root = Box(frame, {}, {}, kids);
  return tree;
}

// An input set in the hook changes that frame's state styles, as the same input set before Render
// does, and the box with neither keeps its resting fill.
void TestHookInputChangesThatFramesStateStyles() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    const auto render = [&](bool hook, bool preset) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Styled tree = BuildStyled(frame);
      if (preset) {
        Check(SetInput(frame, tree.box, ItemInput{.visuals = {.hover = 1.0f}}), "the input is set before Render");
      }
      Place(frame, tree.root, HOOK_ROOT);
      auto list = OwnList();
      const auto result = hook ? Render(frame, tree.root, *list, {}, HoverIt{tree.box}) : Render(frame, tree.root, *list);
      Check(!result.error.has_value(), "the tree renders");
      return list;
    };
    const auto hooked = render(true, false);
    const auto preset = render(false, true);
    const auto resting = render(false, false);
    Check(hooked->VtxBuffer.Size > 0 && NearDrawing(*hooked, *preset, 0.0f), "a hover set in the hook draws as one set before Render");
    Check(!NearDrawing(*hooked, *resting, 0.0f), "and not as the resting box");
  });
}

// SetTextColor only colours text, and refuses anything else with the reason.
void TestSetTextColorRefusals() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Label label = BuildLabel(frame, Rgb(0xE0E0E0));
    ElementError reason = ElementError::None;
    Check(!SetTextColor(frame, label.root, Rgb(0xFF0000), &reason) && reason == ElementError::NotText, "a box is refused as not text");
    reason = ElementError::Stale;
    Check(!SetTextColor(frame, Element{}, Rgb(0xFF0000), &reason) && reason == ElementError::None, "a none handle is refused as none");
    reason = ElementError::None;
    Check(SetTextColor(frame, label.text, Rgb(0xFF0000), &reason) && reason == ElementError::None, "a text node is accepted");
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    reason = ElementError::None;
    Check(!SetTextColor(frame, label.text, Rgb(0xFF0000), &reason) && reason == ElementError::Stale, "and a stale handle is refused as stale");
  });
}

// What building calls did inside the hook, each as "refused with BuildingInHook and changed
// nothing", so a call that slipped through shows which one.
struct BuildAttempts {
  bool flagInside = false;
  bool box = false;
  bool custom = false;
  bool activate = false;
  bool input = false;
  bool makeInteractive = false;
  bool ownKeys = false;
  bool asList = false;
  bool specimen = false;
  bool render = false;
  bool beginIgnored = false;
  bool storageUntouched = false;
  bool flagStillSetAfterNestedRender = false;
};

// Does nothing, to be stored by OnActivate.
struct NoActivate {
  void operator()(Element) const {}
};
struct NoInput {
  void operator()(const NodeInput &) const {}
};
struct NoPaint {
  void operator()(const CustomPaint &) const {}
};

// True when `reason` is the refusal, and sets it back for the next call.
bool Refused(Element result, ElementError &reason) {
  const bool refused = result.IsNone() && reason == ElementError::BuildingInHook;
  reason = ElementError::None;
  return refused;
}

// Tries every call that would add a node, a record or stored bytes, and Render and Begin, from
// inside the hook. `spare` is a plain box made before Render, `clipBox` a clipping box that has no record
// yet, so a list record made for it would show, and `control` an interactive one.
struct BuildProbe {
  UiFrame *frame;
  Element spare;
  Element clipBox;
  Element control;
  BuildAttempts *out;
  void operator()(UiFrame &hooked, Element root) const {
    BuildAttempts &seen = *out;
    seen.flagInside = FrameTestHooks::InHook(*frame);
#ifdef NDEBUG
    // Debug builds assert instead, which would stop the run, so only Release tries these.
    const size_t bytes = FrameTestHooks::CallableBytes(hooked);
    const size_t records = FrameTestHooks::InteractRecords(hooked);
    const size_t nodes = FrameTestHooks::NodeCount(hooked);
    ElementError reason = ElementError::None;
    seen.box = Refused(Box(hooked, {}, {}, {}, &reason), reason);
    seen.custom = Refused(Custom(hooked, {}, NoPaint{}, &reason), reason);
    seen.activate = Refused(OnActivate(hooked, control, NoActivate{}, &reason), reason);
    seen.input = Refused(OnInput(hooked, control, NoInput{}, &reason), reason);
    seen.makeInteractive = Refused(MakeInteractive(hooked, spare, {}, &reason), reason);
    seen.ownKeys = Refused(OwnKeys(hooked, spare, {}, &reason), reason);
    seen.asList = Refused(AsList(hooked, clipBox, &reason), reason);
    seen.specimen = Refused(Specimen(hooked, spare, {}, &reason), reason);
    ImDrawList scratch(ImGui::GetDrawListSharedData());
    const RenderResult nested = Render(hooked, root, scratch, {}, NoRoot{});
    seen.render = nested.error == ElementError::BuildingInHook;
    seen.flagStillSetAfterNestedRender = FrameTestHooks::InHook(hooked);
    const uint32_t serial = FrameTestHooks::Serial(hooked);
    hooked.Begin(hooked.Ui().theme, hooked.Ui().scale, hooked.Ui().fonts, hooked.Store(), hooked.Now());
    seen.beginIgnored = FrameTestHooks::Serial(hooked) == serial && Arranged(hooked, root).width > 0;
    seen.storageUntouched = bytes == FrameTestHooks::CallableBytes(hooked) && records == FrameTestHooks::InteractRecords(hooked) &&
                            nodes == FrameTestHooks::NodeCount(hooked);
#else
    (void)hooked;
    (void)root;
#endif
  }
  // The hook of the nested Render, which must never run.
  struct NoRoot {
    void operator()(UiFrame &, Element) const {}
  };
};

// The hook may restyle but not build. The flag that says so is set while the hook runs and clear
// after Render, and in Release every call that adds a node, a record or stored bytes is refused
// with BuildingInHook and leaves the frame's storage as it was, as are a nested Render, which also
// leaves the flag set, and a Begin, which leaves the frame alone. Outside the hook the same calls
// work, which is the baseline for "refused because of the hook".
void TestHookRefusesBuildingNodes() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const TwoControls tree = BuildTwoControls(frame);
    const Element spare = Box(frame, {}, {});
    Style clipStyle;
    clipStyle.overflow = Overflow::Clip;
    const Element clipBox = Box(frame, {}, clipStyle);
    Place(frame, tree.root, HOOK_ROOT);
    Check(!FrameTestHooks::InHook(frame) && !Box(frame, {}, {}).IsNone(), "before Render the flag is clear and a node can be built");
    Check(!OnActivate(frame, tree.first, NoActivate{}).IsNone() && !OnInput(frame, tree.first, NoInput{}).IsNone(), "and callbacks can be attached");
    Check(!Custom(frame, {}, NoPaint{}).IsNone() && !OwnKeys(frame, spare, {}).IsNone() && !Specimen(frame, spare, {}).IsNone(), "so can custom nodes, keys and specimens");
    BuildAttempts seen;
    auto list = OwnList();
    Check(!Render(frame, tree.root, *list, {}, BuildProbe{&frame, spare, clipBox, tree.first, &seen}).error.has_value(), "the tree renders");
    Check(seen.flagInside, "the flag is set while the hook runs");
#ifdef NDEBUG
    Check(seen.box && seen.custom, "a node or a custom node built in the hook is refused");
    Check(seen.activate && seen.input, "so are callbacks, which would move the bytes the hook runs from");
    Check(seen.makeInteractive && seen.ownKeys && seen.asList && seen.specimen, "so are interaction records");
    Check(seen.render && seen.flagStillSetAfterNestedRender, "a nested Render is refused and leaves the hook flag set");
    Check(seen.beginIgnored, "Begin does nothing in the hook");
    Check(seen.storageUntouched, "and none of it changed the frame's storage");
#endif
    Check(!FrameTestHooks::InHook(frame) && !Box(frame, {}, {}).IsNone(), "after Render the flag is clear and a node can be built again");
    // The baseline for the refusals above, on the same nodes, so each was the hook's doing.
    Check(!MakeInteractive(frame, spare, {}).IsNone(), "outside the hook MakeInteractive works on the same node");
    Check(!AsList(frame, clipBox).IsNone(), "and AsList works on the same clipping box");
  });
}

// ---- AfterInteract on a node -------------------------------------------------------------------

// Adds this callback's tag to a number that records the order they ran in.
struct LogRun {
  int *log;
  int tag;
  void operator()(UiFrame &, Element) const { *log = *log * 10 + tag; }
};

// Keeps the element a callback was called with.
struct KeepElement {
  Element *seen;
  void operator()(UiFrame &, Element element) const { *seen = element; }
};

// A callback on the first control sees the later sibling's input, so it runs after the whole walk,
// and it is given its own node. Before Render the input isn't there, which is the baseline.
void TestAfterInteractRunsAfterTheWalk() {
  NavHarness nav;
  UiFrame frame;
  nav.Frames(2, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const TwoControls tree = BuildTwoControls(frame);
    std::optional<NodeInput> seenFirst, seenSecond;
    Check(!AfterInteract(frame, tree.first, InputSpy{tree.first, tree.second, &seenFirst, &seenSecond}).IsNone(), "the callback is accepted");
    Place(frame, tree.root, HOOK_ROOT);
    Check(!InputOf(frame, tree.second).has_value(), "before Render there is no input");
    Check(!Render(frame, tree.root, context.list).error.has_value(), "the tree renders");
    Check(seenFirst.has_value() && seenSecond.has_value(), "the callback saw both nodes' input");
    Check(seenSecond->id == InputOf(frame, tree.second)->id && seenFirst->id != seenSecond->id, "and they are each its own");
  });
  nav.Frames(1, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const TwoControls tree = BuildTwoControls(frame);
    Element seen;
    (void)AfterInteract(frame, tree.second, KeepElement{&seen});
    Place(frame, tree.root, HOOK_ROOT);
    (void)Render(frame, tree.root, context.list);
    Check(seen == tree.second, "a callback is given its own node");
  });
}

// Each callback runs once a Render, in tree order with children before their parents, and a plain
// box takes one as well as an interactive node. The hook of the whole tree runs last.
void TestAfterInteractOrder() {
  NavHarness nav;
  UiFrame frame;
  nav.Frames(2, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const TwoControls tree = BuildTwoControls(frame);
    int log = 0;
    (void)AfterInteract(frame, tree.root, LogRun{&log, 3});
    (void)AfterInteract(frame, tree.second, LogRun{&log, 2});
    (void)AfterInteract(frame, tree.first, LogRun{&log, 1});
    Place(frame, tree.root, HOOK_ROOT);
    Check(!Render(frame, tree.root, context.list, {}, LogRun{&log, 4}).error.has_value(), "the tree renders");
    Check(log == 1234, "children first, in tree order, then the parent, then the tree's hook, each once");
  });
  // Deeper down too, as a node two levels below the root is reached like one level below it.
  UiFrame deep;
  nav.Frame([&](InteractContext &context) {
    deep.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    int log = 0;
    const Element leaf = Box(deep, {.item = {.width = Dp(HOOK_CONTROL_WIDTH), .height = Dp(HOOK_CONTROL_HEIGHT)}}, {});
    const std::array<Element, 1> leaves{leaf};
    const Element middle = Box(deep, {}, {}, leaves);
    const std::array<Element, 1> middles{middle};
    const Element top = Box(deep, {}, {}, middles);
    (void)AfterInteract(deep, leaf, LogRun{&log, 1});
    (void)AfterInteract(deep, middle, LogRun{&log, 2});
    (void)AfterInteract(deep, top, LogRun{&log, 3});
    Place(deep, top, HOOK_ROOT);
    Check(!Render(deep, top, context.list).error.has_value(), "the deep tree renders");
    Check(log == 123, "from the deepest up");
  });
}

// A node that layout collapsed has no input, as the walk passes it by, and its callback runs all the
// same, so that what it works out can be let go. In a row with room the same node is the baseline,
// with input.
void TestAfterInteractOnACollapsedNode() {
  constexpr float NARROW = 20;
  const auto run = [](float width) {
    NavHarness nav;
    UiFrame frame;
    std::pair<int, bool> found{0, false};
    nav.Frame([&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Element control = Box(frame, {.item = {.width = Dp(HOOK_CONTROL_WIDTH), .height = Dp(HOOK_CONTROL_HEIGHT), .collapses = true}}, {});
      (void)Keyed(frame, control, "control");
      (void)MakeInteractive(frame, control, {});
      (void)AfterInteract(frame, control, LogRun{&found.first, 1});
      const std::array<Element, 1> kids{control};
      const Element root = Box(frame, {.direction = FlexDirection::Row}, {}, kids);
      (void)Measure(frame, root, {width, HOOK_CONTROL_HEIGHT});
      (void)Arrange(frame, root, {HOOK_ROOT.x, HOOK_ROOT.y, width, HOOK_CONTROL_HEIGHT});
      Check(!Render(frame, root, context.list).error.has_value(), "the tree renders");
      found.second = InputOf(frame, control).has_value();
    });
    return found;
  };
  const auto roomy = run(HOOK_ROOT.width);
  const auto narrow = run(NARROW);
  Check(roomy.first == 1 && roomy.second, "with room the node has input and its callback runs");
  Check(narrow.first == 1 && !narrow.second, "collapsed it has none and its callback runs all the same");
}

// A colour set in a callback is in the same frame's paint. Without the callback the label keeps its
// own colour, which is the baseline for it being the callback's doing.
void TestAfterInteractColourShowsInThatFramesPaint() {
  const auto drawn = [](bool recolour) {
    const Color start = Rgb(0x102030);
    const Color other = Rgb(0x405060);
    NavHarness nav;
    UiFrame frame;
    std::pair<bool, bool> found{false, false};
    nav.Frame([&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Label label = BuildLabel(frame, start);
      if (recolour) {
        (void)AfterInteract(frame, label.root, Recolour{label.text, other});
      }
      Place(frame, label.root, HOOK_ROOT);
      auto list = OwnList();
      (void)Render(frame, label.root, *list);
      for (const ImDrawVert &vertex : list->VtxBuffer) {
        found.first = found.first || vertex.col == ToImU32(start);
        found.second = found.second || vertex.col == ToImU32(other);
      }
    });
    return found;
  };
  const auto plain = drawn(false);
  const auto recoloured = drawn(true);
  Check(plain.first && !plain.second, "a label keeps its colour with no callback");
  Check(!recoloured.first && recoloured.second, "and takes the callback's colour in the same frame");
}

// Tries to attach another callback from inside one, and says whether it was refused and why.
struct TryAttach {
  bool *refused;
  ElementError *why;
  void operator()(UiFrame &frame, Element element) const {
    int unused = 0;
    *refused = AfterInteract(frame, element, LogRun{&unused, 1}, why).IsNone();
  }
};

// A none handle is refused and says so. A second call replaces the first. A callback doesn't run
// when Render is refused or the tree is only painted, and one can't be attached from inside another.
void TestAfterInteractRefusals() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    int first = 0;
    int second = 0;
    ElementError reason = ElementError::Stale;
    Check(AfterInteract(frame, Element{}, LogRun{&first, 1}, &reason).IsNone() && reason == ElementError::None, "a none handle is refused and says so");
    const TwoControls tree = BuildTwoControls(frame);
    Check(!AfterInteract(frame, tree.root, LogRun{&first, 1}).IsNone() && !AfterInteract(frame, tree.root, LogRun{&second, 1}).IsNone(), "a second call is accepted");
    Place(frame, tree.root, HOOK_ROOT);
    Check(!Render(frame, tree.root, context.list).error.has_value(), "the tree renders");
    Check(first == 0 && second == 1, "and only the second runs");
    Check(Render(frame, tree.root, context.list).error == ElementError::AlreadyRendered && second == 1, "a refused Render runs nothing");
    int painted = 0;
    const Element alone = Box(frame, {.item = {.width = Dp(HOOK_CONTROL_WIDTH), .height = Dp(HOOK_CONTROL_HEIGHT)}}, {});
    (void)AfterInteract(frame, alone, LogRun{&painted, 1});
    Place(frame, alone, HOOK_ROOT);
    Paint(frame, alone, *OwnList());
    Check(painted == 0, "nor does a tree that is only painted");
    // Two unkeyed roots in a frame collide, so Render refuses this one as it assigns its ids.
    int colliding = 0;
    (void)AfterInteract(frame, alone, LogRun{&colliding, 1});
    Check(Render(frame, alone, context.list).error == ElementError::IdCollision && colliding == 0, "nor when its ids can't be assigned");
#ifdef NDEBUG
    bool refused = false;
    ElementError why = ElementError::None;
    const Element nesting = Box(frame, {.item = {.width = Dp(HOOK_CONTROL_WIDTH), .height = Dp(HOOK_CONTROL_HEIGHT)}}, {});
    (void)Keyed(frame, nesting, "nesting");
    (void)AfterInteract(frame, nesting, TryAttach{&refused, &why});
    Place(frame, nesting, HOOK_ROOT);
    Check(!Render(frame, nesting, context.list).error.has_value(), "a second root renders");
    Check(refused && why == ElementError::BuildingInHook, "one can't be attached from inside another");
#endif
  });
}

// Whether doing `refuse` logs `why`, with no reason asked for, as nearly every caller leaves it out.
// The first report after forgetting is what proves a false answer means "already logged", and
// nothing else.
template <class Refuse> bool LogsRefusal(ElementError why, Refuse refuse) {
  ForgetReportedRefusals();
  Check(ReportRefusal(why) && !ReportRefusal(why), "a reason is logged the first time and not again");
  ForgetReportedRefusals();
  refuse();
  return !ReportRefusal(why);
}

// Each place the element layer refuses something logs it, as most callers drop the refusal. A none
// handle is never logged, since it is only what an earlier refusal handed on.
void TestRefusalsAreReported() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element box = Box(frame, {}, {});
    const Element other = Box(frame, {}, {});
    Check(LogsRefusal(ElementError::InvalidPose, [&] { (void)WithPose(frame, box, {.scale = -1.0f}); }), "a bad pose is logged");
    Check(LogsRefusal(ElementError::NotInteractive, [&] { (void)OnActivate(frame, box, NoActivate{}); }), "a callback on a plain node is logged");
    Check(LogsRefusal(ElementError::NotText, [&] { (void)SetTextColor(frame, box, {}); }), "a text colour on a box is logged");
    Check(LogsRefusal(ElementError::InvalidKey, [&] { (void)Keyed(frame, box, std::string_view{}); }), "an empty key is logged");
    Check(LogsRefusal(ElementError::AlreadyParented, [&] {
            const std::array<Element, 2> twice{other, other};
            (void)Box(frame, {}, {}, twice);
          }), "a child listed twice is logged");
    Check(LogsRefusal(ElementError::IdCollision, [&] {
            const Element first = Box(frame, {}, {});
            const Element second = Box(frame, {}, {});
            (void)AssignIds(frame, first, 7);
            (void)AssignIds(frame, second, 7);
          }), "two roots with one id are logged");
    Check(LogsRefusal(ElementError::IdCollision, [&] {
            // These differ as text, so only the ids can tell, deep in the walk and not at the root.
            const std::array<Element, 2> kids{Keyed(frame, Box(frame, {}, {}), "a###x"), Keyed(frame, Box(frame, {}, {}), "b###x")};
            (void)AssignIds(frame, Box(frame, {}, {}, kids), 11);
          }), "two children with one id are logged");
    Check(LogsRefusal(ElementError::AlreadyRendered, [&] {
            const Element root = Box(frame, {}, {});
            (void)Measure(frame, root, {10, 10});
            (void)Arrange(frame, root, {0, 0, 10, 10});
            auto list = OwnList();
            (void)Render(frame, root, *list);
            (void)Render(frame, root, *list);
          }), "a second Render is logged");
    ForgetReportedRefusals();
    Check(!ReportRefusal(ElementError::None), "a none handle is not logged");
    (void)WithPose(frame, Element{}, {});
    Check(ReportRefusal(ElementError::InvalidPose), "and refusing one doesn't log a bad pose");
  });
}

// The refusal reaches the log, once, by name. Nothing logs to a file here, so it goes to stdout,
// which is pointed at a file. It can't be pointed back everywhere, so this runs after main's last
// line of output.
void TestRefusalReachesTheLog() {
  const auto path = std::filesystem::temp_directory_path() / "kyty_system_ui_refusal_log.txt";
  std::fflush(stdout);
  Check(std::freopen(path.string().c_str(), "wb", stdout) != nullptr, "stdout can be pointed at a file");
  ForgetReportedRefusals();
  (void)ReportRefusal(ElementError::NotClippingBox);
  (void)ReportRefusal(ElementError::NotClippingBox);
  std::fflush(stdout);
  std::ifstream log(path, std::ios::binary);
  std::stringstream text;
  text << log.rdbuf();
  const std::string found = text.str();
  const size_t first = found.find("NotClippingBox");
  Check(first != std::string::npos, "the log names the reason");
  Check(found.find("NotClippingBox", first + 1) == std::string::npos, "and only once");
}

// What each call did inside a callback, as "refused with BuildingInCallback", so a call that slipped
// through shows which one.
struct CallbackAttempts {
  bool ran = false;
  bool box = false;
  bool custom = false;
  bool activate = false;
  bool input = false;
  bool makeInteractive = false;
  bool ownKeys = false;
  bool claimDpad = false;
  bool asList = false;
  bool specimen = false;
  bool decorate = false;
  bool withPose = false;
  bool setInput = false;
  bool setTextColor = false;
  bool keyed = false;
  bool assignIds = false;
  bool render = false;
  bool beginIgnored = false;
  bool storageUntouched = false;
};

// True when `reason` is a callback's refusal, and sets it back for the next call.
bool RefusedInCallback(bool refused, ElementError &reason) {
  const bool right = refused && reason == ElementError::BuildingInCallback;
  reason = ElementError::None;
  return right;
}

// Tries every call that would add to or change the tree, from inside an input callback, which runs
// on every frame. The nodes are made before Render, and are ones each call would otherwise accept.
struct CallbackBuildProbe {
  UiFrame *frame;
  Element spare;
  Element clipBox;
  Element control;
  Element label;
  Element root;
  CallbackAttempts *out;
  void operator()(const NodeInput &) const {
    CallbackAttempts &seen = *out;
    seen.ran = true;
#ifdef NDEBUG
    // Debug builds assert instead, which would stop the run, so only Release tries these.
    UiFrame &inside = *frame;
    const size_t bytes = FrameTestHooks::CallableBytes(inside);
    const size_t records = FrameTestHooks::InteractRecords(inside);
    const size_t decors = FrameTestHooks::DecorRecords(inside);
    const size_t nodes = FrameTestHooks::NodeCount(inside);
    ElementError reason = ElementError::None;
    const auto none = [&](Element result) { return RefusedInCallback(result.IsNone(), reason); };
    seen.box = none(Box(inside, {}, {}, {}, &reason));
    seen.custom = none(Custom(inside, {}, NoPaint{}, &reason));
    seen.activate = none(OnActivate(inside, control, NoActivate{}, &reason));
    seen.input = none(OnInput(inside, control, NoInput{}, &reason));
    seen.makeInteractive = none(MakeInteractive(inside, spare, {}, &reason));
    seen.ownKeys = none(OwnKeys(inside, spare, {}, &reason));
    seen.claimDpad = none(ClaimDpad(inside, control, DpadClaim::Once, &reason));
    seen.asList = none(AsList(inside, clipBox, &reason));
    seen.specimen = none(Specimen(inside, spare, {}, &reason));
    seen.decorate = none(Decorate(inside, spare, {}, &reason));
    seen.withPose = none(WithPose(inside, spare, {}, CENTER, &reason));
    seen.setInput = RefusedInCallback(!SetInput(inside, spare, {}, &reason), reason);
    seen.setTextColor = RefusedInCallback(!SetTextColor(inside, label, {}, &reason), reason);
    seen.keyed = none(Keyed(inside, spare, 1u, &reason));
    seen.assignIds = AssignIds(inside, spare, 0) == ElementError::BuildingInCallback;
    ImDrawList scratch(ImGui::GetDrawListSharedData());
    // The root being walked is the case only Render's own check can answer, as AssignIds refuses a
    // callback too, but is never reached for a root Render has already taken.
    seen.render = Render(inside, spare, scratch).error == ElementError::BuildingInCallback &&
                  Render(inside, root, scratch).error == ElementError::BuildingInCallback;
    const uint32_t serial = FrameTestHooks::Serial(inside);
    inside.Begin(inside.Ui().theme, inside.Ui().scale, inside.Ui().fonts, inside.Store(), inside.Now());
    seen.beginIgnored = FrameTestHooks::Serial(inside) == serial && FrameTestHooks::InCallback(inside);
    seen.storageUntouched = bytes == FrameTestHooks::CallableBytes(inside) && records == FrameTestHooks::InteractRecords(inside) &&
                            decors == FrameTestHooks::DecorRecords(inside) && nodes == FrameTestHooks::NodeCount(inside);
#endif
  }
};

// A callback may neither build nor change the tree, since either could move the storage it runs from
// or the children the walk is going through. In Release every such call is refused and leaves the
// frame's storage as it was, and Begin does nothing. Outside a callback the same calls on the same
// nodes work, which is the baseline for "refused because of the callback".
void TestCallbackRefusesBuilding() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const TwoControls tree = BuildTwoControls(frame);
    const Element spare = Box(frame, {}, {});
    Style clipStyle;
    clipStyle.overflow = Overflow::Clip;
    const Element clipBox = Box(frame, {}, clipStyle);
    const Element label = Text(frame, {"label", DEFAULT_THEME.type.body, Rgb(0xFFFFFF)});
    Place(frame, tree.root, HOOK_ROOT);
    CallbackAttempts seen;
    Check(!OnInput(frame, tree.first, CallbackBuildProbe{&frame, spare, clipBox, tree.first, label, tree.root, &seen}).IsNone(), "the probe is attached");
    auto list = OwnList();
    Check(!Render(frame, tree.root, *list).error.has_value(), "the tree renders");
    Check(seen.ran, "the callback ran");
#ifdef NDEBUG
    Check(seen.box && seen.custom, "a node or a custom node built in a callback is refused");
    Check(seen.activate && seen.input, "so are callbacks, which would move the bytes this one runs from");
    Check(seen.makeInteractive && seen.ownKeys && seen.claimDpad && seen.asList && seen.specimen, "so are interaction records");
    Check(seen.decorate && seen.withPose && seen.setInput && seen.setTextColor, "so are changes to how a node looks");
    Check(seen.keyed && seen.assignIds, "so are changes to its name");
    Check(seen.render, "a nested Render is refused");
    Check(seen.beginIgnored, "Begin does nothing");
    Check(seen.storageUntouched, "and none of it changed the frame's storage");
#endif
    // The baseline for the refusals above, on the same nodes, so each was the callback's doing.
    Check(!FrameTestHooks::InCallback(frame), "after Render the flag is clear");
    Check(!Box(frame, {}, {}).IsNone() && !Custom(frame, {}, NoPaint{}).IsNone(), "and nodes can be built again");
    Check(!OnActivate(frame, tree.first, NoActivate{}).IsNone() && !ClaimDpad(frame, tree.first, DpadClaim::Once).IsNone(), "callbacks and claims work on the same control");
    Check(!Decorate(frame, spare, {}).IsNone() && !WithPose(frame, spare, {}).IsNone(), "decor and poses work on the same node");
    Check(SetInput(frame, spare, {}) && SetTextColor(frame, label, Rgb(0xFFFFFF)), "so do input and text colour");
    Check(!Keyed(frame, spare, 1u).IsNone() && !AssignIds(frame, spare, 0).has_value(), "and keys and ids");
    Check(!MakeInteractive(frame, spare, {}).IsNone() && !AsList(frame, clipBox).IsNone(), "and interaction and lists");
    auto spareList = OwnList();
    Check(!Render(frame, spare, *spareList).error.has_value(), "and it renders");
  });
}

} // namespace

int main() {
  TestChildIdIsIndexSeededByParent();
  TestRootKeyMatchesIdFor();
  TestReorderingFollowsData();
  TestDuplicateKeysRefused();
  TestKeyedRefusals();
  TestNumberKeyCollidesWithIndex();
  TestHashAlikeKeysRefused();
  TestNestedCollisionClearsTree();
  TestSecondRootNeedsKey();
  TestSameRootAssignedTwice();
  TestElementDpadMovesFocus();
  TestElementGridNavigation();
  TestElementDisabledIsSkipped();
  TestElementActivation();
  TestElementPose();
  TestElementPressHighlight();
  TestElementFocusScrollsTheWindow();
  TestElementDisabledKeepsFocusButNoPress();
  TestElementClaimsHorizontal();
  TestElementHitsTheLayoutRect();
  TestNestedInteractiveRefused();
  TestSecondMakeInteractiveRefused();
  TestClaimDpadKeepsTheRestOfTheRecord();
  TestListHitClipIgnoresRoundedCorners();
  TestHitClipEndsWithItsBox();
  TestRenderReportsIdError();
  TestClipLimitsHitTesting();
  TestRenderTwiceRefused();
  TestElementButtonMatchesRealButton();
  TestRefusedKeyInternsNothing();
  TestCallbackToggleMatchesRealToggle();
  TestDisabledNodeGivesBackTheDpad();
  TestCollapsedInteractiveChildIsNotNavigable();
  TestCollapsedAnimationReplaysItsEntrance();
  TestCallbackCountsMatchActivations();
  TestCallbacksKeepReservedBytesSteady();
  TestCallbacksRefusedOnNonInteractive();
  TestDescendantStyleFollowsInteractiveAncestor();
  TestDisabledFadesInheritingBoxesOnce();
  TestCallbackFlagIsSetInsideAndClearedAfter();
  TestSetInputClearsInherited();
  TestHookSeesEveryNodesInputAfterTheWalk();
  TestHookColourShowsInThatFramesPaint();
  TestHookInputChangesThatFramesStateStyles();
  TestSetTextColorRefusals();
  TestHookRefusesBuildingNodes();
  TestAfterInteractRunsAfterTheWalk();
  TestAfterInteractOrder();
  TestAfterInteractOnACollapsedNode();
  TestAfterInteractColourShowsInThatFramesPaint();
  TestAfterInteractRefusals();
  TestCallbackRefusesBuilding();
  TestRefusalsAreReported();
  TestListRememberedRow();
  TestListSelectedWithoutFocus();
  TestListDisabledRowSkipped();
  TestListActivation();
  TestListScrolls();
  TestListHiddenRowsIgnoreTheMouse();
  TestListClickIsNotRedirected();
  TestListDisabledRowNotRestored();
  TestListScreenChange();
  TestListControlBelowTheList();
  TestListHiddenRowsNotDrawn();
  TestListShrinkingScrollsBack();
  TestScrolledChildrenMove();
  TestDataKeysSurviveInsertion();
  TestAsListRefusals();
  TestListCornerHoverMatchesTheListComponent();
  TestPaintedFollowsListScroll();
  TestNestedListReservedBytesSteady();
  TestListReservedBytesSteady();
  TestKeysMatchPushIdNesting();
  TestAnimationMatchesManualPose();
  TestReorderKeepsStartWhenKeyed();
  TestDismissalIsRecordedOnce();
  TestMissingFrameReplaysEntrance();
  TestAnimationComposesInsideExplicitPose();
  TestAnimatingFollowsTimeline();
  TestAnimationRefusalsAndPhase();
  TestUndismissReplaysEntrance();
  TestElementBarOwnsShoulders();
  TestElementModalOverBar();
  TestOwnKeysRefusals();
  TestElementBarAppearingInConsumedFrameStandsDown();
  TestElementBackIsConsumedForLegacyReaders();
  std::printf("System UI element interaction tests passed\n");
  TestRefusalReachesTheLog();
  return 0;
}
