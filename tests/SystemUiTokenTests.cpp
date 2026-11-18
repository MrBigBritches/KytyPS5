// What a component's tokens are: built from the theme by XDefaults, and changed a few fields at a
// time by a TokenPatch, so restyling one thing leaves the rest as the theme gives it.
#include "SystemUiNavHarness.h"

#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/card.h"
#include "graphics/presentation/system/components/hud.h"
#include "graphics/presentation/system/components/list.h"
#include "graphics/presentation/system/components/modal.h"
#include "graphics/presentation/system/components/slider.h"
#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/components/toggle.h"
#include "graphics/presentation/system/components/trophyCup.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/core/units.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/primitives.h"

#include "imgui.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <type_traits>

namespace {

using namespace SystemUiTest;

constexpr float PATCHED_HEIGHT = 70.0f;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiTokenTests: failed: %s\n", message);
    std::abort();
  }
}

// A theme whose scales differ from the default's, so tokens built from it differ from the structs'
// own initial values, which are the default theme's. A patch that merged over the wrong base shows.
Theme OtherTheme() {
  Theme theme = DEFAULT_THEME;
  theme.radius.md = 7.0f;
  theme.radius.lg = 9.0f;
  theme.radius.xl = 11.0f;
  theme.space.s4 = 13.0f;
  theme.space.s5 = 17.0f;
  theme.border.bar = 5.0f;
  theme.motion.base = 222.0f;
  return theme;
}

// True when `merged` is `defaults` with `field` set to `value` and nothing else changed. The expected
// copy is made from `defaults` by assigning the one field, so it has everything else exactly.
template <class Tokens, class Field> bool OnlyFieldChanged(const Tokens &defaults, const Tokens &merged, Field Tokens::*field, Field value) {
  Tokens expected = defaults;
  expected.*field = value;
  return std::memcmp(&expected, &merged, sizeof(Tokens)) == 0 && merged.*field == value;
}

// ---- The patch itself ---------------------------------------------------------------------------

// A patch sets the fields it names and leaves every other one as the defaults it is merged over have
// it. Two patches chained set both, a field named twice keeps its last value, and the defaults come
// from the theme, so another theme's patched button keeps that theme's radius.
void TestPatchMergesOverTheDefaults() {
  const ButtonTokens base = ButtonDefaults(DEFAULT_THEME);
  const auto one = TokenPatch<ButtonTokens>{}.With<&ButtonTokens::height_large>(PATCHED_HEIGHT);
  Check(OnlyFieldChanged(base, one.Apply(base), &ButtonTokens::height_large, PATCHED_HEIGHT), "a patch changes the field it names and nothing else");
  const auto two = one.With<&ButtonTokens::min_width>(99.0f);
  const ButtonTokens both = two.Apply(base);
  Check(both.height_large == PATCHED_HEIGHT && both.min_width == 99.0f && both.radius == base.radius && both.padding_x == base.padding_x, "chained fields are all set, the rest are not");
  const ButtonTokens last = one.With<&ButtonTokens::height_large>(80.0f).Apply(base);
  Check(last.height_large == 80.0f, "a field named twice keeps the last value");
  Check(TokenPatch<ButtonTokens>{}.Empty() && !one.Empty(), "an empty patch says so");
  const Theme other = OtherTheme();
  const ButtonTokens otherBase = ButtonDefaults(other);
  Check(otherBase.radius == 7.0f && otherBase.padding_x == 17.0f && otherBase.padding_x_small == 13.0f, "the defaults follow the theme's scales");
  Check(otherBase.radius != base.radius, "so another theme's defaults differ from the first's");
  Check(OnlyFieldChanged(otherBase, one.Apply(otherBase), &ButtonTokens::height_large, PATCHED_HEIGHT), "a patch over another theme's defaults keeps that theme's other fields");
  Check(ButtonDefaults(DEFAULT_THEME).radius == ButtonTokens{}.radius, "and the default theme's defaults are the struct's own values");
}

// What a patch accepts, which is decided at compile time, so these are static assertions and a
// regression in what compiles breaks the build and not only a run. A field given an int, as in
// `With<&T::radius>(4)`, converts as an assignment would. A Dp field takes a Dp and not a bare
// float, since its constructor is explicit, and an aggregate field such as an Elevation takes no
// scalar, which would otherwise be built into a broken value. The patch is also usable in a constant
// expression.
struct MixedTokens {
  float radius = 1.0f;
  Libs::Graphics::SystemUi::Dp length{};
  int count = 0;
};

template <class Patch, auto Member, class Value>
concept PatchAccepts = requires(Patch patch, Value value) { patch.template With<Member>(value); };

using Libs::Graphics::SystemUi::Dp;
using MixedPatch = TokenPatch<MixedTokens>;
static_assert(PatchAccepts<MixedPatch, &MixedTokens::radius, int>, "an int is accepted for a float field");
static_assert(PatchAccepts<MixedPatch, &MixedTokens::length, Dp>, "a Dp is accepted for a Dp field");
static_assert(!PatchAccepts<MixedPatch, &MixedTokens::length, float>, "a bare float is not accepted for a Dp field");
static_assert(PatchAccepts<TokenPatch<ToastTokens>, &ToastTokens::elevation, Libs::Graphics::SystemUi::Elevation>, "an Elevation is accepted for an Elevation field");
static_assert(!PatchAccepts<TokenPatch<ToastTokens>, &ToastTokens::elevation, float>, "a scalar is not accepted for an aggregate field such as an Elevation");
static_assert(MixedPatch{}.With<&MixedTokens::radius>(4).Apply({}).radius == 4.0f, "an int sets a float field");
static_assert(MixedPatch{}.With<&MixedTokens::length>(Dp(12.0f)).Apply({}).length.value == 12.0f, "a Dp sets a Dp field");
static_assert(MixedPatch{}.With<&MixedTokens::count>(3).Apply({}).count == 3, "an int sets an int field");
static_assert(MixedPatch{}.With<&MixedTokens::radius>(4).Apply({}).count == 0, "and the fields not named keep the defaults");
static_assert(std::is_trivially_copyable_v<MixedPatch>, "a patch is plain data, so props and painters can hold it");
static_assert(TokenPatch<ButtonTokens>{}.With<&ButtonTokens::radius>(4).Apply(ButtonDefaults(DEFAULT_THEME)).radius == 4.0f, "the real tokens take an int too");

// The patch at run time with an int and a Dp field.
void TestPatchTakesConvertibleValues() {
  const auto patch = MixedPatch{}.With<&MixedTokens::length>(Dp(9.0f)).With<&MixedTokens::radius>(2);
  const MixedTokens merged = patch.Apply({});
  Check(merged.length.value == 9.0f && merged.radius == 2.0f && merged.count == 0, "a patch of an int and a Dp merges at run time too");
}

// A patch holds a fixed number of fields, and one given more is marked and keeps the first ones. Debug
// builds assert instead, so this runs only where asserts are off.
void TestPatchCapacity() {
#ifdef NDEBUG
  const auto patch = TokenPatch<ToastTokens>{}
                         .With<&ToastTokens::width>(1.0f)
                         .With<&ToastTokens::min_width>(1.0f)
                         .With<&ToastTokens::max_width_fraction>(1.0f)
                         .With<&ToastTokens::padding>(1.0f)
                         .With<&ToastTokens::icon>(1.0f)
                         .With<&ToastTokens::icon_radius>(1.0f)
                         .With<&ToastTokens::gap>(1.0f)
                         .With<&ToastTokens::min_height>(1.0f)
                         .With<&ToastTokens::edge_bar>(1.0f)
                         .With<&ToastTokens::cup_width>(1.0f)
                         .With<&ToastTokens::cup_height>(1.0f)
                         .With<&ToastTokens::cup_gap>(1.0f)
                         .With<&ToastTokens::radius>(1.0f)
                         .With<&ToastTokens::slide>(1.0f)
                         .With<&ToastTokens::enter_fade_ms>(1.0f)
                         .With<&ToastTokens::enter_move_ms>(1.0f)
                         .With<&ToastTokens::exit_fade_ms>(1.0f);
  Check(patch.Overflowed(), "a patch given more fields than it holds is marked");
  const ToastTokens merged = patch.Apply(ToastDefaults(DEFAULT_THEME));
  Check(merged.width == 1.0f && merged.exit_fade_ms == ToastDefaults(DEFAULT_THEME).exit_fade_ms, "and keeps the first ones and drops the extra");
#endif
}

// ---- One override per component -------------------------------------------------------------------

// Builds `make`'s element in a frame and returns its measured size with no limit.
template <class Make> Libs::Graphics::SystemUi::Vec2 MeasuredSize(Make &&make) {
  NavHarness nav;
  UiFrame frame;
  Libs::Graphics::SystemUi::Vec2 size{};
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element element = make(frame);
    Check(!element.IsNone(), "the component was built");
    const float unlimited = std::numeric_limits<float>::infinity();
    size = Measure(frame, element, {unlimited, unlimited});
  });
  return size;
}

// The same, in a frame begun with `theme` instead of the default.
template <class Make> Libs::Graphics::SystemUi::Vec2 MeasuredSizeIn(const Theme &theme, Make &&make) {
  NavHarness nav;
  UiFrame frame;
  Libs::Graphics::SystemUi::Vec2 size{};
  nav.Frame([&](InteractContext &context) {
    frame.Begin(theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element element = make(frame);
    Check(!element.IsNone(), "the component was built");
    const float unlimited = std::numeric_limits<float>::infinity();
    size = Measure(frame, element, {unlimited, unlimited});
  });
  return size;
}

// A button given a different medium height is that tall, no wider than before, and the same
// button with no override is the default height. The other tokens are the defaults'.
void TestButtonOverride() {
  const ButtonTokens base = ButtonDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<ButtonTokens>{}.With<&ButtonTokens::height_medium>(PATCHED_HEIGHT);
  Check(OnlyFieldChanged(base, patch.Apply(base), &ButtonTokens::height_medium, PATCHED_HEIGHT), "the button's override changes its one token");
  ButtonProps props{.label = "A label wide enough to be past the minimum width"};
  const auto normal = MeasuredSize([&](UiFrame &frame) { return Button(frame, "button", props); });
  props.tokens = patch;
  const auto taller = MeasuredSize([&](UiFrame &frame) { return Button(frame, "button", props); });
  Check(normal.y == base.height_medium && taller.y == PATCHED_HEIGHT, "the overridden button is the new height and the default one is not");
  Check(taller.x == normal.x, "and its width, which comes from other tokens, is unchanged");
}

// An override merges over the defaults of the theme the frame was begun with, and not over the
// struct's own values. In a theme with wider button padding the label's button is wider, with or
// without an override of the height, and it is wider than the default theme's.
void TestOverrideMergesOverTheThemesDefaults() {
  const Theme other = OtherTheme();
  ButtonProps props{.label = "A label wide enough to be past the minimum width"};
  const auto plainDefault = MeasuredSizeIn(DEFAULT_THEME, [&](UiFrame &frame) { return Button(frame, "button", props); });
  const auto plainOther = MeasuredSizeIn(other, [&](UiFrame &frame) { return Button(frame, "button", props); });
  props.tokens = TokenPatch<ButtonTokens>{}.With<&ButtonTokens::height_medium>(PATCHED_HEIGHT);
  const auto patchedOther = MeasuredSizeIn(other, [&](UiFrame &frame) { return Button(frame, "button", props); });
  Check(plainOther.x < plainDefault.x, "the other theme's button is narrower, with its smaller padding");
  Check(patchedOther.x == plainOther.x && patchedOther.y == PATCHED_HEIGHT, "and an override of the height keeps that theme's padding");
}

// A card given other padding insets its content by it, and the same card without is inset by the
// default. The corner radius, which was not named, is the default's.
void TestCardOverride() {
  const CardTokens base = CardDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<CardTokens>{}.With<&CardTokens::padding>(30.0f);
  Check(OnlyFieldChanged(base, patch.Apply(base), &CardTokens::padding, 30.0f), "the card's override changes its one token");
  const auto inset = [&](const TokenPatch<CardTokens> &given) {
    NavHarness nav;
    UiFrame frame;
    float found = 0.0f;
    nav.Frame([&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const std::array<Element, 1> kids{Box(frame, {.item = {.width = Dp(50.0f), .height = Dp(20.0f)}}, {})};
      const Element card = Card(frame, {}, {.tokens = given}, kids);
      const float unlimited = std::numeric_limits<float>::infinity();
      (void)Measure(frame, card, {unlimited, unlimited});
      (void)Arrange(frame, card, {100, 100, 200, 100});
      found = Arranged(frame, kids[0]).x - 100.0f;
    });
    return found;
  };
  Check(inset({}) == base.padding && inset(patch) == 30.0f, "a card's content is inset by its padding, the default or the override");
}

// A cup given a different width is that wide, and without it is the default.
void TestTrophyCupOverride() {
  const ToastTokens base = ToastDefaults(DEFAULT_THEME);
  const auto wide = [](const TokenPatch<ToastTokens> &given) {
    return MeasuredSize([&](UiFrame &frame) { return TrophyCup(frame, {.color = Rgb(0xE0B040), .tokens = given}); }).x;
  };
  Check(wide({}) == base.cup_width && wide(TokenPatch<ToastTokens>{}.With<&ToastTokens::cup_width>(40.0f)) == 40.0f, "a cup is as wide as its token, the default or the override");
}

// A list heading given a different row height is that tall, and without it is the default.
void TestListHeadingOverride() {
  const ListTokens base = ListDefaults(DEFAULT_THEME);
  const auto tall = [](const TokenPatch<ListTokens> &given) {
    return MeasuredSize([&](UiFrame &frame) { return ListHeading(frame, "heading", {.text = "Saves", .tokens = given}); }).y;
  };
  Check(tall({}) == base.row_single && tall(TokenPatch<ListTokens>{}.With<&ListTokens::row_single>(70.0f)) == 70.0f, "a heading is as tall as its token, the default or the override");
}

// A progress bar given a different height is that thick, and without it is the default.
void TestProgressOverride() {
  const ProgressTokens base = ProgressDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<ProgressTokens>{}.With<&ProgressTokens::height>(12.0f);
  Check(OnlyFieldChanged(base, patch.Apply(base), &ProgressTokens::height, 12.0f), "the progress bar's override changes its one token");
  const auto thick = [&](const TokenPatch<ProgressTokens> &given) {
    return MeasuredSize([&](UiFrame &frame) { return ProgressBar(frame, {.value = 0.5f, .fill = Rgb(0x33CC66), .tokens = given}); }).y;
  };
  Check(thick({}) == base.height && thick(patch) == 12.0f, "a progress bar is as thick as its token, the default or the override");
}

// A toggle given another width is that wide and as tall as before.
void TestToggleOverride() {
  const ToggleTokens base = ToggleDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<ToggleTokens>{}.With<&ToggleTokens::width>(100.0f);
  Check(OnlyFieldChanged(base, patch.Apply(base), &ToggleTokens::width, 100.0f), "the toggle's override changes its one token");
  bool on = false;
  const auto normal = MeasuredSize([&](UiFrame &frame) { return Toggle(frame, "toggle", &on); });
  const auto wider = MeasuredSize([&](UiFrame &frame) { return Toggle(frame, "toggle", &on, {.tokens = patch}); });
  Check(normal.x == base.width && wider.x == 100.0f && wider.y == normal.y && normal.y == base.height, "the overridden toggle is wider and as tall");
}

// A slider given a larger knob is as tall as the knob, and the default one is as tall as the default.
void TestSliderOverride() {
  const SliderTokens base = SliderDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<SliderTokens>{}.With<&SliderTokens::knob>(40.0f);
  Check(OnlyFieldChanged(base, patch.Apply(base), &SliderTokens::knob, 40.0f), "the slider's override changes its one token");
  float value = 0.5f;
  const auto normal = MeasuredSize([&](UiFrame &frame) { return Slider(frame, "slider", &value, {}).root; });
  SliderProps props;
  props.tokens = patch;
  const auto larger = MeasuredSize([&](UiFrame &frame) { return Slider(frame, "slider", &value, props).root; });
  Check(normal.y == base.knob && larger.y == 40.0f, "the slider is as tall as its knob, the default or the override");
}

// A row given a different single height is that tall, and a row with two lines, whose height was not
// named, keeps the default.
void TestListOverride() {
  const ListTokens base = ListDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<ListTokens>{}.With<&ListTokens::row_single>(PATCHED_HEIGHT);
  Check(OnlyFieldChanged(base, patch.Apply(base), &ListTokens::row_single, PATCHED_HEIGHT), "the list's override changes its one token");
  const auto row = [&](std::string_view second, const TokenPatch<ListTokens> &given) {
    return MeasuredSize([&](UiFrame &frame) { return ListRow(frame, "row", {"Primary", second, false, false, given}); }).y;
  };
  Check(row("", {}) == base.row_single && row("", patch) == PATCHED_HEIGHT, "a single row is as tall as its token, the default or the override");
  Check(row("Second", patch) == base.row_double, "and a two line row, which the override did not name, keeps the default");
}

// A tab bar given a different height is that tall, and the default bar is a medium button high.
void TestTabsOverride() {
  const TabTokens base = TabDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<TabTokens>{}.With<&TabTokens::height>(PATCHED_HEIGHT);
  Check(OnlyFieldChanged(base, patch.Apply(base), &TabTokens::height, PATCHED_HEIGHT), "the tab bar's override changes its one token");
  Check(base.height == ButtonDefaults(DEFAULT_THEME).height_medium, "the default bar is as tall as a medium button");
  Check(TabDefaults(DEFAULT_THEME).IsValid() && TabDefaults(OtherTheme()).IsValid(), "the tab bar's defaults are valid for the default theme and for another");
  const std::array<std::string_view, 2> labels{"One", "Two"};
  int selected = 0;
  const auto bar = [&](const TokenPatch<TabTokens> &given) {
    TabBarProps props;
    props.tokens = given;
    return MeasuredSize([&](UiFrame &frame) { return TabBar(frame, "tabs", labels, &selected, props).element; }).y;
  };
  Check(bar({}) == base.height && bar(patch) == PATCHED_HEIGHT, "a tab bar is as tall as its token, the default or the override");
}

// A toast given another width is that wide and as tall as the default toast, since its other tokens,
// the padding and the minimum height, are the default's.
void TestToastOverride() {
  const ToastTokens base = ToastDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<ToastTokens>{}.With<&ToastTokens::width>(480.0f);
  Check(OnlyFieldChanged(base, patch.Apply(base), &ToastTokens::width, 480.0f), "the toast's override changes its one token");
  const auto built = [&](const TokenPatch<ToastTokens> &given) {
    return MeasuredSize([&](UiFrame &frame) {
      ToastProps props{"Saved", "Slot 1"};
      props.tokens = given;
      return Toast(frame, props, 1920.0f).root;
    });
  };
  const auto normal = built({});
  const auto narrow = built(patch);
  Check(normal.x == base.width && narrow.x == 480.0f && narrow.y == normal.y, "the overridden toast is narrower and as tall");
}

// A dialog card given another medium width is that wide and as tall as asked, and without it is the
// default width.
void TestDialogOverride() {
  const DialogTokens base = DialogDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<DialogTokens>{}.With<&DialogTokens::width_medium>(500.0f);
  Check(OnlyFieldChanged(base, patch.Apply(base), &DialogTokens::width_medium, 500.0f), "the dialog's override changes its one token");
  NavHarness nav;
  Rect normal, narrow;
  nav.Frame([&](UiFrame &frame) {
    normal = DialogCard(frame, Libs::Graphics::SystemUi::Dp(300.0f));
    narrow = DialogCard(frame, Libs::Graphics::SystemUi::Dp(300.0f), patch);
  });
  Check(normal.width == base.width_medium && narrow.width == 500.0f && narrow.height == normal.height, "the overridden card is narrower and as tall");
}

// The heads-up panel's tokens follow the theme and take a patch like the others, though nothing
// draws one yet.
void TestHudOverride() {
  const HudTokens base = HudDefaults(DEFAULT_THEME);
  const auto patch = TokenPatch<HudTokens>{}.With<&HudTokens::padding_x>(20.0f);
  Check(OnlyFieldChanged(base, patch.Apply(base), &HudTokens::padding_x, 20.0f), "the HUD's override changes its one token");
  Check(HudDefaults(OtherTheme()).radius == OtherTheme().radius.sm, "and its defaults follow the theme");
}

} // namespace

int main() {
  TestPatchMergesOverTheDefaults();
  TestPatchTakesConvertibleValues();
  TestPatchCapacity();
  TestButtonOverride();
  TestOverrideMergesOverTheThemesDefaults();
  TestCardOverride();
  TestProgressOverride();
  TestTrophyCupOverride();
  TestListHeadingOverride();
  TestToggleOverride();
  TestSliderOverride();
  TestListOverride();
  TestTabsOverride();
  TestToastOverride();
  TestDialogOverride();
  TestHudOverride();
  std::printf("System UI token tests passed\n");
  return 0;
}
