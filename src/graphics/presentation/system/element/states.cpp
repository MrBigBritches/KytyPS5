#include "graphics/presentation/system/element/states.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

namespace Libs::Graphics::SystemUi {

namespace {

// How strong each state is, from 0 to 1.
struct Weights {
	float selected = 0.0f;
	float hover    = 0.0f;
	float focus    = 0.0f;
	float pressed  = 0.0f;
	float disabled = 0.0f;
};

// How strong each state is for `input`. Selected and disabled are on or off, so they weigh 0 or 1,
// and the highlights weigh as far as they have come, held to 0 through 1.
Weights WeightsOf(const ItemInput& input) {
	return {input.selected ? 1.0f : 0.0f, ClampUnit(input.visuals.hover),
	        ClampUnit(input.visuals.focus), ClampUnit(input.visuals.press),
	        input.disabled ? 1.0f : 0.0f};
}

// The states with their weights in the order they apply.
using Ordered = std::array<std::pair<const StateStyle*, float>, 5>;

Ordered InOrder(const StateStyles& states, const Weights& weights) {
	return {{{&states.selected, weights.selected},
	         {&states.hover, weights.hover},
	         {&states.focus, weights.focus},
	         {&states.pressed, weights.pressed},
	         {&states.disabled, weights.disabled}}};
}

// The value `t` of the way from `from` to `to`, which is `from` at 0 and `to` at 1.
float Mix(float from, float to, float t) {
	return from + (to - from) * t;
}
// The same for colours, blended channel by channel by Lerp.
Color Mix(Color from, Color to, float t) {
	return Lerp(from, to, t);
}
// The same for lengths, so Blend treats a Dp like a float.
Dp Mix(Dp from, Dp to, float t) {
	return Dp(Mix(from.value, to.value, t));
}

// `base` moved toward the value each state sets for `field`, in the merge order. Hover and focus
// that set the same value approach it once, at the larger of their weights, so blending the two
// of them is the same as blending either alone at that weight.
template <class T>
T Blend(T base, const StateStyles& states, std::optional<T> StateStyle::* field,
        const Weights& weights) {
	T          value  = base;
	const auto toward = [&](const StateStyle& state, float weight) {
		if ((state.*field).has_value()) {
			value = Mix(value, *(state.*field), weight);
		}
	};
	toward(states.selected, weights.selected);
	const auto& hover = states.hover.*field;
	const auto& focus = states.focus.*field;
	if (hover.has_value() && focus.has_value() && *hover == *focus) {
		value = Mix(value, *hover, std::max(weights.hover, weights.focus));
	} else {
		toward(states.hover, weights.hover);
		toward(states.focus, weights.focus);
	}
	toward(states.pressed, weights.pressed);
	toward(states.disabled, weights.disabled);
	return value;
}

// The elevation of the last state that sets one and is at least half on.
Elevation PickElevation(const Elevation& base, const Ordered& ordered) {
	constexpr float HALF = 0.5f;
	Elevation       pick = base;
	for (const auto& [state, weight]: ordered) {
		if (state->elevation.has_value() && weight >= HALF) {
			pick = *state->elevation;
		}
	}
	return pick;
}

// The overlays, the focus ring and the selection the states ask for, at their weights.
void CollectDrawing(const Ordered& ordered, ResolvedStyle& resolved) {
	for (const auto& [state, weight]: ordered) {
		if (weight <= 0.0f) {
			continue;
		}
		if (state->overlay.a > 0) {
			resolved.overlays[resolved.overlay_count++] = {state->overlay, weight};
		}
		if (state->focus_ring) {
			resolved.focus_ring = std::max(resolved.focus_ring, weight);
		}
		if (state->selection) {
			resolved.selection = std::max(resolved.selection, weight);
		}
	}
}

} // namespace

ResolvedStyle ResolveStyle(const Theme& theme, const Style& style, const ItemInput& input) {
	const Weights weights = WeightsOf(input);
	const Ordered ordered = InOrder(style.states, weights);
	ResolvedStyle resolved;
	resolved.fill   = Blend(style.fill, style.states, &StateStyle::fill, weights);
	resolved.border = Blend(style.border, style.states, &StateStyle::border, weights);
	resolved.border_width =
	    Blend(style.border_width, style.states, &StateStyle::border_width, weights).value;
	resolved.radius    = Blend(style.radius, style.states, &StateStyle::radius, weights).value;
	resolved.elevation = PickElevation(style.elevation, ordered);
	resolved.opacity   = Blend(1.0f, style.states, &StateStyle::opacity, weights);
	resolved.scale     = Blend(1.0f, style.states, &StateStyle::scale, weights);
	if (input.disabled && !input.inherited && !style.states.disabled.opacity.has_value()) {
		resolved.opacity *= theme.opacity.disabled;
	}
	// A state that sets nonsense must not poison the box's whole subtree.
	if (!std::isfinite(resolved.scale) || resolved.scale <= 0.0f) {
		resolved.scale = 1.0f;
	}
	if (!std::isfinite(resolved.opacity)) {
		resolved.opacity = 1.0f;
	}
	CollectDrawing(ordered, resolved);
	return resolved;
}

} // namespace Libs::Graphics::SystemUi
