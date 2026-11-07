#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_TOKENPATCH_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_TOKENPATCH_H_

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace Libs::Graphics::SystemUi {

/// A few fields of a component's token struct set by the caller, to be merged over the component's
/// defaults one field at a time, so restyling one thing doesn't mean copying the whole set.
///
/// A patch is built by naming fields as template arguments, with the value as the argument:
/// `TokenPatch<ButtonTokens> {}.With<&ButtonTokens::radius>(4.0f)`. Each call returns a new patch,
/// so several chain. `Apply` takes the defaults and returns them with just those fields replaced,
/// and every other field stays what the defaults say, even when the defaults come from another
/// theme. A field named twice keeps its last value.
///
/// The value is converted to the field's type as if the field were assigned it, so
/// `With<&T::radius>(4)` takes an int for a float field. Only an implicit conversion is allowed, so
/// a field of a type with an explicit constructor, such as a Dp, is set with `Dp(12.0f)` and not a
/// bare float, and an aggregate such as an Elevation can't be built from one number by accident.
///
/// For each field named the patch keeps the value, in a copy of the tokens, and a pointer to a
/// function that assigns that member from the copy. Nothing depends on the layout of the struct.
/// The tokens must be trivially copyable, so the patch is too, and can be in a component's props or
/// be captured by a painter. A patch holds up to CAPACITY fields. A patch that is given more is
/// marked Overflowed, asserts in debug builds, and leaves the extra fields at their defaults, so
/// ask for the narrowest set that restyles the part.
template <class Tokens>
class TokenPatch {
	static_assert(std::is_trivially_copyable_v<Tokens>,
	              "A patch is copied as plain data, so the tokens must be trivially copyable.");

	// The type of the member that `Member` points to.
	template <auto Member>
	using FieldOf = std::remove_cvref_t<decltype(std::declval<Tokens&>().*Member)>;

public:
	/// The most fields one patch can set.
	static constexpr size_t CAPACITY = 16;

	/// This patch with the field `Member` set to `value`, which converts to the field's type
	/// implicitly.
	template <auto Member>
	[[nodiscard]] constexpr TokenPatch With(std::type_identity_t<FieldOf<Member>> value) const {
		return Set<Member>(value);
	}

	/// `defaults` with the fields of this patch replaced and nothing else changed.
	[[nodiscard]] constexpr Tokens Apply(const Tokens& defaults) const {
		Tokens merged = defaults;
		for (size_t index = 0; index < count; ++index) {
			entries[index](merged, values);
		}
		return merged;
	}

	/// True when no field is set, so the defaults are used as they are.
	[[nodiscard]] constexpr bool Empty() const { return count == 0; }
	/// True when a field was dropped for want of room.
	[[nodiscard]] constexpr bool Overflowed() const { return overflowed; }

private:
	// Assigns one member of `merged` from `from`.
	using Copy = void (*)(Tokens& merged, const Tokens& from);

	template <auto Member>
	static constexpr void CopyField(Tokens& merged, const Tokens& from) {
		merged.*Member = from.*Member;
	}

	template <auto Member>
	[[nodiscard]] constexpr TokenPatch Set(const FieldOf<Member>& value) const {
		TokenPatch next     = *this;
		next.values.*Member = value;
		constexpr Copy copy = &CopyField<Member>;
		for (size_t index = 0; index < next.count; ++index) {
			if (next.entries[index] == copy) {
				return next;
			}
		}
		if (next.count == CAPACITY) {
			assert(false && "a token patch was given more fields than it holds");
			next.overflowed = true;
			return next;
		}
		next.entries[next.count++] = copy;
		return next;
	}

	Tokens values {};
	Copy   entries[CAPACITY] {};
	size_t count      = 0;
	bool   overflowed = false;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_TOKENPATCH_H_
