#ifndef GRAPHICS_SHADER_STOREGEN_RESOURCE_KEYS_H
#define GRAPHICS_SHADER_STOREGEN_RESOURCE_KEYS_H

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/storegen/computePredictor.h"

#include <optional>
#include <string>

// Reads a translated compute program's resource lists as descriptor keys. This is route K of the
// plan: Kyty's own translation lists the buffers, images and samplers in its own order, and each
// descriptor's source values say where the descriptor lives.
namespace Libs::Graphics::StoreGen {

/// Classifies the descriptors of a translated program.
/// Known gap: a descriptor made from a user or EUD base moved by an add (python's "derived" key) is
/// not classified, so the program is skipped. No Yumia program has one.
/// The program must be translated with zeroed user data sized to its user SGPR count, as the
/// runtime translates a compute program. Each descriptor's dwords must be user SGPRs, dwords of an
/// extended user data table, or a literal pool table, and the descriptor must not be indirect.
/// @param plan the translated program's resource plan, such as TranslateResult::program.
/// @param why  if given, receives the reason when the result is empty.
/// @return the lists in Kyty's order with the flags Kyty gives them, or nothing when any descriptor
///         is indirect or any buffer or image dword cannot be classified, so the caller skips the
///         program. A sampler that does not classify is kept with no key, as python counts it.
std::optional<ResourceFacts> ExtractResourceFacts(const ShaderRecompiler::IR::ResourcePlan& plan,
                                                  std::string* why = nullptr);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_RESOURCE_KEYS_H
