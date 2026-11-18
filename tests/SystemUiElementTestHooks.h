#ifndef TESTS_SYSTEM_UI_ELEMENT_TEST_HOOKS_H_
#define TESTS_SYSTEM_UI_ELEMENT_TEST_HOOKS_H_

#include "graphics/presentation/system/element/frame.h"

namespace Libs::Graphics::SystemUi {

// Test-only access to what the primitives must not have, a way to forge handles and to move the
// frame serial. It lives under tests/ so no code in src/ includes it.
struct FrameTestHooks {
  // A handle with the given index and serial, to make stale or out of range ones on purpose.
  static Element Forge(uint32_t index, uint32_t serial) { return Element(index, serial); }

  // The serial of the frame's current generation.
  static uint32_t Serial(const UiFrame &frame) { return frame.serial_; }

  // Sets the serial, so a test can reach the wrap-around without UINT32_MAX frames.
  static void SetSerial(UiFrame &frame, uint32_t serial) { frame.serial_ = serial; }

  // How many natural measurements the frame made since Begin.
  static uint32_t NaturalMeasures(const UiFrame &frame) { return frame.naturalMeasures_; }

  // How many times Measure was called on a node since Begin.
  static uint32_t MeasureCalls(const UiFrame &frame) { return frame.measureCalls_; }

  // Whether a callback of Render is running, which the primitives refuse to build elements in.
  static bool InCallback(const UiFrame &frame) { return frame.inCallback_; }

  // Whether Render's afterInteract hook is running, which the primitives refuse to build nodes in.
  static bool InHook(const UiFrame &frame) { return frame.inHook_; }

  // How many bytes of callables the frame holds, which storing one grows.
  static size_t CallableBytes(const UiFrame &frame) { return frame.customBytes_.size(); }

  // How many interaction records the frame holds.
  static size_t InteractRecords(const UiFrame &frame) { return frame.interacts_.size(); }

  // How many decor records the frame holds.
  static size_t DecorRecords(const UiFrame &frame) { return frame.decor_.size(); }

  // How many nodes the frame holds.
  static size_t NodeCount(const UiFrame &frame) { return frame.nodes_.size(); }

  // Makes every layout LaidOutText holds look as if another text hashed to its place, as a real
  // collision is too rare to arrange, by changing the bytes or the key it would be checked against.
  static void CollidePaintedTexts(UiFrame &frame, bool bytes) {
    for (auto &[hash, entry] : frame.paintedTexts_) {
      if (bytes) {
        entry.text += "x";
      } else {
        entry.key.width += 1.0f;
      }
    }
  }

  // How many layouts LaidOutText holds.
  static size_t PaintedTexts(const UiFrame &frame) { return frame.paintedTexts_.size(); }

  // How many text layouts had to be made since Begin, and not reused.
  static uint32_t TextLayoutsBuilt(const UiFrame &frame) { return frame.textLayoutsBuilt_; }

  // How many texts Arrange laid out again since Begin.
  static uint32_t TextRelayouts(const UiFrame &frame) { return frame.textRelayouts_; }
};

} // namespace Libs::Graphics::SystemUi

#endif // TESTS_SYSTEM_UI_ELEMENT_TEST_HOOKS_H_
