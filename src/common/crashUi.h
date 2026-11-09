#ifndef KYTY_COMMON_CRASH_UI_H_
#define KYTY_COMMON_CRASH_UI_H_

// Keeps a background process from raising crash reporting, which would hold it, and its parent's
// wait for it, until a person closes a dialog or the reporter gives up.
namespace Common {

/// Turns the crash reporting of the whole process off. Windows: no fault dialog, no critical error
/// dialog, and an abort neither writes a message nor reports the fault, so an access violation or
/// an abort ends the process at once. POSIX: the core file size limit becomes zero, so a crash
/// does not stop to write a core dump. Failures are tolerated, since the process still ends.
void DisableCrashUi();

} // namespace Common

#endif /* KYTY_COMMON_CRASH_UI_H_ */
