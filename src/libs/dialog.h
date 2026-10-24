#ifndef EMULATOR_INCLUDE_EMULATOR_DIALOG_H_
#define EMULATOR_INCLUDE_EMULATOR_DIALOG_H_

#include "common/abi.h"
#include "common/common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Libs::Dialog {

namespace CommonDialog {

int KYTY_SYSV_ABI  CommonDialogInitialize();
bool KYTY_SYSV_ABI CommonDialogIsUsed();

} // namespace CommonDialog

namespace LoginDialog {

int KYTY_SYSV_ABI  LoginDialogInitialize();
int KYTY_SYSV_ABI  LoginDialogTerminate();
int KYTY_SYSV_ABI  LoginDialogOpen(const void* param);
int KYTY_SYSV_ABI  LoginDialogClose();
int KYTY_SYSV_ABI  LoginDialogUpdateStatus();
int KYTY_SYSV_ABI  LoginDialogGetStatus();
int KYTY_SYSV_ABI  LoginDialogGetResult(void* result);
void KYTY_SYSV_ABI LoginDialogParamInitialize(void* param);

} // namespace LoginDialog

namespace SigninDialog {

int KYTY_SYSV_ABI SigninDialogInitialize();
int KYTY_SYSV_ABI SigninDialogTerminate();
int KYTY_SYSV_ABI SigninDialogOpen(const void* param);
int KYTY_SYSV_ABI SigninDialogClose();
int KYTY_SYSV_ABI SigninDialogUpdateStatus();
int KYTY_SYSV_ABI SigninDialogGetStatus();
int KYTY_SYSV_ABI SigninDialogGetResult(void* result);

} // namespace SigninDialog

namespace SaveDataDialog {

int KYTY_SYSV_ABI SaveDataDialogInitialize();
int KYTY_SYSV_ABI SaveDataDialogGetStatus();
int KYTY_SYSV_ABI SaveDataDialogUpdateStatus();
int KYTY_SYSV_ABI SaveDataDialogGetResult(void* result);
int KYTY_SYSV_ABI SaveDataDialogOpen(const void* param);
int KYTY_SYSV_ABI SaveDataDialogClose(const void* close_param);
int KYTY_SYSV_ABI SaveDataDialogIsReadyToDisplay();
int KYTY_SYSV_ABI SaveDataDialogTerminate();
int KYTY_SYSV_ABI SaveDataDialogProgressBarInc(int target, uint32_t delta);
int KYTY_SYSV_ABI SaveDataDialogProgressBarSetValue(int target, uint32_t rate);

// What the system overlay draws for an open save data dialog.
enum class HostView { List, Message, Progress };
enum class HostButtons { Ok, YesNo, None, OkCancel };

struct HostItem {
	std::string dir_name; // empty for the new-item entry
	std::string title;
	std::string subtitle;
	std::string date;
	std::string size; // e.g. "1.95 MB"
	bool        is_new = false;
	// The save's thumbnail (PNG), shared so snapshots stay cheap; null if there is none.
	std::shared_ptr<const std::vector<uint8_t>> icon_png;
};

struct HostSnapshot {
	uint64_t              generation = 0;
	int32_t               disp_type  = 0; // 1 save, 2 load, 3 delete
	HostView              view       = HostView::List;
	std::vector<HostItem> items;          // the list, or the item a message is about
	size_t                focus      = 0; // initially focused list entry
	// SceSaveDataDialogItemStyle: 0 title, date and size, subtitle; 1 title, subtitle, date and
	// size; 2 title, date and size.
	uint32_t              item_style = 0;
	bool                  can_cancel = true;
	std::string           message;
	bool                  error_message = false;
	HostButtons           buttons       = HostButtons::Ok;
	uint32_t              progress      = 0; // percent
};

struct VisualState {
	bool     active;
	uint64_t revision;
};

bool        GetHostSnapshot(HostSnapshot* snapshot);
VisualState GetVisualState() noexcept;
// The player chose list entry `index` (an index into HostSnapshot::items).
bool HostSelect(uint64_t generation, size_t index);
// The player pressed OK/Yes (1) or No (2).
bool HostButton(uint64_t generation, int32_t button_id);
// The player backed out (Circle, or the Cancel button).
bool HostCancel(uint64_t generation);

} // namespace SaveDataDialog

namespace MsgDialog {

int KYTY_SYSV_ABI MsgDialogInitialize();
int KYTY_SYSV_ABI MsgDialogOpen(const void* param);
int KYTY_SYSV_ABI MsgDialogUpdateStatus();
int KYTY_SYSV_ABI MsgDialogGetStatus();
int KYTY_SYSV_ABI MsgDialogGetResult(void* result);
int KYTY_SYSV_ABI MsgDialogTerminate();
int KYTY_SYSV_ABI MsgDialogClose();
int KYTY_SYSV_ABI MsgDialogProgressBarInc(int target, uint32_t delta);
int KYTY_SYSV_ABI MsgDialogProgressBarSetValue(int target, uint32_t rate);
int KYTY_SYSV_ABI MsgDialogProgressBarSetMsg(int target, const char* msg);

} // namespace MsgDialog

namespace SystemDialog {

enum class Kind { Error, Signin };

struct HostSnapshot {
	Kind     kind;
	uint64_t generation;
	int32_t  error_code;
};

struct VisualState {
	bool     active;
	bool     background;
	uint64_t revision;
};

bool        GetHostSnapshot(HostSnapshot* snapshot);
VisualState GetVisualState() noexcept;
void        SetVisibilityCallback(void (*callback)());
bool        HostClose(uint64_t generation);

} // namespace SystemDialog

namespace ErrorDialog {

int KYTY_SYSV_ABI ErrorDialogInitialize();
int KYTY_SYSV_ABI ErrorDialogOpen(const void* param);
int KYTY_SYSV_ABI ErrorDialogClose();
int KYTY_SYSV_ABI ErrorDialogTerminate();
int KYTY_SYSV_ABI ErrorDialogUpdateStatus();
int KYTY_SYSV_ABI ErrorDialogGetStatus();

} // namespace ErrorDialog

} // namespace Libs::Dialog

#endif /* EMULATOR_INCLUDE_EMULATOR_DIALOG_H_ */
