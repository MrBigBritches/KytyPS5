#include "libs/dialog.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/saveData.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fmt/format.h>
#include <memory>
#include <mutex>

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace Libs::Dialog {

namespace {

constexpr int DIALOG_STATUS_RUNNING  = 2;
constexpr int DIALOG_STATUS_FINISHED = 3;

struct DialogState {
	int      status     = 0;
	uint64_t generation = 0;
};

std::mutex  g_dialog_mutex;
DialogState g_error;
DialogState g_signin;
uint64_t    g_dialog_generation        = 0;
uint64_t    g_dialog_revision          = 0;
int32_t     g_error_code               = 0;
void (*g_dialog_visibility_callback)() = nullptr;

void SetDialogStatus(DialogState& dialog, int status, std::unique_lock<std::mutex>& lock) {
	const bool visibility_changed =
	    (dialog.status == DIALOG_STATUS_RUNNING) != (status == DIALOG_STATUS_RUNNING);
	dialog.status = status;
	if (visibility_changed) {
		g_dialog_revision++;
		if (status == DIALOG_STATUS_RUNNING) {
			dialog.generation = ++g_dialog_generation;
		}
	}
	const auto callback = visibility_changed ? g_dialog_visibility_callback : nullptr;
	lock.unlock();
	if (callback != nullptr) {
		callback();
	}
}

} // namespace

namespace SystemDialog {

bool GetHostSnapshot(HostSnapshot* snapshot) {
	std::scoped_lock lock(g_dialog_mutex);
	if (g_signin.status == DIALOG_STATUS_RUNNING) {
		*snapshot = {Kind::Signin, g_signin.generation, 0};
		return true;
	}
	if (g_error.status == DIALOG_STATUS_RUNNING) {
		*snapshot = {Kind::Error, g_error.generation, g_error_code};
		return true;
	}
	return false;
}

VisualState GetVisualState() noexcept {
	std::scoped_lock lock(g_dialog_mutex);
	const bool       signin = g_signin.status == DIALOG_STATUS_RUNNING;
	return {g_error.status == DIALOG_STATUS_RUNNING || signin, signin, g_dialog_revision};
}

void SetVisibilityCallback(void (*callback)()) {
	std::scoped_lock lock(g_dialog_mutex);
	g_dialog_visibility_callback = callback;
}

bool HostClose(uint64_t generation) {
	std::unique_lock lock(g_dialog_mutex);
	auto&            dialog = g_signin.status == DIALOG_STATUS_RUNNING ? g_signin : g_error;
	if (dialog.status != DIALOG_STATUS_RUNNING || generation != dialog.generation) {
		return false;
	}
	SetDialogStatus(dialog, DIALOG_STATUS_FINISHED, lock);
	return true;
}

} // namespace SystemDialog

namespace CommonDialog {

LIB_NAME("CommonDialog", "CommonDialog");

int KYTY_SYSV_ABI CommonDialogInitialize() {
	PRINT_NAME();

	return OK;
}

bool KYTY_SYSV_ABI CommonDialogIsUsed() {
	PRINT_NAME();

	return false;
}

} // namespace CommonDialog

namespace LoginDialog {

LIB_NAME("LoginDialog", "LoginDialog");

constexpr int LOGIN_STATUS_NONE        = 0;
constexpr int LOGIN_STATUS_INITIALIZED = 1;
constexpr int LOGIN_STATUS_RUNNING     = 2;
constexpr int LOGIN_STATUS_FINISHED    = 3;
constexpr int LOGIN_RESULT_OK          = 0;
constexpr int LOGIN_MODE_ALL_USERS     = 0;
constexpr int LOGIN_MODE_NOT_LOGGED_IN = 1;
constexpr int LOGIN_USER_INVALID       = -1;

constexpr int LOGIN_ERROR_NOT_INITIALIZED     = static_cast<int>(0x81340001u);
constexpr int LOGIN_ERROR_ALREADY_INITIALIZED = static_cast<int>(0x81340002u);
constexpr int LOGIN_ERROR_PARAM_INVALID       = static_cast<int>(0x81340003u);
constexpr int LOGIN_ERROR_INVALID_STATE       = static_cast<int>(0x81340005u);

struct LoginDialogResult {
	int32_t result;
	int32_t selected_user;
	int32_t reserved[2];
};

struct LoginDialogParam {
	int32_t size;
	int32_t mode;
	int32_t exclude_users_from_login_list[4];
	int32_t exclude_users_from_logout_list[4];
	int32_t initial_focus;
	int32_t reserved[5];
};

static_assert(sizeof(LoginDialogResult) == 16);
static_assert(sizeof(LoginDialogParam) == 64);

static int g_login_status        = LOGIN_STATUS_NONE;
static int g_login_selected_user = LOGIN_USER_INVALID;

int KYTY_SYSV_ABI LoginDialogInitialize() {
	PRINT_NAME();

	if (g_login_status != LOGIN_STATUS_NONE) {
		return LOGIN_ERROR_ALREADY_INITIALIZED;
	}

	g_login_status        = LOGIN_STATUS_INITIALIZED;
	g_login_selected_user = Config::GetUserId();

	return OK;
}

int KYTY_SYSV_ABI LoginDialogTerminate() {
	PRINT_NAME();

	if (g_login_status == LOGIN_STATUS_NONE) {
		return LOGIN_ERROR_NOT_INITIALIZED;
	}

	g_login_status        = LOGIN_STATUS_NONE;
	g_login_selected_user = LOGIN_USER_INVALID;

	return OK;
}

int KYTY_SYSV_ABI LoginDialogOpen(const void* param) {
	PRINT_NAME();

	if (g_login_status != LOGIN_STATUS_INITIALIZED && g_login_status != LOGIN_STATUS_FINISHED) {
		return LOGIN_ERROR_INVALID_STATE;
	}
	if (param == nullptr) {
		return LOGIN_ERROR_PARAM_INVALID;
	}

	const auto* p = static_cast<const LoginDialogParam*>(param);
	if (p->size != sizeof(LoginDialogParam) ||
	    (p->mode != LOGIN_MODE_ALL_USERS && p->mode != LOGIN_MODE_NOT_LOGGED_IN)) {
		return LOGIN_ERROR_PARAM_INVALID;
	}

	g_login_selected_user =
	    p->initial_focus != LOGIN_USER_INVALID ? p->initial_focus : Config::GetUserId();

	LOGF("\t size          = %d\n"
	     "\t mode          = %d\n"
	     "\t initial_focus = %d\n"
	     "\t selected_user = %d\n",
	     p->size, p->mode, p->initial_focus, g_login_selected_user);

	g_login_status = LOGIN_STATUS_RUNNING;

	return OK;
}

int KYTY_SYSV_ABI LoginDialogClose() {
	PRINT_NAME();

	if (g_login_status == LOGIN_STATUS_NONE) {
		return LOGIN_ERROR_NOT_INITIALIZED;
	}

	g_login_status = LOGIN_STATUS_FINISHED;

	return OK;
}

int KYTY_SYSV_ABI LoginDialogUpdateStatus() {
	PRINT_NAME();

	if (g_login_status == LOGIN_STATUS_RUNNING) {
		g_login_status = LOGIN_STATUS_FINISHED;
	}

	return g_login_status;
}

int KYTY_SYSV_ABI LoginDialogGetStatus() {
	PRINT_NAME();

	return g_login_status;
}

int KYTY_SYSV_ABI LoginDialogGetResult(void* result) {
	PRINT_NAME();

	if (g_login_status == LOGIN_STATUS_NONE) {
		return LOGIN_ERROR_NOT_INITIALIZED;
	}
	if (result == nullptr) {
		return LOGIN_ERROR_PARAM_INVALID;
	}

	auto* r          = static_cast<LoginDialogResult*>(result);
	r->result        = LOGIN_RESULT_OK;
	r->selected_user = g_login_selected_user;
	r->reserved[0]   = 0;
	r->reserved[1]   = 0;

	return OK;
}

void KYTY_SYSV_ABI LoginDialogParamInitialize(void* param) {
	PRINT_NAME();

	if (param == nullptr) {
		return;
	}

	auto* p = static_cast<LoginDialogParam*>(param);
	std::memset(p, 0, sizeof(LoginDialogParam));
	p->size = sizeof(LoginDialogParam);
	for (int i = 0; i < 4; i++) {
		p->exclude_users_from_login_list[i]  = LOGIN_USER_INVALID;
		p->exclude_users_from_logout_list[i] = LOGIN_USER_INVALID;
	}
	p->initial_focus = LOGIN_USER_INVALID;
}

} // namespace LoginDialog

namespace SigninDialog {

LIB_NAME("SigninDialog", "SigninDialog");

constexpr int SIGNIN_STATUS_NONE          = 0;
constexpr int SIGNIN_STATUS_INITIALIZED   = 1;
constexpr int SIGNIN_STATUS_RUNNING       = 2;
constexpr int SIGNIN_STATUS_FINISHED      = 3;
constexpr int SIGNIN_RESULT_USER_CANCELED = 1;

constexpr int SIGNIN_ERROR_NOT_INITIALIZED     = static_cast<int>(0x81350001u);
constexpr int SIGNIN_ERROR_ALREADY_INITIALIZED = static_cast<int>(0x81350002u);
constexpr int SIGNIN_ERROR_PARAM_INVALID       = static_cast<int>(0x81350003u);
constexpr int SIGNIN_ERROR_INVALID_STATE       = static_cast<int>(0x81350005u);
constexpr int SIGNIN_ERROR_INVALID_USER_ID     = static_cast<int>(0x81350007u);

struct SigninDialogResult {
	int32_t result;
	int32_t reserved[3];
};

struct SigninDialogParam {
	int32_t size;
	int32_t user_id;
	int32_t reserved[2];
};

static_assert(sizeof(SigninDialogResult) == 16);
static_assert(sizeof(SigninDialogParam) == 16);

int KYTY_SYSV_ABI SigninDialogInitialize() {
	PRINT_NAME();
	std::scoped_lock lock(g_dialog_mutex);
	if (g_signin.status != SIGNIN_STATUS_NONE) {
		return SIGNIN_ERROR_ALREADY_INITIALIZED;
	}
	g_signin.status = SIGNIN_STATUS_INITIALIZED;
	return OK;
}

int KYTY_SYSV_ABI SigninDialogTerminate() {
	PRINT_NAME();
	std::unique_lock lock(g_dialog_mutex);
	if (g_signin.status == SIGNIN_STATUS_NONE) {
		return SIGNIN_ERROR_NOT_INITIALIZED;
	}
	SetDialogStatus(g_signin, SIGNIN_STATUS_NONE, lock);
	return OK;
}

int KYTY_SYSV_ABI SigninDialogOpen(const void* param) {
	PRINT_NAME();
	std::unique_lock lock(g_dialog_mutex);
	if (g_signin.status == SIGNIN_STATUS_NONE) {
		return SIGNIN_ERROR_NOT_INITIALIZED;
	}
	if (g_signin.status != SIGNIN_STATUS_INITIALIZED && g_signin.status != SIGNIN_STATUS_FINISHED) {
		return SIGNIN_ERROR_INVALID_STATE;
	}
	if (param == nullptr) {
		return SIGNIN_ERROR_PARAM_INVALID;
	}
	const auto* p = static_cast<const SigninDialogParam*>(param);
	if (p->size != sizeof(SigninDialogParam) || p->reserved[0] != 0 || p->reserved[1] != 0) {
		return SIGNIN_ERROR_PARAM_INVALID;
	}
	if (p->user_id != Config::GetUserId()) {
		return SIGNIN_ERROR_INVALID_USER_ID;
	}
	SetDialogStatus(g_signin, SIGNIN_STATUS_RUNNING, lock);
	return OK;
}

int KYTY_SYSV_ABI SigninDialogClose() {
	PRINT_NAME();
	std::unique_lock lock(g_dialog_mutex);
	if (g_signin.status == SIGNIN_STATUS_NONE) {
		return SIGNIN_ERROR_NOT_INITIALIZED;
	}
	SetDialogStatus(g_signin, SIGNIN_STATUS_FINISHED, lock);
	return OK;
}

int KYTY_SYSV_ABI SigninDialogUpdateStatus() {
	PRINT_NAME();
	std::scoped_lock lock(g_dialog_mutex);
	return g_signin.status;
}

int KYTY_SYSV_ABI SigninDialogGetStatus() {
	PRINT_NAME();
	std::scoped_lock lock(g_dialog_mutex);
	return g_signin.status;
}

int KYTY_SYSV_ABI SigninDialogGetResult(void* result) {
	PRINT_NAME();
	std::scoped_lock lock(g_dialog_mutex);
	if (g_signin.status == SIGNIN_STATUS_NONE) {
		return SIGNIN_ERROR_NOT_INITIALIZED;
	}
	if (g_signin.status != SIGNIN_STATUS_FINISHED) {
		return SIGNIN_ERROR_INVALID_STATE;
	}
	if (result == nullptr) {
		return SIGNIN_ERROR_PARAM_INVALID;
	}
	auto* r        = static_cast<SigninDialogResult*>(result);
	r->result      = SIGNIN_RESULT_USER_CANCELED;
	r->reserved[0] = 0;
	r->reserved[1] = 0;
	r->reserved[2] = 0;
	return OK;
}

} // namespace SigninDialog

namespace SaveDataDialog {

LIB_NAME("SaveDataDialog", "SaveDataDialog");

// The save data dialog is the system UI games use to pick a save slot (list mode) and to show
// save messages and progress. The system overlay draws it from GetHostSnapshot and reports the
// player's choice through HostSelect, HostButton and HostCancel; until then the dialog stays
// RUNNING. Layouts follow SceSaveDataDialog*; result and button semantics are those of
// firmware 4.50 and later.

constexpr int SAVE_STATUS_NONE        = 0;
constexpr int SAVE_STATUS_INITIALIZED = 1;
constexpr int SAVE_STATUS_RUNNING     = 2;
constexpr int SAVE_STATUS_FINISHED    = 3;

constexpr int32_t RESULT_OK            = 0;
constexpr int32_t RESULT_USER_CANCELED = 1;

constexpr int32_t BUTTON_INVALID = 0;
constexpr int32_t BUTTON_OK      = 1; // also YES
constexpr int32_t BUTTON_NO      = 2;

constexpr int32_t MODE_LIST         = 1;
constexpr int32_t MODE_USER_MSG     = 2;
constexpr int32_t MODE_SYSTEM_MSG   = 3;
constexpr int32_t MODE_ERROR_CODE   = 4;
constexpr int32_t MODE_PROGRESS_BAR = 5;

constexpr int32_t TYPE_SAVE   = 1;
constexpr int32_t TYPE_LOAD   = 2;
constexpr int32_t TYPE_DELETE = 3;

struct SaveDataDirName {
	char data[32];
};

struct SaveDataTitleId {
	char data[10];
	char padding[6];
};

struct SaveDataDialogNewItem {
	const char* title;
	void*       icon_buf;
	size_t      icon_size;
	uint8_t     reserved[32];
};

struct SaveDataDialogItems {
	int32_t                      user_id;
	int32_t                      pad0;
	const SaveDataTitleId*       title_id;
	const SaveDataDirName*       dir_names;
	uint32_t                     dir_names_num;
	int32_t                      pad1;
	const SaveDataDialogNewItem* new_item;
	uint32_t                     focus_pos;
	int32_t                      pad2;
	const SaveDataDirName*       focus_pos_dir_name;
	uint32_t                     item_style;
	uint8_t                      reserved[32];
};

struct SaveDataDialogUserMessageParam {
	uint32_t    button_type; // 0 OK, 1 YES/NO, 2 none, 3 OK/cancel
	uint32_t    msg_type;    // 0 normal, 1 error
	const char* msg;
	uint8_t     reserved[32];
};

struct SaveDataDialogSystemMessageParam {
	uint32_t msg_type;
	int32_t  pad0;
	uint64_t value;
	uint8_t  reserved[32];
};

struct SaveDataDialogErrorCodeParam {
	uint32_t error_code;
	uint8_t  reserved[32];
};

struct SaveDataDialogProgressBarParam {
	uint32_t    bar_type;
	int32_t     pad0;
	const char* msg;
	uint32_t    sys_msg_type; // 0 none, 1 progress, 2 restore
	uint8_t     reserved[28];
};

struct SaveDataDialogOptionParam {
	uint32_t back; // 0 enabled, 1 disabled
	uint8_t  reserved[32];
};

struct SaveDataDialogParam {
	uint8_t                                 base_param[48];
	int32_t                                 size;
	int32_t                                 mode;
	int32_t                                 disp_type;
	uint32_t                                pad0;
	void*                                   anim_param;
	const SaveDataDialogItems*              items;
	const SaveDataDialogUserMessageParam*   user_msg_param;
	const SaveDataDialogSystemMessageParam* sys_msg_param;
	const SaveDataDialogErrorCodeParam*     error_code_param;
	const SaveDataDialogProgressBarParam*   prog_bar_param;
	void*                                   user_data;
	const SaveDataDialogOptionParam*        option_param;
	void*                                   wizard_param;
	uint8_t                                 reserved[16];
};

struct SaveDataParam {
	char     title[128];
	char     sub_title[128];
	char     detail[1024];
	uint32_t user_param;
	int32_t  pad;
	int64_t  mtime;
	uint8_t  reserved[32];
};

struct SaveDataDialogResult {
	int32_t          mode;
	int32_t          result;
	int32_t          button_id;
	uint32_t         pad0;
	SaveDataDirName* dir_name;
	SaveDataParam*   param;
	void*            user_data;
	uint8_t          reserved[32];
};

static_assert(sizeof(SaveDataDialogItems) == 96);
static_assert(sizeof(SaveDataDialogParam) == 152);
static_assert(sizeof(SaveDataDialogResult) == 72);
static_assert(sizeof(SaveDataParam) == SaveData::SAVE_DATA_PARAM_SIZE);

struct SaveState {
	int32_t       mode      = 0;
	void*         user_data = nullptr;
	HostSnapshot  host;
	// Parameters of host.items, for the result of a list selection.
	std::vector<SaveDataParam> params;
	// The result once the player chose.
	int32_t       result    = RESULT_OK;
	int32_t       button_id = BUTTON_INVALID;
	std::string   dir_name;
	bool          has_param = false;
	SaveDataParam param {};
	// A RUNNING update before FINISHED, which some games require.
	bool          running_polled = false;
};

static DialogState g_save;
static SaveState   g_save_state;

static std::string FixedString(const char* data, size_t size) {
	return {data, static_cast<size_t>(std::find(data, data + size, '\0') - data)};
}

static std::string FormatDate(int64_t mtime) {
	if (mtime <= 0) {
		return {};
	}
	const auto time = static_cast<std::time_t>(mtime);
	std::tm    local {};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (localtime_s(&local, &time) != 0) {
		return {};
	}
#else
	if (localtime_r(&time, &local) == nullptr) {
		return {};
	}
#endif
	char text[32];
	return std::strftime(text, sizeof(text), "%Y/%m/%d %H:%M", &local) != 0 ? std::string(text)
	                                                                       : std::string();
}

static std::string FormatSize(uint64_t bytes) {
	char text[32];
	if (bytes >= 1024ull * 1024 * 1024) {
		std::snprintf(text, sizeof(text), "%.2f GB",
		              static_cast<double>(bytes) / (1024.0 * 1024 * 1024));
	} else if (bytes >= 1024ull * 1024) {
		std::snprintf(text, sizeof(text), "%.2f MB", static_cast<double>(bytes) / (1024.0 * 1024));
	} else if (bytes >= 1024) {
		std::snprintf(text, sizeof(text), "%.2f KB", static_cast<double>(bytes) / 1024.0);
	} else {
		std::snprintf(text, sizeof(text), "%u B", static_cast<uint32_t>(bytes));
	}
	return text;
}

// Existing save directories among the dialog's items, with their parameters.
static void CollectItems(const SaveDataDialogItems& items, int32_t disp_type, SaveState* state) {
	auto& host = state->host;
	const char* title_id = items.title_id != nullptr && items.title_id->data[0] != '\0'
	                           ? items.title_id->data
	                           : nullptr;
	std::string title_id_storage;
	if (title_id != nullptr) {
		title_id_storage = FixedString(items.title_id->data, sizeof(items.title_id->data));
		title_id         = title_id_storage.c_str();
	}
	// The new item's icon, as the game gives it. Saves without an icon show an empty thumbnail.
	std::shared_ptr<const std::vector<uint8_t>> new_item_icon;
	if (disp_type == TYPE_SAVE && items.new_item != nullptr &&
	    items.new_item->icon_buf != nullptr && items.new_item->icon_size != 0 &&
	    items.new_item->icon_size <= 4 * 1024 * 1024) {
		const auto* bytes = static_cast<const uint8_t*>(items.new_item->icon_buf);
		new_item_icon =
		    std::make_shared<std::vector<uint8_t>>(bytes, bytes + items.new_item->icon_size);
	}
	if (disp_type == TYPE_SAVE && items.new_item != nullptr) {
		HostItem item;
		item.is_new = true;
		item.title  = items.new_item->title != nullptr ? std::string(items.new_item->title)
		                                               : std::string("New Saved Data");
		item.icon_png = new_item_icon;
		host.items.push_back(std::move(item));
		state->params.emplace_back();
	}
	for (uint32_t i = 0; items.dir_names != nullptr && i < items.dir_names_num; i++) {
		const auto name = FixedString(items.dir_names[i].data, sizeof(items.dir_names[i].data));
		SaveDataParam param {};
		if (name.empty() ||
		    !SaveData::ReadSaveDataParam(items.user_id, title_id, name.c_str(), &param,
		                                 sizeof(param))) {
			continue; // no such save yet: the list shows only existing data
		}
		HostItem item;
		item.dir_name = name;
		item.title    = FixedString(param.title, sizeof(param.title));
		item.subtitle = FixedString(param.sub_title, sizeof(param.sub_title));
		item.date     = FormatDate(param.mtime);
		item.size =
		    FormatSize(SaveData::GetSaveDataSize(items.user_id, title_id, name.c_str()));
		if (item.title.empty()) {
			item.title = name;
		}
		auto icon = std::make_shared<std::vector<uint8_t>>();
		if (SaveData::ReadSaveDataIcon(items.user_id, title_id, name.c_str(), icon.get())) {
			item.icon_png = std::move(icon);
		}
		host.items.push_back(std::move(item));
		state->params.push_back(param);
	}
	// Saves newest first, whatever order the game listed them in; the new item stays on top.
	const size_t first_data = !host.items.empty() && host.items.front().is_new ? 1 : 0;
	std::vector<size_t> order(host.items.size() - first_data);
	for (size_t i = 0; i < order.size(); i++) {
		order[i] = first_data + i;
	}
	std::stable_sort(order.begin(), order.end(), [state](size_t a, size_t b) {
		return state->params[a].mtime > state->params[b].mtime;
	});
	std::vector<HostItem>      sorted_items(host.items.begin(), host.items.begin() + first_data);
	std::vector<SaveDataParam> sorted_params(state->params.begin(),
	                                         state->params.begin() + first_data);
	for (const size_t i: order) {
		sorted_items.push_back(std::move(host.items[i]));
		sorted_params.push_back(state->params[i]);
	}
	host.items    = std::move(sorted_items);
	state->params = std::move(sorted_params);
}

// The list entry focused when the dialog opens (SceSaveDataDialogFocusPos).
static size_t InitialFocus(const SaveDataDialogItems& items, const SaveState& state) {
	const auto& list  = state.host.items;
	const bool  has_new = !list.empty() && list.front().is_new;
	const size_t first_data = has_new ? 1 : 0;
	if (list.empty()) {
		return 0;
	}
	switch (items.focus_pos) {
		case 1: return list.size() - 1;                                    // list tail
		case 2: return std::min(first_data, list.size() - 1);              // data head
		case 3: return list.size() - 1;                                    // data tail
		case 4:                                                            // latest data
		case 5: {                                                          // oldest data
			size_t best = first_data;
			for (size_t i = first_data; i < list.size(); i++) {
				const auto mtime = state.params[i].mtime;
				const auto best_mtime = state.params[best].mtime;
				if (items.focus_pos == 4 ? mtime > best_mtime : mtime < best_mtime) {
					best = i;
				}
			}
			return std::min(best, list.size() - 1);
		}
		case 6:                                                            // by directory name
			if (items.focus_pos_dir_name != nullptr) {
				const auto name = FixedString(items.focus_pos_dir_name->data,
				                              sizeof(items.focus_pos_dir_name->data));
				for (size_t i = 0; i < list.size(); i++) {
					if (!list[i].is_new && list[i].dir_name == name) {
						return i;
					}
				}
			}
			return 0;
		default: return 0;                                                  // list head
	}
}

// SceSaveDataDialogSystemMessageType: the text for saving, and for loading and deleting where
// it differs, and the buttons. "{size}" stands for the space the message's value asks for.
struct SystemMessage {
	uint32_t    type;
	HostButtons buttons;
	const char* save;
	const char* load = nullptr;
	const char* del  = nullptr;
};

constexpr uint32_t SYSTEM_MESSAGE_PROGRESS = 5; // stays up until the game closes it

constexpr SystemMessage SYSTEM_MESSAGES[] = {
    {1, HostButtons::Ok, "There is no saved data."},
    {2, HostButtons::YesNo, "Do you want to save?", "Do you want to load this saved data?",
     "Do you want to delete this saved data?"},
    {3, HostButtons::YesNo, "Do you want to overwrite this saved data?"},
    {4, HostButtons::Ok, "There is not enough free space. {size} of free space is required."},
    {SYSTEM_MESSAGE_PROGRESS, HostButtons::None, "Saving...", "Loading...", "Deleting..."},
    {6, HostButtons::Ok, "The saved data is corrupted."},
    {7, HostButtons::Ok, "Saving complete.", "Loading complete.", "Deletion complete."},
    {8, HostButtons::Ok, "There is not enough free space. {size} of free space is required."},
    {10, HostButtons::Ok, "The saved data is corrupted and will be deleted."},
    {11, HostButtons::Ok,
     "The saved data is corrupted. It will be deleted and new saved data will be created."},
    {13, HostButtons::Ok, "The saved data is corrupted. The backup will be restored."},
    {14, HostButtons::Ok, "No more saved data can be created."},
};

static const SystemMessage* FindSystemMessage(uint32_t type) {
	for (const auto& message: SYSTEM_MESSAGES) {
		if (message.type == type) {
			return &message;
		}
	}
	return nullptr;
}

static std::string SystemMessageText(const SystemMessage& message, int32_t disp_type) {
	if (disp_type == TYPE_LOAD && message.load != nullptr) {
		return message.load;
	}
	if (disp_type == TYPE_DELETE && message.del != nullptr) {
		return message.del;
	}
	return message.save;
}

static void SetupSystemMessage(const SaveDataDialogSystemMessageParam& sys, int32_t disp_type,
                               bool back_enabled, HostSnapshot* host) {
	const auto* message = FindSystemMessage(sys.msg_type);
	if (message == nullptr) {
		host->buttons = HostButtons::Ok;
		host->message = fmt::format("Saved data message {}.", sys.msg_type);
		return;
	}
	host->buttons = message->buttons;
	host->message = SystemMessageText(*message, disp_type);
	if (const auto at = host->message.find("{size}"); at != std::string::npos) {
		const double mib = static_cast<double>(sys.value) * 32768.0 / (1024.0 * 1024.0);
		host->message.replace(at, 6, fmt::format("{:.1f} MB", mib));
	}
	if (message->type == SYSTEM_MESSAGE_PROGRESS) {
		host->can_cancel = back_enabled; // Back may cancel
	}
}

static bool FinishLocked(uint64_t generation, int32_t result, int32_t button_id,
                  std::unique_lock<std::mutex>& lock) {
	if (g_save.status != DIALOG_STATUS_RUNNING || g_save.generation != generation) {
		return false;
	}
	g_save_state.result    = result;
	g_save_state.button_id = button_id;
	SetDialogStatus(g_save, DIALOG_STATUS_FINISHED, lock);
	return true;
}

bool GetHostSnapshot(HostSnapshot* snapshot) {
	std::scoped_lock lock(g_dialog_mutex);
	if (g_save.status != DIALOG_STATUS_RUNNING) {
		return false;
	}
	*snapshot            = g_save_state.host;
	snapshot->generation = g_save.generation;
	return true;
}

VisualState GetVisualState() noexcept {
	std::scoped_lock lock(g_dialog_mutex);
	return {g_save.status == DIALOG_STATUS_RUNNING, g_dialog_revision};
}

// The list item last chosen, so a list the game reopens after a declined question (No to
// overwriting) starts on it again. Cleared once the operation goes ahead or the list is left.
static std::string g_reselect_dir;
static bool        g_reselect_new  = false;
static int32_t     g_reselect_type = -1;

bool HostSelect(uint64_t generation, size_t index) {
	std::unique_lock lock(g_dialog_mutex);
	if (g_save.status != DIALOG_STATUS_RUNNING || g_save.generation != generation ||
	    g_save_state.host.view != HostView::List || index >= g_save_state.host.items.size()) {
		return false;
	}
	const auto& item       = g_save_state.host.items[index];
	g_save_state.dir_name  = item.dir_name;
	g_save_state.has_param = !item.is_new;
	g_save_state.param     = g_save_state.params[index];
	LOGF("SaveDataDialog: selected %s\n", item.is_new ? "<new item>" : item.dir_name.c_str());
	g_reselect_dir  = item.dir_name;
	g_reselect_new  = item.is_new;
	g_reselect_type = g_save_state.host.disp_type;
	return FinishLocked(generation, RESULT_OK, BUTTON_INVALID, lock);
}

bool HostButton(uint64_t generation, int32_t button_id) {
	if (button_id != BUTTON_OK && button_id != BUTTON_NO) {
		return false;
	}
	std::unique_lock lock(g_dialog_mutex);
	if (button_id == BUTTON_OK && g_save.generation == generation) {
		g_reselect_type = -1; // going ahead
	}
	return FinishLocked(generation, RESULT_OK, button_id, lock);
}

bool HostCancel(uint64_t generation) {
	std::unique_lock lock(g_dialog_mutex);
	if (g_save_state.host.view == HostView::List) {
		g_save_state.dir_name.clear();
		g_save_state.has_param = false;
		if (g_save.generation == generation) {
			g_reselect_type = -1; // left the list
		}
	}
	return FinishLocked(generation, RESULT_USER_CANCELED, BUTTON_INVALID, lock);
}

int KYTY_SYSV_ABI SaveDataDialogInitialize() {
	PRINT_NAME();

	std::unique_lock lock(g_dialog_mutex);
	g_save_state = {};
	SetDialogStatus(g_save, SAVE_STATUS_INITIALIZED, lock);

	return OK;
}

int KYTY_SYSV_ABI SaveDataDialogGetStatus() {
	PRINT_NAME();

	std::scoped_lock lock(g_dialog_mutex);
	return g_save.status;
}

int KYTY_SYSV_ABI SaveDataDialogUpdateStatus() {
	PRINT_NAME();

	std::scoped_lock lock(g_dialog_mutex);
	// Some games require a RUNNING update before FINISHED: report a choice made before the
	// first update one update later.
	if (g_save.status == SAVE_STATUS_FINISHED && !g_save_state.running_polled) {
		g_save_state.running_polled = true;
		return SAVE_STATUS_RUNNING;
	}
	g_save_state.running_polled = true;
	return g_save.status;
}

int KYTY_SYSV_ABI SaveDataDialogGetResult(void* result) {
	PRINT_NAME();

	std::scoped_lock lock(g_dialog_mutex);
	if (result != nullptr) {
		auto* r      = static_cast<SaveDataDialogResult*>(result);
		r->mode      = g_save_state.mode;
		r->result    = g_save_state.result;
		r->button_id = g_save_state.button_id;
		r->user_data = g_save_state.user_data;
		if (r->dir_name != nullptr) {
			std::snprintf(r->dir_name->data, sizeof(r->dir_name->data), "%s",
			              g_save_state.dir_name.c_str());
		}
		if (r->param != nullptr && g_save_state.has_param) {
			*r->param = g_save_state.param;
		}
		LOGF("\t mode      = %d\n"
		     "\t result    = %d\n"
		     "\t button_id = %d\n"
		     "\t dir_name  = %s\n",
		     r->mode, r->result, r->button_id, g_save_state.dir_name.c_str());
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataDialogOpen(const void* param) {
	PRINT_NAME();

	const auto* p = static_cast<const SaveDataDialogParam*>(param);
	if (p == nullptr) {
		return OK;
	}
	LOGF("\t size      = %d\n"
	     "\t mode      = %d\n"
	     "\t disp_type = %d\n"
	     "\t items     = 0x%016" PRIx64 "\n",
	     p->size, p->mode, p->disp_type, reinterpret_cast<uint64_t>(p->items));

	SaveState state;
	state.mode            = p->mode;
	state.user_data       = p->user_data;
	auto& host            = state.host;
	host.disp_type        = p->disp_type;
	const bool back       = p->option_param == nullptr || p->option_param->back == 0;
	host.can_cancel       = back;
	if (p->items != nullptr) {
		CollectItems(*p->items, p->disp_type, &state);
		host.item_style = p->items->item_style <= 2 ? p->items->item_style : 0;
		for (const auto& item: host.items) {
			LOGF("\t item: %s %s\n", item.is_new ? "<new>" : item.dir_name.c_str(),
			     item.title.c_str());
		}
	}
	switch (p->mode) {
		case MODE_LIST:
			host.view = HostView::List;

			if (p->items != nullptr) {
				host.focus = InitialFocus(*p->items, state);
				std::scoped_lock reselect_lock(g_dialog_mutex);
				if (g_reselect_type == p->disp_type) {
					for (size_t i = 0; i < host.items.size(); i++) {
						const auto& item = host.items[i];
						if (g_reselect_new ? item.is_new
						                   : !item.is_new && item.dir_name == g_reselect_dir) {
							host.focus = i;
							break;
						}
					}
				}
				g_reselect_type = -1;
			}
			if (host.items.empty()) {
				host.view    = HostView::Message;
				host.message = "There is no saved data.";
			}
			break;
		case MODE_USER_MSG:
			host.view = HostView::Message;
			if (p->user_msg_param != nullptr) {
				const auto& user   = *p->user_msg_param;
				host.message       = user.msg != nullptr ? std::string(user.msg) : std::string();
				host.error_message = user.msg_type == 1;
				host.buttons       = user.button_type == 1   ? HostButtons::YesNo
				                     : user.button_type == 2 ? HostButtons::None
				                     : user.button_type == 3 ? HostButtons::OkCancel
				                                             : HostButtons::Ok;
				LOGF("\t user message: buttons = %u, type = %u, %s\n", user.button_type,
				     user.msg_type, host.message.c_str());
			}
			break;
		case MODE_SYSTEM_MSG:
			host.view = HostView::Message;
			if (p->sys_msg_param != nullptr) {
				SetupSystemMessage(*p->sys_msg_param, p->disp_type, back, &host);
				LOGF("\t system message: type = %u, value = %" PRIu64 "\n",
				     p->sys_msg_param->msg_type, p->sys_msg_param->value);
			}
			break;
		case MODE_ERROR_CODE: {
			host.view          = HostView::Message;
			host.error_message = true;
			const uint32_t code =
			    p->error_code_param != nullptr ? p->error_code_param->error_code : 0;
			char text[96];
			if (code == 0x809F0008u) {
				std::snprintf(text, sizeof(text), "There is no saved data.");
			} else if (code == 0x809F000Fu) {
				std::snprintf(text, sizeof(text), "The saved data is corrupted.");
			} else {
				std::snprintf(text, sizeof(text), "An error has occurred. (0x%08X)", code);
			}
			host.message = text;
			LOGF("\t error code = 0x%08X\n", code);
			break;
		}
		case MODE_PROGRESS_BAR:
			{
				std::scoped_lock reselect_lock(g_dialog_mutex);
				g_reselect_type = -1; // the save or load is under way
			}
			host.view       = HostView::Progress;
			host.buttons    = HostButtons::None;
			host.can_cancel = false;
			if (p->prog_bar_param != nullptr) {
				const auto& bar = *p->prog_bar_param;
				if (bar.msg != nullptr) {
					host.message = bar.msg;
				} else if (bar.sys_msg_type == 1) {
					host.message = SystemMessageText(*FindSystemMessage(SYSTEM_MESSAGE_PROGRESS),
					                                 p->disp_type);
				} else if (bar.sys_msg_type == 2) {
					host.message = "Restoring saved data...";
				}
			}
			break;
		default:
			LOGF("SaveDataDialog: unsupported mode %d\n", p->mode);
			host.view    = HostView::Message;
			host.message = "Unsupported saved data dialog.";
			break;
	}
	// Messages and progress are about one save: the result names it, as on the console.
	if (host.view != HostView::List) {
		for (const auto& item: host.items) {
			if (!item.is_new) {
				state.dir_name = item.dir_name;
				break;
			}
		}
	}

	std::unique_lock lock(g_dialog_mutex);
	g_save_state = std::move(state);
	SetDialogStatus(g_save, SAVE_STATUS_RUNNING, lock);

	return OK;
}

int KYTY_SYSV_ABI SaveDataDialogClose(const void* close_param) {
	PRINT_NAME();

	LOGF("\t close_param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(close_param));

	std::unique_lock lock(g_dialog_mutex);
	if (g_save.status != SAVE_STATUS_RUNNING) {
		return OK;
	}
	g_save_state.result    = RESULT_OK;
	g_save_state.button_id = BUTTON_INVALID;
	SetDialogStatus(g_save, SAVE_STATUS_FINISHED, lock);

	return OK;
}

int KYTY_SYSV_ABI SaveDataDialogIsReadyToDisplay() {
	PRINT_NAME();

	return 1;
}

int KYTY_SYSV_ABI SaveDataDialogTerminate() {
	PRINT_NAME();

	std::unique_lock lock(g_dialog_mutex);
	g_save_state = {};
	SetDialogStatus(g_save, SAVE_STATUS_NONE, lock);

	return OK;
}

static int UpdateProgress(int target, uint32_t value, bool add) {
	std::scoped_lock lock(g_dialog_mutex);
	if (g_save.status != SAVE_STATUS_RUNNING || g_save_state.host.view != HostView::Progress ||
	    target != 0) {
		return OK; // nothing to update
	}
	auto& progress = g_save_state.host.progress;
	progress       = std::min<uint32_t>(add ? progress + value : value, 100);
	g_dialog_revision++;
	return OK;
}

int KYTY_SYSV_ABI SaveDataDialogProgressBarInc(int target, uint32_t delta) {
	PRINT_NAME();

	LOGF("\t target = %d\n"
	     "\t delta  = %u\n",
	     target, delta);

	return UpdateProgress(target, delta, true);
}

int KYTY_SYSV_ABI SaveDataDialogProgressBarSetValue(int target, uint32_t rate) {
	PRINT_NAME();

	LOGF("\t target = %d\n"
	     "\t rate   = %u\n",
	     target, rate);

	return UpdateProgress(target, rate, false);
}

} // namespace SaveDataDialog

namespace MsgDialog {

LIB_NAME("MsgDialog.native", "MsgDialog");

constexpr int STATUS_NONE        = 0;
constexpr int STATUS_INITIALIZED = 1;
constexpr int STATUS_FINISHED    = 3;
constexpr int RESULT_OK          = 0;
constexpr int BUTTON_ID_OK       = 1;

struct MsgDialogParam {
	uint8_t  base_param[48];
	uint64_t size;
	int32_t  mode;
	uint32_t pad0;
	void*    user_msg_param;
	void*    prog_bar_param;
	void*    sys_msg_param;
	int32_t  user_id;
	uint8_t  reserved[40];
	uint32_t pad1;
};

struct MsgDialogResult {
	int32_t mode;
	int32_t result;
	int32_t button_id;
	uint8_t reserved[32];
};

static int g_status = STATUS_NONE;
static int g_mode   = 0;

int KYTY_SYSV_ABI MsgDialogInitialize() {
	PRINT_NAME();

	g_status = STATUS_INITIALIZED;
	g_mode   = 0;

	return OK;
}

int KYTY_SYSV_ABI MsgDialogOpen(const void* param) {
	PRINT_NAME();

	const auto* p = static_cast<const MsgDialogParam*>(param);
	if (p != nullptr) {
		g_mode = p->mode;
		LOGF("\t size           = 0x%016" PRIx64 "\n"
		     "\t mode           = %d\n"
		     "\t user_msg_param = 0x%016" PRIx64 "\n"
		     "\t prog_bar_param = 0x%016" PRIx64 "\n"
		     "\t sys_msg_param  = 0x%016" PRIx64 "\n"
		     "\t user_id        = %d\n",
		     p->size, p->mode, reinterpret_cast<uint64_t>(p->user_msg_param),
		     reinterpret_cast<uint64_t>(p->prog_bar_param),
		     reinterpret_cast<uint64_t>(p->sys_msg_param), p->user_id);
	}

	g_status = STATUS_FINISHED;

	return OK;
}

int KYTY_SYSV_ABI MsgDialogUpdateStatus() {
	PRINT_NAME();

	return g_status;
}

int KYTY_SYSV_ABI MsgDialogGetStatus() {
	PRINT_NAME();

	return g_status;
}

int KYTY_SYSV_ABI MsgDialogGetResult(void* result) {
	PRINT_NAME();

	if (result != nullptr) {
		auto* r      = static_cast<MsgDialogResult*>(result);
		r->mode      = g_mode;
		r->result    = RESULT_OK;
		r->button_id = BUTTON_ID_OK;
	}

	return OK;
}

int KYTY_SYSV_ABI MsgDialogTerminate() {
	PRINT_NAME();

	g_status = STATUS_NONE;
	g_mode   = 0;

	return OK;
}

int KYTY_SYSV_ABI MsgDialogClose() {
	PRINT_NAME();

	g_status = STATUS_FINISHED;

	return OK;
}

int KYTY_SYSV_ABI MsgDialogProgressBarInc(int target, uint32_t delta) {
	PRINT_NAME();

	LOGF("\t target = %d\n"
	     "\t delta  = %u\n",
	     target, delta);

	return OK;
}

int KYTY_SYSV_ABI MsgDialogProgressBarSetValue(int target, uint32_t rate) {
	PRINT_NAME();

	LOGF("\t target = %d\n"
	     "\t rate   = %u\n",
	     target, rate);

	return OK;
}

int KYTY_SYSV_ABI MsgDialogProgressBarSetMsg(int target, const char* msg) {
	PRINT_NAME();

	LOGF("\t target = %d\n"
	     "\t msg    = 0x%016" PRIx64 "\n",
	     target, reinterpret_cast<uint64_t>(msg));

	return OK;
}

} // namespace MsgDialog

namespace ErrorDialog {

LIB_NAME("ErrorDialog", "ErrorDialog");

constexpr int STATUS_NONE        = 0;
constexpr int STATUS_INITIALIZED = 1;
constexpr int STATUS_RUNNING     = 2;
constexpr int STATUS_FINISHED    = 3;

constexpr int ERROR_NOT_INITIALIZED     = static_cast<int>(0x80ed0001);
constexpr int ERROR_ALREADY_INITIALIZED = static_cast<int>(0x80ed0002);
constexpr int ERROR_PARAM_INVALID       = static_cast<int>(0x80ed0003);
constexpr int ERROR_INVALID_STATE       = static_cast<int>(0x80ed0005);

struct ErrorDialogParam {
	int32_t size;
	int32_t error_code;
	int32_t user_id;
	int32_t reserved;
};

static_assert(sizeof(ErrorDialogParam) == 16);

int KYTY_SYSV_ABI ErrorDialogInitialize() {
	PRINT_NAME();

	std::scoped_lock lock(g_dialog_mutex);
	if (g_error.status != STATUS_NONE) {
		return ERROR_ALREADY_INITIALIZED;
	}

	g_error.status = STATUS_INITIALIZED;

	return OK;
}

int KYTY_SYSV_ABI ErrorDialogOpen(const void* param) {
	PRINT_NAME();

	std::unique_lock lock(g_dialog_mutex);
	if (g_error.status != STATUS_INITIALIZED && g_error.status != STATUS_FINISHED) {
		return ERROR_INVALID_STATE;
	}
	if (param == nullptr) {
		return ERROR_PARAM_INVALID;
	}

	const auto* p = static_cast<const ErrorDialogParam*>(param);
	if (p->size != sizeof(ErrorDialogParam)) {
		return ERROR_PARAM_INVALID;
	}

	LOGF("\t size       = %d\n"
	     "\t error_code = 0x%08" PRIx32 "\n"
	     "\t user_id    = %d\n",
	     p->size, static_cast<uint32_t>(p->error_code), p->user_id);

	g_error_code = p->error_code;
	SetDialogStatus(g_error, STATUS_RUNNING, lock);

	return OK;
}

int KYTY_SYSV_ABI ErrorDialogClose() {
	PRINT_NAME();

	std::unique_lock lock(g_dialog_mutex);
	if (g_error.status != STATUS_RUNNING) {
		return ERROR_INVALID_STATE;
	}

	SetDialogStatus(g_error, STATUS_FINISHED, lock);

	return OK;
}

int KYTY_SYSV_ABI ErrorDialogTerminate() {
	PRINT_NAME();

	std::unique_lock lock(g_dialog_mutex);
	if (g_error.status == STATUS_NONE) {
		return ERROR_NOT_INITIALIZED;
	}

	SetDialogStatus(g_error, STATUS_NONE, lock);

	return OK;
}

int KYTY_SYSV_ABI ErrorDialogUpdateStatus() {
	PRINT_NAME();

	std::scoped_lock lock(g_dialog_mutex);
	return g_error.status;
}

int KYTY_SYSV_ABI ErrorDialogGetStatus() {
	PRINT_NAME();

	std::scoped_lock lock(g_dialog_mutex);
	return g_error.status;
}

} // namespace ErrorDialog

} // namespace Libs::Dialog
