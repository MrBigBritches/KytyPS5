#ifndef EMULATOR_SRC_LIBS_SAVEDATA_H_
#define EMULATOR_SRC_LIBS_SAVEDATA_H_

#include "common/common.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Libs::SaveData {

// Size of SceSaveDataParam (title, subtitle, detail, user parameter, mtime).
constexpr size_t SAVE_DATA_PARAM_SIZE = 1328;

// Reads the parameter of an existing save directory, with mtime set to its newest file, for the
// save data dialog. `title_id` may be null for the running title. Returns false if the directory
// has no readable parameter.
bool ReadSaveDataParam(int32_t user_id, const char* title_id, const char* dir_name, void* param,
                       size_t param_size);

// Reads the icon (PNG) stored for a save directory by SaveDataSaveIcon.
bool ReadSaveDataIcon(int32_t user_id, const char* title_id, const char* dir_name,
                      std::vector<uint8_t>* png);

// Total size in bytes of the files in a save directory (0 if it does not exist).
uint64_t GetSaveDataSize(int32_t user_id, const char* title_id, const char* dir_name);

} // namespace Libs::SaveData

#endif // EMULATOR_SRC_LIBS_SAVEDATA_H_
