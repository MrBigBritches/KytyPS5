#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <array>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

using Keymap = std::vector<std::string>;
using ControllerColor = std::array<uint8_t, 3>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
// The most workers --warm-workers may ask for. It is above the default cap of 4 so a measurement
// can overcommit a machine, and low enough that a typo cannot start hundreds of compile threads.
constexpr uint32_t MAX_WARM_WORKERS = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;

constexpr int     MAX_PLAYERS         = 4;
constexpr int32_t PLAYER_USER_ID_STEP = 0x100;
constexpr int32_t MAX_USER_ID =
    std::numeric_limits<int32_t>::max() - (MAX_PLAYERS - 1) * PLAYER_USER_ID_STEP;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
	constexpr int32_t USER_ID_EVERYONE = 0xfe;
	constexpr int32_t USER_ID_SYSTEM   = 0xff;
	return user_id >= 0 && user_id <= MAX_USER_ID && user_id != USER_ID_EVERYONE &&
	       user_id != USER_ID_SYSTEM;
}

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	std::string            user_name                   = "Kyty";
	int32_t                user_id                     = DEFAULT_USER_ID;
	std::string            audio_input_device;
	std::optional<ControllerColor> controller_color;
	uint32_t               controller_speaker_volume      = 50;
	uint32_t               controller_vibration_intensity = 100;
	// 0 shares player 1 with the first gamepad; 1..MAX_PLAYERS dedicates that player to the keyboard.
	uint32_t               keyboard_player                = 0;
	PresentMode            present_mode                = PresentMode::Mailbox;
	int32_t                gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   hdr_enabled                 = false;
	bool                   hide_cursor_enabled         = false;
	bool                   stretch_enabled             = false;
	bool                   splash_enabled              = true;
	bool                   vr_enabled                  = false;
	bool                   pro_enabled                 = false;
	bool                   amd_cpu_enabled             = false;
	bool                   guest_tsc_enabled           = false;
	bool                   network_connected           = false;
	bool                   psn_emulation_enabled       = false;
	uint32_t               vblank_frequency            = 60;
	uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	bool                   shader_cache_enabled        = false;
	// Opts in to selecting guarded shader loads instead of branching around them.
	bool                   select_guarded_loads        = false;
	// Opts in to range checking formatted buffer records with one compare.
	bool                   single_compare_record_guard = false;
	// Opts in to eliding compute shader selects that no lane observes outside their mask.
	bool                   elide_exec_selects          = false;
	// Opts in to compiling S_SETPC_B64 jump tables as structured control flow.
	bool                   structured_jump_tables      = false;
	// Opts in to scanning every submission as it is enqueued and comparing the scan with what the
	// command processor runs, for the first-run report. A measurement mode that slows the enqueuing thread.
	bool                   scan_differential           = false;
	// Opts in to reading each command list as the game submits it and building the compute
	// shaders it is about to run in a helper process. Results persist only with
	// shader_cache_enabled.
	bool                   prescan_compute_builds      = false;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	LogDirection           printf_direction            = LogDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	bool                   profiler_enabled            = false;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   renderdoc_enabled           = false;
	bool                   readback_linear_images      = false;
	bool                   readback_tiled_images       = false;
	bool                   readback_fault_wait         = false;
	bool                   sync_raw_image_buffers      = false;
	bool                   skip_redundant_unmap_drains = false;
	bool                   partial_buffer_refresh_enabled = false;
	bool                   batch_hot_readbacks         = false;
	bool                   tessellation_enabled        = false;
	bool                   occlusion_queries_enabled   = false;
	bool                   audio_compile_hold_enabled  = false;
	// Writes _FirstRunReports/<TITLE_ID>-<UTC time>.json at a clean exit.
	bool                   first_run_report_enabled    = false;
	// Stamps each SPIR-V module per run so no shader or driver cache can serve it. Every launch is slow.
	bool                   cold_pipelines_enabled      = false;
	// Builds graphics pipelines on worker threads. A draw whose pipeline is not ready is skipped
	// only where its targets are redrawn every frame, and otherwise waits for the pipeline.
	bool                   async_pipelines_enabled     = false;
	// Like async_pipelines_enabled, which it implies, but only draws with no color target ever go
	// without their pipeline.
	bool                   async_pipelines_depth_only_enabled = false;
	// Like async_pipelines_enabled, which it implies, but a draw the skip rules allow to go without
	// its pipeline is skipped even when it could be deferred, so a resolve never waits on it. A
	// partial view, such as one layer or mip of an image, is judged by the history of its exact
	// range instead of being refused.
	bool                   async_pipelines_prefer_skip_enabled = false;
	// Builds the compute pipelines of the shader store on low priority workers while the game runs,
	// instead of making boot wait for all of them.
	bool                   background_warm_up_enabled  = false;
	// Sets how many workers the background warm-up runs, for measuring it. Zero, the default, sizes
	// them from the hardware threads. Does nothing without background_warm_up_enabled.
	uint32_t warm_workers = 0;
	// Predicts the compute shaders of a Motor title from its eboot and translates them in a low
	// priority helper process, so the warm-up finds them ready. Needs shader_cache_enabled and
	// background_warm_up_enabled, and does nothing without both.
	bool                   generate_shader_store_enabled = false;
	// Lets a draw that misses its graphics pipeline carry on while the pipeline builds, with the
	// draw recorded afterwards. Stands alone and enables no draw skips.
	bool                   deferred_pipeline_draws_enabled = false;
	// Holds the warm-up queue from the first hole of a submission until its resolve ends, so the
	// resolve's compiles do not compete with the warm-up for cores. Needs --background-warm-up and
	// --deferred-pipeline-draws, and does nothing without both.
	bool                   pause_warm_up_for_draws_enabled = false;
	// Lets a draw whose descriptors go in a descriptor set, not pushed, be deferred too. Its set is
	// allocated and written when the hole is recorded. Needs --deferred-pipeline-draws.
	bool                   deferred_non_push_draws_enabled = false;
	// Lets a direct mesh draw be deferred too. Its view keeps the groups, index address and draw
	// dwords its recording reads. Task mesh draws still block. Needs --deferred-pipeline-draws, and
	// usually --deferred-non-push-draws, because mesh pipelines are seldom predicted to push.
	bool                   deferred_mesh_draws_enabled = false;
	// Lets an indirect draw be deferred too. Its view keeps the arguments buffer and offset, so the
	// hole reads the records the GPU sees. Task mesh draws still block. Needs --deferred-pipeline-draws.
	bool                   deferred_indirect_draws_enabled = false;
	// Asks for graphics pipelines built from separately compiled stage libraries, so a stage shared
	// by several pipelines compiles once. It does nothing on a device that misses any requirement.
	bool                   graphics_pipeline_library_enabled = false;
	// Lets the GPU start on a long command stream instead of idling until the end-of-slice submit.
	bool                   gpu_submit_cadence_enabled  = false;
	// Opts in to the host GPU's float min/max on non-AMD GPUs. It is faster, but the sign of a zero
	// result and the NaN returned for two NaN inputs may differ from the guest.
	bool                   native_float_min_max_enabled = false;
	bool                   trophy_enabled              = true;
	uint32_t               trophy_duration_seconds     = 6;
	bool                   perf_overlay_enabled            = false;
	bool                   playgo_hack_enabled         = false;
	bool                   skip_notice_screen          = false;
	// Experimental. Keeps GPU buffer caches when direct memory is released and mapped back unchanged.
	bool                   keep_gpu_cache_on_reuse     = false;
	// Experimental. Also keeps clean texture-cache images across that release and remap.
	bool                   keep_gpu_textures_on_reuse  = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool red_zone_protection_enabled = false;
#endif
	Keymap keymap;
	// Ordinary boots need a non-null argv[0]; --guest-exec supplies the caller's arguments.
	std::vector<std::string> guest_arguments {""};
};

void Load(const ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
const std::string& GetAudioInputDevice();
const std::optional<ControllerColor>& GetControllerColor();
uint32_t GetControllerSpeakerVolume();
uint32_t GetControllerVibrationIntensity();
uint32_t GetKeyboardPlayer();
// Player 0 is the configured user; later players derive their IDs from it.
int32_t  GetPlayerUserId(int player);
// Returns -1 when the user ID does not belong to any player.
int      GetPlayerFromUserId(int32_t user_id);
// Players after the first are guests: they have no profile, trophies or NP account.
bool     IsGuestUserId(int32_t user_id);
PresentMode GetPresentMode();
int32_t GetGpuIndex();
bool     FullscreenEnabled();
bool     HdrEnabled();
bool     HideCursorEnabled();
bool     StretchEnabled();
bool     VrEnabled();
bool     Ps5ProEnabled();
bool     AmdCpuEnabled();
bool     GuestTscEnabled();
bool     IsNetworkConnected();
bool     PsnEmulationEnabled();
uint32_t GetVblankFrequency();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
bool                   ShaderCacheEnabled();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

bool GraphicsDebugDumpEnabled();

// True when guarded shader loads should be selected instead of branched around.
bool SelectGuardedLoadsEnabled();
// True when formatted buffer records should be range checked with one compare.
bool SingleCompareRecordGuardEnabled();
// True when compute shader selects that no lane observes outside their mask should be elided.
bool ElideExecSelectsEnabled();
// True when jump tables should be compiled as structured control flow.
bool StructuredJumpTablesEnabled();
// True when every submission should be scanned and compared with what the command processor runs.
bool ScanDifferentialEnabled();
// True when the compute shaders of each submitted command list should be built ahead in a helper.
bool PrescanComputeBuildsEnabled();

LogDirection          GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool ReadbackTiledImagesEnabled();
/// @return true if guest access to a pending readback's pages should wait for its data.
bool ReadbackFaultWaitEnabled();
bool SyncRawImageBuffersEnabled();
// Skips the GPU drain when a range that is already unmapped is unmapped again.
bool SkipRedundantUnmapDrainsEnabled();
bool PartialBufferRefreshEnabled();
// Makes a fault readback that waits for the GPU also read back GPU-written bytes of buffers that
// recently faulted.
bool BatchHotReadbacksEnabled();
bool TessellationEnabled();
bool OcclusionQueriesEnabled();
bool AudioCompileHoldEnabled();
bool FirstRunReportEnabled();
bool ColdPipelinesEnabled();
/// Whether graphics pipelines are built in the background. The pipeline cache reads it once, when
/// it is made.
bool AsyncPipelinesEnabled();
/// Whether only draws with no color target may go without a pipeline. It implies background builds.
bool AsyncPipelinesDepthOnly();
/// True when --async-pipelines-prefer-skip asks for a draw the skip rules allow to be skipped
/// rather than deferred. It implies background builds. The skip rules then also judge a partial
/// view, such as one layer or mip of an image, by the history of its exact range while the image's
/// table has room, where without the flag they refuse it. They also let a color target pass on
/// enough draws in each of the two completed frames before this one, because a draw early in a color
/// pass sees only the start of its frame. The depth target rule is unchanged. The risk is that a
/// skipped draw into a color target that is redrawn every frame but not cleared leaves that object
/// missing for a frame. The skipped draw can be the frame's first into the target, a full-screen pass
/// that sets it up, so the target keeps stale content for the frame, and for longer when the game
/// reads it back as history, as TAA and trail effects do. The first-run report's
/// skips.sampled_after_skip_by_target_class count is the signal to watch, though it only sees
/// targets that are sampled later. The partial-view and completed-frames parts differ from
/// --async-pipelines on their own, while skipping before deferring only differs when
/// --deferred-pipeline-draws is also on.
bool AsyncPipelinesPreferSkip();
/// True when --background-warm-up builds the shader store's compute pipelines after boot.
bool BackgroundWarmUpEnabled();
/// The warm-up worker count --warm-workers asked for, or zero when the warm-up sizes it itself.
uint32_t GetWarmWorkers();
/// True when --generate-shader-store asks for the shader store to be generated from the eboot. The
/// pipeline cache decides whether it can, and says why not.
bool GenerateShaderStoreEnabled();
/// True when --deferred-pipeline-draws lets draws wait for their pipeline out of line.
bool DeferredPipelineDrawsEnabled();
/// True when --pause-warm-up-for-draws holds the warm-up queue while deferred draws resolve.
bool PauseWarmUpForDrawsEnabled();
/// True when --deferred-non-push-draws lets draws with a descriptor set be deferred. It does nothing
/// without --deferred-pipeline-draws.
bool DeferredNonPushDrawsEnabled();
/// True when --deferred-mesh-draws lets a direct mesh draw be deferred. Task mesh draws still block.
/// It does nothing without --deferred-pipeline-draws, which it does not turn on.
bool DeferredMeshDrawsEnabled();
/// True when --deferred-indirect-draws lets an indirect draw that is not a task mesh draw be
/// deferred. It does nothing without --deferred-pipeline-draws, which it does not turn on.
bool DeferredIndirectDrawsEnabled();
/// True when --graphics-pipeline-library asks for pipelines built from stage libraries. The device
/// decides whether that happens.
bool GraphicsPipelineLibraryEnabled();
bool GpuSubmitCadenceEnabled();
/// Returns whether non-AMD GPUs may use the host's float min/max instead of the exact emulation.
bool NativeFloatMinMaxEnabled();
bool TrophyEnabled();
uint32_t GetTrophyDurationSeconds();
bool PlayGoHackEnabled();
bool SkipNoticeScreen();
// Whether released direct memory keeps its GPU buffer caches for an exact remap. Default off.
bool KeepGpuCacheOnReuse();
// Whether that release also keeps clean texture-cache images. Needs KeepGpuCacheOnReuse. Default off.
bool KeepGpuTexturesOnReuse();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();
const std::vector<std::string>& GetGuestArguments();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
