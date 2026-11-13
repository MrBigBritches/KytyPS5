cmake_minimum_required(VERSION 3.19...4.3)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
	set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "Minimum policy version for bundled dependencies")
endif()

get_property(KYTY_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT KYTY_GENERATOR_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(Kyty)

find_program(SCCACHE_PROGRAM sccache)
find_program(CCACHE_PROGRAM ccache)

foreach(lang IN ITEMS C CXX)
    # An explicitly empty launcher disables caching for this language.
    if(NOT DEFINED CMAKE_${lang}_COMPILER_LAUNCHER)
        if(SCCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
            message(STATUS "sccache enabled for ${lang}: ${SCCACHE_PROGRAM}")
        elseif(CCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
            message(STATUS "ccache enabled for ${lang}: ${CCACHE_PROGRAM}")
        endif()
    endif()
endforeach()

set(KYTY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KYTY_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(KYTY_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

if(CMAKE_SYSTEM_NAME MATCHES ".*Linux")
	set(LINUX TRUE)
endif()

if (NOT (WIN32 OR LINUX OR APPLE))
    message(FATAL_ERROR "only Windows, Linux, and macOS builds are supported")
endif()

if(WIN32)
	# Prevent Windows headers from defining min/max macros in all targets.
	add_compile_definitions(NOMINMAX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include("${KYTY_SOURCE_DIR}/utils.cmake")
include(CTest)

option(KYTY_ENABLE_CLANG_TIDY "Run clang-tidy checks during builds" OFF)

set(KYTY_BUILD_ORIGIN "Source" CACHE STRING "Build provenance shown in diagnostics")
set_property(CACHE KYTY_BUILD_ORIGIN PROPERTY STRINGS Source Official Fork)
if(NOT KYTY_BUILD_ORIGIN STREQUAL "Source" AND NOT KYTY_BUILD_ORIGIN STREQUAL "Official" AND
   NOT KYTY_BUILD_ORIGIN STREQUAL "Fork")
	message(FATAL_ERROR "KYTY_BUILD_ORIGIN must be Source, Official, or Fork")
endif()

set(KYTY_RELEASE_TAG "" CACHE STRING "Release tag embedded in official builds")
set(KYTY_BUILD_REPOSITORY "" CACHE STRING "GitHub repository embedded in fork builds")
set(KYTY_OFFICIAL_BUILD OFF)
set(KYTY_FORK_BUILD OFF)
if(KYTY_BUILD_ORIGIN STREQUAL "Official")
	if(KYTY_RELEASE_TAG STREQUAL "")
		message(FATAL_ERROR "Official builds require KYTY_RELEASE_TAG")
	endif()
	set(KYTY_OFFICIAL_BUILD ON)
elseif(KYTY_BUILD_ORIGIN STREQUAL "Fork")
	if(KYTY_BUILD_REPOSITORY STREQUAL "")
		message(FATAL_ERROR "Fork builds require KYTY_BUILD_REPOSITORY")
	endif()
	set(KYTY_FORK_BUILD ON)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(FATAL_ERROR "only 64-bit builds are supported")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
	set(KYTY_BUILD KYTY_BUILD_DEBUG)
else()
	set(KYTY_BUILD KYTY_BUILD_RELEASE)
endif()

if(LINUX OR APPLE)
	# macOS rides the POSIX/Linux code paths until it gets a dedicated platform
	set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
else()
	set(KYTY_PLATFORM KYTY_PLATFORM_WINDOWS)
endif()

set(CLANG FALSE)
set(GCC FALSE)
set(KYTY_CLANG_CL FALSE)
get_filename_component(KYTY_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME_WE)
string(TOLOWER "${KYTY_CXX_COMPILER_NAME}" KYTY_CXX_COMPILER_NAME)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
	set(CLANG TRUE)
	set(KYTY_COMPILER CLANG)
	if(WIN32 AND KYTY_CXX_COMPILER_NAME STREQUAL "clang-cl")
		set(KYTY_CLANG_CL TRUE)
		set(KYTY_LINKER LLD_LINK)
	elseif(APPLE)
		set(KYTY_LINKER LD64) # Apple's default linker; lld flags don't apply
	else()
		set(KYTY_LINKER LLD)
	endif()
	if ((CMAKE_AR MATCHES "llvm") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0.0) AND (CMAKE_CXX_COMPILER MATCHES "winlibs"))
		string(REPLACE "llvm-ar.exe" "ar.exe" CMAKE_AR ${CMAKE_AR}) # llvm-ar.exe 11.0.0 from winlibs is broken
	endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(GCC TRUE)
	set(KYTY_COMPILER GCC)
	set(KYTY_LINKER LD)
else()
	message(FATAL_ERROR "only Clang and GCC are supported; found ${CMAKE_CXX_COMPILER_ID}")
endif()

set(KYTY_PROJECT_NAME "Emulator" CACHE STRING "Project name suffix")

string(TOLOWER ${KYTY_COMPILER} KYTY_COMPILER_ID)
string(TOLOWER ${KYTY_LINKER} KYTY_LINKER_ID)

if (CLANG AND (KYTY_LINKER STREQUAL LLD))
	set(KYTY_LD_OPTIONS "-fuse-ld=lld")
endif()

if (KYTY_LINKER STREQUAL LD AND NOT LINUX)
	set(KYTY_LD_OPTIONS "-Wl,--image-base=0x100000000000")
endif()

project(Kyty${KYTY_PROJECT_NAME}${CMAKE_BUILD_TYPE}${KYTY_COMPILER} VERSION 0.3.0)

include_directories(
	${KYTY_THIRD_PARTY_DIR}/gtest/include
	${KYTY_THIRD_PARTY_DIR}/gtest
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/magic_enum/include/magic_enum
	${PROJECT_BINARY_DIR}
	${KYTY_SOURCE_DIR}
)

set(KYTY_VERSION "${PROJECT_VERSION}")

configure_file(
  ${KYTY_SOURCE_DIR}/cmake_config.h.in
  ${PROJECT_BINARY_DIR}/cmake_config.h
  )

find_package(Git)

add_custom_target(KytyGitVersion
	COMMAND ${CMAKE_COMMAND}
		-D INPUT_FILE=${KYTY_SOURCE_DIR}/kytyGitVersion.h.in
		-D OUTPUT_FILE=${PROJECT_BINARY_DIR}/kytyGitVersion.h
		-D GIT_EXECUTABLE=${GIT_EXECUTABLE}
		-D GIT_WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}
		-P ${KYTY_SOURCE_DIR}/generate_version.cmake
	BYPRODUCTS ${PROJECT_BINARY_DIR}/kytyGitVersion.h
	COMMENT "Generate kytyGitVersion.h"
)

add_library(kyty_git_version INTERFACE)
add_dependencies(kyty_git_version KytyGitVersion)

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0)
	list(APPEND KYTY_IWYU
		kyty_emulator
		common
		#launcher
	)
	list(APPEND KYTY_CLANG_TIDY
		kyty_emulator
		#common
		launcher
	)
endif()

option(KYTY_BUILD_LAUNCHER "Build Qt launcher" ON)

config_compiler_and_linker()

add_subdirectory("${KYTY_THIRD_PARTY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty")
add_subdirectory("${KYTY_SOURCE_DIR}/common" "${CMAKE_CURRENT_BINARY_DIR}/common")

file(GLOB kyty_emulator_src CONFIGURE_DEPENDS
	"${KYTY_SOURCE_DIR}/libs/*.cpp"
	"${KYTY_SOURCE_DIR}/libs/*.h"
	"${KYTY_SOURCE_DIR}/graphics/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.inc"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.h"
	"${KYTY_SOURCE_DIR}/kernel/*.cpp"
	"${KYTY_SOURCE_DIR}/kernel/*.h"
	"${KYTY_SOURCE_DIR}/loader/*.cpp"
	"${KYTY_SOURCE_DIR}/loader/*.h"
)

find_program(KYTY_GLSLANG_VALIDATOR NAMES glslangValidator glslang REQUIRED)
set(gpu_tiler_shader_dir "${KYTY_SOURCE_DIR}/graphics/host_gpu/shaders")
set(gpu_tiler_generated_dir "${PROJECT_BINARY_DIR}/gpu_tiler_shaders")
set(gpu_tiler_shader_names
	standard256
	standard4
	standard4_3d
	standard64
	standard64_3d
	prt
	prt_3d
	render_target
	depth
	promote_d16
	demote_d16
	color_transform
	color_clear
)
file(GLOB gpu_tiler_shader_includes CONFIGURE_DEPENDS "${gpu_tiler_shader_dir}/gpu_tiler_*.inc")
foreach(shader_name IN LISTS gpu_tiler_shader_names)
	set(shader_source "${gpu_tiler_shader_dir}/gpu_tiler_${shader_name}.comp")
	set(shader_spv "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}.spv")
	set(shader_header "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}_spv.h")
	string(TOUPPER "GPU_TILER_${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" ${gpu_tiler_shader_includes}
		VERBATIM
	)
	list(APPEND gpu_tiler_shader_headers "${shader_header}")
endforeach()

set(fault_buffer_shader_source "${gpu_tiler_shader_dir}/fault_buffer_process.comp")
set(fault_buffer_shader_spv "${gpu_tiler_generated_dir}/fault_buffer_process.spv")
set(fault_buffer_shader_header "${gpu_tiler_generated_dir}/fault_buffer_process_spv.h")
add_custom_command(
	OUTPUT "${fault_buffer_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -DCACHING_PAGEBITS=14 -DMAX_PAGE_FAULTS=1024
	        -o "${fault_buffer_shader_spv}" "${fault_buffer_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${fault_buffer_shader_spv} -DOUTPUT=${fault_buffer_shader_header}
	        -DSYMBOL=FAULT_BUFFER_PROCESS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${fault_buffer_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${fault_buffer_shader_header}")
list(APPEND kyty_emulator_src ${gpu_tiler_shader_headers})

set(gpu_blit_generated_dir "${PROJECT_BINARY_DIR}/gpu_blit_shaders")
function(add_gpu_blit_shader shader_name shader_file)
	set(shader_source "${gpu_tiler_shader_dir}/${shader_file}")
	set(shader_spv "${gpu_blit_generated_dir}/${shader_name}.spv")
	set(shader_header "${gpu_blit_generated_dir}/${shader_name}_spv.h")
	string(TOUPPER "${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_blit_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os ${ARGN}
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" "${gpu_tiler_shader_dir}/gpu_video_out_hdr10_to_sdr.inc"
		VERBATIM
	)
	set(gpu_blit_shader_headers ${gpu_blit_shader_headers} "${shader_header}" PARENT_SCOPE)
endfunction()
add_gpu_blit_shader(gpu_blit_fs_triangle gpu_blit_fs_triangle.vert)
add_gpu_blit_shader(gpu_blit_color_to_ms_depth gpu_blit_color_to_ms_depth.frag)
add_gpu_blit_shader(gpu_video_out_overlay gpu_video_out_overlay.frag)
add_gpu_blit_shader(gpu_imgui_hdr gpu_video_out_overlay.frag -DIMGUI_HDR=1)
add_gpu_blit_shader(gpu_pssr_linear gpu_pssr_linear.comp)
# At SPIR-V 1.6 glslang declares a task shader's workgroup size with LocalSizeId, which needs maintenance4.
add_gpu_blit_shader(gpu_mesh_indirect gpu_mesh_indirect.task --target-env spirv1.5)
list(APPEND kyty_emulator_src ${gpu_blit_shader_headers})

list(APPEND kyty_emulator_src
	"${KYTY_SOURCE_DIR}/emulator.h"
	"${KYTY_SOURCE_DIR}/emulator.cpp"
)
list(REMOVE_DUPLICATES kyty_emulator_src)

if (KYTY_CLANG_CL)
	set_source_files_properties(${kyty_emulator_src} PROPERTIES COMPILE_FLAGS "-Wno-pragma-pack -Wno-deprecated-declarations -D_TIMESPEC_DEFINED")
endif()

set(kyty_emulator_link_libraries common Vulkan::Headers spirv-tools-opt spirv-tools SDL3::SDL3-static imgui xxhash FFmpeg::ffmpeg fmt::fmt nlohmann_json::nlohmann_json LibAtrac9 kyty_git_version Zydis::Zydis xbyak::xbyak opus OpenXR::openxr_loader libzstd_static)

# Linux system libraries.
if(LINUX)
	find_package(Threads REQUIRED)
	list(APPEND kyty_emulator_link_libraries m ${CMAKE_DL_LIBS} Threads::Threads)
endif()

set(inc_headers
	${KYTY_SOURCE_DIR}
	${KYTY_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Tools/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers/include
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/xxHash
	${KYTY_THIRD_PARTY_DIR}/cpuinfo/include
	${KYTY_THIRD_PARTY_DIR}/renderdoc
	${KYTY_THIRD_PARTY_DIR}/stb
)

if (KYTY_CLANG_CL)
	list(APPEND kyty_emulator_link_libraries winpthread)
	list(APPEND inc_headers
		${KYTY_THIRD_PARTY_DIR}/winpthread/include
	)
endif()

list(APPEND check_headers
	${KYTY_SOURCE_DIR}
)

function(add_kyty_full_emulator_test target source)
	add_executable(${target} EXCLUDE_FROM_ALL ${source} ${kyty_emulator_src})
	target_link_libraries(${target} ${kyty_emulator_link_libraries})
	target_include_directories(${target} PRIVATE ${inc_headers})
	if (WIN32)
		target_link_libraries(${target} iphlpapi dxgi)
	endif()
	if (KYTY_CLANG_CL)
		target_link_libraries(${target} onecore)
		add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:${target}>/libwinpthread-1.dll)
	endif()
	# The macOS x86_64 guest address space needs its .zerofill segments anchored
	# by linker flags, or the kernel kills the binary on load (posix_spawn EIO).
	configure_macos_guest_address_space(${target})
endfunction()

function(configure_macos_guest_address_space target)
	if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
			(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")))
		target_sources(${target} PRIVATE "${KYTY_SOURCE_DIR}/kernel/macosGuestAddressSpace.cpp")
		target_compile_definitions(${target} PRIVATE KYTY_LINKED_GUEST_ADDRESS_SPACE=1)
		target_link_options(${target} PRIVATE
			-Wl,-ld_classic,-no_pie,-no_fixup_chains,-no_huge,-pagezero_size,0x40000,-segaddr,SYSTEM_MANAGED,0x40000,-segaddr,SYSTEM_RESERVED,0x7ffffc000,-segaddr,USER_AREA,0x7000000000,-image_base,0x700000000000)
	endif()
endfunction()

add_kyty_full_emulator_test(shader_cfg_tests "${KYTY_TESTS_DIR}/shaderCfgTests.cpp")

add_executable(scalar_provenance_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ScalarProvenanceTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ReadLaneElimination.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(scalar_provenance_tests fmt::fmt xbyak::xbyak)
target_include_directories(scalar_provenance_tests PRIVATE ${inc_headers})

add_executable(build_flags_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BuildFlagsTests.cpp")

add_executable(page_manager_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PageManagerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
)
target_link_libraries(page_manager_tests common)
target_include_directories(page_manager_tests PRIVATE ${inc_headers})

add_executable(readback_guard_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ReadbackGuardTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/readbackGuard.cpp"
)
target_link_libraries(readback_guard_tests common)
target_include_directories(readback_guard_tests PRIVATE ${inc_headers})

add_executable(archive_file_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ArchiveFileTests.cpp"
)
target_link_libraries(archive_file_tests common ZArchive::zarchive)
target_include_directories(archive_file_tests PRIVATE ${inc_headers})

add_executable(trophy_system_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/TrophySystemTests.cpp"
)
target_link_libraries(trophy_system_tests common Vulkan::Headers)
target_include_directories(trophy_system_tests PRIVATE ${inc_headers})

add_executable(avplayer_file_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AvPlayerFileTests.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(avplayer_file_tests common FFmpeg::ffmpeg ZArchive::zarchive)
target_include_directories(avplayer_file_tests PRIVATE ${inc_headers})

add_executable(vdecsw_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/VdecswTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/videoDec2Decoder.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(vdecsw_tests common FFmpeg::ffmpeg)
target_include_directories(vdecsw_tests PRIVATE ${inc_headers})

add_executable(bit_array_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BitArrayTests.cpp"
)
target_include_directories(bit_array_tests PRIVATE ${inc_headers})

add_executable(lru_cache_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/LruCacheTests.cpp"
)
target_include_directories(lru_cache_tests PRIVATE ${inc_headers})

add_executable(slot_vector_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SlotVectorTests.cpp"
)
target_include_directories(slot_vector_tests PRIVATE ${inc_headers})

add_executable(common_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CommonTests.cpp")
target_link_libraries(common_tests common kyty_test_gtest)

add_executable(ime_dialog_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImeDialogTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/imeCommon.cpp"
	"${KYTY_SOURCE_DIR}/libs/imeDialog.cpp"
)
target_include_directories(ime_dialog_tests PRIVATE ${inc_headers})

add_executable(memory_tracker_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/MemoryTrackerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/memoryTracker.cpp"
)
target_link_libraries(memory_tracker_tests fmt::fmt common)
target_include_directories(memory_tracker_tests PRIVATE ${inc_headers})

add_executable(shader_vertex_metadata_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderVertexMetadataTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderVertexMetadata.cpp"
)
target_include_directories(shader_vertex_metadata_tests PRIVATE ${inc_headers})

add_executable(resource_materialization_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceMaterializationTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(resource_materialization_tests fmt::fmt xbyak::xbyak)
target_include_directories(resource_materialization_tests PRIVATE ${inc_headers})

add_executable(resource_tracking_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceTrackingTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ShaderInfoCollection.cpp"
)
target_link_libraries(resource_tracking_tests fmt::fmt xbyak::xbyak)
target_include_directories(resource_tracking_tests PRIVATE ${inc_headers})

add_executable(audio_out2_port_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AudioOut2PortTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/libAudio2.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(audio_out2_port_tests common fmt::fmt)
target_include_directories(audio_out2_port_tests PRIVATE ${inc_headers})

add_executable(audio_out_timing_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AudioOutTimingTests.cpp")
target_link_libraries(audio_out_timing_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(audio_out_timing_tests PRIVATE ${inc_headers})

add_executable(ngs2_sampler_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/Ngs2SamplerTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/ngs2_reverb.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(ngs2_sampler_tests common fmt::fmt LibAtrac9)
target_include_directories(ngs2_sampler_tests PRIVATE ${inc_headers})

add_executable(ajm_resample_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AjmResampleTests.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(ajm_resample_tests common fmt::fmt FFmpeg::ffmpeg LibAtrac9 opus)
target_include_directories(ajm_resample_tests PRIVATE ${inc_headers})

add_executable(pad_haptics_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PadHapticsTests.cpp"
)
target_link_libraries(pad_haptics_tests common fmt::fmt SDL3::SDL3-static opus)
target_include_directories(pad_haptics_tests PRIVATE ${inc_headers})

add_executable(controller_settings_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ControllerSettingsTests.cpp"
)
target_link_libraries(controller_settings_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(controller_settings_tests PRIVATE ${inc_headers})

add_executable(host_input_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/HostInputTests.cpp")
target_link_libraries(host_input_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(host_input_tests PRIVATE ${inc_headers})

add_executable(save_data_memory_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataMemoryTests.cpp"
)
target_link_libraries(save_data_memory_tests common fmt::fmt)
target_include_directories(save_data_memory_tests PRIVATE ${inc_headers})

add_executable(png_enc_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/PngEncTests.cpp")
target_link_libraries(png_enc_tests common fmt::fmt)
target_include_directories(png_enc_tests PRIVATE ${inc_headers})

add_executable(jpeg_dec_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/JpegDecTests.cpp")
target_link_libraries(jpeg_dec_tests common fmt::fmt)
target_include_directories(jpeg_dec_tests PRIVATE ${inc_headers})

add_executable(jpeg_enc_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/JpegEncTests.cpp")
target_link_libraries(jpeg_enc_tests common fmt::fmt)
target_include_directories(jpeg_enc_tests PRIVATE ${inc_headers})

add_executable(occlusion_queries_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/OcclusionQueriesTests.cpp")
target_link_libraries(occlusion_queries_tests common fmt::fmt Vulkan::Headers)
target_include_directories(occlusion_queries_tests PRIVATE ${inc_headers})

add_executable(pipeline_build_queue_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PipelineBuildQueueTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.cpp")
target_link_libraries(pipeline_build_queue_tests common fmt::fmt)
target_include_directories(pipeline_build_queue_tests PRIVATE ${inc_headers})

add_executable(thread_priority_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/ThreadPriorityTests.cpp")
target_link_libraries(thread_priority_tests common fmt::fmt)
target_include_directories(thread_priority_tests PRIVATE ${inc_headers})

add_executable(pipeline_library_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/PipelineLibraryTests.cpp")
target_include_directories(pipeline_library_tests PRIVATE ${inc_headers})

add_executable(pipeline_parts_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PipelinePartsTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/pipelineParts.cpp")
target_link_libraries(pipeline_parts_tests common Vulkan::Headers xxhash fmt::fmt)
target_include_directories(pipeline_parts_tests PRIVATE ${inc_headers})

add_executable(graphics_library_cache_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GraphicsLibraryCacheTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/pipelineParts.cpp")
target_link_libraries(graphics_library_cache_tests common Vulkan::Headers xxhash fmt::fmt)
target_include_directories(graphics_library_cache_tests PRIVATE ${inc_headers})

add_executable(descriptor_set_layout_cache_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/DescriptorSetLayoutCacheTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/descriptorSetLayoutCache.cpp")
target_link_libraries(descriptor_set_layout_cache_tests common Vulkan::Headers xxhash fmt::fmt)
target_include_directories(descriptor_set_layout_cache_tests PRIVATE ${inc_headers})

add_executable(ces_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CesTests.cpp")
target_link_libraries(ces_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(ces_tests PRIVATE ${inc_headers})

add_executable(save_data_dialog_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataDialogTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/dialog.cpp"
)
target_link_libraries(save_data_dialog_tests common fmt::fmt)
target_include_directories(save_data_dialog_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(http_uri_parse_tests
	"${KYTY_TESTS_DIR}/HttpUriParseTests.cpp")

add_executable(rtc_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/RtcTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/libRtc.cpp"
	"${KYTY_SOURCE_DIR}/loader/symbolDatabase.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp")
target_link_libraries(rtc_tests common kyty_test_gtest)
target_include_directories(rtc_tests PRIVATE ${inc_headers})

add_executable(event_queue_lifetime_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EventQueueLifetimeTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/eventQueue.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(event_queue_lifetime_tests common fmt::fmt)
target_include_directories(event_queue_lifetime_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(kernel_file_system_tests
	"${KYTY_TESTS_DIR}/KernelFileSystemTests.cpp")
target_link_libraries(kernel_file_system_tests ZArchive::zarchive)

add_executable(gpu_stall_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/GpuStallTests.cpp")
target_link_libraries(gpu_stall_tests common fmt::fmt)
target_include_directories(gpu_stall_tests PRIVATE ${inc_headers})

add_executable(child_process_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/ChildProcessTests.cpp")
target_link_libraries(child_process_tests common fmt::fmt)
target_include_directories(child_process_tests PRIVATE ${inc_headers})

add_executable(first_run_report_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/FirstRunReportTests.cpp")
target_link_libraries(first_run_report_tests common fmt::fmt nlohmann_json::nlohmann_json)
target_include_directories(first_run_report_tests PRIVATE ${inc_headers})

add_executable(sync_on_address_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SyncOnAddressTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/syncOnAddress.cpp"
)
target_link_libraries(sync_on_address_tests common)
target_include_directories(sync_on_address_tests PRIVATE ${inc_headers})

add_executable(image_page_table_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImagePageTableTests.cpp"
)
target_link_libraries(image_page_table_tests common fmt::fmt)
target_include_directories(image_page_table_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(shader_recompiler_compute_tests "${KYTY_TESTS_DIR}/ShaderRecompilerComputeTests.cpp")
# A light target. It links only the scan, the register decoder and the device-free key and sl00 code.
add_executable(shader_store_scan_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreScanTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/csRegisters.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStaticKey.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/ebootShaderScan.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/computeStaticKey.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/sl00Metadata.cpp"
)
target_link_libraries(shader_store_scan_tests common fmt::fmt xxhash)
target_include_directories(shader_store_scan_tests PRIVATE ${inc_headers})

# The submit-time PM4 walk. It reads buffers through a reader the test supplies, so it links only the
# walk and the register decoder.
add_executable(submit_scan_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SubmitScanTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/csRegisters.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/submitScan.cpp"
)
target_link_libraries(submit_scan_tests common fmt::fmt xxhash)
target_include_directories(submit_scan_tests PRIVATE ${inc_headers})

# The scan differential's alignment, its shared key and its report block. Like the walk it needs no device.
add_executable(scan_differential_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ScanDifferentialTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/csRegisters.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/submitScan.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/scanDifferential.cpp"
)
target_link_libraries(scan_differential_tests common fmt::fmt xxhash nlohmann_json::nlohmann_json)
target_include_directories(scan_differential_tests PRIVATE ${inc_headers})

# The pre-scan's input info builder and the translation setup it shares with the generator. Neither
# runs the recompiler here, so the test links only them and the device-free static key code.
add_executable(compute_translation_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ComputeTranslationTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStaticKey.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/computeTranslation.cpp"
)
target_link_libraries(compute_translation_tests common Vulkan::Headers fmt::fmt xxhash xbyak::xbyak)
target_include_directories(compute_translation_tests PRIVATE ${inc_headers})
if (KYTY_CLANG_CL)
	# The recompiler headers bring in the deprecated codecvt use the store worker test silences too.
	target_compile_options(compute_translation_tests PRIVATE -Wno-deprecated-declarations)
endif()

# The prescan variant tracker and descriptor validation. Both are headers over the recompiler's
# resource types, so the test links no translation, no walk and no device. The image conversion
# predicate the validation shares with the emitter needs the texture format remap table.
add_executable(prescan_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PrescanTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
)
target_link_libraries(prescan_tests common fmt::fmt xxhash)
target_include_directories(prescan_tests PRIVATE ${inc_headers})

# The prescan's scanner-to-preparer ring, its answer table and the preparer's state. All are headers, so
# the test links nothing of the walk, the recompiler or the device.
add_executable(prescan_handoff_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PrescanHandoffTests.cpp"
)
target_link_libraries(prescan_handoff_tests common fmt::fmt xxhash)
target_include_directories(prescan_handoff_tests PRIVATE ${inc_headers})

# The prescan's scanner thread. It reads through a reader the test supplies and posts to a hand-off the
# test drains, so it links only the walk, the register decoder and the scanner. The differential gives
# the report its names for the causes of doubt.
add_executable(prescan_scanner_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PrescanScannerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/csRegisters.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/submitScan.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/scanDifferential.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/prescanScanner.cpp"
)
target_link_libraries(prescan_scanner_tests common fmt::fmt xxhash)
target_include_directories(prescan_scanner_tests PRIVATE ${inc_headers})

# The predictor rules, the header slots and the module id need no device and no translated program.
add_executable(shader_store_predictor_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStorePredictorTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/moduleId.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/computePredictor.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/headerSlots.cpp"
)
target_link_libraries(shader_store_predictor_tests common fmt::fmt xxhash)
target_include_directories(shader_store_predictor_tests PRIVATE ${inc_headers})

# The route K classifier reads IR values, so its test builds a small plan out of the typed IR, like
# scalar_provenance_tests does. It needs no graphics device.
add_executable(shader_store_keys_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreKeysTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/resourceKeys.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(shader_store_keys_tests common fmt::fmt xbyak::xbyak)
target_include_directories(shader_store_keys_tests PRIVATE ${inc_headers})

# The store record format. It encodes and decodes a plan, so it links the plan code like the keys
# test does, plus the zstd frame and the checksum. It needs no graphics device.
add_executable(shader_store_record_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreRecordTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStoreRecord.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(shader_store_record_tests common fmt::fmt xbyak::xbyak xxhash libzstd_static)
target_include_directories(shader_store_record_tests PRIVATE ${inc_headers})

# The tailer that follows a store worker's output file. It checks the generated file's header and
# unframes records, so it links the same pieces as the record test plus the worker's file format. It
# needs no graphics device.
add_executable(shader_store_tailer_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreTailerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/frameTailer.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeWorkerFormat.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStoreRecord.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(shader_store_tailer_tests common fmt::fmt xbyak::xbyak xxhash libzstd_static)
target_include_directories(shader_store_tailer_tests PRIVATE ${inc_headers})

# The frames of the Stream worker mode and the pipe they travel through. The pipes are made inside
# the test, so it starts no worker, links no recompiler and needs no device. The request format is
# linked for the Stream mode's refusal.
add_executable(shader_store_stream_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreStreamTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/streamProtocol.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/streamChannel.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeWorkerFormat.cpp"
)
target_link_libraries(shader_store_stream_tests common Vulkan::Headers fmt::fmt xxhash xbyak::xbyak)
target_include_directories(shader_store_stream_tests PRIVATE ${inc_headers})
if (KYTY_CLANG_CL)
	# The request format's serializer brings in the recompiler headers, with the deprecated codecvt
	# use the translation test silences too.
	target_compile_options(shader_store_stream_tests PRIVATE -Wno-deprecated-declarations)
endif()

# The supervisor of a store worker. It starts this test binary as a fake worker, so it links no
# recompiler and needs no device.
add_executable(shader_store_supervisor_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreSupervisorTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeSupervisor.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeGenerationGate.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/frameTailer.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/progressLog.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeWorkerFormat.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStoreRecord.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(shader_store_supervisor_tests common fmt::fmt xbyak::xbyak xxhash libzstd_static)
target_include_directories(shader_store_supervisor_tests PRIVATE ${inc_headers})

# The supervisor of a Stream worker. It starts this test binary as a fake worker that crashes or
# hangs on demand, so it links no recompiler and needs no device. The generator's supervisor and
# what it needs are linked for WorkerProgramPath, and the typed IR for its module records.
add_executable(shader_store_stream_supervisor_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreStreamSupervisorTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/streamSupervisor.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/streamProtocol.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/streamChannel.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeSupervisor.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/frameTailer.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/progressLog.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeWorkerFormat.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStoreRecord.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/Block.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/Program.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/Type.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/Value.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
)
target_link_libraries(shader_store_stream_supervisor_tests common Vulkan::Headers fmt::fmt xbyak::xbyak xxhash libzstd_static)
target_include_directories(shader_store_stream_supervisor_tests PRIVATE ${inc_headers})
if (KYTY_CLANG_CL)
	# The worker's exit codes in shaderStoreWorker.h bring in the recompiler headers, with the
	# deprecated codecvt use the stream tests silence too.
	target_compile_options(shader_store_stream_supervisor_tests PRIVATE -Wno-deprecated-declarations)
endif()

# The decisions of the drain that takes generated modules into the pipeline cache. They need no
# graphics device, so this target links only the planner.
add_executable(generated_drain_planner_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GeneratedDrainPlannerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/generatedDrainPlanner.cpp"
)
target_include_directories(generated_drain_planner_tests PRIVATE ${inc_headers})

# The classifier that says why a runtime translation was not served by generation, and the rows of
# the first-run report made from the pipeline cache's rows. Both are pure and need no device.
add_executable(generation_miss_classifier_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GenerationMissClassifierTests.cpp"
)
target_link_libraries(generation_miss_classifier_tests common fmt::fmt)
target_include_directories(generation_miss_classifier_tests PRIVATE ${inc_headers})

# Where the generator reads the title's eboot from, including a title in a .zar made by the writer of
# the archive tests. It needs no graphics device.
add_executable(shader_store_eboot_source_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreEbootSourceTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/ebootSource.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeGenerationGate.cpp"
)
target_link_libraries(shader_store_eboot_source_tests common ZArchive::zarchive)
target_include_directories(shader_store_eboot_source_tests PRIVATE ${inc_headers})

# The reader of a store worker's progress log. It reads text lines through Common::File and needs no
# graphics device.
add_executable(shader_store_progress_log_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreProgressLogTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/progressLog.cpp"
)
target_link_libraries(shader_store_progress_log_tests common fmt::fmt)
target_include_directories(shader_store_progress_log_tests PRIVATE ${inc_headers})

# The store worker translates and compiles real programs, so this links the whole recompiler. It
# still needs no graphics device and none of the emulator's libraries, so it does not build the
# emulator's sources as the full emulator tests do.
file(GLOB shader_store_worker_recompiler_src CONFIGURE_DEPENDS
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.cpp"
)
file(GLOB shader_store_worker_storegen_src CONFIGURE_DEPENDS
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/*.cpp"
)
add_executable(shader_store_worker_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreWorkerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/csRegisters.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/moduleId.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStaticKey.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStoreRecord.cpp"
	${shader_store_worker_storegen_src}
	${shader_store_worker_recompiler_src}
)
target_link_libraries(shader_store_worker_tests common Vulkan::Headers fmt::fmt xxhash xbyak::xbyak spirv-tools-opt spirv-tools libzstd_static)
target_include_directories(shader_store_worker_tests PRIVATE ${inc_headers})
if (KYTY_CLANG_CL)
	set_source_files_properties(${shader_store_worker_recompiler_src} PROPERTIES COMPILE_FLAGS "-Wno-pragma-pack -Wno-deprecated-declarations -D_TIMESPEC_DEFINED")
endif()

# The Stream mode of the store worker, its cache of translated programs and the compile path it
# shares with the generator. The worker runs on a thread of the test and talks to it through a real
# pipe, and it translates real programs, so this links the whole recompiler like the worker test.
# It still needs no graphics device.
add_executable(shader_store_stream_worker_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreStreamWorkerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/csRegisters.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/moduleId.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStaticKey.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStoreRecord.cpp"
	${shader_store_worker_storegen_src}
	${shader_store_worker_recompiler_src}
)
target_link_libraries(shader_store_stream_worker_tests common Vulkan::Headers fmt::fmt xxhash xbyak::xbyak spirv-tools-opt spirv-tools libzstd_static)
target_include_directories(shader_store_stream_worker_tests PRIVATE ${inc_headers})
if (KYTY_CLANG_CL)
	# The recompiler headers bring in the deprecated codecvt use the stream test silences too.
	target_compile_options(shader_store_stream_worker_tests PRIVATE -Wno-deprecated-declarations)
endif()

# The prescan's preparer over real programs. Its in-process transport runs the Stream worker's own
# answers here, so this links the whole recompiler like the stream worker test, and still no device.
# Its supervised tests start this binary as a fake helper, and its smoke test starts the real
# emulator in its worker mode, so the emulator is built first.
add_executable(prescan_preparer_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PrescanPreparerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/prescanPreparer.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/csRegisters.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/moduleId.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStaticKey.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStoreRecord.cpp"
	${shader_store_worker_storegen_src}
	${shader_store_worker_recompiler_src}
)
target_link_libraries(prescan_preparer_tests common Vulkan::Headers fmt::fmt xxhash xbyak::xbyak spirv-tools-opt spirv-tools libzstd_static)
target_include_directories(prescan_preparer_tests PRIVATE ${inc_headers})
add_dependencies(prescan_preparer_tests kyty_emulator)
target_compile_definitions(prescan_preparer_tests PRIVATE
	KYTY_EMULATOR_PATH="$<TARGET_FILE:kyty_emulator>")
if (KYTY_CLANG_CL)
	# The recompiler headers bring in the deprecated codecvt use the stream worker test silences too.
	target_compile_options(prescan_preparer_tests PRIVATE -Wno-deprecated-declarations)
endif()

# The store worker mode of the emulator binary. The test starts the real kyty_emulator through
# ChildProcess, so it only builds the requests and eboots it sends and links no recompiler. The
# stream pipe is linked for the Stream mode's smoke test. The emulator is built first, since the
# test runs it.
add_executable(shader_store_dispatch_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderStoreDispatchTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/storeWorkerFormat.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/progressLog.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/streamProtocol.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/storegen/streamChannel.cpp"
)
add_dependencies(shader_store_dispatch_tests kyty_emulator)
target_compile_definitions(shader_store_dispatch_tests PRIVATE
	KYTY_EMULATOR_PATH="$<TARGET_FILE:kyty_emulator>")
target_link_libraries(shader_store_dispatch_tests common Vulkan::Headers fmt::fmt xxhash xbyak::xbyak)
target_include_directories(shader_store_dispatch_tests PRIVATE ${inc_headers})

set(gpu_test_generated_dir "${PROJECT_BINARY_DIR}/gpu_test_shaders")
set(gpu_test_ms_depth_source
	"${gpu_tiler_shader_dir}/gpu_test_ms_depth.comp")
set(gpu_test_ms_depth_spv
	"${gpu_test_generated_dir}/gpu_test_ms_depth.spv")
set(gpu_test_ms_depth_header
	"${gpu_test_generated_dir}/gpu_test_ms_depth_spv.h")
add_custom_command(
	OUTPUT "${gpu_test_ms_depth_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_test_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${gpu_test_ms_depth_spv}" "${gpu_test_ms_depth_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${gpu_test_ms_depth_spv}
	        -DOUTPUT=${gpu_test_ms_depth_header} -DSYMBOL=GPU_TEST_MS_DEPTH_SPV
	        -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${gpu_test_ms_depth_source}"
	VERBATIM
)
target_sources(shader_recompiler_compute_tests PRIVATE
	"${gpu_test_ms_depth_header}")

add_kyty_full_emulator_test(virtual_memory_allocation_tests "${KYTY_TESTS_DIR}/VirtualMemoryAllocationTests.cpp")
add_library(kyty_test_gtest STATIC EXCLUDE_FROM_ALL
	"${KYTY_THIRD_PARTY_DIR}/fmt/test/gtest/gmock-gtest-all.cc")
target_include_directories(kyty_test_gtest PUBLIC "${KYTY_THIRD_PARTY_DIR}/fmt/test/gtest")
target_sources(virtual_memory_allocation_tests PRIVATE "${KYTY_TESTS_DIR}/TrinityTests.cpp")
target_link_libraries(virtual_memory_allocation_tests kyty_test_gtest)

add_kyty_full_emulator_test(net_resolver_tests "${KYTY_TESTS_DIR}/NetResolverTests.cpp")
target_link_libraries(net_resolver_tests kyty_test_gtest)

add_executable(psml_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PsmlTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/libPsml.cpp"
	"${KYTY_SOURCE_DIR}/loader/symbolDatabase.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp")
target_link_libraries(psml_tests common kyty_test_gtest)
target_include_directories(psml_tests PRIVATE ${inc_headers})
target_compile_definitions(virtual_memory_allocation_tests PRIVATE
	KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS=1)

add_executable(gpu_blit_shader_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GpuBlitShaderTests.cpp"
	"${gpu_blit_generated_dir}/gpu_mesh_indirect_spv.h")
target_link_libraries(gpu_blit_shader_tests kyty_test_gtest)
target_include_directories(gpu_blit_shader_tests PRIVATE ${inc_headers})

# These tests use exceptions.
if(NOT KYTY_CLANG_CL)
	foreach(kyty_exception_test scalar_provenance_tests resource_tracking_tests
			virtual_memory_allocation_tests)
		target_compile_options(${kyty_exception_test} PRIVATE -fexceptions)
	endforeach()
endif()

if(BUILD_TESTING)
	add_test(NAME emulator_user_name_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --user-name 1234567890123456)
	add_test(NAME emulator_user_name_too_long
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --user-name 12345678901234567)
	set_tests_properties(emulator_user_name_too_long PROPERTIES WILL_FAIL TRUE)
	add_test(NAME emulator_partial_buffer_refresh_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --partial-buffer-refresh)
	add_test(NAME emulator_present_mode_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --present-mode Immediate)
	add_test(NAME emulator_present_mode_invalid
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --present-mode Invalid)
	set_tests_properties(emulator_present_mode_invalid PROPERTIES WILL_FAIL TRUE)
	add_test(NAME emulator_select_guarded_loads_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --select-guarded-loads)
	add_test(NAME emulator_single_compare_record_guard_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --single-compare-record-guard)
	add_test(NAME emulator_elide_exec_selects_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --elide-exec-selects)
	add_test(NAME emulator_structured_jump_tables_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --structured-jump-tables)
	add_test(NAME emulator_scan_differential_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --scan-differential)
	add_test(NAME emulator_prescan_compute_builds_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --prescan-compute-builds)
	add_test(NAME emulator_audio_compile_hold_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --audio-compile-hold)
	add_test(NAME emulator_first_run_report_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --first-run-report)
	add_test(NAME emulator_cold_pipelines_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --cold-pipelines)
	add_test(NAME emulator_async_pipelines_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --async-pipelines)
	add_test(NAME emulator_async_pipelines_depth_only_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --async-pipelines-depth-only)
	add_test(NAME emulator_background_warm_up_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --background-warm-up)
	add_test(NAME emulator_generate_shader_store_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --generate-shader-store)
	add_test(NAME emulator_warm_workers_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --background-warm-up --warm-workers 3)
	# The limits are the first and last values the option takes, and the values just outside them.
	add_test(NAME emulator_warm_workers_max_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --warm-workers 16)
	foreach(warm_workers_bad IN ITEMS zero over_max not_a_number missing)
		if(warm_workers_bad STREQUAL "zero")
			set(warm_workers_args 0)
		elseif(warm_workers_bad STREQUAL "over_max")
			set(warm_workers_args 17)
		elseif(warm_workers_bad STREQUAL "not_a_number")
			set(warm_workers_args three)
		else()
			set(warm_workers_args "")
		endif()
		add_test(NAME emulator_warm_workers_${warm_workers_bad}
		         COMMAND $<TARGET_FILE:kyty_emulator> --help --warm-workers ${warm_workers_args})
		set_tests_properties(emulator_warm_workers_${warm_workers_bad} PROPERTIES WILL_FAIL TRUE)
	endforeach()
	add_test(NAME emulator_warm_workers_help
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --warm-workers 3)
	set_tests_properties(emulator_warm_workers_help PROPERTIES
	                     PASS_REGULAR_EXPRESSION "--warm-workers <1-16> +Run this many warm-up workers")
	# A pass regex ignores the exit code, and the usage printed on a parse error also has this
	# line, so the help text is checked separately from the flag parsing above.
	add_test(NAME emulator_generate_shader_store_help
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --generate-shader-store)
	set_tests_properties(emulator_generate_shader_store_help PROPERTIES
	                     PASS_REGULAR_EXPRESSION "--generate-shader-store +Predict a Motor title")
	add_test(NAME emulator_deferred_pipeline_draws_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --deferred-pipeline-draws)
	add_test(NAME emulator_pause_warm_up_for_draws_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --pause-warm-up-for-draws)
	add_test(NAME emulator_deferred_non_push_draws_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --deferred-non-push-draws)
	add_test(NAME emulator_graphics_pipeline_library_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --graphics-pipeline-library)
	add_test(NAME emulator_skip_redundant_unmap_drains_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --skip-redundant-unmap-drains)
	add_test(NAME emulator_keep_gpu_cache_on_reuse_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --keep-gpu-cache-on-reuse)
	add_test(NAME emulator_keep_gpu_textures_on_reuse_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --keep-gpu-textures-on-reuse)
	add_test(NAME emulator_gpu_submit_cadence_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --gpu-submit-cadence)
	add_test(NAME emulator_native_float_minmax_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --native-float-minmax)
	add_test(NAME emulator_batch_hot_readbacks_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --batch-hot-readbacks)
	add_test(NAME ime_dialog COMMAND $<TARGET_FILE:ime_dialog_tests>)
	add_test(NAME shader_cfg COMMAND $<TARGET_FILE:shader_cfg_tests>)
	add_test(NAME scalar_provenance COMMAND $<TARGET_FILE:scalar_provenance_tests>)
	add_test(NAME image_page_table COMMAND $<TARGET_FILE:image_page_table_tests>)
	add_test(NAME memory_tracker COMMAND $<TARGET_FILE:memory_tracker_tests>)
	add_test(NAME page_manager COMMAND $<TARGET_FILE:page_manager_tests>)
	add_test(NAME readback_guard COMMAND $<TARGET_FILE:readback_guard_tests>)
	# Shorter than the 60 second wait slice in the tests, so a missing wake-up fails instead of
	# passing slowly when the slice runs out.
	set_tests_properties(readback_guard PROPERTIES TIMEOUT 30)
	add_test(NAME build_flags COMMAND $<TARGET_FILE:build_flags_tests>)
	add_test(NAME archive_file COMMAND $<TARGET_FILE:archive_file_tests>)
	add_test(NAME trophy_system COMMAND $<TARGET_FILE:trophy_system_tests>)
	add_test(NAME avplayer_file COMMAND $<TARGET_FILE:avplayer_file_tests>)
	add_test(NAME vdecsw COMMAND $<TARGET_FILE:vdecsw_tests>)
	add_test(NAME bit_array COMMAND $<TARGET_FILE:bit_array_tests>)
	add_test(NAME lru_cache COMMAND $<TARGET_FILE:lru_cache_tests>)
	add_test(NAME common COMMAND $<TARGET_FILE:common_tests>)
	add_test(NAME shader_vertex_metadata COMMAND $<TARGET_FILE:shader_vertex_metadata_tests>)
	add_test(NAME resource_materialization COMMAND $<TARGET_FILE:resource_materialization_tests>)
	add_test(NAME resource_tracking COMMAND $<TARGET_FILE:resource_tracking_tests>)
	add_test(NAME event_queue_lifetime COMMAND $<TARGET_FILE:event_queue_lifetime_tests>)
	add_test(NAME kernel_file_system COMMAND $<TARGET_FILE:kernel_file_system_tests>)
	add_test(NAME gpu_stall COMMAND $<TARGET_FILE:gpu_stall_tests>)
	# A missed wakeup shows up as a hang, so fail fast instead of at the default timeout.
	set_tests_properties(gpu_stall PROPERTIES TIMEOUT 30)
	add_test(NAME child_process COMMAND $<TARGET_FILE:child_process_tests>)
	# A child that is never killed makes ctest wait on it, so cap the run. The suite takes a few seconds.
	set_tests_properties(child_process PROPERTIES TIMEOUT 120)
	add_test(NAME first_run_report COMMAND $<TARGET_FILE:first_run_report_tests>)
	add_test(NAME sync_on_address COMMAND $<TARGET_FILE:sync_on_address_tests>)
	add_test(NAME audio_out2_port COMMAND $<TARGET_FILE:audio_out2_port_tests>)
	# A push or query wrongly held by GpuStall hangs, so fail fast instead of at the default timeout.
	set_tests_properties(audio_out2_port PROPERTIES TIMEOUT 60)
	add_test(NAME audio_out_timing COMMAND $<TARGET_FILE:audio_out_timing_tests>)
	# A submit or query wrongly held by GpuStall hangs, so fail fast instead of at the default timeout.
	set_tests_properties(audio_out_timing PROPERTIES TIMEOUT 60)
	add_test(NAME ngs2_sampler COMMAND $<TARGET_FILE:ngs2_sampler_tests>)
	add_test(NAME ajm_resample COMMAND $<TARGET_FILE:ajm_resample_tests>)
	add_test(NAME pad_haptics COMMAND $<TARGET_FILE:pad_haptics_tests>)
	add_test(NAME controller_settings COMMAND $<TARGET_FILE:controller_settings_tests>)
	add_test(NAME host_input COMMAND $<TARGET_FILE:host_input_tests>)
	add_test(NAME save_data_memory COMMAND $<TARGET_FILE:save_data_memory_tests>)
	add_test(NAME png_enc COMMAND $<TARGET_FILE:png_enc_tests>)
	add_test(NAME jpeg_dec COMMAND $<TARGET_FILE:jpeg_dec_tests>)
	add_test(NAME jpeg_enc COMMAND $<TARGET_FILE:jpeg_enc_tests>)
	add_test(NAME occlusion_queries COMMAND $<TARGET_FILE:occlusion_queries_tests>)
	add_test(NAME pipeline_build_queue COMMAND $<TARGET_FILE:pipeline_build_queue_tests>)
	# A lost wakeup or a job run twice shows up as a hang, so fail fast.
	set_tests_properties(pipeline_build_queue PROPERTIES TIMEOUT 60)
	add_test(NAME thread_priority COMMAND $<TARGET_FILE:thread_priority_tests>)
	add_test(NAME pipeline_library COMMAND $<TARGET_FILE:pipeline_library_tests>)
	add_test(NAME pipeline_parts COMMAND $<TARGET_FILE:pipeline_parts_tests>)
	add_test(NAME descriptor_set_layout_cache COMMAND $<TARGET_FILE:descriptor_set_layout_cache_tests>)
	# A second build of one library, or a thread that never wakes, shows up as a wrong count or a hang.
	add_test(NAME graphics_library_cache COMMAND $<TARGET_FILE:graphics_library_cache_tests>)
	set_tests_properties(graphics_library_cache PROPERTIES TIMEOUT 60)
	add_test(NAME ces COMMAND $<TARGET_FILE:ces_tests>)
	add_test(NAME save_data_dialog COMMAND $<TARGET_FILE:save_data_dialog_tests>)
	add_test(NAME http_uri_parse COMMAND $<TARGET_FILE:http_uri_parse_tests>)
	add_test(NAME rtc COMMAND $<TARGET_FILE:rtc_tests>)
	add_test(NAME shader_recompiler_compute COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	add_test(NAME shader_store_scan COMMAND $<TARGET_FILE:shader_store_scan_tests>)
	add_test(NAME submit_scan COMMAND $<TARGET_FILE:submit_scan_tests>)
	add_test(NAME scan_differential COMMAND $<TARGET_FILE:scan_differential_tests>)
	add_test(NAME prescan COMMAND $<TARGET_FILE:prescan_tests>)
	# The whole test runs in about a second, so a ring that loses track of an item and leaves the
	# consumer spinning shows up as a hang, which the timeout ends.
	add_test(NAME prescan_handoff COMMAND $<TARGET_FILE:prescan_handoff_tests>)
	set_tests_properties(prescan_handoff PROPERTIES TIMEOUT 60)
	# A scanner thread that never wakes or never stops shows up as a hang, which the timeout ends.
	add_test(NAME prescan_scanner COMMAND $<TARGET_FILE:prescan_scanner_tests>)
	set_tests_properties(prescan_scanner PROPERTIES TIMEOUT 60)
	# A preparer that never wakes or never stops shows up as a hang, which the test's watchdog ends.
	add_test(NAME prescan_preparer COMMAND $<TARGET_FILE:prescan_preparer_tests>)
	set_tests_properties(prescan_preparer PROPERTIES TIMEOUT 300)
	add_test(NAME compute_translation COMMAND $<TARGET_FILE:compute_translation_tests>)
	add_test(NAME shader_store_predictor COMMAND $<TARGET_FILE:shader_store_predictor_tests>)
	add_test(NAME shader_store_keys COMMAND $<TARGET_FILE:shader_store_keys_tests>)
	add_test(NAME shader_store_record COMMAND $<TARGET_FILE:shader_store_record_tests>)
	add_test(NAME shader_store_tailer COMMAND $<TARGET_FILE:shader_store_tailer_tests>)
	add_test(NAME shader_store_stream COMMAND $<TARGET_FILE:shader_store_stream_tests>)
	add_test(NAME shader_store_progress_log COMMAND $<TARGET_FILE:shader_store_progress_log_tests>)
	add_test(NAME shader_store_supervisor COMMAND $<TARGET_FILE:shader_store_supervisor_tests>)
	add_test(NAME shader_store_stream_supervisor COMMAND $<TARGET_FILE:shader_store_stream_supervisor_tests>)
	# A child that is never ended or a Stop that waits shows up as a hang, which the watchdog ends.
	set_tests_properties(shader_store_stream_supervisor PROPERTIES TIMEOUT 300)
	add_test(NAME generated_drain_planner COMMAND $<TARGET_FILE:generated_drain_planner_tests>)
	add_test(NAME generation_miss_classifier COMMAND $<TARGET_FILE:generation_miss_classifier_tests>)
	add_test(NAME shader_store_eboot_source COMMAND $<TARGET_FILE:shader_store_eboot_source_tests>)
	set_tests_properties(shader_store_supervisor PROPERTIES TIMEOUT 300)
	add_test(NAME shader_store_worker COMMAND $<TARGET_FILE:shader_store_worker_tests>)
	add_test(NAME shader_store_stream_worker COMMAND $<TARGET_FILE:shader_store_stream_worker_tests>)
	# A worker or a read that never returns shows up as a hang, which the test's own watchdog ends first.
	set_tests_properties(shader_store_stream_worker PROPERTIES TIMEOUT 300)
	add_test(NAME shader_store_dispatch COMMAND $<TARGET_FILE:shader_store_dispatch_tests>)
	set_tests_properties(shader_store_dispatch PROPERTIES TIMEOUT 300)
	add_test(NAME shader_recompiler_store_generation
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --store-generation-only)
	set_tests_properties(shader_recompiler_store_generation PROPERTIES TIMEOUT 600)
	add_test(NAME shader_recompiler_generated_modules
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --generated-modules-only)
	# The pre-scan's translation setup and static key against the runtime's, on the real device.
	add_test(NAME shader_recompiler_prescan_setup
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --prescan-setup-only)
	set_tests_properties(shader_recompiler_prescan_setup PROPERTIES TIMEOUT 120)
	# The pre-scan's intake on the real device: the GPU thread never waits for it, and the pause for
	# draws never holds its set.
	add_test(NAME shader_recompiler_prescan_intake
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --prescan-intake-only)
	set_tests_properties(shader_recompiler_prescan_intake PROPERTIES TIMEOUT 180)
	# The pre-scan run from GuestGpu: the scanner wiring, the late skip, shutdown and the one producer.
	add_test(NAME shader_recompiler_prescan_gpu
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --prescan-gpu-only)
	set_tests_properties(shader_recompiler_prescan_gpu PROPERTIES TIMEOUT 180)
	add_test(NAME shader_recompiler_submit_cadence
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --submit-cadence-only)
	add_test(NAME shader_recompiler_submit_cadence_gpu
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --submit-cadence-gpu-only)
	set_tests_properties(shader_recompiler_submit_cadence_gpu PROPERTIES TIMEOUT 60)
	add_test(NAME shader_recompiler_submit_cadence_occlusion
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --submit-cadence-occlusion-only)
	set_tests_properties(shader_recompiler_submit_cadence_occlusion PROPERTIES TIMEOUT 60)
	add_test(NAME shader_recompiler_cold_pipelines
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cold-pipelines-only)
	set_tests_properties(shader_recompiler_cold_pipelines PROPERTIES TIMEOUT 60)
	add_test(NAME shader_recompiler_split_layout
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --split-layout-only)
	set_tests_properties(shader_recompiler_split_layout PROPERTIES TIMEOUT 120)
	add_test(NAME shader_recompiler_split_layout_sync
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --split-layout-sync-only)
	set_tests_properties(shader_recompiler_split_layout_sync PROPERTIES TIMEOUT 180)
	add_test(NAME shader_recompiler_async_pipelines
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --async-pipelines-only)
	# A build that never finishes or a worker that is never released shows up as a hang.
	set_tests_properties(shader_recompiler_async_pipelines PROPERTIES TIMEOUT 120)
	add_test(NAME shader_recompiler_tessellation
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --tessellation-only)
	add_test(NAME shader_recompiler_float_min_max
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --float-min-max-only)
	add_test(NAME shader_recompiler_native_float_minmax
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --native-float-minmax-only)
	add_test(NAME shader_recompiler_vop3_conversions
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --vop3-conversions-only)
	add_test(NAME shader_recompiler_quadmask
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --quadmask-only)
	add_test(NAME shader_recompiler_integer16_ternary
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --integer16-ternary-only)
	add_test(NAME shader_recompiler_mad_u16
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --mad-u16-only)
	add_test(NAME shader_recompiler_mul_hi24
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --mul-hi24-only)
	add_test(NAME shader_recompiler_elide_exec_selects
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --elide-exec-selects-only)
	add_test(NAME shader_recompiler_alignbyte
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alignbyte-only)
	add_test(NAME graphics_pipeline_rasterization
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --polygon-mode-only)
	add_test(NAME graphics_draw_offsets
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-offset-only)
	add_test(NAME virtual_memory_allocation
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests>)
	add_test(NAME trinity_mode
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --trinity-only)
	add_test(NAME psml COMMAND $<TARGET_FILE:psml_tests>)
	add_test(NAME pssr COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --pssr-only)
	add_test(NAME guest_red_zone_patcher
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --red-zone-patcher-only)
	add_test(NAME command_scheduler_timeline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --scheduler-only)
	add_test(NAME stream_buffer_ring
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --stream-buffer-only)
	add_test(NAME gpu_command_lane
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-command-lane-only)
	add_test(NAME pm4_flip
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --flip-only)
	add_test(NAME video_out_hdr10_to_sdr
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --video-out-only)
	add_test(NAME depth_bits
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --depth-bits-only)
	add_test(NAME srgb_texture
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --srgb-texture-only)
	add_test(NAME device_creation
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --device-creation-only)
	add_test(NAME msaa_storage
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --msaa-storage-only)
	add_test(NAME pm4_context_state
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --context-state-only)
	add_test(NAME gpu_tiler
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-tiler-only)
	add_test(NAME image_transfer_layout
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-layout-only)

	add_test(NAME texture_cache_layered_image
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --layered-image-only)
	add_test(NAME host_image_allocation
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --host-image-allocation-only)
	add_test(NAME texture_cache_image_views
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-view-cache-only)
	add_test(NAME texture_cache_storage_sampled
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --storage-sampled-only)
	add_test(NAME texture_cache_depth_readback
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --depth-readback-only)
	add_test(NAME compute_meta_clear_classification
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --compute-meta-clear-only)
	add_test(NAME buffer_cache_dirty_gc
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-gc-only)
	add_test(NAME hot_readback_table
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --hot-readback-only)
	add_test(NAME color_render_target_1d
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --rt1d-only)

	add_test(NAME texture_cache_image_overlap
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-overlap-only)
	add_test(NAME texture_cache_htile_clear
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --htile-clear-only)
	add_test(NAME buffer_cache_ranges
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-range-only)

	add_custom_target(kyty_tests DEPENDS
		ime_dialog_tests
		shader_cfg_tests
		scalar_provenance_tests
		image_page_table_tests
		memory_tracker_tests
		page_manager_tests
		readback_guard_tests
		build_flags_tests
		archive_file_tests
		trophy_system_tests
		avplayer_file_tests
		vdecsw_tests
		bit_array_tests
		lru_cache_tests
		common_tests
		shader_vertex_metadata_tests
		resource_materialization_tests
		resource_tracking_tests
		event_queue_lifetime_tests
		kernel_file_system_tests
		gpu_stall_tests
		child_process_tests
		first_run_report_tests
		sync_on_address_tests
		shader_recompiler_compute_tests
		shader_store_scan_tests
		submit_scan_tests
		scan_differential_tests
		prescan_tests
		prescan_handoff_tests
		prescan_scanner_tests
		prescan_preparer_tests
		compute_translation_tests
		shader_store_predictor_tests
		shader_store_keys_tests
		shader_store_record_tests
		shader_store_tailer_tests
		shader_store_stream_tests
		shader_store_progress_log_tests
		shader_store_supervisor_tests
		shader_store_stream_supervisor_tests
		generated_drain_planner_tests
		generation_miss_classifier_tests
		shader_store_eboot_source_tests
		shader_store_worker_tests
		shader_store_stream_worker_tests
		shader_store_dispatch_tests
		virtual_memory_allocation_tests
		audio_out2_port_tests
		audio_out_timing_tests
		ngs2_sampler_tests
		ajm_resample_tests
		pad_haptics_tests
		controller_settings_tests
		host_input_tests
		save_data_memory_tests
		png_enc_tests
		jpeg_dec_tests
		jpeg_enc_tests
		occlusion_queries_tests
		pipeline_build_queue_tests
		thread_priority_tests
		pipeline_library_tests
		pipeline_parts_tests
		descriptor_set_layout_cache_tests
		graphics_library_cache_tests
		ces_tests
		save_data_dialog_tests
		http_uri_parse_tests
		rtc_tests
	)
endif()

# Test Darwin signal-context restoration for SSE4a instructions.
if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
		(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")))
	add_executable(macos_sse4a_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/macosSse4aTests.cpp"
		"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
		"${KYTY_SOURCE_DIR}/loader/guestInstructionPatcher.cpp"
	)
	target_link_libraries(macos_sse4a_tests common fmt::fmt Zydis::Zydis xbyak::xbyak)
	target_include_directories(macos_sse4a_tests PRIVATE ${inc_headers})
	if(BUILD_TESTING)
		add_test(NAME macos_sse4a COMMAND $<TARGET_FILE:macos_sse4a_tests>)
		set_tests_properties(macos_sse4a PROPERTIES TIMEOUT 15)
		add_dependencies(kyty_tests macos_sse4a_tests)
	endif()
endif()

add_executable(kyty_emulator "${KYTY_SOURCE_DIR}/main.cpp" ${kyty_emulator_src})
configure_macos_guest_address_space(kyty_emulator)

set(KYTY_TROPHY_UNLOCK_SOUND "${CMAKE_CURRENT_SOURCE_DIR}/assets/sounds/trophy-unlock.wav")
add_custom_target(kyty_trophy_sound
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:kyty_emulator>/assets/sounds"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_TROPHY_UNLOCK_SOUND}"
		"$<TARGET_FILE_DIR:kyty_emulator>/assets/sounds/trophy-unlock.wav"
	VERBATIM)
add_dependencies(kyty_emulator kyty_trophy_sound)
install(FILES "${KYTY_TROPHY_UNLOCK_SOUND}" DESTINATION assets/sounds)

add_custom_target(kyty_touch_cursors
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/assets/cursors"
		"$<TARGET_FILE_DIR:kyty_emulator>/assets/cursors"
	VERBATIM)
add_dependencies(kyty_emulator kyty_touch_cursors)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/assets/cursors" DESTINATION assets)

set(KYTY_NOTO_SANS_FONTS
	"${KYTY_NOTO_SANS_DIR}/NotoSans-Regular.ttf"
	"${KYTY_NOTO_SANS_DIR}/NotoSans-Bold.ttf"
	"${KYTY_NOTO_SANS_DIR}/NotoSans-Italic.ttf"
	"${KYTY_NOTO_SANS_DIR}/NotoSans-BoldItalic.ttf")
add_custom_target(kyty_noto_sans_fonts
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:kyty_emulator>/assets/fonts"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${KYTY_NOTO_SANS_FONTS}
		"$<TARGET_FILE_DIR:kyty_emulator>/assets/fonts"
	VERBATIM)
add_dependencies(kyty_emulator kyty_noto_sans_fonts)
install(FILES ${KYTY_NOTO_SANS_FONTS} DESTINATION assets/fonts)

target_link_libraries(kyty_emulator ${kyty_emulator_link_libraries})
if (WIN32)
	target_link_libraries(kyty_emulator opengl32)
	target_link_libraries(kyty_emulator iphlpapi)
	target_link_libraries(kyty_emulator setupapi)
	target_link_libraries(kyty_emulator ws2_32)
	target_link_libraries(kyty_emulator dxgi)
endif()
if (KYTY_CLANG_CL)
	target_link_libraries(kyty_emulator onecore)
endif()
if (CLANG AND NOT KYTY_CLANG_CL)
	target_link_libraries(kyty_emulator pthread)
endif()
# dlopen/dlsym/dladdr for RenderDoc.
target_link_libraries(kyty_emulator ${CMAKE_DL_LIBS})
target_include_directories(kyty_emulator PRIVATE ${inc_headers})

clang_tidy_check(kyty_emulator "" "${check_headers}" "${inc_headers}")

include_what_you_use(kyty_emulator "${inc_headers}")

set(KYTY_EMULATOR_MAP_NAME "kyty_emulator_${KYTY_COMPILER_ID}_${KYTY_LINKER_ID}.map")

set(KYTY_EMULATOR_MAP_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/${KYTY_EMULATOR_MAP_NAME}")
set(KYTY_EMULATOR_PDB_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/kyty_emulator.pdb")

if(KYTY_CLANG_CL)
	target_link_options(kyty_emulator PRIVATE
		"/DYNAMICBASE:NO"
		"/DEBUG:FULL"
		"/PDB:${KYTY_EMULATOR_PDB_LINK_PATH}"
		"/lldmap:${KYTY_EMULATOR_MAP_LINK_PATH}"
	)
	add_custom_command(TARGET kyty_emulator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:kyty_emulator>/libwinpthread-1.dll)
elseif(WIN32 OR LINUX)
	set_target_properties(kyty_emulator PROPERTIES LINK_FLAGS "${KYTY_LD_OPTIONS}")
	# LINK_FLAGS is a space-delimited string: a -Map path containing spaces would
	# be split into bogus arguments, so pass it as a discrete link option instead.
	target_link_options(kyty_emulator PRIVATE "-Wl,-Map=${KYTY_EMULATOR_MAP_LINK_PATH}")
endif()

if(LINUX)
    target_link_options(kyty_emulator PRIVATE
        $<$<CONFIG:Release>:-Wl,--strip-all>
    )
endif()

if(APPLE)
	# The emulator writes x86-64 trampolines/PLT stubs into memory and executes them, and
	# runs guest code under Rosetta. That requires the JIT / unsigned-executable-memory
	# entitlements, so re-sign the binary after every link (an unsigned/relinked binary
	# reverts to the hardened defaults and aborts when it first executes written code).
	add_custom_command(TARGET kyty_emulator POST_BUILD
		COMMAND codesign -s - --force --options runtime
			--entitlements "${KYTY_SOURCE_DIR}/macos_jit.entitlements"
			$<TARGET_FILE:kyty_emulator>
		COMMENT "Codesign kyty_emulator with JIT entitlements (macOS)")
endif()

install(TARGETS kyty_emulator DESTINATION .)
if(KYTY_BUILD_LAUNCHER)
	add_subdirectory("${KYTY_SOURCE_DIR}/launcher" "${CMAKE_CURRENT_BINARY_DIR}/launcher")

	if(LINUX)
		target_link_options(launcher PRIVATE $<$<CONFIG:Release>:-Wl,--strip-all>)
		target_link_options(kyty_updater PRIVATE $<$<CONFIG:Release>:-Wl,--strip-all>)
	endif()
endif()
if(KYTY_CLANG_CL)
	install(FILES "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" DESTINATION .)
endif()
