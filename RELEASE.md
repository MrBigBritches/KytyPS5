This is a fork build of KytyPS5. It includes [KytyPS5/KytyPS5](https://github.com/KytyPS5/KytyPS5) at `02b17af3`, plus the open pull requests listed below, each squashed into one commit, and an Experiments tab in the launcher. None of these changes are merged upstream as of this build.

## 🧪 Experiments

These are opt-in features from unmerged pull requests. Turn them on or off in the **Experiments** tab of **Global settings**, or per game under **Edit game config**. In this build they are all on by default.

| Experiment | Emulator flag | What it does |
|---|---|---|
| Audio compile hold | `--audio-compile-hold` | Pauses game audio while a long shader compile freezes the game. Cutscenes that follow the audio clock then mostly stay in step instead of skipping ahead. |
| Structured jump tables | `--structured-jump-tables` | Compiles shader jump tables (`S_SETPC_B64`) as structured control flow instead of a dispatcher loop, which NVIDIA's driver builds in seconds instead of minutes. |
| Selected guarded loads | `--select-guarded-loads` | Replaces the active-lane branch around buffer loads, and the bounds branch around formatted buffer loads, with selects, which NVIDIA's driver compiles faster. On GPUs without robustBufferAccess2 it also turns per-dword buffer bounds checks into selects. |
| Skip redundant unmap drains | `--skip-redundant-unmap-drains` | Skips the GPU queue drain when the game unmaps memory that is already unmapped and holds nothing the GPU still uses. |
| Keep GPU cache on reuse | `--keep-gpu-cache-on-reuse` | Keeps the GPU buffer cache when the game releases memory and maps the same memory back at the same address, instead of uploading it all again. |
| Keep GPU textures on reuse | `--keep-gpu-textures-on-reuse` | Also keeps clean textures across that release and remap. It only has an effect together with Keep GPU cache on reuse. |
| Partial buffer refresh | `--partial-buffer-refresh` | When a shader writes a few bytes at the start of a texture through a buffer, refreshes only those bytes instead of re-uploading the whole texture. |

Measured on an RTX 5090 in Atelier Yumia with both shader experiments on: its jump-table pipelines went from 96–141 s to 4.4–6.6 s (single-digit seconds with structured jump tables alone), and its largest compute shader from 31.6 s to 16.9 s. These were measured before upstream started relying on robustBufferAccess2 for storage buffer bounds, which removes many of the same branches, so the gain from Selected guarded loads is likely smaller in this build. Other games and GPUs have not been measured.

Measured in LEGO Batman: Legacy of the Dark Knight (PPSA16833) before its title screen, per 2 seconds. Keep GPU cache on reuse cut buffer uploads from 4.9–7.2 GB to under 0.5 GB and raised the frame rate from about 24 to 28–30 fps. Keep GPU textures on reuse and Partial buffer refresh then cut texture uploads from 6–9 GB to 1.6–2.0 GB in the steady part of the scene. The scene appears capped at 30 fps, so the later gains show up as steadier frame times rather than a higher frame rate.

## 🔀 Included pull requests

### Graphics
- KytyPS5/KytyPS5#1781: Fixes the texture cache reusing a cached texture when the game asks for the same address with a different number of mip levels. Textures no longer come out corrupt when a shorter mip chain reuses that memory (PPSA09809).
- KytyPS5/KytyPS5#1803: Only treats a compute dispatch as a depth-metadata (HTILE) clear when it really is one, so shaders that write to reused HTILE memory run normally. Models no longer vanish in Octopath Traveler II, which reuses that memory for its GPUScene buffer (PPSA09809).
- KytyPS5/KytyPS5#1932: BDA synchronization now uploads all of a draw's CPU-written pages through one staging copy between a single pair of barriers, instead of one copy and two barriers per buffer, so each synchronization costs less. Together with KytyPS5/KytyPS5#1933, a heavy Octopath Traveler II scene went from about 50 to 55 fps.
- KytyPS5/KytyPS5#1933: Holds back writes from game threads until the command processor resumes, instead of uploading them at the next BDA draw. In frames with many BDA draws, pages the game keeps writing are uploaded and faulted on about a third less often, and Octopath Traveler II's draw-heavy bursts reach the 60 fps cap.
- KytyPS5/KytyPS5#1934: Releases cached buffers that no binding has used for a long time, even without memory pressure. Writes to memory the game has reused mostly stop faulting into buffers nothing reads.
- KytyPS5/KytyPS5#1928: With `--readback-tiled-images` or `--readback-linear-images`, each end-of-pipe label now queues and submits the pending image readbacks, instead of waiting for the next garbage-collector pass. Octopath Traveler II's ambient lighting now reaches the game, so Throné's opening scene no longer stays in shade. Labels are not held back for the readbacks, which keeps Astro Bot's resolution.
- KytyPS5/KytyPS5#2063: DMA fills and copies into memory the GPU hasn't written now invalidate the whole destination range before the CPU write, instead of faulting once per page. In LEGO Batman this removed about 7,500 faults a second from the GPU command thread.
- KytyPS5/KytyPS5#2064: Adds the **Skip redundant unmap drains** experiment. When a fixed remap unmaps a range the release already unmapped, the second unmap no longer waits for the whole GPU queue.
- Branch `keep-gpu-textures-on-reuse` (not yet a pull request, builds on KytyPS5/KytyPS5#2066): Adds the **Keep GPU textures on reuse** experiment. Clean textures in a released range survive an exact remap instead of being rebuilt and uploaded again, and render targets no longer lose their GPU-written contents across it.
- KytyPS5/KytyPS5#2067: Adds the **Partial buffer refresh** experiment. A storage-buffer write at a texture's base refreshes only the bytes it covered, including on tiled and DCC/CMASK colour textures, and the texture keeps its own contents elsewhere instead of being overwritten with stale bytes.

### Shaders
- KytyPS5/KytyPS5#1977: Adds the **Selected guarded loads** experiment. Buffer loads guarded by the active-lane mask, and formatted loads that may run past their range, run unconditionally and select the result instead of branching around the load. Each speculative load is kept inside the bound range. On GPUs with robustBufferAccess2, upstream already drops the per-dword bounds branches, so the experiment now does less there.
- KytyPS5/KytyPS5#1978: Adds the **Structured jump tables** experiment. Shaders with `S_SETPC_B64` jump tables compile to structured control flow instead of the dispatcher loop, so Atelier Yumia no longer freezes for minutes the first time it meets them.

### Audio
- KytyPS5/KytyPS5#1853: Implements the NGS2 sampler voice peak meter, which Trails in the Sky uses for lip-sync. The game no longer stops at startup with "unsupported sampler control" (PPSA28465).
- KytyPS5/KytyPS5#1856: Supports NGS2 submixer compressors with side-chain input, so music ducks while characters speak (PPSA28465).
- KytyPS5/KytyPS5#1865: Implements the NGS2 I3DL2 reverb module that Trails in the Sky builds its submixer racks around, so rooms and caves get their reverb (PPSA28465).
- KytyPS5/KytyPS5#1979: Adds the **Audio compile hold** experiment. Guest audio waits while a pipeline compile has blocked the GPU command thread for at least 250 ms.

### Kernel
- KytyPS5/KytyPS5#2065: On Windows, unmapping part of a memory mapping no longer drops the protection on the parts that stay mapped. Read-only guest memory stays read-only, and CPU writes there keep reaching cached GPU buffers and textures.
- KytyPS5/KytyPS5#2066: Adds the **Keep GPU cache on reuse** experiment. When the game releases direct memory and maps the same memory back at the same address, the GPU buffer cache is kept and its write tracking re-armed, instead of everything being uploaded again.

### Build and tests
- KytyPS5/KytyPS5#1966: Fixes the `vdecsw_tests` build on Windows, where `std::aligned_alloc` is missing, and runs vdecsw in CI.

## Known issues
- A game's first run still freezes while large shaders compile. Later runs load them from the pipeline cache.
- Turning either shader experiment on or off recompiles the game's shaders, so the first-run freezes come back once.
- Cutscenes can still skip short segments during brief shader compiles, which are too short for Audio compile hold to pause audio.
- Audio compile hold has only been tested in Atelier Yumia.
- Structured jump tables: jump tables inside linked `S_SWAPPC` call graphs can still fall back to the dispatcher.
- When a cached buffer that shaders read only through BDA is released, the first read can return zeros for about a frame.
- Freeing one of two depth images that share depth metadata (HTILE) memory can clear depth the other still needs.
- With image readbacks turned on, a readback lands shortly after the end-of-pipe label rather than before it. A game that reads the image straight after the label can still see old data, and a game write to that memory before the readback lands is overwritten.
- Keep GPU cache on reuse, Keep GPU textures on reuse and Partial buffer refresh have only been measured in LEGO Batman. Astro Bot, Trails in the Sky and Octopath Traveler II showed no problems with them on, but they never take the reuse paths.
- Partial buffer refresh still re-uploads depth targets in full.
- With image readbacks turned on, a texture that Keep GPU textures on reuse kept across a release can lose a pending readback, so guest memory for it stays out of date until it is read back again. Without the experiment that texture is freed and its contents lost, so this is no worse than before. The same applies to a texture that Partial buffer refresh is still waiting to refresh.
