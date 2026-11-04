This is a fork build of KytyPS5. It includes [KytyPS5/KytyPS5](https://github.com/KytyPS5/KytyPS5) at `07e87b95`, plus the open pull requests listed below, each squashed into one commit, and an Experiments tab in the launcher. None of these changes are merged upstream as of this build.

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
| GPU submit cadence | `--gpu-submit-cadence` | Submits the GPU work recorded so far about a millisecond after it starts recording draws, while a long command stream records, so the GPU starts on it instead of waiting for the whole stream. |
| Native float min/max | `--native-float-minmax` | On non-AMD GPUs, uses the GPU's own float min/max in shaders instead of the exact emulation, which is faster. The sign of a mixed-zero result and the NaN picked from two NaNs can differ from the PS5. |
| Batch hot readbacks | `--batch-hot-readbacks` | When the game has to wait for GPU-written data, also reads back recently read-back buffers and finished small GPU writes in the same wait, so the next reads do not wait again. |
| Readback fault wait | `--readback-fault-wait` | Makes the game wait for a pending linear image readback instead of reading old data, which fixes flickering objects in Kena: Bridge of Spirits. Does nothing unless **Enable readback** is on in the game's config. |

Measured on an RTX 5090 in Atelier Yumia with both shader experiments on: its jump-table pipelines went from 96–141 s to 4.4–6.6 s (single-digit seconds with structured jump tables alone), and its largest compute shader from 31.6 s to 16.9 s. These were measured before upstream started relying on robustBufferAccess2 for storage buffer bounds, which removes many of the same branches, so the gain from Selected guarded loads is likely smaller in this build. Other games and GPUs have not been measured.

Measured in LEGO Batman: Legacy of the Dark Knight (PPSA16833) before its title screen, per 2 seconds. Keep GPU cache on reuse cut buffer uploads from 4.9–7.2 GB to under 0.5 GB and raised the frame rate from about 24 to 28–30 fps. Keep GPU textures on reuse and Partial buffer refresh then cut texture uploads from 6–9 GB to 1.6–2.0 GB in the steady part of the scene. The scene appears capped at 30 fps, so the later gains show up as steadier frame times rather than a higher frame rate.

Measured on LEGO Batman's main menu on an RTX 5090 at 2560x1440. Native float min/max took it from 21.0 to 25.3 fps, and GPU submit cadence on top to 27.1 fps. With Batch hot readbacks and the colour-clear and texture-lookup changes (KytyPS5/KytyPS5#2094, KytyPS5/KytyPS5#2095) it reached about 29 fps, with about 97% of frames on the 30 fps pace. With GPU submit cadence and Native float min/max on, Batch hot readbacks alone is worth only about 0.6 fps against a build without its code. These were measured on measurement builds of each change, not on this exact build.

Measured in Kena: Bridge of Spirits (PPSA01746), without the other experiments, in single runs. Readback fault wait removed the flickering objects, and the 30-second average went from 31.6 fps to 29.0 and 29.8 fps. Waits averaged about 0.6 ms a frame, which explains only part of that cost. Astro Bot's resolution and anti-aliasing, Octopath Traveler II's lighting, and the frame rates of Astro Bot, LEGO Batman and Atelier Yumia showed no regression, but no game thread ever waited in those games. These were measured on a build before this base, not on this exact build, and not together with the other experiments.

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
- KytyPS5/KytyPS5#2091: Adds the **GPU submit cadence** experiment. The GPU thread submits the command buffer it is recording about 1 ms after its first draw or dispatch, at a command-packet boundary, so the GPU works on earlier draws while the rest records. On LEGO Batman's main menu it raised the frame rate from 19.8 to 24.3 fps when measured on its own, in separate runs.
- KytyPS5/KytyPS5#2093: Adds the **Batch hot readbacks** experiment. A readback that waits for the GPU also downloads the GPU-written bytes of recently read-back buffers and finished small GPU writes, so the reads that follow find clean memory. On LEGO Batman's main menu, with GPU submit cadence and Native float min/max on, slow frames dropped from about 8% to 6% against a build without its code.
- KytyPS5/KytyPS5#2094: Checks a multi-slice DCC texture's fast-clear codes with one guest-memory read and one checked-range lookup instead of one of each per slice, with the same results. On LEGO Batman's main menu the check went from about 90–98 to 40 ms of GPU-thread time per second, part of which was profiling overhead in the earlier measurement.
- KytyPS5/KytyPS5#2095: Texture overlap lookups test each image's range from a span table instead of reading the image objects, and test each image once. Results and their order are unchanged, and lookups are a little faster.
- KytyPS5/KytyPS5#2102: DMA copies whose source and destination overlap now give memmove's result on both copy paths. Such a copy used to abort the emulator with "overlapping self-copy" when it took the GPU path, for example over GPU-written memory, and depended on the C library otherwise.
- KytyPS5/KytyPS5#2120: Adds the **Readback fault wait** experiment, stacked on KytyPS5/KytyPS5#1928. A pending readback of a small linear image keeps its guest pages blocked until its data lands, so a game that reads them waits for the data while labels stay on time, which removes Kena's flickering objects (PPSA01746).

### Shaders
- KytyPS5/KytyPS5#1977: Adds the **Selected guarded loads** experiment. Buffer loads guarded by the active-lane mask, and formatted loads that may run past their range, run unconditionally and select the result instead of branching around the load. Each speculative load is kept inside the bound range. On GPUs with robustBufferAccess2, upstream already drops the per-dword bounds branches, so the experiment now does less there.
- KytyPS5/KytyPS5#1978: Adds the **Structured jump tables** experiment. Shaders with `S_SETPC_B64` jump tables compile to structured control flow instead of the dispatcher loop, so Atelier Yumia no longer freezes for minutes the first time it meets them.
- KytyPS5/KytyPS5#2092: Adds the **Native float min/max** experiment. On non-AMD GPUs that preserve signed zeros and NaNs, shader MIN, MAX, MIN3 and MAX3 use the GPU's own float min/max instead of compares and selects. LEGO Batman's two heaviest main-menu compute shaders run 34% and 53% faster on an RTX 5090.

### Audio
- KytyPS5/KytyPS5#1853: Implements the NGS2 sampler voice peak meter, which Trails in the Sky uses for lip-sync. The game no longer stops at startup with "unsupported sampler control" (PPSA28465).
- KytyPS5/KytyPS5#1856: Supports NGS2 submixer compressors with side-chain input, so music ducks while characters speak (PPSA28465).
- KytyPS5/KytyPS5#1865: Implements the NGS2 I3DL2 reverb module that Trails in the Sky builds its submixer racks around, so rooms and caves get their reverb (PPSA28465). Upstream's own reverb only runs on the stock reverb rack, so this module would otherwise still be ignored.
- KytyPS5/KytyPS5#1979: Adds the **Audio compile hold** experiment. Guest audio waits while a pipeline compile has blocked the GPU command thread for at least 250 ms.

### Kernel
- KytyPS5/KytyPS5#2065: On Windows, unmapping part of a memory mapping no longer drops the protection on the parts that stay mapped. Read-only guest memory stays read-only, and CPU writes there keep reaching cached GPU buffers and textures.
- KytyPS5/KytyPS5#2066: Adds the **Keep GPU cache on reuse** experiment. When the game releases direct memory and maps the same memory back at the same address, the GPU buffer cache is kept and its write tracking re-armed, instead of everything being uploaded again.
- KytyPS5/KytyPS5#2078: On Windows, a partial unmap now records the surviving pieces of a mapping before it unmaps the old view, so the emulator's own reads of the surviving memory in between no longer fail. LEGO Batman reached its main menu in 4–5 runs without stopping with "failed to read color metadata backing", which it hit in about one run in three before.

### Build and tests
- KytyPS5/KytyPS5#1966: Fixes the `vdecsw_tests` build on Windows, where `std::aligned_alloc` is missing, and runs vdecsw in CI.

## Known issues
- A game's first run still freezes while large shaders compile. Later runs load them from the pipeline cache.
- Turning either shader experiment, or Native float min/max, on or off recompiles the game's shaders, so the first-run freezes come back once.
- Cutscenes can still skip short segments during brief shader compiles, which are too short for Audio compile hold to pause audio.
- Audio compile hold has only been tested in Atelier Yumia.
- Structured jump tables: jump tables inside linked `S_SWAPPC` call graphs can still fall back to the dispatcher.
- When a cached buffer that shaders read only through BDA is released, the first read can return zeros for about a frame.
- Freeing one of two depth images that share depth metadata (HTILE) memory can clear depth the other still needs.
- With image readbacks turned on, a readback lands shortly after the end-of-pipe label rather than before it. A game that reads the image straight after the label can still see old data, and a game write to that memory before the readback lands is overwritten. Readback fault wait removes this for linear images with one mip level, layer and depth slice, up to 1 MiB, on their whole pages. Other images, including tiled ones, behave as before.
- Keep GPU cache on reuse, Keep GPU textures on reuse and Partial buffer refresh have only been measured in LEGO Batman. Astro Bot, Trails in the Sky and Octopath Traveler II showed no problems with them on, but they never take the reuse paths.
- Partial buffer refresh still re-uploads depth targets in full.
- With image readbacks turned on, a texture that Keep GPU textures on reuse kept across a release can lose a pending readback, so guest memory for it stays out of date until it is read back again. Without the experiment that texture is freed and its contents lost, so this is no worse than before. The same applies to a texture that Partial buffer refresh is still waiting to refresh.
- GPU submit cadence, Native float min/max and Batch hot readbacks have only been measured on LEGO Batman's main menu on an RTX 5090 under Windows. Astro Bot, Octopath Traveler II and Trails in the Sky showed no visible differences with either of the first two on.
- With GPU submit cadence on, texture ages are counted in frames at the game's suspend points. A game that submits for a long time without one does not age its textures, so they stay cached longer.
- On Windows, a game's CPU access to memory that a partial unmap is splitting can still fault for a moment, as before. The emulator's own reads are fixed, and its writes were already safe.
- With GPU submit cadence on, Batch hot readbacks remembers buffers for fewer frames, because it counts its age in GPU submits and the cadence adds many submits per frame. On LEGO Batman's main menu it still covers more than 10 frames.
- Readback fault wait has only been seen to make a game wait in Kena, only on Windows. If a wait makes no progress for 2 seconds it gives up and the game reads old data, and after three of those in a row it stops for the session. On Linux and macOS it blocks inside the signal handler, which has not been run.
- With Readback fault wait on, a game's fault that no longer has a cause is retried without the normal fault handling, on any mapped page.
- Native float min/max has no effect on AMD GPUs, which already use the GPU's min/max, and is ignored on Apple GPUs, MoltenVK and GPUs that do not preserve signed zeros and NaNs.
