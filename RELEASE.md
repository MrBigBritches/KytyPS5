This is a fork build of KytyPS5. It includes [KytyPS5/KytyPS5](https://github.com/KytyPS5/KytyPS5) at `5e837a9b`, plus the open pull requests listed below, each squashed into one commit, and an Experiments tab in the launcher. None of these changes are merged upstream as of this build.

## 🧪 Experiments

These are opt-in features from unmerged pull requests. Turn them on or off in the **Experiments** tab of **Global settings**, or per game under **Edit game config**. In this build they are all on by default.

| Experiment | Emulator flag | What it does |
|---|---|---|
| Audio compile hold | `--audio-compile-hold` | Pauses game audio while a long shader compile freezes the game. Cutscenes that follow the audio clock then mostly stay in step instead of skipping ahead. |
| Structured jump tables | `--structured-jump-tables` | Compiles shader jump tables (`S_SETPC_B64`) as structured control flow instead of a dispatcher loop, which NVIDIA's driver builds in seconds instead of minutes. |
| Selected guarded loads | `--select-guarded-loads` | Emits bounds-checked shader loads as selects instead of branches, which NVIDIA's driver compiles faster. |

Measured on an RTX 5090 in Atelier Yumia with both shader experiments on: its jump-table pipelines went from 96–141 s to 4.4–6.6 s (single-digit seconds with structured jump tables alone), and its largest compute shader from 31.6 s to 16.9 s. Other games and GPUs have not been measured.

## 🔀 Included pull requests

### Graphics
- KytyPS5/KytyPS5#1781: Fixes the texture cache reusing a cached texture when the game asks for the same address with a different number of mip levels. Textures no longer come out corrupt when a shorter mip chain reuses that memory (PPSA09809).
- KytyPS5/KytyPS5#1803: Only treats a compute dispatch as a depth-metadata (HTILE) clear when it really is one, so shaders that write to reused HTILE memory run normally. Models no longer vanish in Octopath Traveler II, which reuses that memory for its GPUScene buffer (PPSA09809).
- KytyPS5/KytyPS5#1932: BDA synchronization now uploads all of a draw's CPU-written pages through one staging copy between a single pair of barriers, instead of one copy and two barriers per buffer, so each synchronization costs less. Together with #1933, a heavy Octopath Traveler II scene went from about 50 to 55 fps.
- KytyPS5/KytyPS5#1933: Holds back writes from game threads until the command processor resumes, instead of uploading them at the next BDA draw. In frames with many BDA draws, pages the game keeps writing are uploaded and faulted on about a third less often, and Octopath Traveler II's draw-heavy bursts reach the 60 fps cap.
- KytyPS5/KytyPS5#1934: Releases cached buffers that no binding has used for a long time, even without memory pressure. Writes to memory the game has reused mostly stop faulting into buffers nothing reads.
- KytyPS5/KytyPS5#1928: With `--readback-tiled-images` or `--readback-linear-images`, image readbacks now land before the end-of-pipe label the game waits on. In Octopath Traveler II the scene no longer stays in shade because the ambient lighting was read before it was copied.

### Shaders
- KytyPS5/KytyPS5#1977: Adds the **Selected guarded loads** experiment. Bounds-checked buffer loads run unconditionally and select the result, instead of branching around the load, and each speculative load is kept inside the bound range.
- KytyPS5/KytyPS5#1978: Adds the **Structured jump tables** experiment. Shaders with `S_SETPC_B64` jump tables compile to structured control flow instead of the dispatcher loop, so Atelier Yumia no longer freezes for minutes the first time it meets them.

### Audio
- KytyPS5/KytyPS5#1853: Implements the NGS2 sampler voice peak meter, which Trails in the Sky uses for lip-sync. The game no longer stops at startup with "unsupported sampler control" (PPSA28465).
- KytyPS5/KytyPS5#1856: Supports NGS2 submixer compressors with side-chain input, so music ducks while characters speak (PPSA28465).
- KytyPS5/KytyPS5#1865: Implements the NGS2 I3DL2 reverb module that Trails in the Sky builds its submixer racks around, so rooms and caves get their reverb (PPSA28465).
- KytyPS5/KytyPS5#1979: Adds the **Audio compile hold** experiment. Guest audio waits while a pipeline compile has blocked the GPU command thread for at least 250 ms.

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
