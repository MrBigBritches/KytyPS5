<#
.SYNOPSIS
Runs a game once per first-run arm and collects a report for each.

.DESCRIPTION
For each arm the script moves the title's _PipelineCache files aside, launches kyty_emulator with
--first-run-report --cold-pipelines and the arm's flags, asks the player to play the route and quit
normally, then moves the files that run made into a hold folder. The title's original caches are
put back at the end. Nothing is ever deleted or overwritten. If the script is interrupted while the
emulator is still running, it waits for the emulator to exit before it touches any file.

.PARAMETER Game
Path given to --game, either the game folder or its eboot.

.PARAMETER Arms
Names of the arms to run, from the table below. The same arm may be listed twice for two runs.

.PARAMETER TitleId
Name used to find the cache files. Read from the game's sce_sys/param.json when left out.

.PARAMETER BaseArgsFile
Text file holding the settings every arm shares, one argument per line. Blank lines are ignored, so
a value can't be empty. Leave such a flag out instead. The launcher passes resolution, tessellation,
readback settings and experiments, so without these a run is not comparable with normal play.

.PARAMETER Emulator
Path of kyty_emulator. Its folder is the working directory, so reports and caches live there.

.PARAMETER SeedFolder
Optional folder of cache files for the title, such as a shader store made offline from the game's
files. They are copied into the empty cache folder before each arm, so every arm starts from them
instead of nothing. A cold run reads only the shader store, so a seeded driver cache is ignored.
The emulator rewrites the store when it quits, so the hold folder keeps the store as each arm left
it, not the seed. The seed folder itself is never changed, and it can't be the cache folder.
#>
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [Parameter(Mandatory = $true)][string[]]$Arms,
    [Parameter(Mandatory = $true)][string]$BaseArgsFile,
    [string]$TitleId = "",
    [string]$Emulator = (Join-Path $PSScriptRoot "..\_Build\windows\install\kyty_emulator.exe"),
    [string]$SeedFolder = ""
)

$ErrorActionPreference = "Stop"

# The one table of arms. Flags are what the arm adds to the command line. An arm is Ready only when
# the emulator has its flags, and a table entry that is not Ready stops the script before any file
# is touched. Set Ready and fix the flag names here once the strategies exist.
$ArmTable = [ordered]@{
    "baseline"       = @{ Flags = @();                                                                                     Ready = $true }
    "async-graphics" = @{ Flags = @("--async-pipelines");                                                                  Ready = $true }
    "async-depth-only" = @{ Flags = @("--async-pipelines-depth-only");                                                    Ready = $true }
    "warm-up"        = @{ Flags = @("--background-warm-up");                                                               Ready = $true }
    "async-warm-up"  = @{ Flags = @("--async-pipelines", "--background-warm-up");                                         Ready = $true }
    "deferred"       = @{ Flags = @("--deferred-pipeline-draws");                                                          Ready = $true }
    "deferred-warm-up" = @{ Flags = @("--deferred-pipeline-draws", "--background-warm-up");                               Ready = $true }
    "deferred-pause" = @{ Flags = @("--deferred-pipeline-draws", "--background-warm-up", "--pause-warm-up-for-draws");   Ready = $true }
    "deferred-non-push" = @{ Flags = @("--deferred-pipeline-draws", "--background-warm-up", "--pause-warm-up-for-draws", "--deferred-non-push-draws"); Ready = $true }
    "gpl"            = @{ Flags = @("--deferred-pipeline-draws", "--background-warm-up", "--pause-warm-up-for-draws", "--deferred-non-push-draws", "--graphics-pipeline-library"); Ready = $true }
    # The gpl arm plus compute shaders generated from the eboot by a helper process. It needs
    # --shader-cache, which the base args give. The helper's files match the title's cache names,
    # so they move into the hold folder with the rest.
    "gen"            = @{ Flags = @("--deferred-pipeline-draws", "--background-warm-up", "--pause-warm-up-for-draws", "--deferred-non-push-draws", "--graphics-pipeline-library", "--generate-shader-store"); Ready = $true }
    "lookahead"      = @{ Flags = @("--shader-lookahead");                                                                 Ready = $false }
}

# The gen arm with the warm-up held to 3 and to 2 workers, to see how much the warm builds slow the
# graphics compiles at the loading burst. The count and its flag are separate elements, because
# each element becomes one argument. The report keeps the count, so these runs stay apart from gen.
$ArmTable["gen-w3"] = @{ Flags = @($ArmTable["gen"].Flags + @("--warm-workers", "3")); Ready = $true }
$ArmTable["gen-w2"] = @{ Flags = @($ArmTable["gen"].Flags + @("--warm-workers", "2")); Ready = $true }

# The gen arm with the submit-time compute prescan on, so it differs from gen in one flag only.
$ArmTable["gen-prescan"] = @{
    Flags = @($ArmTable["gen"].Flags + @("--prescan-compute-builds"))
    Ready = $true
}

# The gen-prescan arm with pixel shaders and their fragment libraries built ahead from the scanner
# (the experimental graphics prescan), so it differs from gen-prescan in one flag only.
$ArmTable["gen-prescan-gpl"] = @{
    Flags = @($ArmTable["gen-prescan"].Flags + @("--prescan-graphics-libraries"))
    Ready = $true
}

# The gen-prescan-gpl arm where a draw the skip rules allow is skipped instead of deferred, so it
# differs from gen-prescan-gpl in one flag only. The flag implies async pipelines, which the gen arms
# leave off, so a comparison with gen-prescan-gpl measures both together.
$ArmTable["gen-prescan-gpl-skip"] = @{
    Flags = @($ArmTable["gen-prescan-gpl"].Flags + @("--async-pipelines-prefer-skip"))
    Ready = $true
}

# "all" turns on every strategy, so it is Ready only once every strategy arm is, and its flags are
# theirs together. The depth-only arm is left out, because it is the default skip rule with a
# stricter gate. Both flags can be on at once, and then depth-only wins, so "all" would measure the
# depth-only mode instead of the default one. The gpl arm carries the deferred-non-push flags,
# because GPL holes need them, so "all" gets the deferred strategies through it.
$StrategyArms = @("async-graphics", "warm-up", "gpl", "lookahead")
$ArmTable["all"] = @{
    Flags = @($StrategyArms | ForEach-Object { $ArmTable[$_].Flags })
    Ready = @($StrategyArms | Where-Object { -not $ArmTable[$_].Ready }).Count -eq 0
}

# Reads the name the emulator uses for its caches, which is TITLE_ID or else the whole CONTENT_ID.
function Get-GameTitleId([string]$GameFolder) {
    $paramFile = Join-Path $GameFolder "sce_sys\param.json"
    if (-not (Test-Path -LiteralPath $paramFile)) { return "" }
    $text = Get-Content -LiteralPath $paramFile -Raw
    if ($text -match '"titleId"\s*:\s*"([A-Za-z0-9_-]+)"') { return $Matches[1] }
    if ($text -match '"contentId"\s*:\s*"([A-Za-z0-9_-]+)"') { return $Matches[1] }
    return ""
}

# Lists the title's cache files in a folder. A folder that does not exist has none.
function Get-CacheFiles([string]$Folder, [string]$Id) {
    if (-not (Test-Path -LiteralPath $Folder)) { return @() }
    return @(Get-ChildItem -LiteralPath $Folder -File | Where-Object { $_.Name -like "$Id.*" })
}

# Moves every cache file of the title into $Destination without replacing anything there.
function Move-CacheFiles([string]$Folder, [string]$Id, [string]$Destination) {
    $files = Get-CacheFiles $Folder $Id
    if ($files.Count -eq 0) { return }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($file in $files) {
        Move-Item -LiteralPath $file.FullName -Destination (Join-Path $Destination $file.Name)
    }
}

# Lists the running emulator processes by path. A relaunching title starts a child and quits, so
# the process that was started is not enough to know the game has ended.
function Get-EmulatorProcesses([string]$Path) {
    return @(Get-Process | Where-Object { $_.Path -eq $Path })
}

# Quotes one argument for a Windows command line.
function Format-Argument([string]$Value) {
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '"', '\"') + '"'
}

foreach ($arm in $Arms) {
    if (-not $ArmTable.Contains($arm)) {
        throw "Unknown arm '$arm'. Known arms: $($ArmTable.Keys -join ', ')"
    }
    if (-not $ArmTable[$arm].Ready) {
        throw "Arm '$arm' is not implemented yet: the emulator has no flags for it. Nothing was started."
    }
}
if (-not (Test-Path -LiteralPath $Emulator)) { throw "Emulator not found: $Emulator" }
$Emulator = (Resolve-Path -LiteralPath $Emulator).Path
$Game = (Resolve-Path -LiteralPath $Game).Path
$gameFolder = if (Test-Path -LiteralPath $Game -PathType Container) { $Game } else { Split-Path -Parent $Game }
if (-not $TitleId) { $TitleId = Get-GameTitleId $gameFolder }
if (-not $TitleId) { throw "Could not read the title id from the game. Pass -TitleId." }

$workDir = Split-Path -Parent $Emulator
$cacheDir = Join-Path $workDir "_PipelineCache"
$reportDir = Join-Path $workDir "_FirstRunReports"

# The seed is checked before any file moves. Seeding from the cache folder would seed from the
# originals the script is about to move aside, so that is refused.
$seedFiles = @()
if ($SeedFolder) {
    $SeedFolder = (Resolve-Path -LiteralPath $SeedFolder).ProviderPath
    if ([IO.Path]::GetFullPath($SeedFolder).TrimEnd('\') -ieq [IO.Path]::GetFullPath($cacheDir).TrimEnd('\')) {
        throw "The seed folder can't be the cache folder itself. Nothing was started."
    }
    $seedFiles = Get-CacheFiles $SeedFolder $TitleId
    if ($seedFiles.Count -eq 0) { throw "$SeedFolder holds no cache files for $TitleId. Nothing was started." }
}

# Originals still sitting in an old hold folder mean an earlier run did not restore them.
if (Test-Path -LiteralPath $cacheDir) {
    foreach ($old in Get-ChildItem -LiteralPath $cacheDir -Directory | Where-Object { $_.Name -like "_matrix-*" }) {
        $left = Get-CacheFiles (Join-Path $old.FullName "original") $TitleId
        if ($left.Count -gt 0) {
            throw "An earlier run left the original caches in $($old.FullName)\original. Put them back by hand first."
        }
    }
}

$holdDir = Join-Path $cacheDir ("_matrix-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
$originalDir = Join-Path $holdDir "original"
$baseArguments = @(Get-Content -LiteralPath $BaseArgsFile -Encoding UTF8 | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
if ($baseArguments.Count -eq 0) {
    throw "$BaseArgsFile holds no arguments, so a run would not match normal play."
}
$process = $null
$originalsMoved = $false
$leftBehind = $false
try {
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    Move-CacheFiles $cacheDir $TitleId $originalDir
    $originalsMoved = $true
    for ($index = 0; $index -lt $Arms.Count; $index++) {
        $arm = $Arms[$index]
        $arguments = $baseArguments + @("--game", $Game) + $ArmTable[$arm].Flags + @("--first-run-report", "--cold-pipelines")
        # The cache folder holds none of the title's files here, so a copy never replaces one.
        foreach ($seed in $seedFiles) {
            Copy-Item -LiteralPath $seed.FullName -Destination (Join-Path $cacheDir $seed.Name)
        }
        Write-Host ""
        if ($seedFiles.Count -gt 0) { Write-Host "Seeded from ${SeedFolder}: $($seedFiles.Name -join ', ')" }
        Write-Host "Arm '$arm': $($arguments -join ' ')"
        Read-Host "Press Enter to launch, then play the route and quit the game normally"
        $started = Get-Date
        $process = Start-Process -FilePath $Emulator -ArgumentList (($arguments | ForEach-Object { Format-Argument $_ }) -join " ") `
            -WorkingDirectory $workDir -PassThru
        $process.WaitForExit()
        while ((Get-EmulatorProcesses $Emulator).Count -gt 0) { Start-Sleep -Seconds 1 }
        Write-Host "Emulator exit code: $($process.ExitCode)"
        if ($process.ExitCode -ne 0) {
            Write-Warning "The emulator exited with $($process.ExitCode). Check the flags and the game."
        }
        $made = @()
        if (Test-Path -LiteralPath $reportDir) {
            $made = @(Get-ChildItem -LiteralPath $reportDir -File -Filter "$TitleId-*.json" |
                Where-Object { $_.LastWriteTime -ge $started })
        }
        if ($made.Count -eq 0) {
            Write-Warning "No report appeared for '$arm'. The game may not have quit normally."
        } else {
            Write-Host "Report: $($made[-1].FullName)"
        }
        # The next arm must start with no cache, so this run's files wait in the hold folder.
        Move-CacheFiles $cacheDir $TitleId (Join-Path $holdDir "$index-$arm")
    }
} catch {
    Write-Host "Failed: $($_.Exception.Message)"
    Write-Host "Cache files are in the hold folder: $holdDir"
    throw
} finally {
    # Ctrl+C stops the wait but not the emulator, and it saves its cache when it quits, so nothing
    # is touched until every emulator process has gone, including a relaunched child.
    while ((Get-EmulatorProcesses $Emulator).Count -gt 0) {
        Write-Warning "The emulator is still running. Quit the game, then press Enter."
        Read-Host | Out-Null
    }
    # Only after the originals were moved is anything in the cache folder not theirs. Sweeping
    # earlier could file an original under "interrupted". A failed sweep must not stop the restore.
    if ($originalsMoved) {
        try {
            Move-CacheFiles $cacheDir $TitleId (Join-Path $holdDir "interrupted")
        } catch {
            $leftBehind = $true
            Write-Warning "Could not sweep leftover cache files: $($_.Exception.Message)"
        }
    }
    # Restore always runs, even when the first move failed part way and left some files here.
    foreach ($file in @(Get-CacheFiles $originalDir $TitleId)) {
        $target = Join-Path $cacheDir $file.Name
        if (Test-Path -LiteralPath $target) {
            $leftBehind = $true
            Write-Warning "$target exists, so the original stays in $originalDir"
        } else {
            try {
                Move-Item -LiteralPath $file.FullName -Destination $target
            } catch {
                $leftBehind = $true
                Write-Warning "Could not restore $($file.Name): $($_.Exception.Message)"
            }
        }
    }
    if ($leftBehind) {
        Write-Warning "Some caches were NOT restored. Everything is in $holdDir"
    } else {
        Write-Host "Caches for $TitleId restored from $holdDir"
    }
}

Write-Host ""
Write-Host "Compare with: python tools/first_run_compare.py `"$reportDir\$TitleId-*.json`""
