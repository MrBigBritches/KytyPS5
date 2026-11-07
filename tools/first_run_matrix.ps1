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

.PARAMETER Emulator
Path of kyty_emulator. Its folder is the working directory, so reports and caches live there.
#>
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [Parameter(Mandatory = $true)][string[]]$Arms,
    [string]$TitleId = "",
    [string]$Emulator = (Join-Path $PSScriptRoot "..\_Build\windows\install\kyty_emulator.exe")
)

$ErrorActionPreference = "Stop"

# The one table of arms. Flags are what the arm adds to the command line. An arm is Ready only when
# the emulator has its flags, and a table entry that is not Ready stops the script before any file
# is touched. Set Ready and fix the flag names here once the strategies exist.
$ArmTable = [ordered]@{
    "baseline"       = @{ Flags = @();                                                                                     Ready = $true }
    "async-graphics" = @{ Flags = @("--async-graphics-pipelines");                                                         Ready = $false }
    "gpl"            = @{ Flags = @("--graphics-pipeline-library");                                                        Ready = $false }
    "lookahead"      = @{ Flags = @("--shader-lookahead");                                                                 Ready = $false }
    "all"            = @{ Flags = @("--async-graphics-pipelines", "--graphics-pipeline-library", "--shader-lookahead");    Ready = $false }
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
$process = $null
$originalsMoved = $false
$leftBehind = $false
try {
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    Move-CacheFiles $cacheDir $TitleId $originalDir
    $originalsMoved = $true
    for ($index = 0; $index -lt $Arms.Count; $index++) {
        $arm = $Arms[$index]
        $arguments = @("--game", $Game) + $ArmTable[$arm].Flags + @("--first-run-report", "--cold-pipelines")
        Write-Host ""
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
